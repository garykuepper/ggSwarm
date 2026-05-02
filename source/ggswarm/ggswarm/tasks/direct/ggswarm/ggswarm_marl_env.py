"""DirectMARLEnv for ggSwarm Phase 1+ shared-scene multi-drone training.

A=8 drones in one shared PhysX scene per env. CTDE: shared GNN actor across
all drones (decentralized execution), centralized critic at training time
(MAPPO). The capstone single-agent env (`GgswarmEnv` / `ggswarm-v0`) is
preserved alongside this for the replay-gate inputs and any backwards-compat.

Shape contract:
  - self.num_envs                    == real env count (DirectMARLEnv)
  - N_drones = num_envs * num_agents == flat drone instance count
  - self._robot.data.root_pos_w      : [N_drones, 3]   env-major
  - per-agent dict tensors           : [num_envs, *]   one entry per agent
  - per-drone scratch tensors        : [N_drones, *]
  - per-(env, agent) tensors         : [num_envs, num_agents, *]
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import subtract_frame_transforms

from .ggswarm_marl_env_cfg import GgswarmMarlEnvCfg


class GgswarmMarlEnv(DirectMARLEnv):
    cfg: GgswarmMarlEnvCfg

    def __init__(self, cfg: GgswarmMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        A = self.cfg.num_agents
        N_envs = self.num_envs
        N_drones = N_envs * A
        device = self.device

        if A < 2:
            raise ValueError(
                "GgswarmMarlEnv requires num_agents >= 2 (shared-scene). "
                "Use the v1.0.0-capstone tag for single-agent hover-only."
            )

        self._A = A
        self._N_drones = N_drones
        self._agent_ids = list(self.cfg.possible_agents)
        self._cloud_mode = self.cfg.formation_mode == "cloud"
        self._global_step = 0

        # Pair indices for formation pairwise distance error
        self._pair_indices: list[tuple[int, int]] = [(i, j) for i in range(A) for j in range(i + 1, A)]

        # Formation slot offsets — XYZ offset from env centroid per drone
        from ggswarm.formations import get_formation  # noqa: PLC0415

        spacing = self.cfg.formation_target_spacing
        radius = spacing / (2 * math.sin(math.pi / A))
        self._formation_offsets = get_formation(
            self.cfg.formation_shape,
            A,
            radius=radius,
            spacing=spacing,
        ).to(device)  # [A, 3]

        # Per-drone scratch tensors (pre-allocated; reused every step)
        self._actions = torch.zeros(N_drones, 4, device=device)  # [N_drones, 4]
        self._smoothed_actions = torch.zeros(N_drones, 4, device=device)  # [N_drones, 4]
        self._thrust = torch.zeros(N_drones, 1, 3, device=device)  # [N_drones, 1, 3]
        self._moment = torch.zeros(N_drones, 1, 3, device=device)  # [N_drones, 1, 3]
        self._desired_pos_w = torch.zeros(N_drones, 3, device=device)  # [N_drones, 3]

        # MINCO trajectory state (per-drone)
        self._minco_pos = torch.zeros(N_drones, 4, device=device)
        self._minco_vel = torch.zeros(N_drones, 4, device=device)
        self._minco_acc = torch.zeros(N_drones, 4, device=device)

        # Cloud-mode per-env scratch buffers
        if self._cloud_mode:
            self._group_goal_local = torch.zeros(N_envs, 3, device=device)  # [N_envs, 3]
            self._cloud_centroid_dist = torch.zeros(N_envs, device=device)  # [N_envs]
            self._cloud_spacing_penalty = torch.zeros(N_envs, A, device=device)  # [N_envs, A]
            self._cloud_cohesion_reward = torch.zeros(N_envs, A, device=device)  # [N_envs, A]

        # KNN edge buffer for GNN message passing (per-drone, bidirectional)
        K = min(self.cfg.num_neighbors, A - 1)
        self._K = K
        if K > 0:
            num_edges = N_drones * K * 2
            self._knn_edge_index = torch.zeros(2, num_edges, dtype=torch.long, device=device)
            self._knn_src_pattern = torch.arange(A, device=device).unsqueeze(1).expand(A, K)  # [A, K]

        # Pre-allocated reward / collision aggregation buffers
        self._zero_reward_drones = torch.zeros(N_drones, device=device)  # [N_drones]
        self._formation_total_error = torch.zeros(N_envs, device=device)  # [N_envs]
        self._collision_count = torch.zeros(N_envs, device=device)  # [N_envs]

        # SwarmRaft state (per-drone alive mask, per-env dropout step)
        self._agent_alive = torch.ones(N_drones, dtype=torch.bool, device=device)  # [N_drones]
        self._dropout_step = torch.zeros(N_envs, dtype=torch.long, device=device)  # [N_envs]

        # Forest base goal (per-drone, used when forest_enabled)
        if self.cfg.forest_enabled:
            self._forest_base_goal = torch.zeros(N_drones, 3, device=device)  # [N_drones, 3]

        # Per-drone env origin broadcast (avoids per-step repeat_interleave)
        self._env_origins_per_drone = self._terrain.env_origins.repeat_interleave(A, dim=0)  # [N_drones, 3]

        # Body index, mass, weight (regex Articulation: one "body" link per drone)
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Episode reward sums (per-drone) for logging
        self._episode_sums = {
            key: torch.zeros(N_drones, dtype=torch.float, device=device)
            for key in ("lin_vel", "ang_vel", "distance_to_goal", "formation")
        }

        # Debug draw for altitude lines + KNN neighbor edges (best-effort —
        # only available when running with the Isaac Sim GUI).
        try:
            from isaacsim.util.debug_draw import _debug_draw  # noqa: PLC0415

            self._debug_draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            self._debug_draw = None

    # ------------------------------------------------------------------ scene

    def _setup_scene(self):
        # Manually spawn A drone USDs at /World/envs/env_0/Drone_0..Drone_{A-1}.
        # Required because Isaac Lab's spawner skips spawn when the leaf of
        # cfg.robot.prim_path is a regex (asset_base.py:77-83).
        A = self.cfg.num_agents
        spawn_cfg = self.cfg.robot.spawn
        init_state = self.cfg.robot.init_state
        for i in range(A):
            spawn_cfg.func(
                f"/World/envs/env_0/Drone_{i}",
                spawn_cfg,
                translation=init_state.pos,
                orientation=init_state.rot,
            )

        # Single Articulation with regex prim_path; PhysX flattens N_envs * A
        # instances in env-major order (verified by scripts/probe/multi_drone_layout.py).
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Yellow material on each drone body
        import omni.usd  # noqa: PLC0415
        from pxr import UsdShade  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        for i in range(A):
            mat_path = f"/World/envs/env_0/Drone_{i}/body/Looks/DroneMat"
            mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0))
            sim_utils.spawn_preview_surface(mat_path, mat_cfg)
            body_prim = stage.GetPrimAtPath(f"/World/envs/env_0/Drone_{i}/body")
            mat_prim = stage.GetPrimAtPath(mat_path)
            if body_prim.IsValid() and mat_prim.IsValid():
                UsdShade.MaterialBindingAPI.Apply(body_prim)
                UsdShade.MaterialBindingAPI(body_prim).Bind(UsdShade.Material(mat_prim))

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone env_0 to all envs
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Forest obstacles
        self._obstacle_pos = None
        if self.cfg.forest_enabled:
            self._obstacle_pos = self._generate_forest_obstacles()
            self._spawn_forest_cylinders(stage)

    def _generate_forest_obstacles(self) -> torch.Tensor:
        s = self.cfg.forest_cylinder_spacing
        row_a_y = [-2 * s, -s, 0.0, s, 2 * s]
        row_b_y = [-1.5 * s, -0.5 * s, 0.5 * s, 1.5 * s]
        positions = []
        for row_idx in range(self.cfg.forest_num_rows):
            row_x = self.cfg.forest_row_start_x + row_idx * self.cfg.forest_row_spacing
            y_positions = row_a_y if row_idx % 2 == 0 else row_b_y
            for cy in y_positions:
                positions.append([row_x, cy, self.cfg.forest_obstacle_z])
        return torch.tensor(positions, device=self.device, dtype=torch.float32)

    def _spawn_forest_cylinders(self, stage) -> None:
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade  # noqa: PLC0415

        r = self.cfg.forest_obstacle_radius
        h = self.cfg.forest_obstacle_height
        for k in range(self._obstacle_pos.shape[0]):
            pos = self._obstacle_pos[k].cpu().tolist()
            prim_path = f"/World/envs/env_0/Obstacle_{k}"
            cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
            cylinder.GetRadiusAttr().Set(r)
            cylinder.GetHeightAttr().Set(h)
            cylinder.GetAxisAttr().Set("Z")
            xform = UsdGeom.Xformable(cylinder.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
            UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
            mat_path = f"{prim_path}/ObstacleMat"
            mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3))
            sim_utils.spawn_preview_surface(mat_path, mat_cfg)
            body_prim = cylinder.GetPrim()
            mat_prim = stage.GetPrimAtPath(mat_path)
            if body_prim.IsValid() and mat_prim.IsValid():
                UsdShade.MaterialBindingAPI.Apply(body_prim)
                UsdShade.MaterialBindingAPI(body_prim).Bind(UsdShade.Material(mat_prim))

    # ------------------------------------------------------------- per-step

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        # Pack per-agent [num_envs, 4] dict into per-drone [N_drones, 4] env-major.
        A = self._A
        N_envs = self.num_envs
        stacked = torch.stack([actions[f"drone_{i}"].clamp(-1.0, 1.0) for i in range(A)], dim=1)  # [N_envs, A, 4]
        self._actions.copy_(stacked.reshape(N_envs * A, 4))

        # Forest goal deflection (neighbor-velocity-guided)
        if self.cfg.forest_enabled and self._obstacle_pos is not None:
            K_nn = self._K
            dx = self.cfg.centroid_speed * self.step_dt
            deflect_radius = self.cfg.cbf_obstacle_d_safe + self.cfg.forest_obstacle_radius
            blend_lat = self.cfg.forest_deflect_lateral_blend
            vel_eps = self.cfg.forest_deflect_neighbor_vel_eps
            shortfall_scale = self.cfg.forest_deflect_shortfall_scale

            self._forest_base_goal[:, 0] += dx

            drone_local = self._robot.data.root_pos_w - self._env_origins_per_drone  # [N_drones, 3]
            self._forest_base_goal[:, 0] = torch.minimum(
                self._forest_base_goal[:, 0],
                drone_local[:, 0] + self.cfg.forest_max_goal_lead,
            )
            self._desired_pos_w[:] = self._forest_base_goal + self._env_origins_per_drone
            vel_xy = self._robot.data.root_lin_vel_w[:, :2]  # [N_drones, 2]

            pos_g = drone_local.reshape(N_envs, A, 3)
            vel_g = vel_xy.reshape(N_envs, A, 2)
            pair_dist = torch.cdist(pos_g, pos_g)
            pair_dist = pair_dist + torch.eye(A, device=self.device).unsqueeze(0) * 1e6
            _, knn_idx = torch.topk(pair_dist, K_nn, dim=2, largest=False)
            knn_vel = vel_g.gather(1, knn_idx.reshape(N_envs, A * K_nn, 1).expand(N_envs, A * K_nn, 2)).reshape(
                N_envs, A, K_nn, 2
            )
            neighbor_mean_vel = knn_vel.mean(dim=2).reshape(-1, 2)  # [N_drones, 2]

            for k in range(self._obstacle_pos.shape[0]):
                obs_xy = self._obstacle_pos[k, :2]
                diff_xy = drone_local[:, :2] - obs_xy.unsqueeze(0)
                dist = diff_xy.norm(dim=1, keepdim=True).clamp(min=1e-6)
                too_close = dist.squeeze(1) < deflect_radius
                if not too_close.any():
                    continue
                radial = diff_xy[too_close] / dist[too_close]
                lat_pos = torch.stack([-radial[:, 1], radial[:, 0]], dim=1)
                nv = neighbor_mean_vel[too_close]
                own = vel_xy[too_close]
                nv_mag = nv.norm(dim=1, keepdim=True)
                guide_vel = torch.where(nv_mag < vel_eps, own, nv)
                guide_mag = guide_vel.norm(dim=1)
                dot = (lat_pos * guide_vel).sum(dim=1)
                y_sign = torch.sign(diff_xy[too_close, 1])
                y_sign = torch.where(y_sign == 0, torch.ones_like(y_sign), y_sign)
                dot_sign = torch.sign(dot)
                dot_sign = torch.where(dot_sign == 0, torch.ones_like(dot_sign), dot_sign)
                sign = torch.where(guide_mag < vel_eps, y_sign, dot_sign)
                lateral = lat_pos * sign.unsqueeze(1)
                escape = blend_lat * lateral + (1.0 - blend_lat) * radial
                escape = escape / escape.norm(dim=1, keepdim=True).clamp(min=1e-6)
                shortfall = (deflect_radius - dist[too_close].squeeze(1)) * shortfall_scale
                self._desired_pos_w[too_close, 0] += escape[:, 0] * shortfall
                self._desired_pos_w[too_close, 1] += escape[:, 1] * shortfall

        # SwarmRaft: trigger agent dropout at scheduled per-env step
        if self.cfg.dropout_enabled:
            should_trigger = (self.episode_length_buf == self._dropout_step) & (self._dropout_step > 0)
            if should_trigger.any():
                from ggswarm.formations import get_formation  # noqa: PLC0415

                A_local = self._A
                for g in should_trigger.nonzero(as_tuple=True)[0]:
                    drone_start = int(g) * A_local
                    alive_in_env = self._agent_alive[drone_start : drone_start + A_local].nonzero(as_tuple=True)[0]
                    if len(alive_in_env) > 2:
                        victim = alive_in_env[torch.randint(len(alive_in_env), (1,))]
                        self._agent_alive[drone_start + victim] = False
                        n_alive = len(alive_in_env) - 1
                        spacing = self.cfg.formation_target_spacing
                        radius = spacing / (2 * math.sin(math.pi / max(n_alive, 2)))
                        new_offsets = get_formation(
                            self.cfg.formation_shape,
                            n_alive,
                            radius=radius,
                            spacing=spacing,
                        ).to(self.device)
                        alive_mask = self._agent_alive[drone_start : drone_start + A_local]
                        alive_ids = alive_mask.nonzero(as_tuple=True)[0]
                        first_alive = drone_start + int(alive_ids[0])
                        centroid_w = self._desired_pos_w[first_alive] - self._formation_offsets[0]
                        slot_pos = centroid_w.unsqueeze(0) + new_offsets
                        drone_pos = self._robot.data.root_pos_w[drone_start : drone_start + A_local][alive_mask]
                        dist_matrix = torch.cdist(drone_pos, slot_pos)
                        claimed = torch.zeros(n_alive, dtype=torch.bool, device=self.device)
                        for _ in range(n_alive):
                            dist_matrix[:, claimed] = float("inf")
                            flat_idx = dist_matrix.argmin()
                            di = flat_idx // n_alive
                            sj = flat_idx % n_alive
                            drone_id = drone_start + int(alive_ids[di])
                            self._desired_pos_w[drone_id] = slot_pos[sj]
                            claimed[sj] = True
                            dist_matrix[di, :] = float("inf")

        # MINCO smoothing or EMA fallback
        if self.cfg.minco_enabled:
            from ggswarm.minco import apply_minco  # noqa: PLC0415

            act = apply_minco(
                self._actions,
                self._minco_pos,
                self._minco_vel,
                self._minco_acc,
                self.step_dt,
                self.cfg.minco_horizon,
                self.cfg.minco_max_vel,
                self.cfg.minco_max_acc,
            )
        elif self.cfg.smoothing_enabled:
            a = self.cfg.smoothing_alpha
            self._smoothed_actions = a * self._actions + (1 - a) * self._smoothed_actions
            act = self._smoothed_actions
        else:
            act = self._actions

        # CBF safety shield
        if self.cfg.cbf_enabled:
            from ggswarm.cbf import apply_cbf  # noqa: PLC0415

            act = apply_cbf(
                act,
                self._robot.data.root_pos_w,
                self._env_origins_per_drone,
                self._robot.data.root_lin_vel_w,
                self.cfg.num_agents,
                self.cfg.cbf_d_safe,
                self.cfg.cbf_gamma,
                self._agent_alive if self.cfg.dropout_enabled else None,
                self.cfg.cbf_max_correction,
            )
            if self.cfg.minco_enabled:
                self._minco_pos.copy_(act)

        if self.cfg.dropout_enabled:
            act[~self._agent_alive] = 0.0

        # Compose thrust + moment
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (act[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * act[:, 1:]

    def _apply_action(self) -> None:
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )

    # ----------------------------------------------------------- observations

    def _get_observations(self) -> dict[str, torch.Tensor]:
        A = self._A
        N_envs = self.num_envs
        N_drones = self._N_drones

        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )
        # Per-drone 12D base obs
        base_obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,  # [N_drones, 3]
                self._robot.data.root_ang_vel_b,  # [N_drones, 3]
                self._robot.data.projected_gravity_b,  # [N_drones, 3]
                desired_pos_b,  # [N_drones, 3]
            ],
            dim=-1,
        )  # [N_drones, 12]

        # KNN neighbor relative positions + edge publication
        K = self._K
        pos_local = self._robot.data.root_pos_w - self._env_origins_per_drone  # [N_drones, 3]
        pos_grouped = pos_local.reshape(N_envs, A, 3)

        dead_g = ~self._agent_alive.reshape(N_envs, A) if self.cfg.dropout_enabled else None
        rel_parts: list[torch.Tensor] = []
        all_nearest_idx: list[torch.Tensor] = []
        for i in range(A):
            diff = pos_grouped - pos_grouped[:, i : i + 1, :]
            dists = torch.linalg.norm(diff, dim=2)
            dists[:, i] = float("inf")
            if dead_g is not None:
                dists[dead_g] = float("inf")
            _, nearest_idx = torch.topk(dists, K, dim=1, largest=False)  # [N_envs, K]
            all_nearest_idx.append(nearest_idx)
            neighbors = []
            for k in range(K):
                idx = nearest_idx[:, k]
                rel = pos_grouped[torch.arange(N_envs, device=self.device), idx] - pos_grouped[:, i]
                neighbors.append(rel)
            rel_parts.append(torch.cat(neighbors, dim=-1))  # [N_envs, K*3]

        # [N_envs, A, K*3] -> [N_drones, K*3]
        rel_all = torch.stack(rel_parts, dim=1).reshape(N_drones, -1)
        full_obs = torch.cat([base_obs, rel_all], dim=-1)  # [N_drones, 12 + K*3]

        # Publish edge index to GNN policy (when used). Edge construction:
        # global drone idx = env_idx * A + within-env idx.
        knn_idx = torch.stack(all_nearest_idx, dim=1)  # [N_envs, A, K]
        offsets = torch.arange(N_envs, device=self.device).reshape(N_envs, 1, 1) * A
        global_dst = (knn_idx + offsets).reshape(-1)
        global_src = (self._knn_src_pattern.unsqueeze(0) + offsets).reshape(-1)
        half = global_src.shape[0]
        self._knn_edge_index[0, :half] = global_src
        self._knn_edge_index[1, :half] = global_dst
        self._knn_edge_index[0, half:] = global_dst
        self._knn_edge_index[1, half:] = global_src
        try:
            from ggswarm.gnn_policy import GgswarmGNNPolicy  # noqa: PLC0415

            GgswarmGNNPolicy.set_knn_edges(self._knn_edge_index, N_drones)
        except Exception:
            pass  # MLP policy path doesn't need edges

        if self._debug_draw is not None:
            self._draw_debug_overlay(A, K)

        # Reshape per-drone obs to per-agent dict
        obs_per_env_agent = full_obs.reshape(N_envs, A, -1)
        return {f"drone_{i}": obs_per_env_agent[:, i] for i in range(A)}

    def _draw_debug_overlay(self, A: int, K: int) -> None:
        """Render env_0 altitude lines + KNN edges via the Isaac Sim debug-draw API.

        Best-effort visualization — only fires when running with the GUI
        (the debug-draw interface is None in headless training). Each drone
        gets a vertical line down to ground at its XY, colored from an HSV
        wheel; KNN edges inherit the source drone's color so the graph
        structure is readable. Dropped agents (alive mask false) are skipped.
        """
        import colorsys  # noqa: PLC0415

        self._debug_draw.clear_lines()
        colors = [(*colorsys.hsv_to_rgb(i / A, 0.9, 0.9), 0.9) for i in range(A)]
        for i in range(A):
            pos_i = self._robot.data.root_pos_w[i].cpu().tolist()
            self._debug_draw.draw_lines(
                [pos_i],
                [[pos_i[0], pos_i[1], 0.0]],
                [colors[i]],
                [1.0],
            )
        # KNN neighbor edges (env_0 only). The first A*K entries of
        # _knn_edge_index are forward edges, env-major; for env_0 those are
        # indices 0..A*K-1, with src=drone_i, dst=neighbor.
        for i in range(A):
            if self.cfg.dropout_enabled and not self._agent_alive[i]:
                continue
            pos_i = self._robot.data.root_pos_w[i].cpu().tolist()
            src_color = (*colors[i][:3], 0.6)
            for k in range(K):
                j = int(self._knn_edge_index[1, i * K + k])
                if j >= A:  # only draw within env_0
                    continue
                if self.cfg.dropout_enabled and not self._agent_alive[j]:
                    continue
                pos_j = self._robot.data.root_pos_w[j].cpu().tolist()
                self._debug_draw.draw_lines([pos_i], [pos_j], [src_color], [2.0])

    def _get_states(self) -> torch.Tensor:
        # state_space=-1 → DirectMARLEnv concatenates obs_dict for us.
        raise NotImplementedError

    # ----------------------------------------------------------------- rewards

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        self._global_step += 1
        A = self._A
        N_envs = self.num_envs

        # Hover terms (per-drone)
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / self.cfg.distance_to_goal_sigma)
        if self._cloud_mode:
            dtg_reward = self._zero_reward_drones
        else:
            dtg_reward = self.cfg.distance_to_goal_reward_scale * distance_to_goal_mapped * self.step_dt

        rewards_per_drone = {
            "lin_vel": self.cfg.lin_vel_reward_scale * lin_vel * self.step_dt,
            "ang_vel": self.cfg.ang_vel_reward_scale * ang_vel * self.step_dt,
            "distance_to_goal": dtg_reward,
        }

        # Formation reward
        if self._cloud_mode:
            formation_rew = self._compute_cloud_reward()
        else:
            formation_rew = self._compute_formation_reward()
        rewards_per_drone["formation"] = formation_rew

        for key, value in rewards_per_drone.items():
            self._episode_sums[key] += value

        total_per_drone = sum(rewards_per_drone.values())  # [N_drones]
        total_per_env_agent = total_per_drone.reshape(N_envs, A)
        return {f"drone_{i}": total_per_env_agent[:, i] for i in range(A)}

    def _compute_formation_reward(self) -> torch.Tensor:
        A = self._A
        N_envs = self.num_envs
        N_drones = self._N_drones
        pos_local = self._robot.data.root_pos_w - self._env_origins_per_drone  # [N_drones, 3]
        pos_grouped = pos_local.reshape(N_envs, A, 3)
        alpha = min(
            1.0,
            max(
                0.0,
                (self._global_step - self.cfg.formation_curriculum_start)
                / max(1, self.cfg.formation_curriculum_end - self.cfg.formation_curriculum_start),
            ),
        )
        if alpha <= 0.0:
            return self._zero_reward_drones
        self._formation_total_error.zero_()
        for i, j in self._pair_indices:
            dist = torch.linalg.norm(pos_grouped[:, i, :] - pos_grouped[:, j, :], dim=1)
            self._formation_total_error += torch.abs(dist - self.cfg.formation_target_spacing)
        mean_error = self._formation_total_error / len(self._pair_indices)
        formation_mapped = 1 - torch.tanh(mean_error / self.cfg.formation_reward_sigma)
        formation_reward = alpha * self.cfg.formation_reward_scale * formation_mapped * self.step_dt
        # Log to first agent's extras
        log0 = self.extras[self._agent_ids[0]].setdefault("log", {})
        log0["Metrics/mean_formation_error_m"] = mean_error.mean().item()
        log0["Metrics/formation_alpha"] = alpha
        return formation_reward.unsqueeze(1).expand(N_envs, A).reshape(N_drones)

    def _compute_cloud_reward(self) -> torch.Tensor:
        A = self._A
        N_envs = self.num_envs
        N_drones = self._N_drones
        K = self._K
        pos_local = self._robot.data.root_pos_w - self._env_origins_per_drone
        pos_grouped = pos_local.reshape(N_envs, A, 3)
        alpha = min(
            1.0,
            max(
                0.0,
                (self._global_step - self.cfg.formation_curriculum_start)
                / max(1, self.cfg.formation_curriculum_end - self.cfg.formation_curriculum_start),
            ),
        )
        if alpha <= 0.0:
            return self._zero_reward_drones

        if self.cfg.dropout_enabled:
            alive_g = self._agent_alive.reshape(N_envs, A)
            alive_count = alive_g.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(2)
            alive_pos = pos_grouped * alive_g.unsqueeze(2)
            centroid = alive_pos.sum(dim=1, keepdim=True) / alive_count
        else:
            centroid = pos_grouped.mean(dim=1, keepdim=True)
        centroid_squeezed = centroid.squeeze(1)
        self._cloud_centroid_dist[:] = torch.linalg.norm(centroid_squeezed - self._group_goal_local, dim=1)
        centroid_mapped = 1 - torch.tanh(self._cloud_centroid_dist / self.cfg.cloud_centroid_goal_sigma)
        centroid_reward = self.cfg.cloud_centroid_goal_scale * centroid_mapped * self.step_dt
        centroid_reward_broadcast = centroid_reward.unsqueeze(1).expand(N_envs, A)

        self._cloud_cohesion_reward.zero_()
        self._cloud_spacing_penalty.zero_()
        dead_g = ~self._agent_alive.reshape(N_envs, A) if self.cfg.dropout_enabled else None
        for i in range(A):
            diff = pos_grouped - pos_grouped[:, i : i + 1, :]
            dists = torch.linalg.norm(diff, dim=2)
            dists[:, i] = float("inf")
            if dead_g is not None:
                dists[dead_g] = float("inf")
            knn_dists, _ = torch.topk(dists, K, dim=1, largest=False)
            mean_knn = knn_dists.mean(dim=1)
            cohesion_mapped = 1 - torch.tanh(mean_knn / self.cfg.cloud_cohesion_sigma)
            self._cloud_cohesion_reward[:, i] = self.cfg.cloud_cohesion_scale * cohesion_mapped * self.step_dt
            nearest_dist = knn_dists[:, 0]
            too_far = torch.clamp(nearest_dist - self.cfg.cloud_max_neighbor_dist, min=0.0)
            too_close = torch.clamp(self.cfg.cloud_min_spacing - nearest_dist, min=0.0)
            self._cloud_spacing_penalty[:, i] = (
                -(self.cfg.cloud_spacing_penalty * too_far + self.cfg.cloud_separation_penalty * too_close)
                * self.step_dt
            )

        total = alpha * (centroid_reward_broadcast + self._cloud_cohesion_reward + self._cloud_spacing_penalty)
        if dead_g is not None:
            total[dead_g] = 0.0

        log0 = self.extras[self._agent_ids[0]].setdefault("log", {})
        log0["Metrics/centroid_to_goal_dist"] = self._cloud_centroid_dist.mean().item()
        log0["Metrics/formation_alpha"] = alpha

        return total.reshape(N_drones)

    # -------------------------------------------------------------------- dones

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        A = self._A
        N_envs = self.num_envs

        time_out = self.episode_length_buf >= self.max_episode_length - 1  # [N_envs]

        died_drone = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.05,
            self._robot.data.root_pos_w[:, 2] > 2.0,
        )  # [N_drones]
        if self.cfg.dropout_enabled:
            died_drone = died_drone & self._agent_alive

        # Pairwise collisions per env
        if self.cfg.collision_enabled:
            pos_local = self._robot.data.root_pos_w - self._env_origins_per_drone
            pos_g = pos_local.reshape(N_envs, A, 3)
            self._collision_count.zero_()
            collided_env = torch.zeros(N_envs, dtype=torch.bool, device=self.device)
            alive_g = self._agent_alive.reshape(N_envs, A) if self.cfg.dropout_enabled else None
            r = self.cfg.collision_radius
            for i in range(A):
                for j in range(i + 1, A):
                    dist = torch.linalg.norm(pos_g[:, i] - pos_g[:, j], dim=1)
                    pair_collided = dist < r
                    if alive_g is not None:
                        pair_collided = pair_collided & alive_g[:, i] & alive_g[:, j]
                    self._collision_count += pair_collided.float()
                    collided_env = collided_env | pair_collided
            log0 = self.extras[self._agent_ids[0]].setdefault("log", {})
            log0["Metrics/collision_pairs_per_step"] = self._collision_count.sum().item()
        else:
            collided_env = torch.zeros(N_envs, dtype=torch.bool, device=self.device)

        # Per-env termination: any drone dies or env-level collision
        died_env = died_drone.reshape(N_envs, A).any(dim=1) | collided_env  # [N_envs]

        if self.cfg.dropout_enabled:
            log0 = self.extras[self._agent_ids[0]].setdefault("log", {})
            alive_per_env = self._agent_alive.reshape(N_envs, A).sum(dim=1).float()
            log0["Metrics/alive_agents_per_group"] = alive_per_env.mean().item()

        terminated = {agent: died_env for agent in self._agent_ids}
        time_outs = {agent: time_out for agent in self._agent_ids}
        return terminated, time_outs

    # --------------------------------------------------------------------- reset

    def _reset_idx(self, env_ids: Sequence[int] | None):
        A = self._A
        N_envs = self.num_envs

        if env_ids is None or len(env_ids) == N_envs:  # type: ignore[arg-type]
            env_ids = torch.arange(N_envs, device=self.device, dtype=torch.long)
            all_envs = True
        else:
            all_envs = False
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        n_envs_reset = env_ids.shape[0]
        drone_ids = (env_ids.unsqueeze(1) * A + torch.arange(A, device=self.device)).reshape(-1)

        # Log episode metrics
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[drone_ids] - self._robot.data.root_pos_w[drone_ids], dim=1
        ).mean()
        log0 = self.extras[self._agent_ids[0]].setdefault("log", {})
        for key in self._episode_sums:
            avg = torch.mean(self._episode_sums[key][drone_ids])
            log0[f"Episode_Reward/{key}"] = (avg / self.max_episode_length_s).item()
            self._episode_sums[key][drone_ids] = 0.0
        log0["Episode_Termination/died"] = (
            torch.count_nonzero(self.terminated_dict[self._agent_ids[0]][env_ids]).item()
            if hasattr(self, "terminated_dict")
            else 0
        )
        log0["Episode_Termination/time_out"] = (
            torch.count_nonzero(self.time_out_dict[self._agent_ids[0]][env_ids]).item()
            if hasattr(self, "time_out_dict")
            else 0
        )
        log0["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()

        # Articulation reset (drone-level instance ids)
        self._robot.reset(drone_ids)
        super()._reset_idx(env_ids)  # episode_length_buf[env_ids] = 0 + scene.reset(env_ids)

        # Stagger episode lengths on full reset
        if all_envs and N_envs > 1:
            self.episode_length_buf = torch.randint(0, int(self.max_episode_length), (N_envs,), device=self.device)

        # Per-drone scratch reset
        self._actions[drone_ids] = 0.0
        self._smoothed_actions[drone_ids] = 0.0
        self._minco_pos[drone_ids] = 0.0
        self._minco_vel[drone_ids] = 0.0
        self._minco_acc[drone_ids] = 0.0
        self._agent_alive[drone_ids] = True

        # SwarmRaft: per-env dropout step
        if self.cfg.dropout_enabled:
            self._dropout_step[env_ids] = torch.randint(
                self.cfg.dropout_step_min,
                self.cfg.dropout_step_max + 1,
                (n_envs_reset,),
                device=self.device,
            )

        # Sample formation centroid per env
        if self.cfg.formation_centroid is not None:
            fc = self.cfg.formation_centroid
            centroid = torch.tensor([[fc[0], fc[1], fc[2]]], device=self.device).expand(n_envs_reset, 3).clone()
        else:
            centroid_xy = torch.zeros(n_envs_reset, 2, device=self.device).uniform_(-0.5, 0.5)
            centroid_z = torch.zeros(n_envs_reset, 1, device=self.device).uniform_(0.5, 1.5)
            centroid = torch.cat([centroid_xy, centroid_z], dim=-1)

        if self._cloud_mode:
            self._group_goal_local[env_ids] = centroid
            for e_idx in range(n_envs_reset):
                e = int(env_ids[e_idx])
                drone_start = e * A
                origin_e = self._terrain.env_origins[e]
                for i in range(A):
                    self._desired_pos_w[drone_start + i] = centroid[e_idx] + origin_e
        else:
            for e_idx in range(n_envs_reset):
                e = int(env_ids[e_idx])
                drone_start = e * A
                origin_e = self._terrain.env_origins[e]
                drone_pos_local = self._robot.data.root_pos_w[drone_start : drone_start + A] - origin_e.unsqueeze(0)
                slot_pos = centroid[e_idx] + self._formation_offsets
                dist_matrix = torch.cdist(drone_pos_local, slot_pos)
                claimed = torch.zeros(A, dtype=torch.bool, device=self.device)
                for _ in range(A):
                    dist_matrix[:, claimed] = float("inf")
                    flat_idx = dist_matrix.argmin()
                    drone_i = int(flat_idx // A)
                    slot_j = int(flat_idx % A)
                    drone_id = drone_start + drone_i
                    self._desired_pos_w[drone_id] = slot_pos[slot_j] + origin_e
                    if self.cfg.forest_enabled and hasattr(self, "_forest_base_goal"):
                        self._forest_base_goal[drone_id] = slot_pos[slot_j]
                    claimed[slot_j] = True
                    dist_matrix[drone_i, :] = float("inf")

        # Reset robot state with circular spawn (B3 fix)
        joint_pos = self._robot.data.default_joint_pos[drone_ids]
        joint_vel = self._robot.data.default_joint_vel[drone_ids]
        default_root_state = self._robot.data.default_root_state[drone_ids].clone()
        default_root_state[:, :3] += self._env_origins_per_drone[drone_ids]

        r_circle = self.cfg.spawn_radius
        jitter = self.cfg.min_spawn_spacing * 0.15
        slot_idx = drone_ids % A
        theta = 2.0 * math.pi * slot_idx.float() / A
        slot_x = r_circle * torch.cos(theta)
        slot_y = r_circle * torch.sin(theta)
        n_drones_reset = drone_ids.shape[0]
        default_root_state[:, 0] += slot_x + torch.zeros(n_drones_reset, device=self.device).uniform_(-jitter, jitter)
        default_root_state[:, 1] += slot_y + torch.zeros(n_drones_reset, device=self.device).uniform_(-jitter, jitter)
        default_root_state[:, 2] = 0.5

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], drone_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], drone_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, drone_ids)
