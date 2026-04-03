"""ggSwarm: Drone hover + formation environment using DirectRLEnv + PPO.

One Crazyflie per env. PPO trains a shared policy across all envs (CTDE).
When num_agents > 1, consecutive envs are grouped into logical swarms:
  - Observations expanded with neighbor relative positions
  - Formation reward added (curriculum-scaled)
  - Collective resets within each swarm group

Based on Isaac Lab's quadcopter_env.py.
"""

from __future__ import annotations

import math

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import subtract_frame_transforms

from .ggswarm_env_cfg import GgswarmEnvCfg


class GgswarmEnv(DirectRLEnv):
    cfg: GgswarmEnvCfg

    def __init__(self, cfg: GgswarmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        N = self.num_envs
        A = self.cfg.num_agents
        device = self.device

        # Formation grouping
        self._formation_active = A > 1
        self._cloud_mode = self.cfg.formation_mode == "cloud"
        if A > 1 and N % A != 0:
            raise ValueError(f"num_envs ({N}) must be divisible by num_agents ({A})")
        self._num_groups = N // A if A > 1 else N
        self._global_step = 0

        # Pre-build pair indices for formation error
        self._pair_indices = []
        if A > 1:
            for i in range(A):
                for j in range(i + 1, A):
                    self._pair_indices.append((i, j))

        # Formation slot offsets — computed via formations module
        # shape: [num_agents, 3] — XYZ offset from group centroid
        if A > 1:
            from ggswarm.formations import get_formation  # noqa: PLC0415

            spacing = self.cfg.formation_target_spacing
            radius = spacing / (2 * math.sin(math.pi / A))
            self._formation_offsets = get_formation(
                self.cfg.formation_shape, A, radius=radius, spacing=spacing,
            ).to(device)  # [A, 3]
        else:
            self._formation_offsets = None

        # Pre-allocate action tensors (reused every step)
        self._actions = torch.zeros(N, 4, device=device)
        self._smoothed_actions = torch.zeros(N, 4, device=device)
        self._thrust = torch.zeros(N, 1, 3, device=device)
        self._moment = torch.zeros(N, 1, 3, device=device)
        self._desired_pos_w = torch.zeros(N, 3, device=device)

        # MINCO trajectory state (L3 layer)
        self._minco_pos = torch.zeros(N, 4, device=device)
        self._minco_vel = torch.zeros(N, 4, device=device)
        self._minco_acc = torch.zeros(N, 4, device=device)

        # Cloud-mode scratch buffers (pre-allocated to avoid per-step allocation)
        G = self._num_groups
        if self._cloud_mode and A > 1:
            self._group_goal_local = torch.zeros(G, 3, device=device)  # [G, 3]
            self._cloud_centroid_dist = torch.zeros(G, device=device)  # [G]
            self._cloud_spacing_penalty = torch.zeros(G, A, device=device)  # [G, A]
            self._cloud_cohesion_reward = torch.zeros(G, A, device=device)  # [G, A]

        # KNN edge buffer for GNN message passing (pre-allocated)
        K = min(self.cfg.num_neighbors, A - 1) if A > 1 else 0
        if A > 1 and K > 0:
            num_edges = N * K * 2  # K edges per drone, bidirectional
            self._knn_edge_index = torch.zeros(2, num_edges, dtype=torch.long, device=device)
            # Pre-compute source index pattern: each drone repeated K times
            # [A, K] pattern repeated for G groups with offsets
            src_local = torch.arange(A, device=device).unsqueeze(1).expand(A, K)  # [A, K]
            self._knn_src_pattern = src_local  # reused each step

        # Pre-allocated zero-reward buffers (avoid per-step allocation)
        self._zero_reward_N = torch.zeros(N, device=device)  # [N]
        if A > 1:
            self._formation_total_error = torch.zeros(G, device=device)  # [G]

        # Virtual collision detection scratch buffer
        if A > 1:
            self._collision_count = torch.zeros(G, device=device)  # [G]

        # SwarmRaft: agent alive mask and dropout scheduling
        self._agent_alive = torch.ones(N, dtype=torch.bool, device=device)  # [N]
        if A > 1:
            self._dropout_step = torch.zeros(G, dtype=torch.long, device=device)  # [G]

        # Forest: per-group centroid position (local frame, tracks along +X)
        # Initialized on first reset, advanced each step in _pre_physics_step
        if self.cfg.forest_enabled and A > 1:
            self._forest_centroid = torch.zeros(G, 3, device=device)  # [G, 3]
            # Per-drone slot offset from group centroid (set at reset, reused each step)
            self._drone_slot_offset = torch.zeros(N, 3, device=device)  # [N, 3]

        # Body ID, mass, weight
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Episode logging
        log_keys = ["lin_vel", "ang_vel", "distance_to_goal"]
        if self._formation_active:
            log_keys.append("formation")
        if self.cfg.forest_enabled:
            log_keys.append("obstacle")
        self._episode_sums = {
            key: torch.zeros(N, dtype=torch.float, device=device)
            for key in log_keys
        }

        # Debug draw for altitude line
        try:
            from isaacsim.util.debug_draw import _debug_draw  # noqa: PLC0415

            self._debug_draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            self._debug_draw = None

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Apply yellow material to drone body (before clone_environments)
        import omni.usd  # noqa: PLC0415
        from pxr import UsdShade  # noqa: PLC0415

        stage = omni.usd.get_context().get_stage()
        mat_path = "/World/envs/env_0/Drone_0/body/Looks/DroneMat"
        mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0))
        sim_utils.spawn_preview_surface(mat_path, mat_cfg)
        body_prim = stage.GetPrimAtPath("/World/envs/env_0/Drone_0/body")
        mat_prim = stage.GetPrimAtPath(mat_path)
        if body_prim.IsValid() and mat_prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(body_prim)
            UsdShade.MaterialBindingAPI(body_prim).Bind(UsdShade.Material(mat_prim))

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Forest obstacles — generate positions and spawn cylinders in env_0
        self._obstacle_pos = None
        if self.cfg.forest_enabled:
            self._obstacle_pos = self._generate_forest_obstacles()
        if self.cfg.forest_enabled and self._obstacle_pos is not None:
            self._spawn_forest_cylinders(stage)

    def _generate_forest_obstacles(self) -> torch.Tensor:
        """Generate obstacle positions as staggered rows across the flight path.

        Row A: cylinders at Y = -0.8, 0.0, +0.8  (blocks center)
        Row B: cylinders at Y = -0.4, +0.4        (staggered, blocks gaps)

        Rows alternate A-B along X, forcing the swarm to weave. Deterministic
        layout — no randomness.

        Returns:
            [K, 3] obstacle positions in local frame (on self.device).
        """
        obs_z = self.cfg.forest_obstacle_z
        dev = self.device

        # Two row patterns that alternate — staggered to force weaving
        s = self.cfg.forest_cylinder_spacing
        row_a_y = [-s, 0.0, s]           # 3 cylinders — blocks center corridor
        row_b_y = [-s / 2, s / 2]        # 2 cylinders — staggered, blocks gaps in row A

        num_rows = self.cfg.forest_num_rows
        row_spacing = self.cfg.forest_row_spacing
        start_x = self.cfg.forest_row_start_x

        positions = []
        for row_idx in range(num_rows):
            row_x = start_x + row_idx * row_spacing
            y_positions = row_a_y if row_idx % 2 == 0 else row_b_y
            for cy in y_positions:
                positions.append([row_x, cy, obs_z])

        return torch.tensor(positions, device=dev, dtype=torch.float32)  # [K, 3]

    def _spawn_forest_cylinders(self, stage) -> None:
        """Spawn static cylinder prims for forest obstacles."""
        from pxr import Gf, UsdGeom, UsdPhysics  # noqa: PLC0415

        r = self.cfg.forest_obstacle_radius
        h = self.cfg.forest_obstacle_height

        for k in range(self._obstacle_pos.shape[0]):
            pos = self._obstacle_pos[k].cpu().tolist()
            prim_path = f"/World/envs/env_0/Obstacle_{k}"

            # Create cylinder
            cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
            cylinder.GetRadiusAttr().Set(r)
            cylinder.GetHeightAttr().Set(h)
            cylinder.GetAxisAttr().Set("Z")

            # Position
            xform = UsdGeom.Xformable(cylinder.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))

            # Static collider (no rigid body — immovable)
            UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())

            # Dark gray material
            mat_path = f"{prim_path}/ObstacleMat"
            mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3))
            sim_utils.spawn_preview_surface(mat_path, mat_cfg)
            from pxr import UsdShade  # noqa: PLC0415
            body_prim = cylinder.GetPrim()
            mat_prim = stage.GetPrimAtPath(mat_path)
            if body_prim.IsValid() and mat_prim.IsValid():
                UsdShade.MaterialBindingAPI.Apply(body_prim)
                UsdShade.MaterialBindingAPI(body_prim).Bind(UsdShade.Material(mat_prim))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clamp(-1.0, 1.0)

        # Forest: advance centroid along +X, recompute per-drone goals (no deflection)
        if self.cfg.forest_enabled and self._formation_active:
            dx = self.cfg.centroid_speed * self.step_dt  # meters per step
            A = self.cfg.num_agents
            G = self._num_groups

            # Advance group centroids along +X
            self._forest_centroid[:, 0] += dx

            # Recompute per-drone goals from centroid + slot offset (local frame)
            centroid_expanded = self._forest_centroid.unsqueeze(1).expand(G, A, 3)  # [G, A, 3]
            slot_offset = self._drone_slot_offset.reshape(G, A, 3)  # [G, A, 3]
            goals_local = (centroid_expanded + slot_offset).reshape(-1, 3)  # [N, 3]

            # Write back to world frame
            self._desired_pos_w[:, :2] = goals_local[:, :2] + self._terrain.env_origins[:, :2]
            self._desired_pos_w[:, 2] = goals_local[:, 2] + self._terrain.env_origins[:, 2]

            # Keep cloud-mode centroid goal in sync with advancing centroid
            if self._cloud_mode and hasattr(self, '_group_goal_local'):
                self._group_goal_local[:] = self._forest_centroid

        # SwarmRaft: trigger agent dropout at scheduled step
        if self.cfg.dropout_enabled and self.cfg.num_agents > 1:
            A = self.cfg.num_agents
            G = self._num_groups
            ep_len_grouped = self.episode_length_buf.reshape(G, A)[:, 0]  # shape: [G]
            should_trigger = (ep_len_grouped == self._dropout_step) & (self._dropout_step > 0)
            if should_trigger.any():
                from ggswarm.formations import get_formation  # noqa: PLC0415

                for g in should_trigger.nonzero(as_tuple=True)[0]:
                    group_start = g * A
                    alive_in_group = self._agent_alive[group_start:group_start + A].nonzero(as_tuple=True)[0]
                    if len(alive_in_group) > 2:  # keep at least 2 alive
                        victim = alive_in_group[torch.randint(len(alive_in_group), (1,))]
                        self._agent_alive[group_start + victim] = False

                        # Dynamic slot recomputation: new formation for N-1 alive drones
                        n_alive = len(alive_in_group) - 1
                        spacing = self.cfg.formation_target_spacing
                        radius = spacing / (2 * math.sin(math.pi / max(n_alive, 2)))
                        new_offsets = get_formation(
                            self.cfg.formation_shape, n_alive, radius=radius, spacing=spacing,
                        ).to(self.device)  # [n_alive, 3]

                        # Reassign goals to alive drones via nearest-slot
                        alive_mask = self._agent_alive[group_start:group_start + A]
                        alive_ids = alive_mask.nonzero(as_tuple=True)[0]  # [n_alive]
                        # Get centroid from current goal (use first alive drone's goal as reference)
                        first_alive = group_start + alive_ids[0]
                        centroid_w = self._desired_pos_w[first_alive] - self._formation_offsets[0].to(self.device)
                        slot_pos = centroid_w.unsqueeze(0) + new_offsets  # [n_alive, 3]

                        # Greedy nearest-slot assignment for alive drones
                        drone_pos = self._robot.data.root_pos_w[group_start:group_start + A][alive_mask]  # [n_alive, 3]
                        dist_matrix = torch.cdist(drone_pos, slot_pos)  # [n_alive, n_alive]
                        claimed = torch.zeros(n_alive, dtype=torch.bool, device=self.device)
                        for _ in range(n_alive):
                            dist_matrix[:, claimed] = float("inf")
                            flat_idx = dist_matrix.argmin()
                            di = flat_idx // n_alive
                            sj = flat_idx % n_alive
                            drone_id = group_start + alive_ids[di]
                            self._desired_pos_w[drone_id] = slot_pos[sj]
                            claimed[sj] = True
                            dist_matrix[di, :] = float("inf")

        # MINCO minimum-jerk smoothing (L3 layer — replaces EMA)
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
            # Legacy EMA fallback
            a = self.cfg.smoothing_alpha
            self._smoothed_actions = a * self._actions + (1 - a) * self._smoothed_actions
            act = self._smoothed_actions
        else:
            act = self._actions

        # CBF safety shield (Phase 3)
        if self.cfg.cbf_enabled and self.cfg.num_agents > 1:
            from ggswarm.cbf import apply_cbf  # noqa: PLC0415

            act = apply_cbf(
                act,
                self._robot.data.root_pos_w,
                self._terrain.env_origins,
                self._robot.data.root_lin_vel_w,
                self.cfg.num_agents,
                self.cfg.cbf_d_safe,
                self.cfg.cbf_gamma,
                self._agent_alive if self.cfg.dropout_enabled else None,
                self.cfg.cbf_max_correction,
            )
            # Sync MINCO state to post-CBF action so corrections are sticky —
            # without this, MINCO overwrites CBF corrections every step.
            if self.cfg.minco_enabled:
                self._minco_pos.copy_(act)

        # CBF obstacle avoidance (forest cylinders)
        if self.cfg.forest_enabled and self._obstacle_pos is not None:
            from ggswarm.cbf import apply_cbf_obstacles  # noqa: PLC0415

            act = apply_cbf_obstacles(
                act,
                self._robot.data.root_pos_w,
                self._terrain.env_origins,
                self._robot.data.root_lin_vel_w,
                self._obstacle_pos,
                self.cfg.cbf_obstacle_d_safe + self.cfg.forest_obstacle_radius,
                self.cfg.cbf_gamma,
            )

        # Zero out dead drone actions (no thrust/moment for dead drones)
        if self.cfg.dropout_enabled:
            act[~self._agent_alive] = 0.0

        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (act[:, 0] + 1.0) / 2.0
        )
        self._moment[:, 0, :] = self.cfg.moment_scale * act[:, 1:]

        # Forest: count drone-obstacle collisions (actual positions, not goals)
        if self.cfg.forest_enabled and self._obstacle_pos is not None:
            pos_local = self._robot.data.root_pos_w - self._terrain.env_origins  # [N, 3]
            hit_r = self.cfg.forest_obstacle_radius
            step = self.episode_length_buf[0].item()
            for k in range(self._obstacle_pos.shape[0]):
                diff_xy = pos_local[:, :2] - self._obstacle_pos[k, :2].unsqueeze(0)
                dist_xy = diff_xy.norm(dim=1)  # [N]
                hits = (dist_xy < hit_r).sum().item()
                if hits > 0:
                    print(f"[OBSTACLE HIT] step={step} obs={k} hits={hits} "
                          f"min_dist={dist_xy.min():.3f}m")

    def _apply_action(self) -> None:
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
        )
        # shape: [num_envs, 12]
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )

        # Expand obs with neighbor relative positions for formation
        if self.cfg.num_agents > 1:
            obs = self._expand_obs_with_neighbors(obs)

        # Draw altitude lines + KNN neighbor connections (first swarm group)
        if self._debug_draw is not None:
            self._debug_draw.clear_lines()
            import colorsys  # noqa: PLC0415

            A = min(self.cfg.num_agents, self.num_envs)
            colors = [(*colorsys.hsv_to_rgb(i / A, 0.9, 0.9), 0.9) for i in range(A)]
            for i in range(A):
                pos = self._robot.data.root_pos_w[i].cpu().tolist()
                # Altitude line
                self._debug_draw.draw_lines(
                    [pos], [[pos[0], pos[1], 0.0]],
                    [colors[i % len(colors)]], [1.0],
                )

            # KNN neighbor connections (white lines between neighbors)
            if self.cfg.num_agents > 1 and hasattr(self, "_knn_edge_index"):
                K = min(self.cfg.num_neighbors, A - 1)
                edge_color = (1.0, 1.0, 1.0, 0.6)
                for i in range(A):
                    if self.cfg.dropout_enabled and not self._agent_alive[i]:
                        continue
                    pos_i = self._robot.data.root_pos_w[i].cpu().tolist()
                    for k in range(K):
                        # Read from edge_index: first A*K entries are forward edges
                        edge_idx = i * K + k
                        j = int(self._knn_edge_index[1, edge_idx])
                        if j >= A:
                            continue
                        if self.cfg.dropout_enabled and not self._agent_alive[j]:
                            continue
                        pos_j = self._robot.data.root_pos_w[j].cpu().tolist()
                        self._debug_draw.draw_lines(
                            [pos_i], [pos_j], [edge_color], [2.0],
                        )

        return {"policy": obs}

    def _expand_obs_with_neighbors(self, obs: torch.Tensor) -> torch.Tensor:
        """Append K-nearest neighbor relative positions and publish KNN edges.

        Uses fixed K=num_neighbors regardless of swarm size, enabling
        train-with-3 / deploy-with-N scalability. Also builds sparse
        KNN edge_index and publishes it to the GNN policy for K-hop
        message passing.

        Args:
            obs: shape [num_envs, 12]

        Returns:
            shape [num_envs, 12 + num_neighbors*3]
        """
        from ggswarm.gnn_policy import GgswarmGNNPolicy  # noqa: PLC0415

        N = self.num_envs
        A = self.cfg.num_agents
        K = min(self.cfg.num_neighbors, A - 1)  # can't have more neighbors than agents-1
        G = self._num_groups
        # Subtract env origins to get local positions (removes env_spacing offset)
        pos_local = self._robot.data.root_pos_w - self._terrain.env_origins  # shape: [N, 3]

        # Reshape to [num_groups, num_agents, 3]
        pos_grouped = pos_local.reshape(G, A, 3)

        # Pre-compute dead drone mask for dropout exclusion
        dead_g = None
        if self.cfg.dropout_enabled:
            dead_g = ~self._agent_alive.reshape(G, A)  # shape: [G, A]

        rel_parts = []
        all_nearest_idx = []  # collect KNN indices for edge construction
        for i in range(A):
            # Compute distances from drone i to all others in group
            # shape: [G, A, 3] - [G, 1, 3] -> [G, A, 3] -> norm -> [G, A]
            diff = pos_grouped - pos_grouped[:, i:i+1, :]  # [G, A, 3]
            dists = torch.linalg.norm(diff, dim=2)  # [G, A]
            dists[:, i] = float("inf")  # exclude self
            # Exclude dead drones from neighbor selection
            if dead_g is not None:
                dists[dead_g] = float("inf")

            # Find K nearest indices
            _, nearest_idx = torch.topk(dists, K, dim=1, largest=False)  # [G, K]
            all_nearest_idx.append(nearest_idx)

            # Gather relative positions of K nearest
            neighbors = []
            for k in range(K):
                idx = nearest_idx[:, k]  # [G]
                rel = pos_grouped[torch.arange(G, device=self.device), idx] - pos_grouped[:, i]  # [G, 3]
                neighbors.append(rel)
            rel_parts.append(torch.cat(neighbors, dim=-1))  # [G, K*3]

        # Stack and flatten: [G, A, K*3] -> [N, K*3]
        rel_all = torch.stack(rel_parts, dim=1).reshape(N, -1)

        # --- Build KNN sparse edge_index and publish to GNN policy ---
        # all_nearest_idx: list of A tensors, each [G, K] (group-local indices)
        knn_idx = torch.stack(all_nearest_idx, dim=1)  # [G, A, K]

        # Convert group-local indices to global batch indices
        offsets = torch.arange(G, device=self.device).reshape(G, 1, 1) * A  # [G, 1, 1]
        global_dst = (knn_idx + offsets).reshape(-1)  # [G*A*K]

        # Source indices: drone i repeated K times, offset by group
        # _knn_src_pattern: [A, K], unsqueeze(0): [1, A, K], offsets: [G, 1, 1]
        global_src = (self._knn_src_pattern.unsqueeze(0) + offsets).reshape(-1)  # [G*A*K]

        # Write bidirectional edges into pre-allocated buffer
        half = global_src.shape[0]  # G*A*K
        self._knn_edge_index[0, :half] = global_src
        self._knn_edge_index[1, :half] = global_dst
        self._knn_edge_index[0, half:] = global_dst
        self._knn_edge_index[1, half:] = global_src

        # Publish to GNN policy
        GgswarmGNNPolicy.set_knn_edges(self._knn_edge_index, N)

        return torch.cat([obs, rel_all], dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        self._global_step += 1

        # --- Hover reward (always active) ---
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1
        )
        distance_to_goal_mapped = 1 - torch.tanh(
            distance_to_goal / self.cfg.distance_to_goal_sigma
        )

        # In cloud mode, suppress per-drone goal reward — centroid-to-goal is in
        # _compute_cloud_reward instead, so drones aren't all pulled to one point.
        if self._cloud_mode:
            dtg_reward = self._zero_reward_N  # shape: [N]
        else:
            dtg_reward = (
                self.cfg.distance_to_goal_reward_scale
                * distance_to_goal_mapped * self.step_dt
            )  # shape: [N]

        rewards = {
            "lin_vel": self.cfg.lin_vel_reward_scale * lin_vel * self.step_dt,
            "ang_vel": self.cfg.ang_vel_reward_scale * ang_vel * self.step_dt,
            "distance_to_goal": dtg_reward,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # --- Formation reward (curriculum-scaled) ---
        if self._formation_active:
            if self._cloud_mode:
                formation_rew = self._compute_cloud_reward()
            else:
                formation_rew = self._compute_formation_reward()
            reward = reward + formation_rew
            rewards["formation"] = formation_rew

        # --- Obstacle proximity penalty (forest) ---
        if self.cfg.forest_enabled and self._obstacle_pos is not None:
            pos_local = self._robot.data.root_pos_w - self._terrain.env_origins  # [N, 3]
            obstacle_pen = self._zero_reward_N.clone()  # [N]
            for k in range(self._obstacle_pos.shape[0]):
                diff_xy = pos_local[:, :2] - self._obstacle_pos[k, :2].unsqueeze(0)  # [N, 2]
                dist_xy = diff_xy.norm(dim=1)  # [N]
                # Linear penalty: max at obstacle center, zero at penalty_radius
                penetration = torch.clamp(
                    self.cfg.obstacle_penalty_radius - dist_xy, min=0.0
                )  # [N]
                obstacle_pen = obstacle_pen - (
                    self.cfg.obstacle_penalty_scale * penetration * self.step_dt
                )
            reward = reward + obstacle_pen
            rewards["obstacle"] = obstacle_pen

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _compute_formation_reward(self) -> torch.Tensor:
        """Compute formation reward based on inter-drone spacing error.

        Returns:
            shape [num_envs] — formation reward per drone (same for all in group)
        """
        N = self.num_envs
        A = self.cfg.num_agents
        G = self._num_groups
        # Subtract env origins to get local positions (removes env_spacing offset)
        pos_local = self._robot.data.root_pos_w - self._terrain.env_origins  # shape: [N, 3]
        pos_grouped = pos_local.reshape(G, A, 3)

        # Curriculum alpha: ramps 0 -> 1 over training
        alpha = min(1.0, max(0.0,
            (self._global_step - self.cfg.formation_curriculum_start)
            / max(1, self.cfg.formation_curriculum_end - self.cfg.formation_curriculum_start)
        ))

        if alpha <= 0.0:
            return self._zero_reward_N  # shape: [N]

        # Compute mean pairwise distance error
        self._formation_total_error.zero_()  # shape: [G]
        for i, j in self._pair_indices:
            dist = torch.linalg.norm(pos_grouped[:, i, :] - pos_grouped[:, j, :], dim=1)
            self._formation_total_error += torch.abs(dist - self.cfg.formation_target_spacing)
        mean_error = self._formation_total_error / len(self._pair_indices)

        # Tanh mapping (same pattern as distance_to_goal)
        formation_mapped = 1 - torch.tanh(mean_error / self.cfg.formation_reward_sigma)
        formation_reward = (
            alpha * self.cfg.formation_reward_scale * formation_mapped * self.step_dt
        )

        # Log mean formation error in meters
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["Metrics/mean_formation_error_m"] = mean_error.mean().item()
        self.extras["log"]["Metrics/formation_alpha"] = alpha

        # Broadcast from [G] -> [N] (same reward for all agents in group)
        return formation_reward.unsqueeze(1).expand(G, A).reshape(N)

    def _compute_cloud_reward(self) -> torch.Tensor:
        """Compute cloud/mesh formation reward (boids-like).

        Three components:
        - Centroid-to-goal: shared reward for group centroid reaching target
        - Cohesion: reward for staying near K-nearest neighbors (not centroid)
        - Spacing: penalty for nearest neighbor being too close or too far

        Returns:
            shape [num_envs] — per-drone reward
        """
        N = self.num_envs
        A = self.cfg.num_agents
        G = self._num_groups
        K = min(self.cfg.num_neighbors, A - 1)
        pos_local = self._robot.data.root_pos_w - self._terrain.env_origins  # [N, 3]
        pos_grouped = pos_local.reshape(G, A, 3)

        # Curriculum alpha
        alpha = min(1.0, max(0.0,
            (self._global_step - self.cfg.formation_curriculum_start)
            / max(1, self.cfg.formation_curriculum_end - self.cfg.formation_curriculum_start)
        ))
        if alpha <= 0.0:
            return self._zero_reward_N  # shape: [N]

        # --- Centroid-to-goal: reward group centroid for reaching target ---
        # Compute centroid from alive drones only when dropout is active
        if self.cfg.dropout_enabled:
            alive_g = self._agent_alive.reshape(G, A)  # shape: [G, A]
            alive_count = alive_g.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(2)  # [G, 1, 1]
            alive_pos = pos_grouped * alive_g.unsqueeze(2)  # zero dead positions [G, A, 3]
            centroid = alive_pos.sum(dim=1, keepdim=True) / alive_count  # [G, 1, 3]
        else:
            centroid = pos_grouped.mean(dim=1, keepdim=True)  # [G, 1, 3]
        centroid_squeezed = centroid.squeeze(1)  # [G, 3]
        self._cloud_centroid_dist[:] = torch.linalg.norm(
            centroid_squeezed - self._group_goal_local, dim=1
        )  # [G]
        centroid_mapped = 1 - torch.tanh(
            self._cloud_centroid_dist / self.cfg.cloud_centroid_goal_sigma
        )  # [G]
        centroid_reward = (
            self.cfg.cloud_centroid_goal_scale * centroid_mapped * self.step_dt
        )  # [G]
        # Broadcast to all drones in group: [G] -> [G, A]
        centroid_reward_broadcast = centroid_reward.unsqueeze(1).expand(G, A)

        # --- KNN cohesion + spacing (merged loop) ---
        # Cohesion: reward for mean KNN distance being small (stay near neighbors)
        # Spacing: penalty for nearest neighbor being too close or too far
        self._cloud_cohesion_reward.zero_()  # [G, A]
        self._cloud_spacing_penalty.zero_()  # [G, A]
        dead_g = None
        if self.cfg.dropout_enabled:
            dead_g = ~self._agent_alive.reshape(G, A)  # shape: [G, A]
        for i in range(A):
            diff = pos_grouped - pos_grouped[:, i:i+1, :]  # [G, A, 3]
            dists = torch.linalg.norm(diff, dim=2)  # [G, A]
            dists[:, i] = float("inf")
            # Exclude dead drones from neighbor distances
            if dead_g is not None:
                dists[dead_g] = float("inf")

            # KNN cohesion: mean distance to K nearest neighbors
            knn_dists, _ = torch.topk(dists, K, dim=1, largest=False)  # [G, K]
            mean_knn = knn_dists.mean(dim=1)  # [G]
            cohesion_mapped = 1 - torch.tanh(mean_knn / self.cfg.cloud_cohesion_sigma)
            self._cloud_cohesion_reward[:, i] = (
                self.cfg.cloud_cohesion_scale * cohesion_mapped * self.step_dt
            )

            # Spacing: nearest neighbor penalties
            nearest_dist = knn_dists[:, 0]  # [G] — closest neighbor
            too_far = torch.clamp(nearest_dist - self.cfg.cloud_max_neighbor_dist, min=0.0)
            too_close = torch.clamp(self.cfg.cloud_min_spacing - nearest_dist, min=0.0)
            self._cloud_spacing_penalty[:, i] = -(
                self.cfg.cloud_spacing_penalty * too_far
                + self.cfg.cloud_separation_penalty * too_close
            ) * self.step_dt

        total = alpha * (
            centroid_reward_broadcast + self._cloud_cohesion_reward + self._cloud_spacing_penalty
        )  # [G, A]
        # Zero reward for dead drones
        if dead_g is not None:
            total[dead_g] = 0.0

        # Log metrics
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["Metrics/centroid_to_goal_dist"] = self._cloud_centroid_dist.mean().item()
        self.extras["log"]["Metrics/mean_knn_dist"] = mean_knn.mean().item()
        self.extras["log"]["Metrics/formation_alpha"] = alpha

        return total.reshape(N)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.05,
            self._robot.data.root_pos_w[:, 2] > 2.0,
        )
        # Don't count dropped-out drones as dying (they fall but are already "dead")
        if self.cfg.dropout_enabled:
            died = died & self._agent_alive

        # Virtual collision detection: check pairwise distances within swarm groups
        if self.cfg.num_agents > 1 and self.cfg.collision_enabled:
            A = self.cfg.num_agents
            G = self._num_groups
            pos_local = self._robot.data.root_pos_w - self._terrain.env_origins  # shape: [N, 3]
            pos_g = pos_local.reshape(G, A, 3)  # shape: [G, A, 3]

            # Check all pairs for collision (alive pairs only)
            self._collision_count.zero_()  # shape: [G]
            collided_group = self._collision_count > 0  # init false, shape: [G]
            alive_g = self._agent_alive.reshape(G, A) if self.cfg.dropout_enabled else None
            r = self.cfg.collision_radius
            for i in range(A):
                for j in range(i + 1, A):
                    dist = torch.linalg.norm(
                        pos_g[:, i] - pos_g[:, j], dim=1
                    )  # shape: [G]
                    pair_collided = dist < r  # shape: [G]
                    # Skip collisions involving dead drones
                    if alive_g is not None:
                        pair_collided = pair_collided & alive_g[:, i] & alive_g[:, j]
                    self._collision_count += pair_collided.float()
                    collided_group = collided_group | pair_collided

            # Kill all drones in groups with collision
            collided_flat = collided_group.unsqueeze(1).expand(G, A).reshape(-1)  # shape: [N]
            died = died | collided_flat

            # Log collision count
            if "log" not in self.extras:
                self.extras["log"] = {}
            self.extras["log"]["Metrics/collision_pairs_per_step"] = self._collision_count.sum().item()

        # Log alive agent count for SwarmRaft monitoring
        if self.cfg.dropout_enabled and self.cfg.num_agents > 1:
            if "log" not in self.extras:
                self.extras["log"] = {}
            alive_per_group = self._agent_alive.reshape(self._num_groups, self.cfg.num_agents).sum(dim=1).float()
            self.extras["log"]["Metrics/alive_agents_per_group"] = alive_per_group.mean().item()

        # Collective resets: if any drone in a swarm group dies, all die
        if self.cfg.num_agents > 1 and self.cfg.collective_resets:
            A = self.cfg.num_agents
            G = self._num_groups
            died_grouped = died.reshape(G, A)
            any_died = died_grouped.any(dim=1)
            died = any_died.unsqueeze(1).expand(G, A).reshape(-1)

        # Debug: log which envs die and why (only first few envs, during play)
        if self._debug_draw is not None:
            A = min(self.cfg.num_agents, self.num_envs)
            for idx in range(A):
                if died[idx]:
                    z = self._robot.data.root_pos_w[idx, 2].item()
                    print(f"[DEBUG] env {idx} DIED: z={z:.4f}")
                if time_out[idx]:
                    print(f"[DEBUG] env {idx} TIMEOUT: ep_len={self.episode_length_buf[idx].item()}")

        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):  # noqa: C901
        if env_ids is None or len(env_ids) == self.num_envs:  # type: ignore[arg-type]
            env_ids = self._robot._ALL_INDICES

        # Log episode metrics
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums:
            avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out resets to avoid training spikes (skip for single group / play)
            if self.cfg.num_agents > 1 and self._num_groups > 1:
                # Sync episode length within each swarm group
                A = self.cfg.num_agents
                G = self._num_groups
                group_lengths = torch.randint(0, int(self.max_episode_length), (G,), device=self.device)
                self.episode_length_buf = group_lengths.unsqueeze(1).expand(G, A).reshape(-1).clone()
            elif self.cfg.num_agents <= 1:
                self.episode_length_buf = torch.randint_like(
                    self.episode_length_buf, high=int(self.max_episode_length)
                )

        self._actions[env_ids] = 0.0
        self._smoothed_actions[env_ids] = 0.0
        self._minco_pos[env_ids] = 0.0
        self._minco_vel[env_ids] = 0.0
        self._minco_acc[env_ids] = 0.0
        # SwarmRaft: revive all drones on reset
        self._agent_alive[env_ids] = True

        # Sample new goal positions
        if self.cfg.num_agents > 1 and self._formation_offsets is not None:
            # Group-aware: sample one centroid per group, assign formation offsets
            A = self.cfg.num_agents
            env_ids_t = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
            # Find unique groups being reset
            group_ids = torch.unique(env_ids_t // A)
            n_groups = len(group_ids)

            # SwarmRaft: sample random dropout step for each group
            if self.cfg.dropout_enabled:
                self._dropout_step[group_ids] = torch.randint(
                    self.cfg.dropout_step_min, self.cfg.dropout_step_max + 1,
                    (n_groups,), device=self.device,
                )

            # Sample or use fixed centroid per group
            if self.cfg.formation_centroid is not None:
                fc = self.cfg.formation_centroid
                centroid = torch.tensor([[fc[0], fc[1], fc[2]]], device=self.device).expand(n_groups, 3).clone()
            else:
                centroid_xy = torch.zeros(n_groups, 2, device=self.device).uniform_(-0.5, 0.5)
                centroid_z = torch.zeros(n_groups, 1, device=self.device).uniform_(0.5, 1.5)
                centroid = torch.cat([centroid_xy, centroid_z], dim=-1)  # [n_groups, 3]

            # Initialize forest centroid tracker from reset centroid
            if self.cfg.forest_enabled and hasattr(self, '_forest_centroid'):
                for g_idx in range(n_groups):
                    g = group_ids[g_idx]
                    self._forest_centroid[g] = centroid[g_idx]

            # Assign each drone in each group (nearest-slot matching)
            for g_idx in range(n_groups):
                g = group_ids[g_idx]
                # Store group goal in local coords for centroid-to-goal reward
                if self._cloud_mode:
                    self._group_goal_local[g] = centroid[g_idx]  # [3]

                if not self._cloud_mode and self._formation_offsets is not None:
                    # Greedy nearest-slot: each drone claims the closest unclaimed slot
                    group_start = g * A
                    drone_pos = (
                        self._robot.data.root_pos_w[group_start:group_start + A]
                        - self._terrain.env_origins[group_start:group_start + A]
                    )  # [A, 3] local positions
                    slot_pos = centroid[g_idx] + self._formation_offsets  # [A, 3]
                    # Compute pairwise distance matrix [A_drones, A_slots]
                    dist_matrix = torch.cdist(drone_pos, slot_pos)  # [A, A]
                    claimed = torch.zeros(A, dtype=torch.bool, device=self.device)
                    for _ in range(A):
                        # Mask already-claimed slots
                        dist_matrix[:, claimed] = float("inf")
                        # Find closest drone-slot pair
                        flat_idx = dist_matrix.argmin()
                        drone_i = flat_idx // A
                        slot_j = flat_idx % A
                        # Assign
                        drone_id = group_start + drone_i
                        self._desired_pos_w[drone_id] = (
                            slot_pos[slot_j] + self._terrain.env_origins[drone_id]
                        )
                        # Store slot offset from centroid for forest goal recomputation
                        if self.cfg.forest_enabled and hasattr(self, '_drone_slot_offset'):
                            self._drone_slot_offset[drone_id] = (
                                slot_pos[slot_j] - centroid[g_idx]
                            )  # [3]
                        claimed[slot_j] = True
                        dist_matrix[drone_i, :] = float("inf")  # drone assigned
                else:
                    for i in range(A):
                        drone_id = g * A + i
                        offset = torch.zeros(3, device=self.device)
                        self._desired_pos_w[drone_id] = (
                            centroid[g_idx]
                            + self._terrain.env_origins[drone_id]
                            + offset
                        )
        else:
            # Single-agent: independent random goals
            self._desired_pos_w[env_ids, :2] = torch.zeros_like(
                self._desired_pos_w[env_ids, :2]
            ).uniform_(-2.0, 2.0)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] = torch.zeros_like(
                self._desired_pos_w[env_ids, 2]
            ).uniform_(0.5, 1.5)

        # Reset robot state with random spawn position
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # Random XY offset within [-0.5, 0.5]m of env origin
        # Random XY offset scaled by spawn_radius (auto-scales with num_agents)
        r = self.cfg.spawn_radius
        default_root_state[:, 0] += torch.zeros(len(env_ids), device=self.device).uniform_(-r, r)
        default_root_state[:, 1] += torch.zeros(len(env_ids), device=self.device).uniform_(-r, r)
        # Spawn below goal height so drones fly up to formation
        default_root_state[:, 2] = 0.5

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
