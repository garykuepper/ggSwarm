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
        self._formation_active = A > 1 and self.cfg.formation_reward_scale > 0.0
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

        # Formation slot offsets — equilateral arrangement with target_spacing
        # Circumradius = target_spacing / (2 * sin(pi/A)) so pairwise distance = target_spacing
        # shape: [num_agents, 3] — XYZ offset from group centroid
        if A > 1:
            spacing = self.cfg.formation_target_spacing
            radius = spacing / (2 * math.sin(math.pi / A))
            offsets = []
            for i in range(A):
                angle = 2 * math.pi * i / A
                offsets.append([
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                    0.0,
                ])
            self._formation_offsets = torch.tensor(offsets, device=device)  # [A, 3]
        else:
            self._formation_offsets = None

        # Pre-allocate action tensors (reused every step)
        self._actions = torch.zeros(N, 4, device=device)
        self._thrust = torch.zeros(N, 1, 3, device=device)
        self._moment = torch.zeros(N, 1, 3, device=device)
        self._desired_pos_w = torch.zeros(N, 3, device=device)

        # Body ID, mass, weight
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Episode logging
        log_keys = ["lin_vel", "ang_vel", "distance_to_goal"]
        if self._formation_active:
            log_keys.append("formation")
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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = (
            self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        )
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

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

        # Draw altitude lines (first swarm group)
        if self._debug_draw is not None:
            self._debug_draw.clear_lines()
            colors = [
                (0.12, 0.47, 0.71, 0.9),  # tab:blue
                (1.0, 0.50, 0.05, 0.9),   # tab:orange
                (0.17, 0.63, 0.17, 0.9),   # tab:green
            ]
            A = min(self.cfg.num_agents, len(colors), self.num_envs)
            for i in range(A):
                pos = self._robot.data.root_pos_w[i].cpu().tolist()
                self._debug_draw.draw_lines(
                    [pos], [[pos[0], pos[1], 0.0]],
                    [colors[i % len(colors)]], [1.0],
                )

        return {"policy": obs}

    def _expand_obs_with_neighbors(self, obs: torch.Tensor) -> torch.Tensor:
        """Append relative neighbor positions to each drone's observation.

        Groups consecutive envs into swarms of num_agents. Each drone gets
        (num_agents-1) * 3 extra dims: relative XYZ to each neighbor.

        Args:
            obs: shape [num_envs, 12]

        Returns:
            shape [num_envs, 12 + (num_agents-1)*3]
        """
        N = self.num_envs
        A = self.cfg.num_agents
        G = self._num_groups
        # Subtract env origins to get local positions (removes env_spacing offset)
        pos_local = self._robot.data.root_pos_w - self._terrain.env_origins  # shape: [N, 3]

        # Reshape to [num_groups, num_agents, 3]
        pos_grouped = pos_local.reshape(G, A, 3)

        # For each drone, compute relative positions to all neighbors
        rel_parts = []
        for i in range(A):
            neighbors = []
            for j in range(A):
                if i == j:
                    continue
                rel = pos_grouped[:, j, :] - pos_grouped[:, i, :]  # [G, 3]
                neighbors.append(rel)
            rel_parts.append(torch.cat(neighbors, dim=-1))  # [G, (A-1)*3]

        # Stack and flatten: [G, A, (A-1)*3] -> [N, (A-1)*3]
        rel_all = torch.stack(rel_parts, dim=1).reshape(N, -1)

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

        rewards = {
            "lin_vel": self.cfg.lin_vel_reward_scale * lin_vel * self.step_dt,
            "ang_vel": self.cfg.ang_vel_reward_scale * ang_vel * self.step_dt,
            "distance_to_goal": self.cfg.distance_to_goal_reward_scale
            * distance_to_goal_mapped * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # --- Formation reward (Phase 2B, curriculum-scaled) ---
        if self._formation_active:
            formation_rew = self._compute_formation_reward()
            reward = reward + formation_rew
            rewards["formation"] = formation_rew

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
            return torch.zeros(N, device=self.device)

        # Compute mean pairwise distance error
        total_error = torch.zeros(G, device=self.device)
        for i, j in self._pair_indices:
            dist = torch.linalg.norm(pos_grouped[:, i, :] - pos_grouped[:, j, :], dim=1)
            total_error += torch.abs(dist - self.cfg.formation_target_spacing)
        mean_error = total_error / len(self._pair_indices)

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

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.05,
            self._robot.data.root_pos_w[:, 2] > 2.0,
        )

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
            # Spread out resets to avoid training spikes
            if self.cfg.num_agents > 1:
                # Sync episode length within each swarm group
                A = self.cfg.num_agents
                G = self._num_groups
                group_lengths = torch.randint(0, int(self.max_episode_length), (G,), device=self.device)
                self.episode_length_buf = group_lengths.unsqueeze(1).expand(G, A).reshape(-1).clone()
            else:
                self.episode_length_buf = torch.randint_like(
                    self.episode_length_buf, high=int(self.max_episode_length)
                )

        self._actions[env_ids] = 0.0

        # Sample new goal positions
        if self.cfg.num_agents > 1 and self._formation_offsets is not None:
            # Group-aware: sample one centroid per group, assign formation offsets
            A = self.cfg.num_agents
            env_ids_t = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
            # Find unique groups being reset
            group_ids = torch.unique(env_ids_t // A)
            n_groups = len(group_ids)

            # Sample or use fixed centroid per group
            if self.cfg.formation_centroid is not None:
                fc = self.cfg.formation_centroid
                centroid = torch.tensor([[fc[0], fc[1], fc[2]]], device=self.device).expand(n_groups, 3).clone()
            else:
                centroid_xy = torch.zeros(n_groups, 2, device=self.device).uniform_(-0.5, 0.5)
                centroid_z = torch.zeros(n_groups, 1, device=self.device).uniform_(0.5, 1.5)
                centroid = torch.cat([centroid_xy, centroid_z], dim=-1)  # [n_groups, 3]

            # Assign each drone in each group
            for g_idx in range(n_groups):
                g = group_ids[g_idx]
                for i in range(A):
                    drone_id = g * A + i
                    self._desired_pos_w[drone_id] = (
                        centroid[g_idx]
                        + self._terrain.env_origins[drone_id]
                        + self._formation_offsets[i]
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
        default_root_state[:, 0] += torch.zeros(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
        default_root_state[:, 1] += torch.zeros(len(env_ids), device=self.device).uniform_(-0.5, 0.5)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
