# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Single-drone hover baseline environment."""

from collections.abc import Sequence

import torch
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform

from .drone_hover_env_cfg import GGSwarmHoverEnvCfg
from .drone_swarm_env import GGSwarmMarlEnv


class GGSwarmHoverEnv(GGSwarmMarlEnv):
    """Hover-only env that learns stable spawn-hold flight."""

    cfg: GGSwarmHoverEnvCfg

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # shape: [num_envs, num_agents, 3]
        pos_w = self.robot.data.root_pos_w.view(self.num_envs, self.cfg.num_agents, 3)
        # shape: [num_envs, num_agents, 3]
        lin_vel_b = self.robot.data.root_lin_vel_b.view(
            self.num_envs, self.cfg.num_agents, 3
        )
        # shape: [num_envs, num_agents, 3]
        ang_vel_b = self.robot.data.root_ang_vel_b.view(
            self.num_envs, self.cfg.num_agents, 3
        )

        # shape: [num_envs, num_agents]
        dist_to_goal = torch.norm(self._desired_pos_w - pos_w, dim=-1)
        # shape: [num_envs, num_agents]
        above_hover_floor = (pos_w[:, :, 2] >= self.cfg.hover_reward_min_height).float()
        rew_pos = (
            torch.exp(-dist_to_goal / self.cfg.rew_pos_sigma)
            * self.cfg.rew_scale_pos
            * above_hover_floor
        )

        # shape: [num_envs, num_agents]
        rew_vel = torch.norm(lin_vel_b, dim=-1) * self.cfg.rew_scale_vel
        rew_ang_vel = torch.norm(ang_vel_b, dim=-1) * self.cfg.rew_scale_ang_vel
        rew_alive = torch.full_like(rew_pos, self.cfg.rew_scale_alive)

        # Major crash penalty near/at ground.
        # shape: [num_envs, num_agents]
        ground_hit = pos_w[:, :, 2] < self.cfg.ground_hit_height
        rew_ground_hit = ground_hit.float() * self.cfg.rew_scale_ground_hit
        rew_terminated = ground_hit.float() * self.cfg.rew_scale_terminated

        # shape: [num_envs, num_agents]
        total_rewards = (
            rew_pos + rew_vel + rew_ang_vel + rew_alive + rew_ground_hit + rew_terminated
        )
        return {
            agent: total_rewards[:, i] for i, agent in enumerate(self.cfg.possible_agents)
        }

    def _get_dones(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # shape: [num_envs, num_agents, 3]
        pos_w = self.robot.data.root_pos_w.view(self.num_envs, self.cfg.num_agents, 3)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # shape: [num_envs, num_agents]
        ground_hit = pos_w[:, :, 2] < self.cfg.ground_hit_height
        out_of_bounds_top = pos_w[:, :, 2] > self.cfg.max_height
        out_of_bounds = ground_hit | out_of_bounds_top

        terminated = {}
        time_outs = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            terminated[agent] = out_of_bounds[:, i]
            time_outs[agent] = time_out
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_tensor = env_ids

        robot_indices = (
            env_ids_tensor.unsqueeze(-1) * self.cfg.num_agents
            + torch.arange(self.cfg.num_agents, device=self.device)
        ).view(-1)

        num_resets = len(env_ids_tensor)
        # shape: [num_resets, num_agents, 3]
        random_pos = sample_uniform(
            -self.cfg.spawn_dist,
            self.cfg.spawn_dist,
            (num_resets, self.cfg.num_agents, 3),
            self.device,
        )
        random_pos[:, :, 2] = sample_uniform(
            0.7, 1.3, (num_resets, self.cfg.num_agents), self.device
        )

        # shape: [num_resets, 1, 3]
        env_origins = self.scene.env_origins[env_ids_tensor].unsqueeze(1)
        # shape: [num_resets, num_agents, 3]
        root_pos_w = random_pos + env_origins

        default_root_state = self.robot.data.default_root_state[: self.cfg.num_agents]
        root_state = default_root_state.clone().repeat(num_resets, 1, 1)
        root_state[:, :, :3] = root_pos_w

        random_yaw = sample_uniform(
            -self.cfg.spawn_yaw_range,
            self.cfg.spawn_yaw_range,
            (num_resets, self.cfg.num_agents),
            self.device,
        )
        root_state[:, :, 3:7] = quat_from_euler_xyz(
            torch.zeros_like(random_yaw),
            torch.zeros_like(random_yaw),
            random_yaw,
        )

        # Spawn-hold target: keep each episode target at reset pose.
        self._desired_pos_w[env_ids_tensor] = root_pos_w

        self.robot.write_root_pose_to_sim(root_state[:, :, :7].view(-1, 7), robot_indices)
        self.robot.write_root_velocity_to_sim(
            root_state[:, :, 7:13].view(-1, 6), robot_indices
        )
        joint_pos = self.robot.data.default_joint_pos[0].repeat(
            num_resets * self.cfg.num_agents, 1
        )
        joint_vel = self.robot.data.default_joint_vel[0].repeat(
            num_resets * self.cfg.num_agents, 1
        )
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, robot_indices)
