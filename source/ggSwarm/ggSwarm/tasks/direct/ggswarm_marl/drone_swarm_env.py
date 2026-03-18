# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import logging
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import (
    GroundPlaneCfg,
    spawn_ground_plane,
)
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    sample_uniform,
    subtract_frame_transforms,
)

from .drone_swarm_env_cfg import GgswarmMarlEnvCfg

# import logger
logger = logging.getLogger("isaaclab")


class GgswarmMarlEnv(DirectMARLEnv):
    """MARL environment for drone swarm formation control."""

    cfg: GgswarmMarlEnvCfg

    def __init__(
        self,
        cfg: GgswarmMarlEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Get specific body indices (using "body" from Crazyflie asset)
        self._body_indices = self.robot.find_bodies("body")[0]
        # Get propeller body indices for force application
        self._prop_body_indices = self.robot.find_bodies("m.*_prop")[0]

        self._robot_mass = self.robot.root_physx_view.get_masses()[0].sum()
        gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)
        self._gravity_magnitude = gravity.norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Buffers for actions and goal positions
        self._desired_pos_w = torch.zeros(self.num_envs, self.cfg.num_agents, 3, device=self.device)
        self._prop_forces = torch.zeros(self.num_envs * self.cfg.num_agents, 4, 3, device=self.device)
        self._prop_torques = torch.zeros(self.num_envs * self.cfg.num_agents, 4, 3, device=self.device)

    def _setup_scene(self):
        # Phase 1: Spawn source drones in environment 0
        for i in range(self.cfg.num_agents):
            self.cfg.robot_cfg.spawn.func(
                f"/World/envs/env_0/drone_{i}",
                self.cfg.robot_cfg.spawn,
            )

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Phase 2: Clone and replicate environments
        # This repeats the drones across all environments
        logger.info("Cloning environments...")
        self.scene.clone_environments(copy_from_source=True)

        # Phase 3: Initialize Articulation view
        prim_path = self.cfg.robot_cfg.prim_path
        logger.info(f"Initializing Articulation with path: {prim_path}")
        self.robot = Articulation(self.cfg.robot_cfg)
        logger.info("Articulation initialized")

        # Add articulation to scene
        self.scene.articulations["robot"] = self.robot

        logger.info("GgswarmMarlEnv._setup_scene done")

        # Filter collisions for CPU
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        # Stack actions into [num_envs, num_agents, 4]
        agent_actions = []
        for agent in self.cfg.possible_agents:
            agent_actions.append(actions[agent].unsqueeze(1))
        all_actions = torch.cat(agent_actions, dim=1).clamp(-1.0, 1.0)

        # Reshape to (num_instances, 4) (num_instances = num_envs * num_agents)
        flat_actions = all_actions.view(-1, 4)

        # Map actions to thrust (Z-axis) and moments (XYZ)
        # Issue 4: Document thrust mapping (neutral hover action is ~0.0)
        thrust_val = (flat_actions[:, 0] + 1.0) / 2.0
        weight = self._robot_weight
        total_thrust = self.cfg.thrust_to_weight * weight * thrust_val
        self._prop_forces[:, :, 2] = total_thrust.unsqueeze(-1) / 4.0

        # Apply moments
        torques = self.cfg.moment_scale * flat_actions[:, 1:]
        self._prop_torques[:, :, :] = torques.unsqueeze(-2) / 4.0

    def _apply_action(self) -> None:
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._prop_body_indices,
            forces=self._prop_forces,
            torques=self._prop_torques,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Issue 5: Cache GPU memory reads once per step
        pos_w = self.robot.data.root_pos_w.view(self.num_envs, self.cfg.num_agents, 3)
        quat_w = self.robot.data.root_quat_w.view(self.num_envs, self.cfg.num_agents, 4)
        lin_vel_b = self.robot.data.root_lin_vel_b.view(self.num_envs, self.cfg.num_agents, 3)
        ang_vel_b = self.robot.data.root_ang_vel_b.view(self.num_envs, self.cfg.num_agents, 3)
        proj_grav_b = self.robot.data.projected_gravity_b.view(self.num_envs, self.cfg.num_agents, 3)

        # Calculate relative position to individual goals in body frame
        target_pos = self._desired_pos_w.view(-1, 3)
        rel_pos_b, _ = subtract_frame_transforms(pos_w.view(-1, 3), quat_w.view(-1, 4), target_pos)
        rel_pos_b = rel_pos_b.view(self.num_envs, self.cfg.num_agents, 3)

        # Calculate Adjacency Matrix (Distance-based)
        # Inter-drone distances: [num_envs, num_agents, num_agents]
        diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        # Threshold at 2.0m for connectivity
        adj_matrix = (dist < 2.0).float()
        # Issue 3: Remove self-connections
        eye = torch.eye(self.cfg.num_agents, device=self.device).unsqueeze(0)
        adj_matrix = adj_matrix * (1.0 - eye)

        observations = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            observations[agent] = torch.cat(
                [
                    lin_vel_b[:, i],
                    ang_vel_b[:, i],
                    proj_grav_b[:, i],
                    rel_pos_b[:, i],
                ],
                dim=-1,
            )
        # We store the adjacency matrix in extras for now, used by GATv2 later
        self.extras["adj_matrix"] = adj_matrix
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # Issue 5: Cache GPU memory reads once per step
        pos_w = self.robot.data.root_pos_w.view(self.num_envs, self.cfg.num_agents, 3)
        lin_vel_b = self.robot.data.root_lin_vel_b.view(self.num_envs, self.cfg.num_agents, 3)
        ang_vel_b = self.robot.data.root_ang_vel_b.view(self.num_envs, self.cfg.num_agents, 3)

        dist_to_goal = torch.norm(self._desired_pos_w - pos_w, dim=-1)

        rewards = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            # Distance reward (Gaussian-like)
            rew_pos = torch.exp(-dist_to_goal[:, i] / 0.5) * self.cfg.rew_scale_pos
            # Velocity penalties
            rew_vel = torch.norm(lin_vel_b[:, i], dim=-1) * self.cfg.rew_scale_vel
            rew_scale = self.cfg.rew_scale_ang_vel
            rew_ang_vel = torch.norm(ang_vel_b[:, i], dim=-1) * rew_scale
            # Alive bonus
            rew_alive = self.cfg.rew_scale_alive

            # Issue 1: Apply termination penalty
            is_terminated = (pos_w[:, i, 2] < self.cfg.min_height) | (pos_w[:, i, 2] > self.cfg.max_height)
            rew_terminated = is_terminated.float() * self.cfg.rew_scale_terminated

            rewards[agent] = rew_pos + rew_vel + rew_ang_vel + rew_alive + rew_terminated
        return rewards

    def _get_dones(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        pos_w = self.robot.data.root_pos_w.view(self.num_envs, self.cfg.num_agents, 3)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Check for crashes or out of bounds
        out_of_bounds = (pos_w[:, :, 2] < self.cfg.min_height) | (pos_w[:, :, 2] > self.cfg.max_height)

        terminated = {}
        time_outs = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            # Terminate if individual drone goes out of bounds
            terminated[agent] = out_of_bounds[:, i]
            time_outs[agent] = time_out

        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids_tensor = env_ids

        # Get robot indices corresponding to the environment IDs
        robot_indices = (
            env_ids_tensor.unsqueeze(-1) * self.cfg.num_agents + torch.arange(self.cfg.num_agents, device=self.device)
        ).view(-1)

        # Reset robot states to random positions centered around origin
        num_resets = len(env_ids)
        # [num_resets, num_agents, 3]
        random_pos = sample_uniform(
            -self.cfg.spawn_dist,
            self.cfg.spawn_dist,
            (num_resets, self.cfg.num_agents, 3),
            self.device,
        )
        # Set Z height comfortably
        random_pos[:, :, 2] = sample_uniform(0.5, 1.5, (num_resets, self.cfg.num_agents), self.device)

        # Apply environment origins
        env_origins = self.scene.env_origins[env_ids_tensor].unsqueeze(1)
        root_pos_w = random_pos + env_origins

        # Set default root state
        # Issue 6: Safer default root state indexing
        default_root_state = self.robot.data.default_root_state[: self.cfg.num_agents]
        root_state = default_root_state.clone().repeat(num_resets, 1, 1)
        root_state[:, :, :3] = root_pos_w
        # Random yaw rotations
        random_yaw = sample_uniform(
            -3.14,
            3.14,
            (num_resets, self.cfg.num_agents),
            self.device,
        )
        root_state[:, :, 3:7] = quat_from_euler_xyz(
            torch.zeros_like(random_yaw),
            torch.zeros_like(random_yaw),
            random_yaw,
        )

        # Goals: hovering at initial position for Phase 1
        self._desired_pos_w[env_ids_tensor] = root_pos_w

        # Write to sim using robot_indices
        self.robot.write_root_pose_to_sim(root_state[:, :, :7].view(-1, 7), robot_indices)
        self.robot.write_root_velocity_to_sim(root_state[:, :, 7:13].view(-1, 6), robot_indices)
        # Crazyflie joint states (rotors)
        joint_pos = self.robot.data.default_joint_pos[0].repeat(num_resets * self.cfg.num_agents, 1)
        joint_vel = self.robot.data.default_joint_vel[0].repeat(num_resets * self.cfg.num_agents, 1)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, robot_indices)
