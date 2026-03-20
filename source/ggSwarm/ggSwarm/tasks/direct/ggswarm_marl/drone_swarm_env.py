# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import logging
import math
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

from .drone_swarm_env_cfg import GGSwarmMarlEnvCfg
from .contract_logic import (
    MarlRewardParams,
    compute_adjacency_matrix,
    compute_marl_rewards,
)

# import logger
logger = logging.getLogger("isaaclab")


class GGSwarmMarlEnv(DirectMARLEnv):
    """MARL environment for drone swarm formation control."""

    cfg: GGSwarmMarlEnvCfg

    def __init__(
        self,
        cfg: GGSwarmMarlEnvCfg,
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
        # Create a bright yellow material for high visibility
        yellow_material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.5, roughness=0.5)
        yellow_material_cfg.func("/World/Materials/DroneYellow", yellow_material_cfg)

        # Phase 1: Spawn source drones in environment 0
        for i in range(self.cfg.num_agents):
            prim_path = f"/World/envs/env_0/drone_{i}"
            self.cfg.robot_cfg.spawn.func(
                prim_path,
                self.cfg.robot_cfg.spawn,
            )
            # Apply the yellow material to the drone visuals
            # Note: For Crazyflie, visuals are nested under the body
            sim_utils.bind_visual_material(prim_path, "/World/Materials/DroneYellow")

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

        logger.info("GGSwarmMarlEnv._setup_scene done")

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
        # Ensure the composed wrenches are pushed to the simulator this step.
        # The NVIDIA quadcopter demo calls `robot.write_data_to_sim()` after setting forces/torques.
        self.robot.write_data_to_sim()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Issue 5: Cache GPU memory reads once per step
        # shape: [num_envs, num_agents, 3]
        pos_w = self.robot.data.root_pos_w.view(self.num_envs, self.cfg.num_agents, 3)
        # shape: [num_envs, num_agents, 4]
        quat_w = self.robot.data.root_quat_w.view(self.num_envs, self.cfg.num_agents, 4)
        # shape: [num_envs, num_agents, 3]
        lin_vel_b = self.robot.data.root_lin_vel_b.view(self.num_envs, self.cfg.num_agents, 3)
        # shape: [num_envs, num_agents, 3]
        ang_vel_b = self.robot.data.root_ang_vel_b.view(self.num_envs, self.cfg.num_agents, 3)
        # shape: [num_envs, num_agents, 3]
        proj_grav_b = self.robot.data.projected_gravity_b.view(self.num_envs, self.cfg.num_agents, 3)

        # Calculate relative position to individual goals in body frame
        # shape: [num_envs * num_agents, 3]
        target_pos = self._desired_pos_w.view(-1, 3)
        rel_pos_b, _ = subtract_frame_transforms(pos_w.view(-1, 3), quat_w.view(-1, 4), target_pos)
        # shape: [num_envs, num_agents, 3]
        rel_pos_b = rel_pos_b.view(self.num_envs, self.cfg.num_agents, 3)

        # Calculate Adjacency Matrix (Distance-based)
        # shape: [num_envs, num_agents, num_agents]
        adj_matrix = compute_adjacency_matrix(
            pos_w, graph_connectivity_radius=self.cfg.graph_connectivity_radius
        )

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
        # shape: [num_envs, num_agents, 3]
        pos_w = self.robot.data.root_pos_w.view(self.num_envs, self.cfg.num_agents, 3)
        # shape: [num_envs, num_agents, 3]
        lin_vel_b = self.robot.data.root_lin_vel_b.view(self.num_envs, self.cfg.num_agents, 3)
        # shape: [num_envs, num_agents, 3]
        ang_vel_b = self.robot.data.root_ang_vel_b.view(self.num_envs, self.cfg.num_agents, 3)
        # shape: [num_envs, num_agents, 3]
        desired_pos_w = self._desired_pos_w

        params = MarlRewardParams(
            curriculum_start_step=self.cfg.curriculum_start_step,
            curriculum_end_step=self.cfg.curriculum_end_step,
            rew_scale_pos=self.cfg.rew_scale_pos,
            rew_scale_formation=self.cfg.rew_scale_formation,
            rew_scale_cohesion=self.cfg.rew_scale_cohesion,
            rew_scale_separation=self.cfg.rew_scale_separation,
            rew_scale_vel=self.cfg.rew_scale_vel,
            rew_scale_ang_vel=self.cfg.rew_scale_ang_vel,
            rew_scale_alive=self.cfg.rew_scale_alive,
            rew_scale_terminated=self.cfg.rew_scale_terminated,
            rew_pos_sigma=self.cfg.rew_pos_sigma,
            rew_formation_sigma=self.cfg.rew_formation_sigma,
            target_formation_dist=self.cfg.target_formation_dist,
            graph_connectivity_radius=self.cfg.graph_connectivity_radius,
            min_separation_dist=self.cfg.min_separation_dist,
            min_height=self.cfg.min_height,
            max_height=self.cfg.max_height,
        )

        # shape: [num_envs, num_agents]
        total_rewards = compute_marl_rewards(
            pos_w=pos_w,
            desired_pos_w=desired_pos_w,
            lin_vel_b=lin_vel_b,
            ang_vel_b=ang_vel_b,
            common_step_counter=int(self.common_step_counter),
            params=params,
        )

        rewards: dict[str, torch.Tensor] = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            rewards[agent] = total_rewards[:, i]
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

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
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
        num_resets = len(env_ids_tensor)
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

        # Goals: assign deterministic formation slots (Phase 2)
        # This provides a stable per-agent reference for `rel_pos_b` so the
        # formation task is learnable and doesn't rely purely on emergent symmetry breaking.
        #
        # shape: [num_agents]
        angles = torch.linspace(
            0.0,
            2.0 * math.pi,
            self.cfg.num_agents + 1,
            device=self.device,
            dtype=torch.float32,
        )[:-1]
        # For desired spacing `d`, choose a circle radius with circumference ~= N*d.
        radius = (self.cfg.num_agents * self.cfg.target_formation_dist) / (2.0 * math.pi)
        # shape: [num_agents, 3]
        offsets = torch.stack(
            [
                radius * torch.cos(angles),
                radius * torch.sin(angles),
                torch.zeros_like(angles),
            ],
            dim=-1,
        )
        # shape: [num_resets, num_agents, 3]
        desired_pos_w = env_origins + offsets.unsqueeze(0)
        # Keep Z goal equal to each agent's spawn height to avoid fighting altitude early.
        desired_pos_w[:, :, 2] = root_pos_w[:, :, 2]
        self._desired_pos_w[env_ids_tensor] = desired_pos_w

        # Write to sim using robot_indices
        self.robot.write_root_pose_to_sim(root_state[:, :, :7].view(-1, 7), robot_indices)
        self.robot.write_root_velocity_to_sim(root_state[:, :, 7:13].view(-1, 6), robot_indices)
        # Crazyflie joint states (rotors)
        joint_pos = self.robot.data.default_joint_pos[0].repeat(num_resets * self.cfg.num_agents, 1)
        joint_vel = self.robot.data.default_joint_vel[0].repeat(num_resets * self.cfg.num_agents, 1)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, robot_indices)
