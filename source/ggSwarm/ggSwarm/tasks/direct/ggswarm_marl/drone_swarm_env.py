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
    StableHoverRewardParams,
    compute_adjacency_matrix,
    compute_marl_rewards,
    compute_stable_hover_rewards,
)
from .cbf_safety import CbfParams, apply_cbf_safety, apply_cbf_obstacle_safety
from .minco_trajectory import MincoParams, MincoSmoother
from .swarm_raft import SwarmRaft, SwarmRaftParams
from .attitude_controller import AttitudeControllerParams, compute_attitude_control

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

        # Get main body index for force/torque application (matches Isaac Lab reference)
        self._body_indices = self.robot.find_bodies("body")[0]

        self._robot_mass = self.robot.root_physx_view.get_masses()[0].sum()
        gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)
        self._gravity_magnitude = gravity.norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Build attitude controller params from cfg (Rule 6 — no magic numbers here)
        self._att_ctrl_params = AttitudeControllerParams(
            kp_att=self.cfg.kp_att,
            kd_att=self.cfg.kd_att,
            kp_yaw=self.cfg.kp_yaw,
            max_tilt_angle=self.cfg.max_tilt_angle,
            max_yaw_rate=self.cfg.max_yaw_rate,
            max_moment=self.cfg.max_moment,
            thrust_to_weight=self.cfg.thrust_to_weight,
        )
        logger.info(
            "AttitudeController active | kp_att=%.4f kd_att=%.4f kp_yaw=%.4f",
            self.cfg.kp_att,
            self.cfg.kd_att,
            self.cfg.kp_yaw,
        )
        if self.cfg.use_stable_hover_rewards:
            logger.info(
                "Reward path: compute_stable_hover_rewards (tanh pos, squared vel, dt-scaled)"
            )

        num_instances = self.num_envs * self.cfg.num_agents
        # Buffers for actions and goal positions
        self._desired_pos_w = torch.zeros(self.num_envs, self.cfg.num_agents, 3, device=self.device)
        # Thrust applied to main body (body-frame Z); pre-allocated, reused every step
        self._thrust = torch.zeros(num_instances, 1, 3, device=self.device)  # pre-allocated; reused every step
        # Roll/pitch/yaw moments applied to main body; pre-allocated, reused every step
        self._moment = torch.zeros(num_instances, 1, 3, device=self.device)  # pre-allocated; reused every step
        # PD moments before max_moment clamp; only used when action telemetry is enabled
        self._moment_pre_clamp: torch.Tensor | None = None
        if self.cfg.action_telemetry_max_env_steps > 0:
            self._moment_pre_clamp = torch.zeros(
                num_instances, 1, 3, device=self.device
            )  # pre-allocated; reused every step
        self._pending_action_telemetry: dict[str, torch.Tensor] = {}

        # --- Phase 3: SwarmRaft Consensus (L3, gated by cfg.raft_enabled) ---
        if self.cfg.raft_enabled:
            raft_params = SwarmRaftParams(
                heartbeat_timeout=self.cfg.raft_heartbeat_timeout,
                replan_cooldown=self.cfg.raft_replan_cooldown,
                target_formation_dist=self.cfg.target_formation_dist,
                formation_height=self.cfg.raft_formation_height,  # Rule 6: cfg param, not hardcoded
            )
            self._swarm_raft = SwarmRaft(
                num_envs=self.num_envs,
                num_agents=self.cfg.num_agents,
                params=raft_params,
                env_origins=self.scene.env_origins,
                device=self.device,
            )
            # Cache latest consensus obs features for use in _get_observations
            # shape: [num_envs, num_agents]
            self._raft_is_leader = torch.zeros(
                self.num_envs, self.cfg.num_agents, device=self.device
            )
            self._raft_alive_frac = torch.ones(
                self.num_envs, self.cfg.num_agents, device=self.device
            )

        # --- Phase 4: Obstacle positions buffer (gated by cfg.obstacle_enabled) ---
        # Populated in _setup_scene; used by CBF obstacle safety in _pre_physics_step.
        # shape: [num_obstacles, 3]  (world-frame, env_0 reference -- replicated across envs)
        self._obstacle_positions: torch.Tensor = torch.zeros(0, 3, device=self.device)

        # --- Phase 4: Agent loss simulation state (gated by cfg.agent_loss_enabled) ---
        if self.cfg.agent_loss_enabled:
            # Per-environment step at which to kill a random agent this episode.
            # -1 means no kill scheduled yet for this episode.
            # shape: [num_envs]
            self._kill_step = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
            # Which agent to kill per env (-1 = none).
            # shape: [num_envs]
            self._kill_agent_idx = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
            # Track which agents have been forcibly terminated this episode.
            # shape: [num_envs, num_agents]
            self._force_terminated = torch.zeros(
                self.num_envs, self.cfg.num_agents, dtype=torch.bool, device=self.device
            )

        # --- Phase 3: MINCO Trajectory Smoother (L5, gated by cfg.minco_enabled) ---
        if self.cfg.minco_enabled:
            minco_params = MincoParams(
                smoothing_alpha=self.cfg.minco_smoothing_alpha,
                buffer_size=self.cfg.minco_buffer_size,
            )
            self._minco = MincoSmoother(
                num_envs=self.num_envs,
                num_agents=self.cfg.num_agents,
                action_dim=4,
                params=minco_params,
                device=self.device,
            )

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

        # --- Phase 4: Spawn obstacles (gated by cfg.obstacle_enabled) ---
        if self.cfg.obstacle_enabled:
            import random as _random
            _random.seed(42)
            obstacle_positions = []
            half = self.cfg.obstacle_field_size
            for k in range(self.cfg.obstacle_count):
                x = _random.uniform(-half, half)
                y = _random.uniform(-half, half)
                pos = (x, y, self.cfg.obstacle_height / 2.0)
                obstacle_positions.append(pos)
                prim_path = f"/World/envs/env_0/obstacle_{k}"
                obstacle_cfg = sim_utils.CapsuleCfg(
                    radius=self.cfg.obstacle_radius,
                    height=self.cfg.obstacle_height,
                    axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.8, 0.2, 0.2), metallic=0.1
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                )
                obstacle_cfg.func(prim_path, obstacle_cfg, translation=pos)
            # Store obstacle centres for CBF (world frame, env_0 offset = env_origins[0])
            obs_tensor = torch.tensor(obstacle_positions, dtype=torch.float32)
            self._obstacle_positions = obs_tensor.to(self.device)
            logger.info(f"Spawned {self.cfg.obstacle_count} obstacles.")

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
        raw_actions = torch.cat(agent_actions, dim=1)
        # Clamp to gym action space; SKRL Gaussian policies may sample outside [-1, 1].
        all_actions = raw_actions.clamp(-1.0, 1.0)

        # --- L4: CBF Safety Shield (Phase 3, gated by cfg.cbf_enabled) ---
        if self.cfg.cbf_enabled:
            # shape: [num_envs, num_agents, 3]
            pos_w = self.robot.data.root_pos_w.view(self.num_envs, self.cfg.num_agents, 3)
            lin_vel_w = self.robot.data.root_lin_vel_w.view(
                self.num_envs, self.cfg.num_agents, 3
            )
            cbf_params = CbfParams(
                d_safe=self.cfg.cbf_d_safe,
                gamma=self.cfg.cbf_gamma,
                activation_margin=self.cfg.cbf_activation_margin,
            )
            all_actions, cbf_rate = apply_cbf_safety(all_actions, pos_w, lin_vel_w, cbf_params)
            if "log" not in self.extras:
                self.extras["log"] = {}
            self.extras["log"]["cbf_intervention_rate"] = cbf_rate

            # --- Phase 4: CBF obstacle avoidance extension ---
            if self.cfg.obstacle_enabled and self._obstacle_positions.shape[0] > 0:
                all_actions, obs_cbf_rate = apply_cbf_obstacle_safety(
                    all_actions,
                    pos_w,
                    lin_vel_w,
                    obstacle_positions=self._obstacle_positions,
                    obstacle_d_safe=self.cfg.cbf_obstacle_d_safe,
                    gamma=self.cfg.cbf_gamma,
                )
                self.extras["log"]["cbf_obstacle_intervention_rate"] = obs_cbf_rate

        # --- L5: MINCO Trajectory Smoother (Phase 3, gated by cfg.minco_enabled) ---
        if self.cfg.minco_enabled and hasattr(self, "_minco"):
            all_actions = self._minco.smooth(all_actions)

        # Reshape to (num_instances, 4) (num_instances = num_envs * num_agents)
        flat_actions = all_actions.view(-1, 4)

        # Read current attitude state for the PD controller
        # shape: [num_instances, 3]
        proj_grav_b = self.robot.data.projected_gravity_b
        ang_vel_b = self.robot.data.root_ang_vel_b

        # PD attitude controller: converts [thrust_cmd, desired_roll, desired_pitch,
        # desired_yaw_rate] into body-frame thrust force and moments.
        # Writes in-place into pre-allocated self._thrust and self._moment (Rule 15).
        compute_attitude_control(
            actions=flat_actions,
            proj_grav_b=proj_grav_b,
            ang_vel_b=ang_vel_b,
            robot_weight=self._robot_weight,
            params=self._att_ctrl_params,
            thrust_buf=self._thrust,
            moment_buf=self._moment,
            moment_pre_clamp_buf=self._moment_pre_clamp,
        )

        # Optional TensorBoard telemetry (first N env steps only).
        # Values stored as 0-dim tensors so SKRL's isinstance+numel check passes
        # and they appear as "Info / *" scalars in TensorBoard.
        self._pending_action_telemetry = {}
        if (
            self.cfg.action_telemetry_max_env_steps > 0
            and int(self.common_step_counter) < self.cfg.action_telemetry_max_env_steps
        ):
            mm = float(self.cfg.max_moment)
            ra = raw_actions[..., 0]
            ca = all_actions[..., 0]
            self._pending_action_telemetry["act_raw_thrust_mean"] = ra.mean()
            self._pending_action_telemetry["act_raw_thrust_std"] = ra.std(unbiased=False)
            self._pending_action_telemetry["act_raw_thrust_min"] = ra.min()
            self._pending_action_telemetry["act_raw_thrust_max"] = ra.max()
            self._pending_action_telemetry["act_clamped_thrust_mean"] = ca.mean()
            self._pending_action_telemetry["act_clamp_hit_frac"] = (raw_actions != all_actions).float().mean()
            thrust_val = (flat_actions[:, 0] + 1.0) * 0.5
            self._pending_action_telemetry["thrust_val_mean"] = thrust_val.mean()
            if self._moment_pre_clamp is not None:
                pre = self._moment_pre_clamp[:, 0, :]
                self._pending_action_telemetry["moment_pre_abs_max_mean"] = pre.abs().max(dim=1).values.mean()
                self._pending_action_telemetry["moment_saturated_frac"] = (
                    (pre.abs() >= mm - 1e-6).any(dim=-1).float().mean()
                )

    def _apply_action(self) -> None:
        # Apply thrust (body-frame Z) and attitude moments to the main body in a
        # single call, matching the Isaac Lab Isaac-Quadcopter-Direct-v0 reference.
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_indices,
            forces=self._thrust,
            torques=self._moment,
        )
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

        # Calculate Adjacency Matrix (Distance-based)
        # shape: [num_envs, num_agents, num_agents]
        adj_matrix = compute_adjacency_matrix(
            pos_w, graph_connectivity_radius=self.cfg.graph_connectivity_radius
        )

        # --- L3: SwarmRaft Consensus tick (Phase 3, gated by cfg.raft_enabled) ---
        if self.cfg.raft_enabled and hasattr(self, "_swarm_raft"):
            if int(self.common_step_counter) % self.cfg.raft_tick_interval == 0:
                # Derive alive mask from height bounds (same condition as _get_dones)
                # shape: [num_envs, num_agents]
                agent_alive_step = (
                    (pos_w[:, :, 2] >= self.cfg.min_height)
                    & (pos_w[:, :, 2] <= self.cfg.max_height)
                )
                self._desired_pos_w, self._raft_is_leader, self._raft_alive_frac = (
                    self._swarm_raft.tick(agent_alive_step, self._desired_pos_w)
                )

        # Recalculate rel_pos_b after potential SwarmRaft goal update
        target_pos = self._desired_pos_w.view(-1, 3)
        rel_pos_b, _ = subtract_frame_transforms(pos_w.view(-1, 3), quat_w.view(-1, 4), target_pos)
        # shape: [num_envs, num_agents, 3]
        rel_pos_b = rel_pos_b.view(self.num_envs, self.cfg.num_agents, 3)

        observations = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            base_obs = torch.cat(
                [
                    lin_vel_b[:, i],
                    ang_vel_b[:, i],
                    proj_grav_b[:, i],
                    rel_pos_b[:, i],
                ],
                dim=-1,
            )
            if self.cfg.raft_enabled and hasattr(self, "_raft_is_leader"):
                # Extend obs to 14 dims: [12-dim base, is_leader, num_alive_frac]
                # shape: [num_envs, 14]
                observations[agent] = torch.cat(
                    [
                        base_obs,
                        self._raft_is_leader[:, i].unsqueeze(-1),
                        self._raft_alive_frac[:, i].unsqueeze(-1),
                    ],
                    dim=-1,
                )
            else:
                observations[agent] = base_obs

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

        if self.cfg.use_stable_hover_rewards:
            sh_params = StableHoverRewardParams(
                rew_scale_pos=self.cfg.rew_scale_pos,
                rew_scale_vel=self.cfg.rew_scale_vel,
                rew_scale_ang_vel=self.cfg.rew_scale_ang_vel,
                step_dt=self.step_dt,
                min_height=self.cfg.min_height,
                low_clearance_margin_m=self.cfg.low_clearance_margin_m,
                rew_scale_low_clearance=self.cfg.rew_scale_low_clearance,
                rew_scale_terminated=self.cfg.rew_scale_terminated,
            )
            total_rewards, terms_dict = compute_stable_hover_rewards(
                pos_w=pos_w,
                desired_pos_w=desired_pos_w,
                lin_vel_b=lin_vel_b,
                ang_vel_b=ang_vel_b,
                params=sh_params,
                return_terms=True,
            )
        else:
            # shape: [num_envs, num_agents, 3]
            proj_grav_b = self.robot.data.projected_gravity_b.view(
                self.num_envs, self.cfg.num_agents, 3
            )
            params = MarlRewardParams(
                curriculum_start_step=self.cfg.curriculum_start_step,
                curriculum_end_step=self.cfg.curriculum_end_step,
                curriculum_pos_floor=self.cfg.curriculum_pos_floor,
                rew_scale_pos=self.cfg.rew_scale_pos,
                rew_scale_formation=self.cfg.rew_scale_formation,
                rew_scale_cohesion=self.cfg.rew_scale_cohesion,
                rew_scale_separation=self.cfg.rew_scale_separation,
                rew_scale_vel=self.cfg.rew_scale_vel,
                rew_scale_ang_vel=self.cfg.rew_scale_ang_vel,
                rew_scale_alive=self.cfg.rew_scale_alive,
                rew_scale_terminated=self.cfg.rew_scale_terminated,
                rew_scale_upright=self.cfg.rew_scale_upright,
                rew_pos_sigma=self.cfg.rew_pos_sigma,
                rew_formation_sigma=self.cfg.rew_formation_sigma,
                target_formation_dist=self.cfg.target_formation_dist,
                graph_connectivity_radius=self.cfg.graph_connectivity_radius,
                min_separation_dist=self.cfg.min_separation_dist,
                min_height=self.cfg.min_height,
                max_height=self.cfg.max_height,
                rew_scale_low_clearance=self.cfg.rew_scale_low_clearance,
                low_clearance_margin_m=self.cfg.low_clearance_margin_m,
            )

            # shape: [num_envs, num_agents]
            total_rewards, terms_dict = compute_marl_rewards(
                pos_w=pos_w,
                desired_pos_w=desired_pos_w,
                lin_vel_b=lin_vel_b,
                ang_vel_b=ang_vel_b,
                proj_grav_b=proj_grav_b,
                common_step_counter=int(self.common_step_counter),
                params=params,
                return_terms=True,
            )

        # Compute curriculum alpha for logging
        from .contract_logic import compute_curriculum_alpha
        alpha = compute_curriculum_alpha(
            int(self.common_step_counter),
            curriculum_start_step=self.cfg.curriculum_start_step,
            curriculum_end_step=self.cfg.curriculum_end_step,
        )

        # Log per-term rewards for TensorBoard visualization.
        # Values must be 0-dim torch.Tensor (not Python float) so SKRL's
        # isinstance(v, torch.Tensor) and v.numel() == 1 check passes and
        # they appear as "Info / *" scalars in TensorBoard.
        low_clearance_z = self.cfg.min_height + self.cfg.low_clearance_margin_m
        self.extras["log"] = {
            "rew_pos": terms_dict["rew_pos"].mean(),
            "rew_formation": terms_dict["rew_formation"].mean(),
            "rew_cohesion": terms_dict["rew_cohesion"].mean(),
            "rew_separation": terms_dict["rew_separation"].mean(),
            "rew_upright": terms_dict["rew_upright"].mean(),
            "rew_vel": terms_dict["rew_vel"].mean(),
            "rew_ang_vel": terms_dict["rew_ang_vel"].mean(),
            "rew_alive": terms_dict["rew_alive"].mean(),
            "rew_terminated": terms_dict["rew_terminated"].mean(),
            "rew_low_clearance": terms_dict["rew_low_clearance"].mean(),
            "mean_world_z": pos_w[:, :, 2].mean(),
            "low_clearance_frac": (pos_w[:, :, 2] < low_clearance_z).float().mean(),
            "curriculum_alpha": torch.tensor(alpha, device=self.device),
        }
        if self._pending_action_telemetry:
            self.extras["log"].update(self._pending_action_telemetry)

        rewards: dict[str, torch.Tensor] = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            rewards[agent] = total_rewards[:, i]
        return rewards

    def _get_dones(  # noqa: C901  # grandfathered — do not add branches
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        pos_w = self.robot.data.root_pos_w.view(self.num_envs, self.cfg.num_agents, 3)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Check for crashes or out of bounds
        out_of_bounds = (pos_w[:, :, 2] < self.cfg.min_height) | (pos_w[:, :, 2] > self.cfg.max_height)

        # --- Phase 4: Simulated agent loss (gated by cfg.agent_loss_enabled) ---
        if self.cfg.agent_loss_enabled and hasattr(self, "_kill_step"):
            ep_step = self.episode_length_buf  # shape: [num_envs]

            # Schedule kills for envs where no kill is scheduled yet
            unscheduled = self._kill_step < 0
            if unscheduled.any():
                lo = self.cfg.agent_loss_interval_min
                hi = max(self.cfg.agent_loss_interval_max, lo + 1)
                rand_steps = torch.randint(
                    lo, hi, (self.num_envs,), device=self.device
                )
                rand_agents = torch.randint(
                    0, self.cfg.num_agents, (self.num_envs,), device=self.device
                )
                self._kill_step = torch.where(unscheduled, rand_steps, self._kill_step)
                self._kill_agent_idx = torch.where(
                    unscheduled, rand_agents, self._kill_agent_idx
                )

            # Fire kills where scheduled step has been reached
            fire = (ep_step >= self._kill_step) & (self._kill_step >= 0)
            if fire.any():
                env_ids = fire.nonzero(as_tuple=False).squeeze(-1)
                for env_id in env_ids:
                    agent_idx = int(self._kill_agent_idx[env_id].item())
                    if not self._force_terminated[env_id, agent_idx]:
                        self._force_terminated[env_id, agent_idx] = True
                        # Teleport the drone below the floor so it triggers OOB termination
                        self._desired_pos_w[env_id, agent_idx, 2] = -10.0

            # Merge forced terminations into out_of_bounds
            out_of_bounds = out_of_bounds | self._force_terminated

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

        # Stagger episode lengths on full reset to prevent correlated data spikes
        # (Isaac Lab pattern: spread out resets so many envs don't terminate simultaneously).
        if len(env_ids_tensor) == self.num_envs:
            self.episode_length_buf = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

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
        # Set Z height from cfg params; changing min_height/max_height alone does NOT affect spawn (Rule 6).
        random_pos[:, :, 2] = sample_uniform(
            self.cfg.spawn_z_min, self.cfg.spawn_z_max, (num_resets, self.cfg.num_agents), self.device
        )

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
        # Chord-length formula: place agents on a circle such that the straight-line
        # distance between adjacent agents equals target_formation_dist.
        # d = 2*r*sin(pi/N)  =>  r = d / (2*sin(pi/N))
        # Using arc-length (old formula) would misalign the formation reward target.
        radius = self.cfg.target_formation_dist / (2.0 * math.sin(math.pi / self.cfg.num_agents))
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
        # Phase 2A: override goal XY to spawn XY — pure hover in place, no lateral gradient.
        # Without this, formation circle XY offsets create a persistent lateral reward gradient
        # that causes the policy to command roll/pitch (observed ~21°/24° tilt in PD1–PD6).
        if self.cfg.hover_in_place:
            desired_pos_w[:, :, :2] = root_pos_w[:, :, :2]
        self._desired_pos_w[env_ids_tensor] = desired_pos_w

        # --- L3: Reset SwarmRaft state for these environments (Phase 3) ---
        if self.cfg.raft_enabled and hasattr(self, "_swarm_raft"):
            self._swarm_raft.reset_envs(env_ids_tensor, self._desired_pos_w)
            self._raft_is_leader[env_ids_tensor] = 0.0
            self._raft_alive_frac[env_ids_tensor] = 1.0

        # --- L5: Reset MINCO smoother state for these environments (Phase 3) ---
        if self.cfg.minco_enabled and hasattr(self, "_minco"):
            self._minco.reset_envs(env_ids_tensor)

        # --- Phase 4: Reset agent-loss state for these environments ---
        if self.cfg.agent_loss_enabled and hasattr(self, "_kill_step"):
            self._kill_step[env_ids_tensor] = -1
            self._kill_agent_idx[env_ids_tensor] = -1
            self._force_terminated[env_ids_tensor] = False

        # Write to sim using robot_indices
        self.robot.write_root_pose_to_sim(root_state[:, :, :7].view(-1, 7), robot_indices)
        self.robot.write_root_velocity_to_sim(root_state[:, :, 7:13].view(-1, 6), robot_indices)
        # Crazyflie joint states (rotors)
        joint_pos = self.robot.data.default_joint_pos[0].repeat(num_resets * self.cfg.num_agents, 1)
        joint_vel = self.robot.data.default_joint_vel[0].repeat(num_resets * self.cfg.num_agents, 1)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, robot_indices)
