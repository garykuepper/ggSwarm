"""ggSwarm: Single-drone hover environment using DirectRLEnv + PPO.

One Crazyflie per env. PPO trains a shared policy across all envs.
For hover training, this is equivalent to multi-drone-per-env since
drones don't interact. Formation (Phase 2B+) will add a wrapper for
multi-drone-per-env with inter-drone observations.

Based on Isaac Lab's quadcopter_env.py.
"""

from __future__ import annotations

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

        # Pre-allocate action tensors (reused every step)
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Body ID, mass, weight
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Episode logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_goal"]
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
        mat_path = "/World/envs/env_0/Robot/body/Looks/DroneMat"
        mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0))
        sim_utils.spawn_preview_surface(mat_path, mat_cfg)
        body_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/body")
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
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        obs = torch.cat([
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            desired_pos_b,
        ], dim=-1)

        # Draw altitude line (env 0 only)
        if self._debug_draw is not None:
            self._debug_draw.clear_lines()
            pos = self._robot.data.root_pos_w[0].cpu().tolist()
            self._debug_draw.draw_lines(
                [pos], [[pos[0], pos[1], 0.0]],
                [(0.12, 0.47, 0.71, 0.9)], [1.0],
            )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1
        )
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / self.cfg.distance_to_goal_sigma)

        rewards = {
            "lin_vel": self.cfg.lin_vel_reward_scale * lin_vel * self.step_dt,
            "ang_vel": self.cfg.ang_vel_reward_scale * ang_vel * self.step_dt,
            "distance_to_goal": self.cfg.distance_to_goal_reward_scale * distance_to_goal_mapped * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.1,
            self._robot.data.root_pos_w[:, 2] > 2.0,
        )
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
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out resets to avoid training spikes
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0

        # Sample new goal position
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(
            self._desired_pos_w[env_ids, :2]
        ).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(
            self._desired_pos_w[env_ids, 2]
        ).uniform_(0.5, 1.5)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
