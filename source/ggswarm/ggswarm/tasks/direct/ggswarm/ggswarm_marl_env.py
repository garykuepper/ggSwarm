"""Phase 1a MARL skeleton — DirectMARLEnv with shared-scene multi-drone layout.

Task 2 of the Phase 1a re-cut plan: the smallest env that loads, spawns A=8
drones in one shared PhysX scene per env, steps the simulator with hover
thrust, and returns trivially-shaped per-agent dicts. No formation reward,
CBF, MINCO, dropout, forest. Those land in Task 3.

Shape contract:
  - self.num_envs                    == real env count (DirectMARLEnv)
  - N_drones = num_envs * num_agents == flat drone instance count
  - self._robot.data.root_pos_w      : [N_drones, 3]   (env-major)
  - per-agent dict tensors           : [num_envs, *]
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv

from .ggswarm_marl_env_cfg import GgswarmMarlEnvCfg


class GgswarmMarlEnv(DirectMARLEnv):
    cfg: GgswarmMarlEnvCfg

    def __init__(self, cfg: GgswarmMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        A = self.cfg.num_agents
        N_envs = self.num_envs
        N_drones = N_envs * A
        device = self.device

        self._A = A
        self._N_drones = N_drones
        self._agent_ids = list(self.cfg.possible_agents)

        # Per-drone scratch tensors. Pre-allocated; reused every step.
        self._actions = torch.zeros(N_drones, 4, device=device)               # [N_drones, 4]
        self._thrust = torch.zeros(N_drones, 1, 3, device=device)             # [N_drones, 1, 3]
        self._moment = torch.zeros(N_drones, 1, 3, device=device)             # [N_drones, 1, 3]

        # Per-drone env origins (broadcast helper to avoid per-step allocation).
        self._env_origins_per_drone = self._terrain.env_origins.repeat_interleave(
            A, dim=0
        )  # [N_drones, 3]

        # Body index, mass, weight (regex Articulation has one "body" link per drone).
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Pre-allocated zero buffer for stub rewards / dones (returned per-agent each step).
        self._zero_per_env = torch.zeros(N_envs, device=device)               # [N_envs]
        self._false_per_env = torch.zeros(N_envs, dtype=torch.bool, device=device)
        self._stub_obs = torch.zeros(N_envs, 18, device=device)               # [N_envs, 18]

    def _setup_scene(self):
        # Manually spawn A drone USDs at /World/envs/env_0/Drone_0..Drone_{A-1}.
        # Required because the Isaac Lab spawner skips spawning when the leaf
        # of cfg.robot.prim_path is a regex (asset_base.py:77-83).
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

        # One Articulation with regex prim_path. Constructor sees the leaf
        # regex and skips its own spawn; PhysX flattens N_drones instances
        # in env-major order.
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

        # Clone env_0 to all other envs
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        # actions[agent] is [num_envs, 4]. Pack into per-drone [N_drones, 4]
        # in env-major order so it lines up with the Articulation's instance order.
        A = self._A
        N_envs = self.num_envs
        # [N_envs, A, 4] from stacking per-agent
        stacked = torch.stack(
            [actions[f"drone_{i}"].clamp(-1.0, 1.0) for i in range(A)], dim=1
        )
        self._actions.copy_(stacked.reshape(N_envs * A, 4))

        # Hover thrust + zero moment (stub — Task 3 reintroduces MINCO/CBF/etc.)
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

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Stub: return zeros per-agent. Task 3 reintroduces real obs.
        return {agent: self._stub_obs for agent in self._agent_ids}

    def _get_states(self) -> torch.Tensor:
        # state_space=-1 means DirectMARLEnv auto-concatenates from obs_dict.
        # This method is only invoked if state_space > 0; our stub never reaches it.
        raise NotImplementedError

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        return {agent: self._zero_per_env for agent in self._agent_ids}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {agent: self._false_per_env for agent in self._agent_ids}
        time_outs = {agent: time_out for agent in self._agent_ids}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        # Stub: no goal sampling, no spawn jitter. Drones use their default
        # root state (set up at spawn). Task 3 wires the formation goal logic.
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        A = self._A
        drone_ids = (
            env_ids.unsqueeze(1) * A + torch.arange(A, device=self.device)
        ).reshape(-1)

        default_root_state = self._robot.data.default_root_state[drone_ids].clone()
        default_root_state[:, :3] += self._env_origins_per_drone[drone_ids]
        # Place drones on a small circle so they're not stacked on top of one another.
        import math  # noqa: PLC0415
        slot_idx = drone_ids % A
        theta = 2.0 * math.pi * slot_idx.float() / A
        default_root_state[:, 0] += self.cfg.spawn_radius * torch.cos(theta)
        default_root_state[:, 1] += self.cfg.spawn_radius * torch.sin(theta)
        default_root_state[:, 2] = 0.5

        joint_pos = self._robot.data.default_joint_pos[drone_ids]
        joint_vel = self._robot.data.default_joint_vel[drone_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], drone_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], drone_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, drone_ids)
