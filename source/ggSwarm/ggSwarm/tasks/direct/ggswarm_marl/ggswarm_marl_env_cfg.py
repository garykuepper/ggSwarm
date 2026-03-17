# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets import CRAZYFLIE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class GgswarmMarlEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # multi-agent specification and spaces definition
    num_agents = 4

    possible_agents: list[str] = []
    action_spaces: dict = {}
    observation_spaces: dict = {}
    state_space = -1

    def __post_init__(self):
        self.possible_agents = [f"drone_{i}" for i in range(self.num_agents)]
        self.action_spaces = {agent: 4 for agent in self.possible_agents}
        self.observation_spaces = {agent: 12 for agent in self.possible_agents}

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
    )

    # robot(s)
    robot_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/drone_.*"
    )

    # swarm specific
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, env_spacing=5.0, replicate_physics=True
    )

    # reward scales
    rew_scale_pos = 1.0
    rew_scale_vel = -0.05
    rew_scale_ang_vel = -0.01
    rew_scale_alive = 0.1
    rew_scale_terminated = -2.0

    # reset states/conditions
    min_height = 0.1
    max_height = 3.0
    spawn_dist = 1.5  # max distance from origin for spawning drones
