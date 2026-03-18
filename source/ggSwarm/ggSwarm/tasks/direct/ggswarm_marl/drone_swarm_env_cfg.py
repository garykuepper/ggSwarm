# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets import CRAZYFLIE_CFG


@configclass
class GGSwarmMarlEnvCfg(DirectMARLEnvCfg):
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

    # viewer
    viewer: ViewerCfg = ViewerCfg(
        eye=(4.0, 4.0, 3.0),    # camera position (metres)
        lookat=(0.0, 0.0, 1.0), # look-at point (near drone spawn height)
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
    )

    # robot(s)
    robot_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/drone_.*")

    # swarm specific
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01
    graph_connectivity_radius: float = 2.0  # (metres) for L2 adjacency matrix
    spawn_yaw_range: float = 3.14159        # ± range for random yaw (rad)
    target_formation_dist: float = 1.5      # desired inter-agent spacing (m)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=32, env_spacing=5.0, replicate_physics=True)

    # reward scales
    rew_scale_pos = 1.0
    rew_scale_vel = -0.05
    rew_scale_ang_vel = -0.01
    rew_scale_alive = 0.1
    rew_scale_terminated = -2.0
    # Phase 2 rewards
    rew_scale_formation = 2.0
    rew_scale_cohesion = 0.5
    rew_scale_separation = -5.0
    
    # reward sigmas
    rew_pos_sigma = 0.5
    rew_formation_sigma = 0.3

    # reset states/conditions
    min_height = 0.1
    max_height = 3.0
    spawn_dist = 1.5  # max distance from origin for spawning drones

    # curriculum
    curriculum_start_step: int = 25000
    curriculum_end_step: int = 50000
