# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

try:
    # Common Isaac Lab layout
    from isaaclab_assets.robots import CRAZYFLIE_CFG
except ImportError:  # pragma: no cover
    try:
        # Alternative layout used by some releases
        from isaaclab_assets.robots.crazyflie import CRAZYFLIE_CFG
    except ImportError:  # pragma: no cover
        # Backwards-compat for older asset layouts
        from isaaclab_assets import CRAZYFLIE_CFG


@configclass
class GGSwarmMarlEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # multi-agent specification and spaces definition
    num_agents = 3

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
        eye=(1.2, 1.2, 1.4),  # closer camera for small drones
        lookat=(0.0, 0.0, 0.9),  # look-at near typical hover/formation altitude
    )

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
    # With the current thrust mapping in `drone_swarm_env.py`,
    # `action_z = 0.0 -> thrust_val = 0.5`, so `thrust_to_weight = 1.9`
    # makes the neutral action hover (total thrust ~= 1.0 * weight).
    # Aligned with Isaac Lab's Isaac-Quadcopter-Direct-v0 reference baseline.
    thrust_to_weight: float = 1.9
    # Restored to 0.01 (Isaac Lab reference for Crazyflie).
    # Previous reduction to 0.001 was 10x overcorrection; tumbling was caused by
    # wide spawn yaw and lack of upright reward, both now addressed.
    moment_scale: float = 0.01
    graph_connectivity_radius: float = 2.0  # (metres) for L2 adjacency matrix
    # Tighter yaw range (0.1 vs 0.3) keeps drones more level at spawn, reducing early tumble pressure.
    spawn_yaw_range: float = 0.1  # ± range for random yaw (rad)
    target_formation_dist: float = 0.20  # desired inter-agent spacing (m)
    drone_radius: float = 0.05  # (metres) approximate collision radius
    min_separation_dist: float = 0.10  # (metres) minimum allowed inter-agent distance

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=5.0, replicate_physics=True
    )

    # reward scales
    rew_scale_pos = 3.0
    rew_scale_vel = -0.15
    rew_scale_ang_vel = -0.5
    # Stronger alive bonus to reinforce staying airborne.
    rew_scale_alive = 1.0
    # Aggressive crash penalty to prioritize recovery over all else.
    rew_scale_terminated = -20.0
    # Uprightness reward: now exceeds position reward (5.0 vs 3.0) to make "stay level" the top priority.
    # This ensures drones prioritize orientation control over reaching the goal while tumbling.
    rew_scale_upright: float = 5.0
    # Phase 2 rewards
    rew_scale_formation = 1.0
    rew_scale_cohesion = 0.2
    rew_scale_separation = -5.0

    # reward sigmas
    rew_pos_sigma = 0.5
    rew_formation_sigma = 0.3

    # reset states/conditions
    min_height = 0.1
    max_height = 3.0
    # Smaller spawn radius so drones start close to goals for cleaner gradients.
    spawn_dist = 0.5  # max distance from origin for spawning drones

    # curriculum: formation rewards fade in later to give hover more time to stabilize.
    # Start at 80k (later than before), reach full strength by 250k.
    # curriculum_pos_floor ensures hover signal never fully disappears.
    curriculum_start_step: int = 80000
    curriculum_end_step: int = 250000
    curriculum_pos_floor: float = 0.4
