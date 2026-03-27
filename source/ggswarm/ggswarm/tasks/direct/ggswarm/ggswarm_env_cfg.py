from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip


def compute_spawn_offsets(num_agents: int, radius: float, z: float = 0.5) -> list[tuple[float, float, float]]:
    """Compute spawn positions arranged in a circle."""
    return [
        (radius * math.cos(2 * math.pi * i / num_agents),
         radius * math.sin(2 * math.pi * i / num_agents),
         z)
        for i in range(num_agents)
    ]


@configclass
class GgswarmEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    num_agents = 3

    # Single-agent spaces (PPO sees each drone as an independent instance)
    action_space = 4       # thrust + 3 moments
    observation_space = 12  # lin_vel_b(3) + ang_vel_b(3) + proj_grav_b(3) + desired_pos_b(3)
    state_space = 0

    # simulation — match Isaac Lab quadcopter reference
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # viewer camera
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.0, 2.0, 2.0),
        lookat=(0.0, 0.0, 0.5),
        resolution=(1920, 1080),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    # robot — base config, _setup_scene creates one per agent
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Drone_0")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # spawn offsets — computed dynamically as circle of radius 0.3m
    spawn_radius = 0.3  # meters, distance from env center to each drone

    # reward scales
    lin_vel_reward_scale = -0.10
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0
    distance_to_goal_sigma = 0.5  # tanh sharpness (lower = tighter position hold)
