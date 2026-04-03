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


def compute_spawn_offsets(
    num_agents: int, radius: float, z: float = 0.7
) -> list[tuple[float, float, float]]:
    """Compute spawn positions arranged in a circle."""
    return [
        (
            radius * math.cos(2 * math.pi * i / num_agents),
            radius * math.sin(2 * math.pi * i / num_agents),
            z,
        )
        for i in range(num_agents)
    ]


@configclass
class GgswarmEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    num_agents = 8

    # Single-agent spaces (PPO sees each drone as an independent instance)
    action_space = 4   # thrust + 3 moments
    # obs = 12 (local) + (num_agents-1)*3 (neighbor rel_pos) = 18 for 3 agents
    # Set to 12 for hover-only (num_agents=1), 18 for formation (num_agents=3)
    observation_space = 12
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
        eye=(1.5, 1.5, 2.0),
        lookat=(0.0, 0.0, 1.0),
        resolution=(1920, 1080),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    # robot — base config, _setup_scene creates one per agent
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Drone_0"
    )
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # spawn offsets — radius computed to maintain min spacing between drones
    # R = min_spawn_spacing / (2 * sin(pi/N)), ensures nearest-neighbor >= 0.75m
    min_spawn_spacing = 0.75  # minimum distance between adjacent drones at spawn (m)

    @property
    def spawn_radius(self) -> float:
        """Compute spawn radius to maintain min_spawn_spacing for num_agents."""
        if self.num_agents <= 1:
            return 0.5
        return self.min_spawn_spacing / (2 * math.sin(math.pi / self.num_agents))

    # reward scales — hover (quadcopter reference)
    lin_vel_reward_scale = -0.2
    ang_vel_reward_scale = -0.2
    distance_to_goal_reward_scale = 15.0
    distance_to_goal_sigma = 0.8

    # formation (Phase 2B/2C/3)
    num_neighbors = 2                 # K-nearest neighbors in obs (fixed obs size)
    formation_mode = "polygon"        # "polygon" (rigid slots) or "cloud" (boids-like)
    formation_shape = "triangle"      # training shape: polygon, grid, triangle (play overrides with --formation)
    formation_target_spacing = 0.5    # target inter-drone distance (m) — polygon mode
    formation_reward_scale = 2.0      # formation reward scale — polygon mode
    formation_reward_sigma = 0.3      # tanh sharpness — polygon mode
    formation_curriculum_start = 0    # immediate ramp
    formation_curriculum_end = 5000   # full formation by ~208 iterations
    collective_resets = True          # if any drone in group dies, all reset
    formation_centroid = None         # fixed centroid (x,y,z) or None for random

    # cloud formation (Phase 3)
    cloud_cohesion_scale = 3.0        # reward for staying near group centroid
    cloud_cohesion_sigma = 0.8        # tanh sharpness for KNN cohesion
    cloud_min_spacing = 0.50           # min distance to nearest neighbor (separation)
    cloud_separation_penalty = 20.0   # penalty for being too close (threshold-based)
    cloud_max_neighbor_dist = 1.0     # max distance to nearest neighbor before penalty
    cloud_spacing_penalty = 2.0       # penalty scale for straying too far
    cloud_centroid_goal_scale = 15.0  # centroid-to-goal reward (replaces per-drone in cloud)
    cloud_centroid_goal_sigma = 0.8   # tanh sharpness for centroid-to-goal distance

    # action smoothing (Phase 3 — EMA, legacy fallback)
    smoothing_enabled = True          # EMA on policy actions (used when minco_enabled=False)
    smoothing_alpha = 0.3             # 0=full smooth, 1=no smooth

    # MINCO trajectory smoothing (L3 layer — replaces EMA)
    minco_enabled = True              # minimum-jerk trajectory filter
    minco_horizon = 0.04              # planning horizon T (seconds)
    minco_max_vel = 5.0               # velocity clamp (action-units/s)
    minco_max_acc = 25.0              # acceleration clamp (action-units/s²)

    # CBF safety shield (Phase 3)
    cbf_enabled = True                # collision avoidance barrier
    cbf_d_safe = 0.30                 # min safe distance (m) — safety floor
    cbf_gamma = 2.0                   # barrier decay rate (QP enforcement strength)
    cbf_max_correction = 0.15         # max per-pair action correction per step

    # Virtual collision detection
    collision_radius = 0.10           # body collision radius (m) — terminates episode
    collision_enabled = True          # enable virtual collision termination

    # Obstacle forest (Phase 4 — Step 5)
    forest_enabled = False            # spawn static cylinder obstacles
    forest_obstacle_radius = 0.12     # cylinder radius (m)
    forest_obstacle_height = 1.5      # cylinder height (m)
    forest_obstacle_z = 0.5           # cylinder center Z (m)
    cbf_obstacle_d_safe = 0.25        # goal deflection radius beyond obstacle body (m)
    centroid_speed = 0.5            # moving centroid speed (m/s)
    forest_cylinder_spacing = 1.2     # Y spacing between cylinders within a row (m)
    forest_row_spacing = 1.2          # X spacing between rows (m)
    forest_num_rows = 2               # number of obstacle rows
    forest_row_start_x = 3.0          # X position of first row (m)
    forest_viewer_eye = (4.0, -1.8, 5.0)   # isometric camera for forest mode
    forest_viewer_lookat = (4.0, 0.0, 1.0)

    # SwarmRaft agent dropout (L3 consensus)
    dropout_enabled = True            # enable random agent dropout during training
    dropout_step_min = 200            # earliest step to trigger dropout (after formation settles)
    dropout_step_max = 350            # latest step to trigger dropout
    dropout_count = 1                 # agents to kill per group
