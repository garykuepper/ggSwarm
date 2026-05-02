"""DirectMARLEnv config for ggSwarm Phase 1+ shared-scene multi-drone training.

Coexists with GgswarmEnvCfg (single-agent DirectRLEnv used by ggswarm-v0,
which the capstone replay gate keys off). This MARL variant powers the new
`ggswarm-marl-v0` task with A=8 drones in one shared PhysX scene per env,
trained via SKRL MAPPO.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip


NUM_AGENTS = 8
OBS_PER_AGENT = 12 + 2 * 3  # 12 local + K=2 neighbor rel_pos = 18


@configclass
class GgswarmMarlEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    num_agents = NUM_AGENTS

    # MARL spaces — one entry per drone
    possible_agents: list[str] = [f"drone_{i}" for i in range(NUM_AGENTS)]
    action_spaces: dict[str, int] = {f"drone_{i}": 4 for i in range(NUM_AGENTS)}
    observation_spaces: dict[str, int] = {
        f"drone_{i}": OBS_PER_AGENT for i in range(NUM_AGENTS)
    }
    state_space = -1  # auto-concat per-agent obs for the centralized critic

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

    # scene — num_envs is real env count; each env has A drones in shared PhysX
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=2.5, replicate_physics=True
    )

    # robot — leaf regex matches Drone_0..Drone_{A-1} per env. Spawning of the
    # individual Drone_i USDs happens manually in _setup_scene because the
    # Isaac Lab spawner skips the spawn step when the leaf is a regex
    # (asset_base.py:77-83). The regex Articulation flattens to env-major
    # `[num_envs * A, *]` order — verified by scripts/probe/multi_drone_layout.py.
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Drone_.*"
    )
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # spawn radius — circle of radius `spawn_radius` keeps inter-drone spacing
    # at min_spawn_spacing on a regular A-gon. Replaces capstone-era stacked
    # spawn (closes B3).
    min_spawn_spacing = 0.75

    @property
    def spawn_radius(self) -> float:
        """Spawn radius that keeps adjacent drones at min_spawn_spacing on a regular A-gon."""
        return self.min_spawn_spacing / (2 * math.sin(math.pi / self.num_agents))

    # reward scales — hover (quadcopter reference)
    lin_vel_reward_scale = -0.2
    ang_vel_reward_scale = -0.2
    distance_to_goal_reward_scale = 15.0
    distance_to_goal_sigma = 0.8

    # formation
    num_neighbors = 2                 # K-nearest neighbors in obs (fixed obs size)
    formation_mode = "polygon"        # "polygon" (rigid slots) or "cloud" (boids-like)
    formation_shape = "triangle"      # training shape: polygon, grid, triangle
    formation_target_spacing = 0.5
    formation_reward_scale = 2.0
    formation_reward_sigma = 0.3
    formation_curriculum_start = 0
    formation_curriculum_end = 5000
    formation_centroid = None         # fixed centroid (x,y,z) or None for random

    # cloud formation
    cloud_cohesion_scale = 3.0
    cloud_cohesion_sigma = 0.8
    cloud_min_spacing = 0.50
    cloud_separation_penalty = 20.0
    cloud_max_neighbor_dist = 1.0
    cloud_spacing_penalty = 2.0
    cloud_centroid_goal_scale = 15.0
    cloud_centroid_goal_sigma = 0.8

    # action smoothing (legacy EMA fallback)
    smoothing_enabled = True
    smoothing_alpha = 0.3

    # MINCO trajectory smoothing
    minco_enabled = True
    minco_horizon = 0.04
    minco_max_vel = 5.0
    minco_max_acc = 25.0

    # CBF safety shield
    cbf_enabled = True
    cbf_d_safe = 0.30
    cbf_gamma = 2.0
    cbf_max_correction = 0.15

    # Virtual collision detection
    collision_radius = 0.10
    collision_enabled = True

    # Obstacle forest (Phase 4 — toggled via play.py --forest)
    forest_enabled = False
    forest_obstacle_radius = 0.20
    forest_obstacle_height = 1.5
    forest_obstacle_z = 0.5
    cbf_obstacle_d_safe = 0.60
    centroid_speed = 0.5
    forest_cylinder_spacing = 1.2
    forest_row_spacing = 1.2
    forest_num_rows = 2
    forest_row_start_x = 3.0
    forest_viewer_eye = (4.0, -1.8, 5.0)
    forest_viewer_lookat = (4.0, 0.0, 1.0)

    forest_deflect_lateral_blend = 0.7
    forest_deflect_neighbor_vel_eps = 0.05
    forest_deflect_shortfall_scale = 1.5
    forest_max_goal_lead = 0.5

    obstacle_max_correction = 0.25
    obstacle_lateral_blend = 0.7
    obstacle_dampen_floor = 0.5
    obstacle_dampen_strength = 0.5

    # SwarmRaft agent dropout
    dropout_enabled = False
    dropout_step_min = 200
    dropout_step_max = 350
    dropout_count = 1
