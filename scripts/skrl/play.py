# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play a checkpoint of an RL agent from skrl."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Length of the recorded video (in steps). 500 = 10s at dt=0.02.",
)
parser.add_argument(
    "--prefix",
    type=str,
    default="ggswarm",
    help="Filename prefix for video, trajectories, and CSV (e.g. p4-5-dropout).",
)
# Keep --video_prefix as alias for backward compatibility
parser.add_argument("--video_prefix", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument(
    "--num_agents", type=int, default=8,
    help="Drones per swarm group. >1 enables SwarmWrapper for formation.",
)
parser.add_argument(
    "--policy", type=str, default="gnn", choices=["mlp", "gnn"],
    help="Policy architecture: gnn (default, GATv2) or mlp.",
)
parser.add_argument(
    "--play_length",
    type=int,
    default=500,
    help="Number of steps to play. 500 = 10s at dt=0.02.",
)
parser.add_argument(
    "--trajectories",
    action="store_true",
    default=True,
    help="Record and plot drone trajectories.",
)
parser.add_argument(
    "--formation",
    type=str,
    default=None,
    help="Formation shape at play time: polygon, grid, triangle, letter_G, etc.",
)
parser.add_argument(
    "--dropout",
    action="store_true",
    default=False,
    help="Enable DropoutGuard dropout during play (disabled by default).",
)
parser.add_argument(
    "--no-minco",
    action="store_true",
    default=False,
    help="Disable MINCO trajectory filter for A/B jitter comparison.",
)
parser.add_argument(
    "--forest",
    action="store_true",
    default=False,
    help="Enable forest obstacle navigation with moving centroid.",
)
parser.add_argument(
    "--tron",
    action="store_true",
    default=False,
    help="Enable Tron-styled cinematic environment (black void, fog, emissive grid, orbit camera).",
)
parser.add_argument(
    "--cloud",
    action="store_true",
    default=False,
    help="Use cloud/boid formation_mode at play time (centroid + KNN cohesion + separation, no rigid slots). Caveat: production checkpoint trained in polygon mode.",
)
parser.add_argument(
    "--cam_mode",
    type=str,
    default="orbit",
    choices=["orbit", "top_down", "low_angle", "chase"],
    help="Camera mode for --tron play. orbit: rotate around centroid. top_down: locked overhead. low_angle: dramatic ground-level looking up. chase: trail the centroid.",
)
parser.add_argument(
    "--no_drones",
    action="store_true",
    default=False,
    help="Hide all drone visuals (cold-open style — env only). Drones still execute physics, the camera still tracks their centroid, but they're not rendered.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint."
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# Resolve --video_prefix alias → --prefix
if args_cli.video_prefix is not None:
    args_cli.prefix = args_cli.video_prefix
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
# Sync play_length and video_length — use the larger value for both
if args_cli.video:
    max_len = max(args_cli.play_length, args_cli.video_length)
    args_cli.play_length = max_len
    args_cli.video_length = max_len

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import ggswarm.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = (
        "skrl_cfg_entry_point"
        if algorithm in ["ppo"]
        else f"skrl_{algorithm}_cfg_entry_point"
    )
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    experiment_cfg: dict,
):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    elif args_cli.num_agents > 1:
        env_cfg.scene.num_envs = args_cli.num_agents  # one swarm group for play
    else:
        env_cfg.scene.num_envs = 1
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # Extend episode length to match play_length so no mid-play resets
    if args_cli.play_length:
        env_cfg.episode_length_s = args_cli.play_length * env_cfg.decimation * env_cfg.sim.dt + 1.0

    # Apply num_agents and expand observation space for formation
    env_cfg.num_agents = args_cli.num_agents
    if args_cli.num_agents > 1:
        env_cfg.observation_space = 12 + env_cfg.num_neighbors * 3
        env_cfg.scene.env_spacing = 0.01  # drones visually in same space
        env_cfg.collective_resets = False  # no group teleport during play
        env_cfg.formation_centroid = (0.0, 0.0, 1.0)  # hover over origin during play
        env_cfg.dropout_enabled = args_cli.dropout  # off by default, --dropout to enable
    else:
        env_cfg.observation_space = 12

    # MINCO toggle for A/B jitter comparison
    if args_cli.no_minco:
        env_cfg.minco_enabled = False
        print("[INFO] MINCO trajectory filter DISABLED (--no-minco)")

    # Cloud / boid mode at play time. Caveat: production checkpoint p4-revert-4
    # was trained in formation_mode="polygon" so cloud-mode play is an
    # off-distribution test — drones may behave clumsily. This is the cheap
    # experiment, not a trained-from-scratch boid policy.
    if args_cli.cloud:
        env_cfg.formation_mode = "cloud"
        print("[INFO] Cloud / boid formation_mode enabled (off-distribution for polygon-trained checkpoint)")

    # Forest obstacle navigation
    if args_cli.forest:
        env_cfg.forest_enabled = True
        env_cfg.formation_centroid = (0.0, 0.0, 1.0)  # start at origin, moves +X
        # Forest camera from cfg
        env_cfg.viewer = type(env_cfg.viewer)(
            eye=env_cfg.forest_viewer_eye,
            lookat=env_cfg.forest_viewer_lookat,
            resolution=(1920, 1080),
        )
        print(f"[INFO] Forest mode: {env_cfg.forest_num_rows} rows, "
              f"spacing {env_cfg.forest_cylinder_spacing}m, "
              f"centroid speed {env_cfg.centroid_speed} m/s")

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = (
        args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    )
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join(
        "logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print(
                "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
            )
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
            other_dirs=["checkpoints"],
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # Tron-styled cinematic environment (Phase 5 sub-phase A)
    # Phase B incremental rebuild: start with ONLY an orbit camera, no scene
    # changes (vanilla Isaac Lab visuals). The orbit positions are pushed to
    # env.sim.set_camera_view() each frame — that's the same API Isaac Lab's
    # ViewportCameraController uses internally, so the recorded video will
    # actually follow the orbit (the Kit-viewport set_active_camera approach
    # only updated the live UI, not the env.render() pipeline that
    # NvencRecorder reads from).
    tron_orbit = None
    if args_cli.tron:
        tron_orbit = {
            "mode": args_cli.cam_mode,
            "angle_deg": 0.0,
            # orbit
            "radius": 2.0,            # XY orbit radius (m)
            "height": 0.6,            # Z offset above centroid for orbit (m)
            "speed_deg": 0.6,         # orbit angular speed (deg/frame)
            # top_down
            "top_down_height": 3.0,   # Z above centroid for overhead view (m)
            "top_down_yaw_deg": 0.0,  # subtle rotation about vertical (start facing +X)
            "top_down_yaw_speed": 0.15,  # slow yaw drift per frame
            # low_angle
            "low_radius": 4.0,        # XY distance for low_angle (m)
            "low_height": -0.4,       # camera Z (negative = below centroid, slight up-look)
            "low_target_z": 0.4,      # look-at z above centroid for upward angle
            # chase
            "chase_back": 3.5,        # distance behind the centroid in -X direction
            "chase_height": 1.2,      # Z above centroid
        }

        # Tron iter 1: remove all UsdLux lights from the stage (any type, any
        # path) via traversal. Standalone — does not touch render mode, fog,
        # ground, materials, or anything else.
        import omni.usd  # noqa: PLC0415
        from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade, Gf  # noqa: PLC0415

        _LIGHT_TYPE_NAMES = {
            "DistantLight", "DomeLight", "RectLight", "SphereLight",
            "DiskLight", "CylinderLight", "GeometryLight",
        }
        _stage = omni.usd.get_context().get_stage()
        _to_remove = [
            str(p.GetPath()) for p in _stage.Traverse()
            if p.GetTypeName() in _LIGHT_TYPE_NAMES
        ]
        for _path in _to_remove:
            _stage.RemovePrim(_path)
        print(f"[INFO] Tron iter 1: removed {len(_to_remove)} light(s): {_to_remove}")

        # Tron iter 4: make Drone instances uninstanceable so material edits
        # propagate to the renderer. Per IsaacLab issue #622 and the
        # sim_utils.make_uninstanceable docstring, instanceable assets cache
        # their visuals at the prototype level — any GetInput("diffuseColor")
        # .Set() call below would silently no-op without this. Must run AFTER
        # gym.make() (env exists) but BEFORE env.reset() (physics not yet
        # active), which is exactly where this --tron block lives.
        import isaaclab.sim as sim_utils  # noqa: PLC0415
        for _i in range(args_cli.num_agents):
            _drone_path = f"/World/envs/env_{_i}/Drone_0"
            if _stage.GetPrimAtPath(_drone_path).IsValid():
                sim_utils.make_uninstanceable(_drone_path)
        print(f"[INFO] Tron iter 4: made {args_cli.num_agents} drone(s) uninstanceable")

        # Tron iter 2: change the existing DroneMat's diffuseColor in place.
        # The env's _setup_scene already creates a UsdPreviewSurface material
        # at /World/envs/env_{i}/Drone_0/body/Looks/DroneMat and binds it to
        # the body prim. Rather than creating a new material with a different
        # shader type and fighting USD binding strength, we just edit the
        # existing material's diffuseColor and emissiveColor attributes.
        _A_iter = args_cli.num_agents
        # UsdPreviewSurface inputs are LINEAR — gamma-decode the desired sRGB
        # display color #ff8c00 (1.0, 0.549, 0.0) → linear (1.0, 0.262, 0.0).
        # Without this correction the rendered color skews yellow.
        _AMBER = Gf.Vec3f(1.0, 0.262, 0.0)        # linear → #ff8c00 sRGB
        _AMBER_BRIGHT = Gf.Vec3f(1.5, 0.39, 0.0)  # 1.5x amber, still under clip
        _updated = 0
        for _i in range(_A_iter):
            _shader_path = f"/World/envs/env_{_i}/Drone_0/body/Looks/DroneMat/Shader"
            _shader_prim = _stage.GetPrimAtPath(_shader_path)
            if not _shader_prim.IsValid():
                continue
            _shader = UsdShade.Shader(_shader_prim)
            _diffuse = _shader.GetInput("diffuseColor")
            if _diffuse:
                _diffuse.Set(_AMBER)
                _updated += 1
            _emissive = _shader.GetInput("emissiveColor")
            if _emissive:
                _emissive.Set(_AMBER_BRIGHT)
            else:
                _shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(_AMBER_BRIGHT)
        print(f"[INFO] Tron iter 2: updated DroneMat on {_updated}/{_A_iter} drones to amber #ff8c00")

        # Tron iter 8: optional --no_drones for cold-open shots. Hides every
        # drone prim entirely (visibility = invisible) AND disables the
        # KNN/altitude debug overlay. Drones still execute physics and the
        # orbit camera still tracks their centroid, but they don't render.
        # Use with --tron to capture environment-only clips.
        if args_cli.no_drones:
            _hidden = 0
            for _i in range(_A_iter):
                _drone_root = f"/World/envs/env_{_i}/Drone_0"
                _root_prim = _stage.GetPrimAtPath(_drone_root)
                if _root_prim.IsValid():
                    UsdGeom.Imageable(_root_prim).MakeInvisible()
                    _hidden += 1
            # Suppress the KNN debug edge overlay (drawn per-frame by the env
            # from drone positions). The env guards every draw with
            # `if self._debug_draw is not None`, so setting it to None is enough.
            env.unwrapped._debug_draw = None
            print(f"[INFO] Tron iter 8: hid {_hidden} drone(s) + disabled debug_draw for --no_drones cold-open")

        # Tron iter 5: replace the terrain grid with a custom emissive teal
        # plane. The default Isaac Lab terrain uses a baked grid texture that
        # overrides any constant diffuseColor we set (and the iter 3 cyan-flash
        # confirmed this). Easier to remove the terrain entirely and spawn a
        # simple plane with our own UsdPreviewSurface material via Isaac Lab's
        # spawn_preview_surface (the same path the env uses for DroneMat).
        _ground_root = _stage.GetPrimAtPath("/World/ground")
        if _ground_root.IsValid():
            _stage.RemovePrim("/World/ground")
            print("[INFO] Tron iter 5: removed /World/ground (terrain)")

        # 50m × 50m flat quad in the XY plane at z=0 (Isaac Lab Z-up).
        _grid_path = "/World/TronGrid"
        _grid_mesh = UsdGeom.Mesh.Define(_stage, _grid_path)
        _size = 50.0
        _grid_mesh.CreatePointsAttr([
            Gf.Vec3f(-_size, -_size, 0.0),
            Gf.Vec3f( _size, -_size, 0.0),
            Gf.Vec3f( _size,  _size, 0.0),
            Gf.Vec3f(-_size,  _size, 0.0),
        ])
        _grid_mesh.CreateFaceVertexCountsAttr([4])
        _grid_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        _grid_mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)

        # Pure black base plane — invisible against the black void, but kept
        # as a flat surface so the grid lines have something to live on.
        # The cyan lines from iter 6 are the only thing that should render here.
        _grid_mat_path = "/World/Looks/TronGridMat"
        sim_utils.spawn_preview_surface(
            _grid_mat_path,
            sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0),
                emissive_color=(0.0, 0.0, 0.0),
                roughness=1.0,
            ),
        )
        _grid_mat_prim = _stage.GetPrimAtPath(_grid_mat_path)
        UsdShade.MaterialBindingAPI.Apply(_grid_mesh.GetPrim())
        UsdShade.MaterialBindingAPI(_grid_mesh.GetPrim()).Bind(
            UsdShade.Material(_grid_mat_prim)
        )
        print("[INFO] Tron iter 5: spawned 50m teal base plane at /World/TronGrid")

        # Tron iter 6: bright teal grid line geometry on top of the base plane.
        # 1m spacing, ~5cm wide lines as thin quads at z=0.005 to avoid
        # z-fighting with the base. One mesh containing all line quads.
        _line_w = 0.01       # half-width: 2cm wide lines
        _line_z = 0.005      # slight offset above base plane
        _spacing = 2.0       # 2m grid cells (wider than before)
        _n = int(_size / _spacing)  # lines from -_n*spacing .. +_n*spacing

        _line_points: list[Gf.Vec3f] = []
        _line_face_counts: list[int] = []
        _line_face_indices: list[int] = []
        _line_normals: list[Gf.Vec3f] = []
        _vidx = 0

        # Vertical lines (constant X, span all Y)
        for _i in range(-_n, _n + 1):
            _x = _i * _spacing
            _line_points.extend([
                Gf.Vec3f(_x - _line_w, -_size, _line_z),
                Gf.Vec3f(_x + _line_w, -_size, _line_z),
                Gf.Vec3f(_x + _line_w,  _size, _line_z),
                Gf.Vec3f(_x - _line_w,  _size, _line_z),
            ])
            _line_face_counts.append(4)
            _line_face_indices.extend([_vidx, _vidx + 1, _vidx + 2, _vidx + 3])
            _line_normals.extend([Gf.Vec3f(0, 0, 1)] * 4)
            _vidx += 4

        # Horizontal lines (constant Y, span all X)
        for _i in range(-_n, _n + 1):
            _y = _i * _spacing
            _line_points.extend([
                Gf.Vec3f(-_size, _y - _line_w, _line_z),
                Gf.Vec3f( _size, _y - _line_w, _line_z),
                Gf.Vec3f( _size, _y + _line_w, _line_z),
                Gf.Vec3f(-_size, _y + _line_w, _line_z),
            ])
            _line_face_counts.append(4)
            _line_face_indices.extend([_vidx, _vidx + 1, _vidx + 2, _vidx + 3])
            _line_normals.extend([Gf.Vec3f(0, 0, 1)] * 4)
            _vidx += 4

        _lines_path = "/World/TronGridLines"
        _lines_mesh = UsdGeom.Mesh.Define(_stage, _lines_path)
        _lines_mesh.CreatePointsAttr(_line_points)
        _lines_mesh.CreateFaceVertexCountsAttr(_line_face_counts)
        _lines_mesh.CreateFaceVertexIndicesAttr(_line_face_indices)
        _lines_mesh.CreateNormalsAttr(_line_normals)

        # Bright emissive teal material — these are the glowing grid lines
        _lines_mat_path = "/World/Looks/TronGridLinesMat"
        sim_utils.spawn_preview_surface(
            _lines_mat_path,
            sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.018, 0.665, 0.652),
                emissive_color=(0.3, 3.0, 2.8),   # bright glow
                roughness=0.4,
            ),
        )
        _lines_mat_prim = _stage.GetPrimAtPath(_lines_mat_path)
        UsdShade.MaterialBindingAPI.Apply(_lines_mesh.GetPrim())
        UsdShade.MaterialBindingAPI(_lines_mesh.GetPrim()).Bind(
            UsdShade.Material(_lines_mat_prim)
        )
        print(f"[INFO] Tron iter 6: spawned {2 * (2 * _n + 1)} grid line quads at /World/TronGridLines")

        # Tron iter 7: black forest cylinders wrapped with red emissive ring
        # stack. Same dark-base-with-bright-glowing-structure idea as the
        # floor. Two parts:
        #   (a) make existing cylinders uninstanceable + set their material
        #       to pure black so the body itself is invisible against the void
        #   (b) spawn 5 thin red emissive rings per unique obstacle, slightly
        #       outside the cylinder surface, evenly distributed in height
        if args_cli.forest:
            base_unwrapped_iter = env.unwrapped

            # (a) Set existing obstacle cylinder bodies to black. The body is
            # still a solid that occludes anything behind it, but we want
            # that — it gives the wireframe its silhouette and reads as a
            # 3D object instead of a translucent mesh.
            _BLACK = Gf.Vec3f(0.0, 0.0, 0.0)
            for _prim in _stage.Traverse():
                _path_str = str(_prim.GetPath())
                if "/Obstacle_" not in _path_str or _prim.GetTypeName() != "Cylinder":
                    continue
                sim_utils.make_uninstanceable(_path_str)
                _shader_prim = _stage.GetPrimAtPath(_path_str + "/ObstacleMat/Shader")
                if not _shader_prim.IsValid():
                    continue
                _cyl_shader = UsdShade.Shader(_shader_prim)
                _cyl_diff = _cyl_shader.GetInput("diffuseColor")
                if _cyl_diff:
                    _cyl_diff.Set(_BLACK)
                _cyl_em = _cyl_shader.GetInput("emissiveColor")
                if _cyl_em:
                    _cyl_em.Set(_BLACK)
                else:
                    _cyl_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(_BLACK)

            # (b) Spawn red wireframe on each unique obstacle: dense vertical
            # strips + horizontal rings + a thicker bright ring at the top
            # edge for emphasis. Strips are tangent quads in one combined mesh.
            # Rings are thin Cylinder primitives.
            import math as _cyl_math  # noqa: PLC0415

            _h = base_unwrapped_iter.cfg.forest_obstacle_height
            _r = base_unwrapped_iter.cfg.forest_obstacle_radius
            _strip_w = 0.018      # half-width: 3.6cm wide strips (thicker so each one is visible at more angles)
            _strip_r = _r * 1.02  # 2% outside the cylinder surface
            _n_strips = 24        # denser vertical lines around each cylinder

            _obs_pos = (
                base_unwrapped_iter._obstacle_pos.cpu().tolist()
                if base_unwrapped_iter._obstacle_pos is not None else []
            )

            _strip_points: list[Gf.Vec3f] = []
            _strip_face_counts: list[int] = []
            _strip_face_indices: list[int] = []
            _strip_normals: list[Gf.Vec3f] = []
            _vidx = 0
            for _opos in _obs_pos:
                _cx, _cy, _cz = float(_opos[0]), float(_opos[1]), float(_opos[2])
                _z_top = _cz + _h / 2.0
                _z_bot = _cz - _h / 2.0
                for _i in range(_n_strips):
                    _theta = (_i / _n_strips) * 2.0 * _cyl_math.pi
                    _ct, _st = _cyl_math.cos(_theta), _cyl_math.sin(_theta)
                    # Strip center on the cylinder surface at angle theta
                    _scx = _cx + _strip_r * _ct
                    _scy = _cy + _strip_r * _st
                    # Tangent direction = (-sin theta, cos theta, 0)
                    _tx, _ty = -_st, _ct
                    # Quad corners: (-w,-h), (+w,-h), (+w,+h), (-w,+h) along tangent×Z
                    _strip_points.extend([
                        Gf.Vec3f(_scx - _strip_w * _tx, _scy - _strip_w * _ty, _z_bot),
                        Gf.Vec3f(_scx + _strip_w * _tx, _scy + _strip_w * _ty, _z_bot),
                        Gf.Vec3f(_scx + _strip_w * _tx, _scy + _strip_w * _ty, _z_top),
                        Gf.Vec3f(_scx - _strip_w * _tx, _scy - _strip_w * _ty, _z_top),
                    ])
                    _strip_face_counts.append(4)
                    _strip_face_indices.extend([_vidx, _vidx + 1, _vidx + 2, _vidx + 3])
                    # Outward-facing radial normal
                    _strip_normals.extend([Gf.Vec3f(_ct, _st, 0.0)] * 4)
                    _vidx += 4

            _strips_path = "/World/TronObstacleStrips"
            _strips_mesh = UsdGeom.Mesh.Define(_stage, _strips_path)
            _strips_mesh.CreatePointsAttr(_strip_points)
            _strips_mesh.CreateFaceVertexCountsAttr(_strip_face_counts)
            _strips_mesh.CreateFaceVertexIndicesAttr(_strip_face_indices)
            _strips_mesh.CreateNormalsAttr(_strip_normals)

            _strip_mat_path = "/World/Looks/TronCylStripMat"
            sim_utils.spawn_preview_surface(
                _strip_mat_path,
                sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.5, 0.0, 0.0),
                    emissive_color=(2.0, 0.0, 0.0),
                    roughness=0.4,
                ),
            )
            _strip_mat_prim = _stage.GetPrimAtPath(_strip_mat_path)
            UsdShade.MaterialBindingAPI.Apply(_strips_mesh.GetPrim())
            UsdShade.MaterialBindingAPI(_strips_mesh.GetPrim()).Bind(
                UsdShade.Material(_strip_mat_prim)
            )
            _strip_count = len(_obs_pos) * _n_strips
            print(f"[INFO] Tron iter 7a: {_strip_count} red vertical strips ({len(_obs_pos)} obstacles × {_n_strips})")

            # (c) Horizontal annulus rings on each cylinder. Built as flat
            # ring meshes (NOT solid cylinders — those rendered as disks
            # because Cylinder is a solid volume). Each annulus is N angular
            # segments between an inner and outer radius, all at one z height.
            _ring_segs = 32
            _ring_inner = _r * 1.04
            _ring_outer = _r * 1.10
            _top_inner = _r * 1.05
            _top_outer = _r * 1.18         # thicker top edge ring
            _ring_fracs = [0.1, 0.275, 0.45, 0.625, 0.8]  # 5 mid-height rings

            def _ring_pts_and_faces(cx, cy, z, r_in, r_out, n_seg, base_idx):
                pts: list[Gf.Vec3f] = []
                idx_pairs: list[int] = []
                for j in range(n_seg):
                    th = (j / n_seg) * 2.0 * _cyl_math.pi
                    ct, st = _cyl_math.cos(th), _cyl_math.sin(th)
                    pts.append(Gf.Vec3f(cx + r_in * ct, cy + r_in * st, z))
                    pts.append(Gf.Vec3f(cx + r_out * ct, cy + r_out * st, z))
                faces: list[int] = []
                fcounts: list[int] = []
                for j in range(n_seg):
                    i_in = base_idx + 2 * j
                    i_out = base_idx + 2 * j + 1
                    nj = (j + 1) % n_seg
                    n_in = base_idx + 2 * nj
                    n_out = base_idx + 2 * nj + 1
                    faces.extend([i_in, i_out, n_out, n_in])
                    fcounts.append(4)
                return pts, faces, fcounts

            _ring_points: list[Gf.Vec3f] = []
            _ring_face_indices: list[int] = []
            _ring_face_counts: list[int] = []
            _ring_normals: list[Gf.Vec3f] = []
            _vidx_r = 0
            _ring_idx = 0
            for _k, _opos in enumerate(_obs_pos):
                _cx, _cy, _cz = float(_opos[0]), float(_opos[1]), float(_opos[2])
                _z_top = _cz + _h / 2.0
                _z_bot = _cz - _h / 2.0
                # Mid-height rings
                for _frac in _ring_fracs:
                    _z = _z_bot + _frac * _h
                    _pts, _faces, _fcs = _ring_pts_and_faces(
                        _cx, _cy, _z, _ring_inner, _ring_outer, _ring_segs, _vidx_r
                    )
                    _ring_points.extend(_pts)
                    _ring_face_indices.extend(_faces)
                    _ring_face_counts.extend(_fcs)
                    _ring_normals.extend([Gf.Vec3f(0, 0, 1)] * len(_pts))
                    _vidx_r += len(_pts)
                    _ring_idx += 1
                # Top edge ring (thicker)
                _pts, _faces, _fcs = _ring_pts_and_faces(
                    _cx, _cy, _z_top, _top_inner, _top_outer, _ring_segs, _vidx_r
                )
                _ring_points.extend(_pts)
                _ring_face_indices.extend(_faces)
                _ring_face_counts.extend(_fcs)
                _ring_normals.extend([Gf.Vec3f(0, 0, 1)] * len(_pts))
                _vidx_r += len(_pts)
                _ring_idx += 1

            _rings_path = "/World/TronObstacleRings"
            _rings_mesh = UsdGeom.Mesh.Define(_stage, _rings_path)
            _rings_mesh.CreatePointsAttr(_ring_points)
            _rings_mesh.CreateFaceVertexCountsAttr(_ring_face_counts)
            _rings_mesh.CreateFaceVertexIndicesAttr(_ring_face_indices)
            _rings_mesh.CreateNormalsAttr(_ring_normals)
            _rings_mesh.GetPrim().CreateAttribute(
                "doubleSided", Sdf.ValueTypeNames.Bool
            ).Set(True)
            UsdShade.MaterialBindingAPI.Apply(_rings_mesh.GetPrim())
            UsdShade.MaterialBindingAPI(_rings_mesh.GetPrim()).Bind(
                UsdShade.Material(_strip_mat_prim)
            )
            print(f"[INFO] Tron iter 7b: {_ring_idx} annulus rings (3 mid + 1 top per cylinder)")

        print("[INFO] Tron orbit camera enabled (sim.set_camera_view, vanilla scene).")

    # Override formation shape at play time (--formation arg)
    if args_cli.formation and args_cli.num_agents > 1:
        from ggswarm.formations import get_formation  # noqa: PLC0415

        base = env.unwrapped
        A = base.cfg.num_agents
        spacing = base.cfg.formation_target_spacing
        import math  # noqa: PLC0415
        radius = spacing / (2 * math.sin(math.pi / A))
        new_offsets = get_formation(args_cli.formation, A, radius=radius, spacing=spacing, size=2.0)
        base._formation_offsets = new_offsets.to(base.device)
        print(f"[INFO] Formation override: {args_cli.formation} ({A} agents)")

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording (NVENC H.264 hardware encoder)
    if args_cli.video:
        from ggswarm.viz.nvenc_recorder import NvencRecorder

        # Phase 5 cinematic clips go to a clean top-level videos/showcase/ dir
        # so they're easy to find for stitching. Standard play videos stay
        # nested in the run directory.
        if args_cli.tron:
            video_folder = os.path.join("videos", "showcase")
        else:
            video_folder = os.path.join(log_dir, "videos", "play")
        env = NvencRecorder(
            env,
            video_folder=video_folder,
            video_length=args_cli.video_length,
            name_prefix=args_cli.prefix,
        )
        print(
            f"[INFO] Recording video to {video_folder} (NVENC H.264, {args_cli.video_length} steps)"
        )

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(
        env, ml_framework=args_cli.ml_framework
    )  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"][
        "write_interval"
    ] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"][
        "checkpoint_interval"
    ] = 0  # don't generate checkpoints

    if args_cli.policy == "gnn":
        # GNN policy: manually create agent with GATv2 model
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
        from skrl.memories.torch import RandomMemory
        from skrl.resources.preprocessors.torch import RunningStandardScaler

        from ggswarm.gnn_policy import GgswarmGNNPolicy

        memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=env.device)
        gnn_model = GgswarmGNNPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            num_neighbors=env_cfg.num_neighbors,
            num_agents=args_cli.num_agents,
        )
        GgswarmGNNPolicy.init_edge_cache(memory_size=24, num_envs=env.num_envs)
        models = {"policy": gnn_model, "value": gnn_model}

        ppo_cfg = PPO_DEFAULT_CONFIG.copy()
        ppo_cfg.update({
            "state_preprocessor": RunningStandardScaler,
            "state_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
            "value_preprocessor": RunningStandardScaler,
            "value_preprocessor_kwargs": {"size": 1, "device": env.device},
        })

        agent = PPO(
            models=models,
            memory=memory,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            cfg=ppo_cfg,
        )

        class _GNNRunner:
            """Minimal runner interface for GNN play compatibility."""
            def __init__(self, a):
                self.agent = a

        runner = _GNNRunner(agent)
        print("[INFO] Using GATv2 GNN policy for play")
    else:
        runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # Debug: verify preprocessor stats loaded correctly
    if hasattr(runner.agent, '_state_preprocessor') and runner.agent._state_preprocessor is not None:
        sp = runner.agent._state_preprocessor
        if hasattr(sp, 'running_mean'):
            print(f"[DEBUG] state_preprocessor running_mean: {sp.running_mean.mean():.4f}, "
                  f"running_var: {sp.running_variance.mean():.4f}, "
                  f"count: {sp.current_count.item():.0f}")
        else:
            print("[WARN] state_preprocessor has no running_mean — stats may not be loaded!")
    else:
        print("[WARN] No state_preprocessor found on agent!")

    # Trajectory recording buffers
    traj_pos: list[torch.Tensor] = []
    traj_quat: list[torch.Tensor] = []
    traj_goal: list[torch.Tensor] = []

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {
                    a: outputs[-1][a].get("mean_actions", outputs[0][a])
                    for a in env.possible_agents
                }
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)

        # Advance Tron camera (Z-up, drives env render camera so the
        # recorded video follows it, not just the live UI viewport).
        # Branches on tron_orbit["mode"] to support orbit / top_down /
        # low_angle / chase modes.
        if tron_orbit is not None:
            import math  # noqa: PLC0415

            base_for_cam = env.unwrapped
            A_cam = args_cli.num_agents
            centroid = base_for_cam._robot.data.root_pos_w[:A_cam].mean(dim=0).cpu()
            cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
            _mode = tron_orbit["mode"]

            if _mode == "orbit":
                tron_orbit["angle_deg"] = (tron_orbit["angle_deg"] + tron_orbit["speed_deg"]) % 360.0
                rad = math.radians(tron_orbit["angle_deg"])
                eye = (
                    cx + tron_orbit["radius"] * math.sin(rad),
                    cy + tron_orbit["radius"] * math.cos(rad),
                    cz + tron_orbit["height"],
                )
                target = (cx, cy, cz)

            elif _mode == "top_down":
                # Locked overhead with a subtle yaw drift so the formation
                # rotates within the frame slowly. Eye directly above
                # centroid; lookat at centroid; the yaw is encoded by
                # offsetting the lookat slightly along the rotated axis.
                tron_orbit["top_down_yaw_deg"] = (
                    tron_orbit["top_down_yaw_deg"] + tron_orbit["top_down_yaw_speed"]
                ) % 360.0
                eye = (cx, cy, cz + tron_orbit["top_down_height"])
                # We need a target slightly off-center so the camera has a
                # defined orientation that we can rotate via the yaw drift.
                # Offset of 1mm along the rotated axis is enough to control roll.
                yaw_rad = math.radians(tron_orbit["top_down_yaw_deg"])
                target = (
                    cx + 0.001 * math.sin(yaw_rad),
                    cy + 0.001 * math.cos(yaw_rad),
                    cz,
                )

            elif _mode == "low_angle":
                # Dramatic ground-level shot looking up at the swarm. Slowly
                # arcs around at a slow speed (use the orbit angle for this).
                tron_orbit["angle_deg"] = (tron_orbit["angle_deg"] + tron_orbit["speed_deg"] * 0.5) % 360.0
                rad = math.radians(tron_orbit["angle_deg"])
                eye = (
                    cx + tron_orbit["low_radius"] * math.sin(rad),
                    cy + tron_orbit["low_radius"] * math.cos(rad),
                    cz + tron_orbit["low_height"],
                )
                target = (cx, cy, cz + tron_orbit["low_target_z"])

            elif _mode == "chase":
                # Trail behind the centroid in -X direction (the centroid
                # advances +X in forest mode), slightly above. No rotation —
                # camera just follows.
                eye = (
                    cx - tron_orbit["chase_back"],
                    cy,
                    cz + tron_orbit["chase_height"],
                )
                target = (cx, cy, cz)

            else:  # fallback
                eye = (cx + 6.0, cy, cz + 2.0)
                target = (cx, cy, cz)

            base_for_cam.sim.set_camera_view(eye=eye, target=target)

        # Record trajectory data (first swarm group)
        if args_cli.trajectories:
            base_env = env.unwrapped
            A = base_env.cfg.num_agents
            pos = base_env._robot.data.root_pos_w[:A].detach().cpu().clone()    # [A, 3]
            quat = base_env._robot.data.root_quat_w[:A].detach().cpu().clone()  # [A, 4]
            goal = base_env._desired_pos_w[:A].detach().cpu().clone()            # [A, 3]
            # Mask out dead drones with NaN so plots skip them cleanly
            if base_env.cfg.dropout_enabled:
                dead = ~base_env._agent_alive[:A].cpu()
                pos[dead] = float("nan")
                quat[dead] = float("nan")
                goal[dead] = float("nan")
            traj_pos.append(pos)
            traj_quat.append(quat)
            traj_goal.append(goal)

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break
        if args_cli.play_length and timestep >= args_cli.play_length:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Generate trajectory plots
    if args_cli.trajectories and traj_pos:
        euler_fn = None
        try:
            from isaaclab.utils.math import euler_xyz_from_quat

            euler_fn = euler_xyz_from_quat
        except ImportError:
            print(
                "[WARN] euler_xyz_from_quat not available; attitude subplot will be empty."
            )

        from ggswarm.viz.trajectory_plots import generate_trajectory_plots

        traj_dir = os.path.join(log_dir, "trajectories")
        base_env = env.unwrapped
        A = base_env.cfg.num_agents
        env_origins = base_env._terrain.env_origins[:A].cpu() if A > 1 else None
        generate_trajectory_plots(
            traj_pos,
            traj_quat,
            out_dir=traj_dir,
            agent_names=[f"drone_{i}" for i in range(A)],
            euler_fn=euler_fn,
            goal_data=traj_goal if traj_goal else None,
            env_origins=env_origins,
            target_spacing=base_env.cfg.formation_target_spacing if A > 1 else None,
            centroid=base_env.cfg.formation_centroid if A > 1 else None,
            collision_radius=base_env.cfg.collision_radius if A > 1 else None,
            prefix=args_cli.prefix,
        )

        # Save trajectory data as CSV for post-processing
        import csv  # noqa: PLC0415

        pos = torch.stack(traj_pos)  # [T, A, 3]
        goal = torch.stack(traj_goal) if traj_goal else None  # [T, A, 3]
        if env_origins is not None:
            pos = pos - env_origins.unsqueeze(0)
            if goal is not None:
                goal = goal - env_origins.unsqueeze(0)
        T = pos.shape[0]

        csv_fname = f"{args_cli.prefix}-trajectory_data.csv" if args_cli.prefix != "ggswarm" else "trajectory_data.csv"
        csv_path = os.path.join(traj_dir, csv_fname)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["step"]
            for a in range(A):
                header += [f"d{a}_x", f"d{a}_y", f"d{a}_z"]
                if goal is not None:
                    header += [f"d{a}_gx", f"d{a}_gy", f"d{a}_gz"]
            writer.writerow(header)
            for t in range(T):
                row = [t]
                for a in range(A):
                    p = pos[t, a].tolist()
                    row += [f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"]
                    if goal is not None:
                        g = goal[t, a].tolist()
                        row += [f"{g[0]:.4f}", f"{g[1]:.4f}", f"{g[2]:.4f}"]
                writer.writerow(row)
        print(f"[INFO] Trajectory CSV saved: {csv_path}")

        # Save obstacle positions if forest mode
        if args_cli.forest and hasattr(base_env, '_obstacle_pos') and base_env._obstacle_pos is not None:
            obs_fname = f"{args_cli.prefix}-obstacles.csv" if args_cli.prefix != "ggswarm" else "obstacles.csv"
            obs_path = os.path.join(traj_dir, obs_fname)
            obs_pos = base_env._obstacle_pos.cpu()
            obs_r = base_env.cfg.forest_obstacle_radius
            with open(obs_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "x", "y", "z", "radius"])
                for k in range(obs_pos.shape[0]):
                    p = obs_pos[k].tolist()
                    writer.writerow([k, f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}", f"{obs_r:.4f}"])
            print(f"[INFO] Obstacle CSV saved: {obs_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    # Force exit — Isaac Sim viewer can linger after close
    import sys  # noqa: PLC0415, E402

    sys.exit(0)
