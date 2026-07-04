"""
ggSwarm Cinematic Showcase — Tron-inspired demo video for capstone.

Orchestrates an 8-scene cinematic sequence with camera cuts, formation
morphing, DropoutGuard dropout, and scale-up. Uses tron_env.py for visuals
and the trained GNN policy for drone control.

Usage:
    env_isaaclab/Scripts/python.exe scripts/showcase.py \
        --task ggswarm-v0 \
        --checkpoint logs/skrl/ggswarm/p4/<run>/checkpoints/best_agent.pt \
        --num_agents 8

Scenes:
    1. Black void cold open, grid materializes (3s)
    2. 8 drones spawn, GNN edges visible as teal lines (5s)
    3. Formation snaps into octagon (5s)
    4. Formation morphing: octagon → triangle → grid → letter G (20s)
    5. One drone fails — LED cuts out, edges snap (5s)
    6. DropoutGuard recovery — heptagon reforms (10s)
    7. Scale-up: 20-agent polygon fills frame (15s)
    8. Title card fade (5s)

Total: ~70s at 60fps = ~4200 frames
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

# --- Isaac Lab bootstrap (must be before any omni imports) ---
parser = argparse.ArgumentParser(description="ggSwarm Cinematic Showcase")
parser.add_argument("--task", type=str, default="ggswarm-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_agents", type=int, default=8)
parser.add_argument("--output_dir", type=str, default="videos/showcase")
parser.add_argument("--prefix", type=str, default="ggswarm-showcase")
parser.add_argument("--test", action="store_true", default=False,
                    help="Quick 5s test run — one scene, polygon formation only.")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Now safe to import omni/torch/etc ---
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils  # noqa: F401
from isaaclab.envs import DirectRLEnv

# Register ggswarm task
import ggswarm.tasks  # noqa: F401

from tron_env import setup_tron_environment, TronCameraRig


# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------

SCENES = [
    {"name": "cold_open",      "duration_s": 3.0,  "camera": "orbit",    "formation": None},
    {"name": "drone_spawn",    "duration_s": 5.0,  "camera": "orbit",    "formation": "triangle"},
    {"name": "octagon_form",   "duration_s": 5.0,  "camera": "orbit",    "formation": "polygon"},
    {"name": "morph_triangle", "duration_s": 5.0,  "camera": "top_down", "formation": "triangle"},
    {"name": "morph_grid",     "duration_s": 5.0,  "camera": "top_down", "formation": "grid"},
    {"name": "morph_letter_G", "duration_s": 5.0,  "camera": "top_down", "formation": "letter_G"},
    {"name": "drone_fail",     "duration_s": 5.0,  "camera": "low_angle","formation": None},
    {"name": "swarmraft",      "duration_s": 10.0, "camera": "orbit",    "formation": "polygon"},
    {"name": "title_card",     "duration_s": 5.0,  "camera": "orbit",    "formation": None},
]


TEST_SCENES = [
    {"name": "test_polygon", "duration_s": 5.0, "camera": "orbit", "formation": "polygon"},
]


def get_scene_at_step(step: int, dt: float, scene_list: list) -> tuple[dict, float]:
    """Return the active scene and progress (0-1) within it."""
    elapsed = step * dt
    cumulative = 0.0
    for scene in scene_list:
        if elapsed < cumulative + scene["duration_s"]:
            progress = (elapsed - cumulative) / scene["duration_s"]
            return scene, progress
        cumulative += scene["duration_s"]
    return SCENES[-1], 1.0


def main():
    # --- Environment setup ---
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device)
    env_cfg.num_agents = args_cli.num_agents
    env_cfg.observation_space = 12 + env_cfg.num_neighbors * 3
    env_cfg.scene.num_envs = args_cli.num_agents
    env_cfg.scene.env_spacing = 0.01
    env_cfg.collective_resets = False
    env_cfg.dropout_enabled = False
    env_cfg.formation_centroid = (0.0, 0.0, 1.0)

    # Long episode for full showcase
    scenes = TEST_SCENES if args_cli.test else SCENES
    total_duration = sum(s["duration_s"] for s in scenes)
    env_cfg.episode_length_s = total_duration + 5.0

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base_env = env.unwrapped

    A = args_cli.num_agents
    device = base_env.device
    num_envs = base_env.num_envs

    # --- Apply Tron environment (before any wrapping) ---
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    drone_paths = []
    for i in range(A):
        path = f"/World/envs/env_{i}/Drone_0"
        if stage.GetPrimAtPath(path).IsValid():
            drone_paths.append(path)

    setup_tron_environment(base_env.sim, drone_paths)
    cam = TronCameraRig(base_env.sim, mode="orbit", radius=6.0, height=2.0)
    print("[SHOWCASE] Tron environment applied.")

    # Set active viewport camera
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport:
            viewport.set_active_camera(cam.prim_path)
            print(f"[SHOWCASE] Active camera set to {cam.prim_path}")
    except Exception as e:
        print(f"[SHOWCASE] Could not set viewport camera: {e}")

    # --- Video recorder (wraps gym.Env — must be BEFORE SkrlVecEnvWrapper) ---
    from ggswarm.viz.nvenc_recorder import NvencRecorder

    os.makedirs(args_cli.output_dir, exist_ok=True)
    env = NvencRecorder(
        env,
        video_folder=args_cli.output_dir,
        video_length=int(total_duration / (env_cfg.decimation * env_cfg.sim.dt)),
        name_prefix=args_cli.prefix,
    )

    # --- Wrap for SKRL (must be AFTER NvencRecorder) ---
    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # --- Load trained policy ---
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    from ggswarm.gnn_policy import GgswarmGNNPolicy

    memory = RandomMemory(memory_size=24, num_envs=num_envs, device=device)
    gnn_model = GgswarmGNNPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        num_neighbors=env_cfg.num_neighbors,
        num_agents=A,
    )
    GgswarmGNNPolicy.init_edge_cache(memory_size=24, num_envs=num_envs)
    models = {"policy": gnn_model, "value": gnn_model}

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update({
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": device},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
    })

    agent = PPO(
        models=models,
        memory=memory,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        cfg=ppo_cfg,
    )
    agent.load(args_cli.checkpoint)
    agent.set_running_mode("eval")

    print(f"[SHOWCASE] Loaded checkpoint: {args_cli.checkpoint}")
    print(f"[SHOWCASE] {A} agents, {total_duration:.0f}s total duration")

    # --- Run showcase ---
    obs, _ = env.reset()
    dt = base_env.step_dt
    total_steps = int(total_duration / dt)
    prev_scene_name = None
    prev_formation = None

    print(f"[SHOWCASE] Running {total_steps} steps ({total_duration:.0f}s)...")

    for step in range(total_steps):
        scene, progress = get_scene_at_step(step, dt, scenes)

        # Scene change events
        if scene["name"] != prev_scene_name:
            print(f"[SCENE] {scene['name']} (camera: {scene['camera']})")
            cam.set_mode(scene["camera"])
            prev_scene_name = scene["name"]

        # Formation change
        if scene["formation"] and scene["formation"] != prev_formation:
            from ggswarm.formations import get_formation

            spacing = base_env.cfg.formation_target_spacing
            radius = spacing / (2 * math.sin(math.pi / max(A, 2)))
            new_offsets = get_formation(
                scene["formation"], A, radius=radius, spacing=spacing, size=2.0
            ).to(device)
            base_env._formation_offsets = new_offsets

            # Reassign goals with new formation via nearest-slot
            centroid_w = torch.tensor([0.0, 0.0, 1.0], device=device)
            slot_pos = centroid_w.unsqueeze(0) + new_offsets
            drone_pos = base_env._robot.data.root_pos_w[:A] - base_env._terrain.env_origins[:A]
            dist_matrix = torch.cdist(drone_pos, slot_pos)
            claimed = torch.zeros(A, dtype=torch.bool, device=device)
            for _ in range(A):
                dist_matrix[:, claimed] = float("inf")
                flat_idx = dist_matrix.argmin()
                di = flat_idx // A
                sj = flat_idx % A
                base_env._desired_pos_w[di] = slot_pos[sj] + base_env._terrain.env_origins[di]
                claimed[sj] = True
                dist_matrix[di, :] = float("inf")

            prev_formation = scene["formation"]
            print(f"[FORMATION] → {scene['formation']}")

        # Drone failure scene
        if scene["name"] == "drone_fail" and progress < 0.1:
            # Kill one drone
            if base_env._agent_alive[:A].all():
                victim = A - 1  # kill last drone
                base_env._agent_alive[victim] = False
                print(f"[SWARMRAFT] Drone {victim} killed!")

        # DropoutGuard recovery: recompute for N-1
        if scene["name"] == "swarmraft" and progress < 0.05 and prev_formation != "polygon_recovery":
            alive_count = base_env._agent_alive[:A].sum().item()
            if alive_count < A:
                from ggswarm.formations import get_formation

                spacing = base_env.cfg.formation_target_spacing
                radius = spacing / (2 * math.sin(math.pi / max(alive_count, 2)))
                new_offsets = get_formation("polygon", alive_count, radius=radius).to(device)

                alive_ids = base_env._agent_alive[:A].nonzero(as_tuple=True)[0]
                centroid_w = torch.tensor([0.0, 0.0, 1.0], device=device)
                slot_pos = centroid_w.unsqueeze(0) + new_offsets
                drone_pos = base_env._robot.data.root_pos_w[alive_ids] - base_env._terrain.env_origins[alive_ids]
                dist_matrix = torch.cdist(drone_pos, slot_pos)
                claimed = torch.zeros(alive_count, dtype=torch.bool, device=device)
                for _ in range(alive_count):
                    dist_matrix[:, claimed] = float("inf")
                    flat_idx = dist_matrix.argmin()
                    di = flat_idx // alive_count
                    sj = flat_idx % alive_count
                    drone_id = alive_ids[di]
                    base_env._desired_pos_w[drone_id] = slot_pos[sj] + base_env._terrain.env_origins[drone_id]
                    claimed[sj] = True
                    dist_matrix[di, :] = float("inf")

                prev_formation = "polygon_recovery"
                print(f"[SWARMRAFT] Recovery: {alive_count} drones → heptagon")

        # Policy inference
        with torch.no_grad():
            actions = agent.act(obs, timestep=step, timesteps=total_steps)[0]

        obs, _, _, _, _ = env.step(actions)

        # Advance camera
        centroid = base_env._robot.data.root_pos_w[:A].mean(dim=0).cpu()
        from pxr import Gf

        cam.step(centroid=Gf.Vec3f(centroid[0].item(), centroid[1].item(), centroid[2].item()))

    # --- Cleanup ---
    env.close()
    print(f"\n[SHOWCASE] Complete! Video saved to {args_cli.output_dir}/")


if __name__ == "__main__":
    main()
    simulation_app.close()
    sys.exit(0)
