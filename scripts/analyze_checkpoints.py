"""Analyze checkpoint progression over a training run.

Evaluates multiple checkpoints from a training directory and produces a table
showing how metrics evolve over training. This helps diagnose when the policy
degrades, e.g., at curriculum transitions.

Usage:
    python scripts/analyze_checkpoints.py --run_dir logs/skrl/ggswarm_marl/<timestamp> \
        --interval 50000 --num_episodes 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Analyze checkpoint progression over training."
)
parser.add_argument(
    "--run_dir",
    type=str,
    required=True,
    help="Path to training run directory (containing checkpoints/ subdirectory).",
)
parser.add_argument(
    "--interval",
    type=int,
    default=50000,
    help="Checkpoint interval to analyze (e.g. 50000 means analyze every 50k steps).",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=3,
    help="Number of episodes to evaluate per checkpoint.",
)
parser.add_argument(
    "--output_csv",
    type=str,
    default=None,
    help="Output CSV file for results. Defaults to run_dir/checkpoint_progression.csv.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import random

import gymnasium as gym
import skrl
from packaging import version

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import ggSwarm.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

if version.parse(skrl.__version__) < version.parse("1.4.3"):
    raise RuntimeError(
        f"Unsupported skrl version: {skrl.__version__}. Install skrl>=1.4.3"
    )

from skrl.utils.runner.torch import Runner


def _collect_checkpoints(run_dir: Path, interval: int) -> list[tuple[int, Path]]:
    """Collect checkpoint paths at regular intervals.

    Returns:
        List of (step_number, checkpoint_path) tuples, sorted by step.
    """
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    checkpoints = []

    # Collect agent_*.pt files
    for ckpt_file in checkpoints_dir.glob("agent_*.pt"):
        try:
            step = int(ckpt_file.stem.split("_")[1])
            if step % interval == 0:
                checkpoints.append((step, ckpt_file))
        except (ValueError, IndexError):
            continue

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


@hydra_task_config("Template-GGSwarm-Marl-Direct-v0", "skrl_mappo_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, cfg: dict):
    """Analyze checkpoint progression."""
    run_dir = Path(args_cli.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    checkpoints = _collect_checkpoints(run_dir, args_cli.interval)
    if not checkpoints:
        print(
            f"No checkpoints found at interval {args_cli.interval} in {run_dir / 'checkpoints'}"
        )
        return

    print(f"Found {len(checkpoints)} checkpoints to analyze:")
    for step, ckpt_path in checkpoints:
        print(f"  {step:,} steps: {ckpt_path.name}")

    # Load eval function from eval_phase2
    # Import here to avoid circular dependencies
    from scripts.eval_phase2 import (
        EvalStats,
        _pairwise_mean_abs_spacing_error,
        _compute_orientation_metrics,
        _compute_altitude_metrics,
    )

    # Override environment config for evaluation
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    cfg["seed"] = random.randint(0, 10000)
    env_cfg.seed = cfg["seed"]

    # Disable checkpoint writing during eval
    cfg["trainer"]["checkpoint_interval"] = 0

    # Create environment once
    env = gym.make("Template-GGSwarm-Marl-Direct-v0", cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        pass  # MARL is ok
    base_env = env.unwrapped
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # Set up runner (use it as a checkpoint loader)
    runner = Runner(env, cfg)
    agent = getattr(runner, "agent", None)
    agents_attr = getattr(runner, "agents", None)
    if agent is None and isinstance(agents_attr, (list, tuple)) and agents_attr:
        agent = agents_attr[0]
    if agent is None:
        raise RuntimeError("Could not resolve skrl agent for evaluation")

    agent.set_running_mode("eval")
    max_episode_length = getattr(base_env, "max_episode_length", None)
    if max_episode_length is None:
        raise RuntimeError("Could not determine max_episode_length from env")

    # Results table
    results = []

    for step, ckpt_path in checkpoints:
        print(f"\n--- Evaluating checkpoint at {step:,} steps ---")

        # Load checkpoint
        checkpoint = torch.load(str(ckpt_path), map_location=agent.device)

        def _load_policy(state_dict):
            """Try to load policy state dict."""
            candidates = []

            if hasattr(agent, "policy"):
                candidates.append(getattr(agent, "policy"))

            if hasattr(agent, "models"):
                models = getattr(agent, "models")
                if isinstance(models, dict):
                    policy_like = [m for k, m in models.items() if "policy" in str(k).lower()]
                    for m in policy_like or list(models.values()):
                        candidates.append(m)
                else:
                    if hasattr(models, "policy"):
                        candidates.append(getattr(models, "policy"))

            for name in dir(agent):
                try:
                    v = getattr(agent, name)
                except Exception:
                    continue
                if isinstance(v, torch.nn.Module):
                    candidates.append(v)
                elif isinstance(v, dict):
                    for vv in v.values():
                        if isinstance(vv, torch.nn.Module):
                            candidates.append(vv)

            for m in candidates:
                try:
                    m.load_state_dict(state_dict)
                    return True
                except Exception:
                    continue
            return False

        if "policy" in checkpoint:
            _load_policy(checkpoint["policy"])
        elif "policy_0" in checkpoint:
            _load_policy(checkpoint["policy_0"])
        else:
            loaded = False
            if isinstance(checkpoint, dict):
                for k, v in checkpoint.items():
                    if isinstance(v, dict) and "policy" in v:
                        loaded = _load_policy(v["policy"])
                        if loaded:
                            break
            if not loaded:
                print(f"  WARNING: Could not load policy from checkpoint")

        # Run evaluation
        stats = EvalStats()
        obs, _ = env.reset()
        total_steps = args_cli.num_episodes * max_episode_length
        steps = 0

        while simulation_app.is_running() and steps < total_steps:
            with torch.inference_mode():
                outputs = agent.act(obs, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents"):
                    actions = {
                        a: outputs[-1][a].get("mean_actions", outputs[0][a])
                        for a in base_env.possible_agents
                    }
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])

                obs, _, _, _, _ = env.step(actions)

                pos_w = base_env.robot.data.root_pos_w.view(
                    base_env.num_envs, base_env.cfg.num_agents, 3
                )
                formation_error = _pairwise_mean_abs_spacing_error(
                    pos_w, float(base_env.cfg.target_formation_dist)
                )

                diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
                dist = torch.norm(diff, dim=-1)
                eye = torch.eye(base_env.cfg.num_agents, device=dist.device).unsqueeze(0)
                separation_event_rate = (
                    ((dist < float(base_env.cfg.min_separation_dist)) & (1.0 - eye).bool())
                    .any(dim=2)
                    .any(dim=1)
                    .float()
                    .mean()
                )

                lin_vel_b = base_env.robot.data.root_lin_vel_b.view(
                    base_env.num_envs, base_env.cfg.num_agents, 3
                )
                mean_speed = torch.norm(lin_vel_b, dim=-1).mean()

                desired_altitude = base_env._desired_pos_w[:, :, 2]
                current_altitude = pos_w[:, :, 2]
                mean_altitude_error = torch.abs(current_altitude - desired_altitude).mean()
                ground_hit_rate = (current_altitude < float(base_env.cfg.min_height)).any(
                    dim=1
                ).float().mean()
                airborne_threshold = float(base_env.cfg.min_height) + 0.2
                airborne_ratio = (current_altitude > airborne_threshold).float().mean()

                proj_grav_b = base_env.robot.data.projected_gravity_b.view(
                    base_env.num_envs, base_env.cfg.num_agents, 3
                )
                mean_roll_deg, mean_pitch_deg, orientation_violation_rate = (
                    _compute_orientation_metrics(proj_grav_b)
                )

                altitude_std, _ = _compute_altitude_metrics(pos_w, base_env._desired_pos_w)

                episode_survival = torch.tensor(steps + 1, dtype=torch.float32)

                stats.update(
                    formation_error=formation_error,
                    separation_event_rate=separation_event_rate,
                    mean_speed=mean_speed,
                    mean_altitude_error=mean_altitude_error,
                    ground_hit_rate=ground_hit_rate,
                    airborne_ratio=airborne_ratio,
                    mean_roll_deg=mean_roll_deg,
                    mean_pitch_deg=mean_pitch_deg,
                    orientation_violation_rate=orientation_violation_rate,
                    altitude_std=altitude_std,
                    episode_survival_steps=episode_survival,
                )

            steps += 1

        summary = stats.summarize()
        results.append({
            "step": step,
            "formation_error_m": summary["mean_formation_error_m"],
            "separation_event_rate": summary["separation_event_rate"],
            "mean_speed_mps": summary["mean_speed_mps"],
            "altitude_error_m": summary["mean_altitude_error_m"],
            "ground_hit_rate": summary["ground_hit_rate"],
            "airborne_ratio": summary["airborne_ratio"],
            "mean_roll_deg": summary["mean_roll_deg"],
            "mean_pitch_deg": summary["mean_pitch_deg"],
            "orientation_violation_rate": summary["orientation_violation_rate"],
            "altitude_std_m": summary["altitude_std_m"],
        })

        print(f"  ✓ Formation error: {summary['mean_formation_error_m']:.4f}m")
        print(f"  ✓ Separation event rate: {summary['separation_event_rate']:.4f}")
        print(f"  ✓ Mean roll/pitch: {summary['mean_roll_deg']:.1f}°/{summary['mean_pitch_deg']:.1f}°")
        print(f"  ✓ Airborne ratio: {summary['airborne_ratio']:.4f}")

    env.close()

    # Write results to CSV
    output_csv = args_cli.output_csv or str(run_dir / "checkpoint_progression.csv")
    import csv

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ Results written to: {output_csv}")
    print("\nSummary:")
    print(f"  Checkpoints analyzed: {len(results)}")
    if results:
        best_formation = min(results, key=lambda x: x["formation_error_m"])
        worst_airborne = min(results, key=lambda x: x["airborne_ratio"])
        print(
            f"  Best formation error: {best_formation['formation_error_m']:.4f}m at step {best_formation['step']:,}"
        )
        print(
            f"  Best airborne ratio: {1 - worst_airborne['airborne_ratio']:.4f} at step {worst_airborne['step']:,}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
