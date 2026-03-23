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

import torch
from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ggswarm_utils.checkpoint import load_policy_from_checkpoint, resolve_agent  # noqa: E402
from ggswarm_utils.phases.phase2 import Phase2Collector  # noqa: E402
from ggswarm_utils.sim_helpers import (  # noqa: E402
    PHASE_REGISTRY,
    configure_gnn_policy,
    extract_actions,
)


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
parser.add_argument(
    "--task",
    type=str,
    default="Template-GGSwarm-Marl-Direct-v0",
    help="Gym task ID to evaluate checkpoints against.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Override scene.num_envs (default: keep value from env cfg).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Env / trainer seed (match skrl_mappo_cfg.yaml seed for train–eval parity).",
)
parser.add_argument(
    "--gnn",
    dest="use_gnn",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Use GGSwarmGNNPolicy (default: true when --task matches a PHASE_REGISTRY entry "
        "with gnn_default=True, e.g. hover-stability / phase2)."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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


def _gnn_default_for_task(task: str) -> bool:
    """Return True if the task is registered with ``gnn_default=True``."""
    for p in PHASE_REGISTRY.values():
        if p.task == task:
            return p.gnn_default
    return False


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


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
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

    # Override environment config for evaluation
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    use_gnn = args_cli.use_gnn
    if use_gnn is None:
        use_gnn = _gnn_default_for_task(args_cli.task)
    if use_gnn:
        configure_gnn_policy(cfg, Runner)

    # Disable checkpoint writing during eval
    cfg["trainer"]["checkpoint_interval"] = 0

    # Create environment once
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        pass  # MARL is ok
    base_env = env.unwrapped
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env, cfg)
    agent = resolve_agent(runner)

    agent.set_running_mode("eval")
    max_episode_length = getattr(base_env, "max_episode_length", None)
    if max_episode_length is None:
        raise RuntimeError("Could not determine max_episode_length from env")

    # Results table
    results = []

    for step, ckpt_path in checkpoints:
        print(f"\n--- Evaluating checkpoint at {step:,} steps ---")

        load_policy_from_checkpoint(agent, ckpt_path)

        # Use Phase2Collector instead of inline metric computation
        collector = Phase2Collector(airborne_height_margin=0.2)
        obs, _ = env.reset()
        total_steps = args_cli.num_episodes * max_episode_length
        sim_step = 0
        ep_step = 0
        episode_num = 0

        while simulation_app.is_running() and sim_step < total_steps:
            with torch.inference_mode():
                actions = extract_actions(agent, obs, base_env)
                obs, _, _, _, _ = env.step(actions)
                ep_step += 1

                collector.on_step(
                    base_env=base_env,
                    obs=obs,
                    step=sim_step,
                    episode_step=ep_step,
                    episode_num=episode_num,
                )

            sim_step += 1
            if ep_step >= max_episode_length:
                collector.on_episode_end(episode_num)
                ep_step = 0
                episode_num += 1

        summary = collector.summarize()
        results.append({"step": step, **summary})

        print(f"  - Formation error: {summary['mean_formation_error_m']:.4f}m")
        print(f"  - Separation event rate: {summary['separation_event_rate']:.4f}")
        print(f"  - Mean roll/pitch: {summary['mean_roll_deg']:.1f}deg/{summary['mean_pitch_deg']:.1f}deg")
        print(f"  - Airborne ratio: {summary['airborne_ratio']:.4f}")

    env.close()

    # Write results to CSV
    output_csv = args_cli.output_csv or str(run_dir / "checkpoint_progression.csv")
    import csv

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_csv}")
    print("\nSummary:")
    print(f"  Checkpoints analyzed: {len(results)}")
    if results:
        best_formation = min(results, key=lambda x: x["formation_error_m"])
        worst_airborne = min(results, key=lambda x: x["airborne_ratio"])
        print(
            f"  Best formation error: {best_formation['formation_error_m']:.4f}m at step {best_formation['step']:,}"
        )
        print(
            f"  Worst airborne ratio: {worst_airborne['airborne_ratio']:.4f} at step {worst_airborne['step']:,}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
