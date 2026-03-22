"""PD-only baseline: constant neutral RL commands (zeros) with no trained policy.

Zeros in [-1, 1] action space correspond to nominal hover collective and zero
attitude setpoints; the inner-loop PD converts these into thrust and moments.

Usage:
    python scripts/pd_neutral_baseline.py --headless \\
        --task Template-GGSwarm-Marl-HoverStability-v0 --num_envs 8 --num_steps 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    type=str,
    default="Template-GGSwarm-Marl-HoverStability-v0",
    help="Gym task id (default: hover-stability).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=8,
    help="Parallel envs (keep small for a readable trace).",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=500,
    help="Decimated env steps to simulate after reset.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Env seed (match skrl_mappo_cfg.yaml for train parity).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import ggSwarm.tasks  # noqa: F401, E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import DirectMARLEnv  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    """Build env, step with neutral actions, print altitude / ground / airborne stats."""
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    if not isinstance(base, DirectMARLEnv):
        raise TypeError(f"Expected DirectMARLEnv, got {type(base).__name__}")

    device = base.device
    n_env = base.num_envs
    agents = list(base.cfg.possible_agents)
    neutral: dict[str, torch.Tensor] = {
        a: torch.zeros(n_env, 4, device=device, dtype=torch.float32) for a in agents
    }

    mh = float(base.cfg.min_height)
    airborne_z = mh + 0.2

    env.reset()
    sum_mean_z = 0.0
    sum_ground = 0.0
    sum_airborne = 0.0
    n_samples = 0

    for _ in range(args_cli.num_steps):
        env.step(neutral)
        pos_w = base.robot.data.root_pos_w.view(n_env, base.cfg.num_agents, 3)
        z = pos_w[:, :, 2]
        mean_z = float(z.mean().item())
        ground = float((z < mh).any(dim=1).float().mean().item())
        air = float((z > airborne_z).float().mean().item())
        sum_mean_z += mean_z
        sum_ground += ground
        sum_airborne += air
        n_samples += 1

    env.close()

    avg_z = sum_mean_z / max(n_samples, 1)
    avg_ground = sum_ground / max(n_samples, 1)
    avg_air = sum_airborne / max(n_samples, 1)

    print("\n=== PD neutral baseline (zero RL commands) ===")
    print(f"  task         : {args_cli.task}")
    print(f"  seed         : {args_cli.seed}")
    print(f"  num_envs     : {n_env}")
    print(f"  num_steps    : {args_cli.num_steps}")
    print(f"  min_height   : {mh} m")
    print(f"  airborne_z   : > {airborne_z} m (scorecard-style band)")
    print(f"  mean_world_z : {avg_z:.4f} m (time avg)")
    print(f"  ground_hit_rate (any agent / env): {avg_ground:.4f}")
    print(f"  airborne_ratio (agent-steps)     : {avg_air:.4f}")
    print(
        "\nInterpret: if ground_hit_rate is high or mean_z drifts below the band, "
        "tune thrust/PD/spawn (see docs/ops/phase2a_diagnostics.md) before long MAPPO."
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
