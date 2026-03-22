"""Phase 4 scale benchmark: measures formation convergence, VRAM, and inference
latency as num_agents grows from 3 → 5 → 10 → 15 → 20.

Run with:
    python scripts/bench_scale.py --checkpoint <path> --headless

Outputs a CSV table to stdout and optionally to --output_csv.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ggswarm_utils.checkpoint import load_policy_from_checkpoint, resolve_agent  # noqa: E402
from ggswarm_utils.eval_stats import pairwise_mean_abs_spacing_error  # noqa: E402
from ggswarm_utils.sim_helpers import configure_gnn_policy, extract_actions, override_agent_count  # noqa: E402


@dataclass
class ScaleResult:
    num_agents: int
    num_envs: int
    mean_formation_error_m: float = float("nan")
    std_formation_error_m: float = float("nan")
    mean_vel_norm_mps: float = float("nan")
    cbf_intervention_rate: float = float("nan")
    steps_per_sec: float = float("nan")
    vram_mb: float = float("nan")
    episode_success_rate: float = float("nan")



def _run_benchmark(
    agent,
    env,
    base_env,
    num_steps: int,
    sim_dt: float,
) -> ScaleResult:
    """Run a fixed number of steps and collect metrics."""
    obs, _ = env.reset()

    formation_errors: list[float] = []
    vel_norms: list[float] = []
    cbf_rates: list[float] = []
    episodes_total = 0
    episodes_succeeded = 0

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()

    for step in range(num_steps):
        with torch.inference_mode():
            actions = extract_actions(agent, obs, base_env)
            obs, _, terminated, truncated, _ = env.step(actions)

            pos_w = base_env.robot.data.root_pos_w.view(
                base_env.num_envs, base_env.cfg.num_agents, 3
            )
            lin_vel_b = base_env.robot.data.root_lin_vel_b.view(
                base_env.num_envs, base_env.cfg.num_agents, 3
            )

            # pairwise_mean_abs_spacing_error returns a scalar; wrap in tensor for consistency
            err_scalar = pairwise_mean_abs_spacing_error(
                pos_w, float(base_env.cfg.target_formation_dist)
            )
            # Use a 1-element tensor to keep downstream code uniform
            err = err_scalar.unsqueeze(0) if err_scalar.dim() == 0 else err_scalar
            formation_errors.append(float(err.mean().item()))
            vel_norms.append(float(torch.norm(lin_vel_b, dim=-1).mean().item()))

            cbf_rate = base_env.extras.get("log", {}).get("cbf_intervention_rate")
            if cbf_rate is not None:
                cbf_rates.append(float(cbf_rate))

            # Count episode completions
            if isinstance(terminated, dict):
                done = any(v.any().item() for v in terminated.values())
            else:
                done = terminated.any().item() if hasattr(terminated, "any") else bool(terminated)
            if done:
                episodes_total += 1
                mean_err = float(err.mean().item())
                if mean_err < 0.5:
                    episodes_succeeded += 1

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0
    steps_per_sec = num_steps / elapsed

    # VRAM
    vram_mb = float("nan")
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()

    err_tensor = torch.tensor(formation_errors)
    success_rate = (
        (episodes_succeeded / max(1, episodes_total)) if episodes_total > 0 else float("nan")
    )

    return ScaleResult(
        num_agents=base_env.cfg.num_agents,
        num_envs=base_env.num_envs,
        mean_formation_error_m=float(err_tensor.mean().item()),
        std_formation_error_m=float(err_tensor.std().item()),
        mean_vel_norm_mps=float(sum(vel_norms) / max(1, len(vel_norms))),
        cbf_intervention_rate=float(sum(cbf_rates) / max(1, len(cbf_rates))) if cbf_rates else float("nan"),
        steps_per_sec=steps_per_sec,
        vram_mb=vram_mb,
        episode_success_rate=success_rate,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 scale benchmark for ggSwarm."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Template-GGSwarm-Marl-Direct-v0",
        help="Base task ID (Phase 3 or Phase 2).",
    )
    parser.add_argument("--algorithm", type=str, default="MAPPO")
    parser.add_argument("--ml_framework", type=str, default="torch")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--agent_counts",
        nargs="+",
        type=int,
        default=[3, 5, 10, 15, 20],
        help="List of num_agents values to benchmark.",
    )
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument(
        "--steps_per_run",
        type=int,
        default=500,
        help="Number of simulation steps per agent-count benchmark.",
    )
    parser.add_argument("--gnn", action="store_true", default=True)
    parser.add_argument("--output_csv", type=str, default=None)

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
    from isaaclab_tasks.utils.hydra import hydra_task_config

    from skrl.utils.runner.torch import Runner

    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = (
        "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
    )

    results: list[ScaleResult] = []

    for n_agents in args_cli.agent_counts:
        if not simulation_app.is_running():
            break

        print(f"\n[BENCH] Running with num_agents={n_agents} ...")

        @hydra_task_config(args_cli.task, agent_cfg_entry_point)
        def _run(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, cfg: dict):
            env_cfg.scene.num_envs = args_cli.num_envs
            env_cfg.sim.device = args_cli.device or env_cfg.sim.device
            obs_dim = 14 if getattr(env_cfg, "raft_enabled", False) else 12
            override_agent_count(env_cfg, n_agents, obs_dim=obs_dim)

            if args_cli.gnn:
                configure_gnn_policy(cfg, Runner)

            cfg["agent"]["experiment"]["directory"] = "logs/skrl/ggswarm_bench"
            cfg["agent"]["experiment"]["experiment_name"] = f"bench_n{n_agents}"
            cfg["trainer"]["checkpoint_interval"] = 0

            env = gym.make(args_cli.task, cfg=env_cfg)
            if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
                env = multi_agent_to_single_agent(env)
            base_env = env.unwrapped
            env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

            runner = Runner(env, cfg)
            agent = resolve_agent(runner)
            load_policy_from_checkpoint(agent, args_cli.checkpoint)

            agent.set_running_mode("eval")
            sim_dt = float(base_env.cfg.sim.dt) * float(base_env.cfg.decimation)
            result = _run_benchmark(agent, env, base_env, args_cli.steps_per_run, sim_dt)
            results.append(result)

            print(
                f"  form_err={result.mean_formation_error_m:.3f}m  "
                f"steps/s={result.steps_per_sec:.1f}  "
                f"VRAM={result.vram_mb:.0f}MB  "
                f"success_rate={result.episode_success_rate:.2f}"
            )
            env.close()

        _run()

    # Print table
    print("\n=== Phase 4 Scale Benchmark Results ===")
    headers = [
        "num_agents", "num_envs", "mean_form_err_m", "std_form_err_m",
        "mean_vel_mps", "cbf_rate", "steps_per_sec", "vram_mb", "success_rate",
    ]
    print("  " + "  ".join(f"{h:>18}" for h in headers))
    for r in results:
        row = [
            r.num_agents, r.num_envs,
            f"{r.mean_formation_error_m:.4f}", f"{r.std_formation_error_m:.4f}",
            f"{r.mean_vel_norm_mps:.4f}", f"{r.cbf_intervention_rate:.4f}",
            f"{r.steps_per_sec:.1f}", f"{r.vram_mb:.0f}", f"{r.episode_success_rate:.3f}",
        ]
        print("  " + "  ".join(f"{str(v):>18}" for v in row))

    if args_cli.output_csv:
        with open(args_cli.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "num_agents": r.num_agents,
                    "num_envs": r.num_envs,
                    "mean_form_err_m": r.mean_formation_error_m,
                    "std_form_err_m": r.std_formation_error_m,
                    "mean_vel_mps": r.mean_vel_norm_mps,
                    "cbf_rate": r.cbf_intervention_rate,
                    "steps_per_sec": r.steps_per_sec,
                    "vram_mb": r.vram_mb,
                    "success_rate": r.episode_success_rate,
                })
        print(f"\n[BENCH] Results written to {args_cli.output_csv}")

    simulation_app.close()



if __name__ == "__main__":
    main()
