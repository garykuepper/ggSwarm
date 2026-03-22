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

import torch
from isaaclab.app import AppLauncher


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


def _pairwise_formation_error(pos_w: torch.Tensor, target_dist: float) -> torch.Tensor:
    """Mean |d_ij - target| over i<j, per environment. Shape [num_envs]."""
    diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)
    num_agents = dist.shape[-1]
    mask = torch.triu(
        torch.ones(num_agents, num_agents, device=pos_w.device, dtype=torch.bool),
        diagonal=1,
    )
    # shape: [num_envs, num_pairs]
    per_env = torch.abs(dist[:, mask] - target_dist).mean(dim=-1)
    return per_env


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
            outputs = agent.act(obs, timestep=0, timesteps=0)
            if hasattr(base_env, "possible_agents"):
                actions = {
                    a: outputs[-1][a].get("mean_actions", outputs[0][a])
                    for a in base_env.possible_agents
                }
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)

            pos_w = base_env.robot.data.root_pos_w.view(
                base_env.num_envs, base_env.cfg.num_agents, 3
            )
            lin_vel_b = base_env.robot.data.root_lin_vel_b.view(
                base_env.num_envs, base_env.cfg.num_agents, 3
            )

            err = _pairwise_formation_error(pos_w, float(base_env.cfg.target_formation_dist))
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
            env_cfg.num_agents = n_agents
            env_cfg.possible_agents = [f"drone_{i}" for i in range(n_agents)]
            env_cfg.action_spaces = {a: 4 for a in env_cfg.possible_agents}
            obs_dim = 14 if getattr(env_cfg, "raft_enabled", False) else 12
            env_cfg.observation_spaces = {a: obs_dim for a in env_cfg.possible_agents}

            if args_cli.gnn:
                cfg["models"]["policy"]["class"] = "GGSwarmGNNPolicy"
                if "network" in cfg["models"]["policy"]:
                    del cfg["models"]["policy"]["network"]
                original_component = Runner._component

                def custom_component(self, name: str):
                    if name.lower() == "ggswarmgnnpolicy":
                        from ggSwarm.tasks.direct.ggswarm_marl.agents.skrl_gnn_policy import (
                            GGSwarmGNNPolicy,
                        )
                        return GGSwarmGNNPolicy
                    return original_component(self, name)

                Runner._component = custom_component

            cfg["agent"]["experiment"]["directory"] = "logs/skrl/ggswarm_bench"
            cfg["agent"]["experiment"]["experiment_name"] = f"bench_n{n_agents}"
            cfg["trainer"]["checkpoint_interval"] = 0

            env = gym.make(args_cli.task, cfg=env_cfg)
            if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
                env = multi_agent_to_single_agent(env)
            base_env = env.unwrapped
            env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

            runner = Runner(env, cfg)
            agent = getattr(runner, "agent", None)
            agents_attr = getattr(runner, "agents", None)
            if agent is None and isinstance(agents_attr, (list, tuple)) and agents_attr:
                agent = agents_attr[0]
            if agent is None:
                raise RuntimeError("Could not resolve skrl agent")

            ckpt = torch.load(args_cli.checkpoint, map_location=agent.device)
            key = "policy" if "policy" in ckpt else ("policy_0" if "policy_0" in ckpt else None)
            if key is None:
                raise KeyError(f"Checkpoint keys: {list(ckpt.keys())}")
            for m in _collect_modules(agent):
                try:
                    m.load_state_dict(ckpt[key])
                    break
                except Exception:
                    continue

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


def _collect_modules(agent) -> list[torch.nn.Module]:
    candidates: list[torch.nn.Module] = []
    for attr in ["policy", "models"]:
        obj = getattr(agent, attr, None)
        if isinstance(obj, torch.nn.Module) and obj not in candidates:
            candidates.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, torch.nn.Module) and v not in candidates:
                    candidates.append(v)
    return candidates


if __name__ == "__main__":
    main()
