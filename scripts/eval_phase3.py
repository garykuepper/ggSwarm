"""Evaluate Phase 3 metrics: collision avoidance (CBF), gap-fill latency
(SwarmRaft), and velocity jitter reduction (MINCO).

Metrics map directly to proposal objectives:
  O1 -- zero collisions (CBF)
  O2 -- >= 20% jitter reduction (MINCO)
  O3 -- gap-fill latency < 2.0 s (SwarmRaft)
  P3.4 -- mean formation error < 0.5 m (maintained from Phase 2)

Pass/fail thresholds:
  collision_rate       == 0.0      (P3.1 / O1)
  gap_fill_latency_s   < 2.0       (P3.2 / O3)
  jitter_reduction_pct >= 20.0     (P3.3 / O2)
  mean_formation_error < 0.5 m    (P3.4)

Usage:
  python scripts/eval_phase3.py \\
      --checkpoint logs/skrl/ggswarm_marl/<run>/checkpoints/best_agent.pt \\
      --num_episodes 10 \\
      --kill_agent_step 50    # step at which to kill agent 0 (0 = disable)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import torch
from isaaclab.app import AppLauncher


# ---------------------------------------------------------------------------
# Stats containers
# ---------------------------------------------------------------------------


@dataclass
class Phase3Stats:
    """Accumulates per-step metrics across all evaluation episodes."""

    # Collision / separation
    collision_events: int = 0
    total_steps: int = 0

    # Formation error
    formation_error_sum: float = 0.0
    formation_error_count: int = 0

    # Velocity jitter (std of ||lin_vel|| per step)
    vel_norm_samples: list = field(default_factory=list)

    # Gap-fill latency: steps from kill event to formation recovery
    gap_fill_latencies_steps: list = field(default_factory=list)
    _kill_step: int = -1          # global step index when last kill happened
    _post_kill: bool = False       # True once kill has fired this episode
    _recovery_threshold: float = 0.5  # formation error below which = recovered

    # CBF intervention rate
    cbf_intervention_rate_sum: float = 0.0
    cbf_intervention_rate_count: int = 0

    def record_step(
        self,
        *,
        pos_w: torch.Tensor,
        lin_vel_b: torch.Tensor,
        target_formation_dist: float,
        min_separation_dist: float,
        global_step: int,
        cbf_intervention_rate: float | None,
    ) -> None:
        num_agents = pos_w.shape[1]
        self.total_steps += 1

        # -- Collision detection --
        diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        eye = torch.eye(num_agents, device=dist.device).unsqueeze(0).bool()
        collision_mask = (dist < min_separation_dist) & ~eye
        if collision_mask.any():
            self.collision_events += 1

        # -- Formation error --
        upper_mask = torch.triu(
            torch.ones(num_agents, num_agents, device=pos_w.device, dtype=torch.bool),
            diagonal=1,
        )
        pair_errors = torch.abs(dist[:, upper_mask] - target_formation_dist)
        self.formation_error_sum += float(pair_errors.mean().item())
        self.formation_error_count += 1

        # -- Velocity jitter --
        vel_norm = torch.norm(lin_vel_b, dim=-1)  # [num_envs, num_agents]
        self.vel_norm_samples.append(float(vel_norm.mean().item()))

        # -- Gap-fill latency tracking --
        if self._post_kill and self._kill_step >= 0:
            form_err = float(pair_errors.mean().item())
            if form_err < self._recovery_threshold:
                latency_steps = global_step - self._kill_step
                self.gap_fill_latencies_steps.append(latency_steps)
                self._post_kill = False
                self._kill_step = -1

        # -- CBF intervention rate --
        if cbf_intervention_rate is not None:
            self.cbf_intervention_rate_sum += cbf_intervention_rate
            self.cbf_intervention_rate_count += 1

    def record_kill(self, global_step: int) -> None:
        """Record the step at which an agent was killed."""
        self._kill_step = global_step
        self._post_kill = True

    def reset_episode(self) -> None:
        """Reset per-episode tracking at episode boundary."""
        self._kill_step = -1
        self._post_kill = False

    def summarize(self, sim_dt: float = 0.02) -> dict[str, float]:
        """Return summary dict.  sim_dt = decimation * physics_dt (s/step)."""
        collision_rate = self.collision_events / max(1, self.total_steps)
        mean_form_err = self.formation_error_sum / max(1, self.formation_error_count)

        vel_tensor = torch.tensor(self.vel_norm_samples) if self.vel_norm_samples else torch.zeros(1)
        mean_vel = float(vel_tensor.mean().item())
        std_vel = float(vel_tensor.std().item())

        gap_fill_latencies_s = [s * sim_dt for s in self.gap_fill_latencies_steps]
        mean_gap_fill_s = (
            sum(gap_fill_latencies_s) / len(gap_fill_latencies_s)
            if gap_fill_latencies_s
            else float("nan")
        )

        cbf_rate = (
            self.cbf_intervention_rate_sum / max(1, self.cbf_intervention_rate_count)
            if self.cbf_intervention_rate_count > 0
            else float("nan")
        )

        return {
            "collision_rate": collision_rate,
            "collision_events": self.collision_events,
            "mean_formation_error_m": mean_form_err,
            "mean_vel_norm_mps": mean_vel,
            "std_vel_norm_mps": std_vel,
            "mean_gap_fill_latency_s": mean_gap_fill_s,
            "gap_fill_samples": len(gap_fill_latencies_s),
            "cbf_intervention_rate": cbf_rate,
        }


# ---------------------------------------------------------------------------
# Pass/fail checks
# ---------------------------------------------------------------------------

PASS_THRESHOLDS = {
    "collision_rate": (0.0, "=="),           # P3.1 / O1
    "mean_gap_fill_latency_s": (2.0, "<"),   # P3.2 / O3
    "mean_formation_error_m": (0.5, "<"),    # P3.4
}


def _check_pass_fail(summary: dict[str, float]) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for key, (threshold, op) in PASS_THRESHOLDS.items():
        val = summary.get(key)
        if val is None or (isinstance(val, float) and val != val):
            results[key] = False
            continue
        if op == "==":
            results[key] = abs(val - threshold) < 1e-9
        elif op == "<":
            results[key] = val < threshold
        elif op == ">":
            results[key] = val > threshold
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pairwise_distances(pos_w: torch.Tensor) -> torch.Tensor:
    """Compute all pairwise distances. Shape [num_envs, num_agents, num_agents]."""
    diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
    return torch.norm(diff, dim=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 3 metrics (CBF, SwarmRaft, MINCO) for ggSwarm."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Template-GGSwarm-Marl-Direct-v0",
        help="Gym task ID. Use Template-GGSwarm-Marl-Phase3-v0 for Phase 3 features.",
    )
    parser.add_argument("--algorithm", type=str, default="MAPPO")
    parser.add_argument("--ml_framework", type=str, default="torch")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_agents", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument(
        "--kill_agent_step",
        type=int,
        default=0,
        help="Episode step at which to forcibly terminate agent 0 (0 = disabled).",
    )
    parser.add_argument(
        "--gnn",
        action="store_true",
        default=True,
        help="Use GGSwarmGNNPolicy (GATv2).",
    )
    parser.add_argument("--seed", type=int, default=42)

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
        raise RuntimeError(f"Unsupported skrl version: {skrl.__version__}. Install skrl>=1.4.3")

    from skrl.utils.runner.torch import Runner

    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = (
        "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
    )

    @hydra_task_config(args_cli.task, agent_cfg_entry_point)
    def _eval(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, cfg: dict):
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if hasattr(env_cfg, "num_agents") and args_cli.num_agents is not None:
            env_cfg.num_agents = args_cli.num_agents
            env_cfg.possible_agents = [f"drone_{i}" for i in range(env_cfg.num_agents)]
            env_cfg.action_spaces = {agent: 4 for agent in env_cfg.possible_agents}
            obs_dim = 14 if getattr(env_cfg, "raft_enabled", False) else 12
            env_cfg.observation_spaces = {agent: obs_dim for agent in env_cfg.possible_agents}

        cfg["seed"] = args_cli.seed
        env_cfg.seed = args_cli.seed

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

        task_name = args_cli.task.split(":")[-1]
        log_root_path = os.path.abspath(
            os.path.join("logs", "skrl", cfg["agent"]["experiment"]["directory"])
        )

        if args_cli.checkpoint is not None:
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(
                log_root_path,
                run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
                other_dirs=["checkpoints"],
            )

        cfg["agent"]["experiment"]["directory"] = log_root_path
        cfg["agent"]["experiment"]["experiment_name"] = os.path.basename(
            os.path.dirname(os.path.dirname(resume_path)).rstrip("/\\")
        )

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
            env = multi_agent_to_single_agent(env)

        base_env = env.unwrapped
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        cfg["trainer"]["checkpoint_interval"] = 0
        runner = Runner(env, cfg)

        # Resolve agent
        agent = getattr(runner, "agent", None)
        agents_attr = getattr(runner, "agents", None)
        if agent is None and isinstance(agents_attr, (list, tuple)) and agents_attr:
            agent = agents_attr[0]
        if agent is None:
            raise RuntimeError("Could not resolve skrl agent for evaluation")

        # Load checkpoint
        checkpoint = torch.load(resume_path, map_location=agent.device)

        def _load(sd: dict, src: str) -> bool:
            for m in _collect_modules(agent):
                try:
                    m.load_state_dict(sd)
                    print(f"[EVAL] Loaded policy from {src}")
                    return True
                except Exception:
                    continue
            return False

        if "policy" in checkpoint:
            if not _load(checkpoint["policy"], "checkpoint['policy']"):
                raise KeyError("Could not load policy from checkpoint")
        elif "policy_0" in checkpoint:
            if not _load(checkpoint["policy_0"], "checkpoint['policy_0']"):
                raise KeyError("Could not load policy from checkpoint")
        else:
            raise KeyError(f"Checkpoint keys: {list(checkpoint.keys())}")

        agent.set_running_mode("eval")

        max_episode_length = getattr(base_env, "max_episode_length", None)
        if max_episode_length is None:
            raise RuntimeError("Could not determine max_episode_length from env")

        sim_dt = float(base_env.cfg.sim.dt) * float(base_env.cfg.decimation)
        total_steps = int(args_cli.num_episodes) * int(max_episode_length)

        stats = Phase3Stats()
        obs, _ = env.reset()
        global_step = 0
        episode_step = 0
        episode_num = 0

        print(f"\n[EVAL Phase3] checkpoint: {resume_path}")
        print(f"[EVAL Phase3] task: {args_cli.task}")
        print(f"[EVAL Phase3] episodes: {args_cli.num_episodes}")
        print(f"[EVAL Phase3] kill_agent_step: {args_cli.kill_agent_step}")
        print(f"[EVAL Phase3] sim_dt: {sim_dt:.4f} s/step\n")

        # Baseline velocity std (first episode, no kill) for jitter comparison
        baseline_vel_samples: list[float] = []
        collecting_baseline = True

        while simulation_app.is_running() and global_step < total_steps:
            with torch.inference_mode():
                outputs = agent.act(obs, timestep=0, timesteps=0)
                if hasattr(base_env, "possible_agents"):
                    actions = {
                        a: outputs[-1][a].get("mean_actions", outputs[0][a])
                        for a in base_env.possible_agents
                    }
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])

                # --- Simulated agent loss (kill_agent_step > 0) ---
                if (
                    args_cli.kill_agent_step > 0
                    and episode_step == args_cli.kill_agent_step
                    and not stats._post_kill
                    and episode_num > 0  # skip first episode (baseline)
                ):
                    # Force agent 0 out of bounds by setting its desired_pos_w to floor
                    base_env._desired_pos_w[:, 0, 2] = -10.0
                    stats.record_kill(global_step)
                    print(
                        f"[EVAL Phase3] Killed agent 0 at episode {episode_num} "
                        f"step {episode_step} (global {global_step})"
                    )

                obs, _, _, _, _ = env.step(actions)

                # Gather tensors
                pos_w = base_env.robot.data.root_pos_w.view(
                    base_env.num_envs, base_env.cfg.num_agents, 3
                )
                lin_vel_b = base_env.robot.data.root_lin_vel_b.view(
                    base_env.num_envs, base_env.cfg.num_agents, 3
                )

                cbf_rate = base_env.extras.get("log", {}).get("cbf_intervention_rate")

                stats.record_step(
                    pos_w=pos_w,
                    lin_vel_b=lin_vel_b,
                    target_formation_dist=float(base_env.cfg.target_formation_dist),
                    min_separation_dist=float(base_env.cfg.min_separation_dist),
                    global_step=global_step,
                    cbf_intervention_rate=cbf_rate,
                )

                # Collect baseline vel norms during first episode
                if collecting_baseline and episode_num == 0:
                    vel_norm = torch.norm(lin_vel_b, dim=-1).mean()
                    baseline_vel_samples.append(float(vel_norm.item()))

            global_step += 1
            episode_step += 1

            if episode_step >= max_episode_length:
                episode_num += 1
                episode_step = 0
                stats.reset_episode()
                if collecting_baseline and episode_num == 1:
                    collecting_baseline = False
                if episode_num % max(1, args_cli.num_episodes // 5) == 0:
                    s = stats.summarize(sim_dt)
                    print(
                        f"[EVAL Phase3] episode={episode_num}/{args_cli.num_episodes} "
                        f"collisions={stats.collision_events} "
                        f"form_err={s['mean_formation_error_m']:.3f}m "
                        f"std_vel={s['std_vel_norm_mps']:.4f} "
                        f"gap_fill_s={s['mean_gap_fill_latency_s']:.2f}"
                    )

        summary = stats.summarize(sim_dt)

        # Compute jitter reduction vs. baseline
        jitter_reduction_pct = float("nan")
        if baseline_vel_samples and stats.vel_norm_samples:
            baseline_tensor = torch.tensor(baseline_vel_samples)
            all_tensor = torch.tensor(stats.vel_norm_samples)
            baseline_std = float(baseline_tensor.std().item())
            overall_std = float(all_tensor.std().item())
            if baseline_std > 1e-6:
                jitter_reduction_pct = 100.0 * (baseline_std - overall_std) / baseline_std
        summary["jitter_reduction_pct"] = jitter_reduction_pct

        pass_fail = _check_pass_fail(summary)
        # Jitter reduction check requires baseline
        pass_fail["jitter_reduction_pct"] = (
            (jitter_reduction_pct >= 20.0)
            if not (jitter_reduction_pct != jitter_reduction_pct)
            else False
        )

        print("\n=== Phase 3 Evaluation Summary ===")
        print(f"checkpoint     : {resume_path}")
        print(f"task           : {args_cli.task}")
        print(f"episodes       : {args_cli.num_episodes}")
        print(f"kill_agent_step: {args_cli.kill_agent_step}")
        print()
        _print_metric(
            "collision_rate          (target == 0.0)",
            summary["collision_rate"],
            pass_fail.get("collision_rate", False),
        )
        _print_metric(
            "collision_events        (total)",
            summary["collision_events"],
        )
        _print_metric(
            "mean_formation_error_m  (target < 0.5)",
            summary["mean_formation_error_m"],
            pass_fail.get("mean_formation_error_m", False),
        )
        _print_metric(
            "mean_vel_norm_mps",
            summary["mean_vel_norm_mps"],
        )
        _print_metric(
            "std_vel_norm_mps",
            summary["std_vel_norm_mps"],
        )
        _print_metric(
            "jitter_reduction_pct    (target >= 20%)",
            summary["jitter_reduction_pct"],
            pass_fail.get("jitter_reduction_pct", False),
        )
        _print_metric(
            "mean_gap_fill_latency_s (target < 2.0)",
            summary["mean_gap_fill_latency_s"],
            pass_fail.get("mean_gap_fill_latency_s", False),
        )
        _print_metric(
            "gap_fill_samples        (kill events recovered)",
            summary["gap_fill_samples"],
        )
        _print_metric(
            "cbf_intervention_rate",
            summary["cbf_intervention_rate"],
        )

        all_pass = all(pass_fail.values())
        print(f"\n{'[PASS]' if all_pass else '[FAIL]'} Phase 3 objectives")
        for k, v in pass_fail.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")

        env.close()

    _eval()
    simulation_app.close()


def _collect_modules(agent) -> list[torch.nn.Module]:
    """Collect all nn.Module candidates from an skrl agent."""
    candidates: list[torch.nn.Module] = []

    def _add(x: object) -> None:
        if isinstance(x, torch.nn.Module) and x not in candidates:
            candidates.append(x)

    for attr in ["policy", "models"]:
        obj = getattr(agent, attr, None)
        if isinstance(obj, torch.nn.Module):
            _add(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _add(v)

    for name in dir(agent):
        try:
            v = getattr(agent, name)
        except Exception:
            continue
        if isinstance(v, torch.nn.Module):
            _add(v)
        elif isinstance(v, dict):
            for vv in v.values():
                _add(vv)

    return candidates


def _print_metric(label: str, value: float, passed: bool | None = None) -> None:
    status = ""
    if passed is not None:
        status = "  [PASS]" if passed else "  [FAIL]"
    if isinstance(value, float) and value != value:
        print(f"  {label:<50}: N/A (no data){status}")
    else:
        print(f"  {label:<50}: {value:.6g}{status}")


if __name__ == "__main__":
    main()
