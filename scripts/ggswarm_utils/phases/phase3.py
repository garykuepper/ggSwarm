"""Phase 3 metric collector (CBF safety, SwarmRaft gap-fill, MINCO jitter).

Covers Phase 3 (muscle refinement) and Phase 4 (stress testing) tasks.
Implements the PhaseCollector protocol from eval_runner.

Metrics map to proposal objectives:
  O1 -- zero collisions (CBF)         -> collision_rate == 0
  O2 -- >=20% jitter reduction (MINCO) -> jitter_reduction_pct >= 20
  O3 -- gap-fill latency < 2.0 s       -> mean_gap_fill_latency_s < 2.0
  P3.4 -- formation error < 0.5 m      -> mean_formation_error_m < 0.5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


PASS_THRESHOLDS: dict[str, tuple[float, str]] = {
    "collision_rate": (0.0, "=="),           # P3.1 / O1
    "mean_gap_fill_latency_s": (2.0, "<"),   # P3.2 / O3
    "mean_formation_error_m": (0.5, "<"),    # P3.4
}


@dataclass
class Phase3Stats:
    """Accumulates per-step metrics across Phase 3 evaluation episodes."""

    # Collision / separation
    collision_events: int = 0
    total_steps: int = 0

    # Formation error
    formation_error_sum: float = 0.0
    formation_error_count: int = 0

    # Velocity jitter (std of ||lin_vel|| across steps)
    vel_norm_samples: list[float] = field(default_factory=list)

    # Gap-fill latency: steps from kill event to formation recovery
    gap_fill_latencies_steps: list[int] = field(default_factory=list)
    _kill_step: int = -1
    _post_kill: bool = False
    _recovery_threshold: float = 0.5

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
        """Accumulate one step worth of Phase 3 metrics."""
        num_agents = pos_w.shape[1]
        self.total_steps += 1

        # Collision detection
        diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        eye = torch.eye(num_agents, device=dist.device).unsqueeze(0).bool()
        collision_mask = (dist < min_separation_dist) & ~eye
        if collision_mask.any():
            self.collision_events += 1

        # Formation error (upper-triangle mean)
        upper_mask = torch.triu(
            torch.ones(num_agents, num_agents, device=pos_w.device, dtype=torch.bool),
            diagonal=1,
        )
        pair_errors = torch.abs(dist[:, upper_mask] - target_formation_dist)
        self.formation_error_sum += float(pair_errors.mean().item())
        self.formation_error_count += 1

        # Velocity jitter
        vel_norm = torch.norm(lin_vel_b, dim=-1)  # [num_envs, num_agents]
        self.vel_norm_samples.append(float(vel_norm.mean().item()))

        # Gap-fill latency tracking (only active after a kill)
        if self._post_kill and self._kill_step >= 0:
            form_err = float(pair_errors.mean().item())
            if form_err < self._recovery_threshold:
                latency_steps = global_step - self._kill_step
                self.gap_fill_latencies_steps.append(latency_steps)
                self._post_kill = False
                self._kill_step = -1

        if cbf_intervention_rate is not None:
            self.cbf_intervention_rate_sum += cbf_intervention_rate
            self.cbf_intervention_rate_count += 1

    def record_kill(self, global_step: int) -> None:
        """Record the step at which an agent was forcibly killed."""
        self._kill_step = global_step
        self._post_kill = True

    def reset_episode(self) -> None:
        """Reset per-episode kill tracking at episode boundary."""
        self._kill_step = -1
        self._post_kill = False

    def summarize(self, sim_dt: float = 0.02) -> dict[str, float]:
        """Return summary dict.  sim_dt = decimation * physics_dt (s/step)."""
        collision_rate = self.collision_events / max(1, self.total_steps)
        mean_form_err = self.formation_error_sum / max(1, self.formation_error_count)

        vel_tensor = (
            torch.tensor(self.vel_norm_samples)
            if self.vel_norm_samples
            else torch.zeros(1)
        )
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
            "collision_events": float(self.collision_events),
            "mean_formation_error_m": mean_form_err,
            "mean_vel_norm_mps": mean_vel,
            "std_vel_norm_mps": std_vel,
            "mean_gap_fill_latency_s": mean_gap_fill_s,
            "gap_fill_samples": float(len(gap_fill_latencies_s)),
            "cbf_intervention_rate": cbf_rate,
        }


def check_pass_fail(summary: dict[str, float]) -> dict[str, bool]:
    """Return pass/fail verdict for each Phase 3 threshold key."""
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


class Phase3Collector:
    """PhaseCollector for Phase 3 / Phase 4 tasks.

    Supports an optional agent-kill scenario (kill_agent_step > 0) to measure
    gap-fill latency.  The kill is injected by setting _desired_pos_w for agent 0
    to floor level — this approach doesn't require env code changes.

    Args:
        kill_agent_step: Episode step at which to kill agent 0 (0 = disabled).
        sim_dt:          Simulation timestep (decimation * physics_dt) in seconds.
    """

    def __init__(
        self,
        kill_agent_step: int = 0,
        sim_dt: float = 0.02,
    ) -> None:
        self._stats = Phase3Stats()
        self._kill_agent_step = kill_agent_step
        self._sim_dt = sim_dt
        self._baseline_vel_samples: list[float] = []
        self._collecting_baseline: bool = True
        self._global_step: int = 0

    def on_step(
        self,
        *,
        base_env: object,
        obs: object,
        step: int,
        episode_step: int,
        episode_num: int,
    ) -> None:
        """Accumulate Phase 3 metrics; optionally inject agent kill."""
        self._global_step = step

        # Inject simulated agent loss at the specified episode step
        if (
            self._kill_agent_step > 0
            and episode_step == self._kill_agent_step
            and not self._stats._post_kill
            and episode_num > 0  # skip first episode (used as baseline)
        ):
            base_env._desired_pos_w[:, 0, 2] = -10.0  # type: ignore[attr-defined]
            self._stats.record_kill(step)

        # shape: [num_envs, num_agents, 3]
        pos_w = base_env.robot.data.root_pos_w.view(  # type: ignore[attr-defined]
            base_env.num_envs, base_env.cfg.num_agents, 3  # type: ignore[attr-defined]
        )
        lin_vel_b = base_env.robot.data.root_lin_vel_b.view(  # type: ignore[attr-defined]
            base_env.num_envs, base_env.cfg.num_agents, 3  # type: ignore[attr-defined]
        )
        cbf_rate = (
            base_env.extras.get("log", {}).get("cbf_intervention_rate")  # type: ignore[attr-defined]
        )

        self._stats.record_step(
            pos_w=pos_w,
            lin_vel_b=lin_vel_b,
            target_formation_dist=float(base_env.cfg.target_formation_dist),  # type: ignore[attr-defined]
            min_separation_dist=float(base_env.cfg.min_separation_dist),  # type: ignore[attr-defined]
            global_step=step,
            cbf_intervention_rate=cbf_rate,
        )

        # Collect baseline velocity during first episode
        if self._collecting_baseline and episode_num == 0:
            vel_norm = torch.norm(lin_vel_b, dim=-1).mean()
            self._baseline_vel_samples.append(float(vel_norm.item()))

    def on_episode_end(self, episode_num: int) -> None:
        """Reset per-episode kill state; stop collecting baseline after ep 0."""
        self._stats.reset_episode()
        if self._collecting_baseline and episode_num == 0:
            self._collecting_baseline = False

    def summarize(self) -> dict[str, float]:
        """Return Phase 3 summary with jitter reduction computed vs. baseline."""
        summary = self._stats.summarize(self._sim_dt)

        # Compute jitter reduction vs. first-episode baseline
        jitter_reduction_pct = float("nan")
        if self._baseline_vel_samples and self._stats.vel_norm_samples:
            baseline_tensor = torch.tensor(self._baseline_vel_samples)
            all_tensor = torch.tensor(self._stats.vel_norm_samples)
            baseline_std = float(baseline_tensor.std().item())
            overall_std = float(all_tensor.std().item())
            if baseline_std > 1e-6:
                jitter_reduction_pct = (
                    100.0 * (baseline_std - overall_std) / baseline_std
                )
        summary["jitter_reduction_pct"] = jitter_reduction_pct

        return summary
