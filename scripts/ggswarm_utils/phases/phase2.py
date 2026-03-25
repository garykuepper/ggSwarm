"""Phase 2 metric collector (hover-stability, formation, and perturbation tasks).

Handles Phase 2A (hover-stability), 2B (formation), and 2C (perturbation)
identically at the metrics level — they differ only in their task ID.

Implements the PhaseCollector protocol from eval_runner.
"""

from __future__ import annotations

import torch

from ggswarm_utils.eval_stats import (
    EvalStats,
    compute_altitude_metrics,
    compute_orientation_metrics,
    pairwise_mean_abs_spacing_error,
)

_AIRBORNE_HEIGHT_MARGIN_DEFAULT: float = 0.2


class Phase2Collector:
    """PhaseCollector for Phase 2 formation / hover-stability tasks.

    Collects formation error, separation events, speed, altitude error,
    orientation metrics, and **episode survival** (mean steps until first batch
    ground hit per eval episode, or full horizon — recorded in on_episode_end only).

    Args:
        airborne_height_margin: Extra height (m) above min_height to count
            a drone as stably airborne.  Matches the CLI default.
        log_interval_frac: Fraction of total steps between progress prints
            (0 = never print).
    """

    def __init__(
        self,
        airborne_height_margin: float = _AIRBORNE_HEIGHT_MARGIN_DEFAULT,
        log_interval_frac: float = 0.1,
    ) -> None:
        self._stats = EvalStats()
        self._airborne_margin = airborne_height_margin
        self._log_interval_frac = log_interval_frac
        self._episode_step_count: int = 0
        # First simulation step index (1-based within episode) with any env ground hit.
        self._first_ground_hit_step: int | None = None

    def on_step(
        self,
        *,
        base_env: object,
        obs: object,
        step: int,
        episode_step: int,
        episode_num: int,
    ) -> None:
        """Accumulate one step of Phase 2 metrics from base_env tensors."""
        # shape: [num_envs, num_agents, 3]
        pos_w = base_env.robot.data.root_pos_w.view(  # type: ignore[attr-defined]
            base_env.num_envs, base_env.cfg.num_agents, 3  # type: ignore[attr-defined]
        )

        formation_error = pairwise_mean_abs_spacing_error(
            pos_w, float(base_env.cfg.target_formation_dist)  # type: ignore[attr-defined]
        )

        diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        eye = torch.eye(
            base_env.cfg.num_agents,  # type: ignore[attr-defined]
            device=dist.device,
        ).unsqueeze(0)
        separation_event_rate = (
            (
                (dist < float(base_env.cfg.min_separation_dist))  # type: ignore[attr-defined]
                & (1.0 - eye).bool()
            )
            .any(dim=2)
            .any(dim=1)
            .float()
            .mean()
        )

        # shape: [num_envs, num_agents, 3]
        lin_vel_b = base_env.robot.data.root_lin_vel_b.view(  # type: ignore[attr-defined]
            base_env.num_envs, base_env.cfg.num_agents, 3  # type: ignore[attr-defined]
        )
        mean_speed = torch.norm(lin_vel_b, dim=-1).mean()

        # shape: [num_envs, num_agents]
        desired_altitude = base_env._desired_pos_w[:, :, 2]  # type: ignore[attr-defined]
        current_altitude = pos_w[:, :, 2]
        mean_altitude_error = torch.abs(current_altitude - desired_altitude).mean()
        ground_hit_rate = (
            (current_altitude < float(base_env.cfg.min_height))  # type: ignore[attr-defined]
            .any(dim=1)
            .float()
            .mean()
        )
        airborne_threshold = (
            float(base_env.cfg.min_height) + self._airborne_margin  # type: ignore[attr-defined]
        )
        airborne_ratio = (current_altitude > airborne_threshold).float().mean()

        # Orientation metrics
        proj_grav_b = base_env.robot.data.projected_gravity_b.view(  # type: ignore[attr-defined]
            base_env.num_envs, base_env.cfg.num_agents, 3  # type: ignore[attr-defined]
        )
        mean_roll_deg, mean_pitch_deg, orientation_violation_rate = compute_orientation_metrics(
            proj_grav_b
        )

        # Altitude std deviation
        altitude_std, _ = compute_altitude_metrics(pos_w, base_env._desired_pos_w)  # type: ignore[attr-defined]

        self._episode_step_count += 1
        # Survival metric: only check env 0 (the eval env), not the full batch.
        # The batch-wide ground_hit_rate includes 4096 envs — any spawn-time dip
        # in any of 12,288 agents would trigger a false early survival cutoff.
        env0_ground_hit = (current_altitude[0] < float(base_env.cfg.min_height)).any()  # type: ignore[attr-defined]
        if env0_ground_hit and self._first_ground_hit_step is None:
            self._first_ground_hit_step = self._episode_step_count

        self._stats.update(
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
        )

    def on_episode_end(self, episode_num: int) -> None:
        """Finalize episode survival metric and reset per-episode counters."""
        if self._episode_step_count > 0:
            survival = float(
                self._first_ground_hit_step
                if self._first_ground_hit_step is not None
                else self._episode_step_count
            )
            self._stats.record_episode_survival(survival)
        self._episode_step_count = 0
        self._first_ground_hit_step = None

    def summarize(self) -> dict[str, float]:
        """Return mean Phase 2 metrics over all accumulated steps."""
        return self._stats.summarize()
