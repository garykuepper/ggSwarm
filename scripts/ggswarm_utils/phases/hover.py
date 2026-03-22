"""Hover-phase metric collector (GGS-Hover-v0).

Implements the PhaseCollector protocol from eval_runner.
Collects airborne ratio, altitude error, position error, speed, and ground hits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HoverEvalStats:
    """Running accumulators for hover evaluation metrics."""

    airborne_ratio_sum: float = 0.0
    airborne_ratio_count: int = 0
    altitude_error_sum: float = 0.0
    altitude_error_count: int = 0
    position_error_sum: float = 0.0
    position_error_count: int = 0
    mean_speed_sum: float = 0.0
    mean_speed_count: int = 0
    mean_ang_speed_sum: float = 0.0
    mean_ang_speed_count: int = 0
    ground_hit_rate_sum: float = 0.0
    ground_hit_rate_count: int = 0

    def update(
        self,
        *,
        airborne_ratio: torch.Tensor,
        altitude_error: torch.Tensor,
        position_error: torch.Tensor,
        mean_speed: torch.Tensor,
        mean_ang_speed: torch.Tensor,
        ground_hit_rate: torch.Tensor,
    ) -> None:
        """Accumulate one step's worth of metrics."""
        self.airborne_ratio_sum += float(airborne_ratio.item())
        self.airborne_ratio_count += 1
        self.altitude_error_sum += float(altitude_error.item())
        self.altitude_error_count += 1
        self.position_error_sum += float(position_error.item())
        self.position_error_count += 1
        self.mean_speed_sum += float(mean_speed.item())
        self.mean_speed_count += 1
        self.mean_ang_speed_sum += float(mean_ang_speed.item())
        self.mean_ang_speed_count += 1
        self.ground_hit_rate_sum += float(ground_hit_rate.item())
        self.ground_hit_rate_count += 1

    def summarize(self) -> dict[str, float]:
        """Return mean metrics over all accumulated steps."""
        return {
            "airborne_ratio": self.airborne_ratio_sum / max(1, self.airborne_ratio_count),
            "mean_altitude_error_m": self.altitude_error_sum / max(1, self.altitude_error_count),
            "mean_position_error_m": self.position_error_sum / max(1, self.position_error_count),
            "mean_speed_mps": self.mean_speed_sum / max(1, self.mean_speed_count),
            "mean_ang_speed_rps": self.mean_ang_speed_sum / max(1, self.mean_ang_speed_count),
            "ground_hit_rate": self.ground_hit_rate_sum / max(1, self.ground_hit_rate_count),
        }


class HoverCollector:
    """PhaseCollector for the GGS-Hover-v0 task.

    Measures single-drone hover stability: altitude accuracy, airborne time,
    speed, and ground contact rate.
    """

    def __init__(self) -> None:
        self._stats = HoverEvalStats()

    def on_step(
        self,
        *,
        base_env: object,
        obs: object,
        step: int,
        episode_step: int,
        episode_num: int,
    ) -> None:
        """Extract hover metrics from base_env tensors each step."""
        # shape: [num_envs, num_agents, 3]
        pos_w = base_env.robot.data.root_pos_w.view(  # type: ignore[attr-defined]
            base_env.num_envs, base_env.cfg.num_agents, 3  # type: ignore[attr-defined]
        )
        # shape: [num_envs, num_agents, 3]
        lin_vel_b = base_env.robot.data.root_lin_vel_b.view(  # type: ignore[attr-defined]
            base_env.num_envs, base_env.cfg.num_agents, 3  # type: ignore[attr-defined]
        )
        # shape: [num_envs, num_agents, 3]
        ang_vel_b = base_env.robot.data.root_ang_vel_b.view(  # type: ignore[attr-defined]
            base_env.num_envs, base_env.cfg.num_agents, 3  # type: ignore[attr-defined]
        )
        desired_pos_w = base_env._desired_pos_w  # type: ignore[attr-defined]

        # shape: [num_envs, num_agents]
        z = pos_w[:, :, 2]
        airborne_ratio = (
            z >= float(base_env.cfg.hover_reward_min_height)  # type: ignore[attr-defined]
        ).float().mean()
        altitude_error = torch.abs(desired_pos_w[:, :, 2] - z).mean()
        position_error = torch.norm(desired_pos_w - pos_w, dim=-1).mean()
        mean_speed = torch.norm(lin_vel_b, dim=-1).mean()
        mean_ang_speed = torch.norm(ang_vel_b, dim=-1).mean()
        ground_hit_rate = (
            z < float(base_env.cfg.ground_hit_height)  # type: ignore[attr-defined]
        ).float().mean()

        self._stats.update(
            airborne_ratio=airborne_ratio,
            altitude_error=altitude_error,
            position_error=position_error,
            mean_speed=mean_speed,
            mean_ang_speed=mean_ang_speed,
            ground_hit_rate=ground_hit_rate,
        )

    def on_episode_end(self, episode_num: int) -> None:
        """No per-episode state to reset for the hover task."""

    def summarize(self) -> dict[str, float]:
        """Return mean hover metrics over all steps."""
        return self._stats.summarize()
