"""Eval metric helpers shared across all phase evaluation scripts.

Contains:
    EvalStats                        -- running-average accumulator for Phase 2/2A/2B metrics
    pairwise_mean_abs_spacing_error  -- formation error from position tensor
    compute_orientation_metrics      -- roll, pitch, violation rate from gravity projection
    compute_altitude_metrics         -- altitude std dev and mean altitude
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EvalStats:
    """Running averages for Phase 2 / hover-stability evaluation metrics.

    Call update() once per simulation step for per-step metrics.
    Call record_episode_survival() once per completed eval episode for
    ``survival_steps`` (mean time until first batch ground hit or full horizon).
    """

    formation_error_sum: float = 0.0
    formation_error_count: int = 0
    separation_event_sum: float = 0.0
    separation_event_count: int = 0
    mean_speed_sum: float = 0.0
    mean_speed_count: int = 0
    altitude_error_sum: float = 0.0
    altitude_error_count: int = 0
    ground_hit_rate_sum: float = 0.0
    ground_hit_rate_count: int = 0
    airborne_ratio_sum: float = 0.0
    airborne_ratio_count: int = 0
    mean_roll_deg_sum: float = 0.0
    mean_roll_deg_count: int = 0
    mean_pitch_deg_sum: float = 0.0
    mean_pitch_deg_count: int = 0
    orientation_violation_rate_sum: float = 0.0
    orientation_violation_rate_count: int = 0
    altitude_std_sum: float = 0.0
    altitude_std_count: int = 0
    episode_survival_steps_sum: float = 0.0
    episode_survival_steps_count: int = 0

    def update(
        self,
        *,
        formation_error: torch.Tensor,
        separation_event_rate: torch.Tensor,
        mean_speed: torch.Tensor,
        mean_altitude_error: torch.Tensor,
        ground_hit_rate: torch.Tensor,
        airborne_ratio: torch.Tensor,
        mean_roll_deg: torch.Tensor,
        mean_pitch_deg: torch.Tensor,
        orientation_violation_rate: torch.Tensor,
        altitude_std: torch.Tensor,
        episode_survival_steps: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one step of per-step metrics.

        Args:
            episode_survival_steps: Deprecated path; prefer record_episode_survival().
                When omitted, survival aggregates are unchanged (correct behaviour).
        """
        self.formation_error_sum += float(formation_error.item())
        self.formation_error_count += 1
        self.separation_event_sum += float(separation_event_rate.item())
        self.separation_event_count += 1
        self.mean_speed_sum += float(mean_speed.item())
        self.mean_speed_count += 1
        self.altitude_error_sum += float(mean_altitude_error.item())
        self.altitude_error_count += 1
        self.ground_hit_rate_sum += float(ground_hit_rate.item())
        self.ground_hit_rate_count += 1
        self.airborne_ratio_sum += float(airborne_ratio.item())
        self.airborne_ratio_count += 1
        self.mean_roll_deg_sum += float(mean_roll_deg.item())
        self.mean_roll_deg_count += 1
        self.mean_pitch_deg_sum += float(mean_pitch_deg.item())
        self.mean_pitch_deg_count += 1
        self.orientation_violation_rate_sum += float(orientation_violation_rate.item())
        self.orientation_violation_rate_count += 1
        self.altitude_std_sum += float(altitude_std.item())
        self.altitude_std_count += 1
        if episode_survival_steps is not None:
            self.episode_survival_steps_sum += float(episode_survival_steps.item())
            self.episode_survival_steps_count += 1

    def record_episode_survival(self, steps: float) -> None:
        """Record steps-until-first-ground-hit (or full horizon) for one eval episode."""
        self.episode_survival_steps_sum += float(steps)
        self.episode_survival_steps_count += 1

    def summarize(self) -> dict[str, float]:
        """Return mean of all accumulated metrics."""
        return {
            "mean_formation_error_m": self.formation_error_sum / max(1, self.formation_error_count),
            "separation_event_rate": self.separation_event_sum / max(1, self.separation_event_count),
            "mean_speed_mps": self.mean_speed_sum / max(1, self.mean_speed_count),
            "mean_altitude_error_m": self.altitude_error_sum / max(1, self.altitude_error_count),
            "ground_hit_rate": self.ground_hit_rate_sum / max(1, self.ground_hit_rate_count),
            "airborne_ratio": self.airborne_ratio_sum / max(1, self.airborne_ratio_count),
            "mean_roll_deg": self.mean_roll_deg_sum / max(1, self.mean_roll_deg_count),
            "mean_pitch_deg": self.mean_pitch_deg_sum / max(1, self.mean_pitch_deg_count),
            "orientation_violation_rate": (
                self.orientation_violation_rate_sum / max(1, self.orientation_violation_rate_count)
            ),
            "altitude_std_m": self.altitude_std_sum / max(1, self.altitude_std_count),
            "survival_steps": (
                self.episode_survival_steps_sum / max(1, self.episode_survival_steps_count)
            ),
        }


def pairwise_mean_abs_spacing_error(
    pos_w: torch.Tensor, target_dist: float
) -> torch.Tensor:
    """Compute mean |d_ij - target| over i<j pairs, averaged over envs.

    Args:
        pos_w: Agent world positions, shape [num_envs, num_agents, 3].
        target_dist: Desired inter-agent spacing in metres.

    Returns:
        Scalar tensor: mean formation error (metres).
    """
    # shape: [num_envs, num_agents, num_agents]
    diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)
    num_agents = dist.shape[-1]
    mask = torch.triu(
        torch.ones((num_agents, num_agents), device=dist.device, dtype=torch.bool),
        diagonal=1,
    )
    per_env = torch.abs(dist[:, mask] - target_dist).mean(dim=1)
    return per_env.mean()


def compute_orientation_metrics(
    proj_grav_b: torch.Tensor, orientation_threshold_deg: float = 45.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute roll, pitch, and orientation violation rate from projected gravity.

    Args:
        proj_grav_b: Projected gravity in body frame, shape [num_envs, num_agents, 3].
                     When level, z-component is -1.0.
        orientation_threshold_deg: Threshold in degrees for violation detection.

    Returns:
        (mean_roll_deg, mean_pitch_deg, violation_rate) — all scalar tensors.
    """
    grav_x = proj_grav_b[:, :, 0]  # [num_envs, num_agents]
    grav_y = proj_grav_b[:, :, 1]
    grav_z = proj_grav_b[:, :, 2]

    # atan2(y, -z) gives roll; atan2(x, -z) gives pitch
    roll_rad = torch.atan2(grav_y, -grav_z)
    pitch_rad = torch.atan2(grav_x, -grav_z)

    mean_roll = torch.abs(roll_rad * 180.0 / torch.pi).mean()
    mean_pitch = torch.abs(pitch_rad * 180.0 / torch.pi).mean()

    threshold_rad = orientation_threshold_deg * torch.pi / 180.0
    violations = (torch.abs(roll_rad) > threshold_rad) | (torch.abs(pitch_rad) > threshold_rad)
    return mean_roll, mean_pitch, violations.float().mean()


def compute_altitude_metrics(
    pos_w: torch.Tensor, desired_pos_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute altitude standard deviation and mean altitude.

    Args:
        pos_w: Current positions, shape [num_envs, num_agents, 3].
        desired_pos_w: Desired positions, shape [num_envs, num_agents, 3].

    Returns:
        (altitude_std, mean_altitude) — both scalar tensors.
    """
    altitude = pos_w[:, :, 2]  # [num_envs, num_agents]
    return altitude.std(), altitude.mean()
