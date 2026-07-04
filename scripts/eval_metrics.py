"""Post-processing metrics script for ggSwarm evaluation.

Computes presentation-ready metrics from trajectory data and TB logs.
Run after play.py with --trajectories to analyze recorded data.

Usage:
    python scripts/eval_metrics.py --run_dir logs/skrl/ggswarm/p4/<run>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def load_trajectory_data(traj_dir: Path) -> dict:
    """Load trajectory tensors saved by play.py."""
    # Trajectory plots save a PNG but not raw tensors — we need to
    # re-derive metrics from the TB event file instead
    return {}


def compute_metrics_from_tb(run_dir: Path) -> dict:
    """Extract key metrics from TensorBoard event file."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(run_dir))
    ea.Reload()

    metrics = {}
    for tag in ea.Tags()["scalars"]:
        vals = ea.Scalars(tag)
        if len(vals) >= 2:
            values = [v.value for v in vals]
            metrics[tag] = {
                "start": values[0],
                "end": values[-1],
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
            }
    return metrics


def compute_formation_convergence(pos_data: list[torch.Tensor],
                                   goal_data: list[torch.Tensor],
                                   threshold: float = 0.3,
                                   dt: float = 0.02) -> float | None:
    """Time (seconds) until mean formation error < threshold.

    Args:
        pos_data: list of [A, 3] position tensors per step
        goal_data: list of [A, 3] goal tensors per step
        threshold: formation error threshold (m)
        dt: env step time (s)

    Returns:
        Convergence time in seconds, or None if never converged
    """
    for step, (pos, goal) in enumerate(zip(pos_data, goal_data)):
        errors = torch.linalg.norm(pos - goal, dim=1)  # [A]
        mean_error = errors.mean().item()
        if mean_error < threshold:
            return step * dt
    return None


def compute_knn_stats(pos_data: list[torch.Tensor], K: int = 2) -> dict:
    """Compute KNN distance statistics across all steps.

    Returns:
        dict with min_knn, mean_knn, max_knn, floor (5th percentile)
    """
    all_knn = []
    for pos in pos_data:
        A = pos.shape[0]
        if A < 2:
            continue
        # Pairwise distances
        dists = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)  # [A, A]
        dists.fill_diagonal_(float("inf"))
        knn_dists, _ = torch.topk(dists, min(K, A - 1), dim=1, largest=False)  # [A, K]
        all_knn.append(knn_dists.mean(dim=1))  # [A] mean per drone

    if not all_knn:
        return {}

    all_knn_flat = torch.cat(all_knn)
    return {
        "min_knn": all_knn_flat.min().item(),
        "max_knn": all_knn_flat.max().item(),
        "mean_knn": all_knn_flat.mean().item(),
        "floor_5pct": all_knn_flat.quantile(0.05).item(),
        "collision_count": (all_knn_flat < 0.10).sum().item(),
    }


def compute_velocity_jitter(vel_data: list[torch.Tensor]) -> dict:
    """Compute velocity jitter (std of linear velocity magnitude).

    Returns:
        dict with mean_vel, std_vel, max_vel
    """
    all_vel_mag = []
    for vel in vel_data:
        vel_mag = torch.linalg.norm(vel, dim=1)  # [A]
        all_vel_mag.append(vel_mag)

    if not all_vel_mag:
        return {}

    all_vel_flat = torch.cat(all_vel_mag)
    return {
        "mean_vel": all_vel_flat.mean().item(),
        "std_vel": all_vel_flat.std().item(),
        "max_vel": all_vel_flat.max().item(),
    }


def compute_energy_proxy(thrust_data: list[torch.Tensor], dt: float = 0.02) -> float:
    """Total thrust integral as energy proxy.

    Args:
        thrust_data: list of [A] thrust command tensors per step
        dt: env step time (s)

    Returns:
        Total thrust-seconds (energy proxy)
    """
    total = 0.0
    for thrust in thrust_data:
        total += thrust.abs().sum().item() * dt
    return total


def print_report(run_dir: Path, tb_metrics: dict):
    """Print a formatted metrics report."""
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION REPORT: {run_dir.name}")
    print(f"{'=' * 60}\n")

    # Training metrics from TB
    reward = tb_metrics.get("Reward / Total reward (mean)", {})
    ep_len = tb_metrics.get("Episode / Total timesteps (mean)", {})
    std = tb_metrics.get("Policy / Standard deviation", {})

    print("Training Metrics:")
    print(f"  Reward (mean):    start={reward.get('start', '?'):.1f}  end={reward.get('end', '?'):.1f}")
    print(f"  Episode length:   start={ep_len.get('start', '?'):.1f}  end={ep_len.get('end', '?'):.1f}")
    print(f"  Policy std:       start={std.get('start', '?'):.2f}  end={std.get('end', '?'):.2f}")

    # Collision metrics
    collisions = tb_metrics.get("Metrics/collision_pairs_per_step", {})
    if collisions:
        print(f"\nCollision Metrics:")
        print(f"  Collision pairs/step: mean={collisions.get('mean', '?'):.2f}  max={collisions.get('max', '?'):.0f}")

    # Formation metrics
    formation = tb_metrics.get("Metrics/mean_formation_error_m", {})
    if formation:
        print(f"\nFormation Metrics:")
        print(f"  Formation error (m): mean={formation.get('mean', '?'):.3f}  end={formation.get('end', '?'):.3f}")

    # Alive agents (DropoutGuard)
    alive = tb_metrics.get("Metrics/alive_agents_per_group", {})
    if alive:
        print(f"\nDropoutGuard Metrics:")
        print(f"  Alive agents/group:  mean={alive.get('mean', '?'):.1f}  min={alive.get('min', '?'):.1f}")

    print(f"\n{'=' * 60}")

    # Pass/fail criteria
    print("\nPass Criteria (M3 Gate):")
    checks = [
        ("Formation error < 0.3m", formation.get("end", 999) < 0.3 if formation else None, "O1"),
        ("Episode length >= 400", ep_len.get("end", 0) >= 400, "O1"),
        ("Zero collisions (end)", collisions.get("end", 1) == 0 if collisions else None, "O1"),
        ("Policy std < 0.3", std.get("end", 1) < 0.3, "Converged"),
    ]
    for name, passed, obj in checks:
        if passed is None:
            status = "N/A"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        print(f"  [{status:4s}] {name} ({obj})")

    print()


def main():
    parser = argparse.ArgumentParser(description="ggSwarm evaluation metrics")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to training run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: {run_dir} not found")
        return

    # Compute TB metrics
    tb_metrics = compute_metrics_from_tb(run_dir)

    # Print report
    print_report(run_dir, tb_metrics)


if __name__ == "__main__":
    main()
