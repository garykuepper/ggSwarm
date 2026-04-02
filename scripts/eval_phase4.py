"""Phase 4 evaluation metrics from trajectory CSVs.

Computes:
  - Formation error (mean distance to goal slot, steady-state)
  - Velocity jitter (std of inter-step velocity)
  - Convergence time (steps until formation error < threshold)
  - Inter-agent min distance (collision proxy)
  - Per-agent goal tracking error over time

Usage:
  python scripts/eval_phase4.py <csv_path> [--steady_start 200] [--threshold 0.3]
  python scripts/eval_phase4.py --compare <csv_a> <csv_b>   # MINCO A/B
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def load_trajectory(csv_path: str) -> dict:
    """Load trajectory CSV into structured arrays.

    Returns dict with keys:
      - pos: [T, A, 3] drone positions
      - goal: [T, A, 3] goal positions (or None)
      - num_agents: int
      - num_steps: int
    """
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader]

    # Detect number of agents from header
    # Columns: step, d0_x, d0_y, d0_z, [d0_gx, d0_gy, d0_gz], d1_x, ...
    has_goals = "d0_gx" in header
    cols_per_agent = 6 if has_goals else 3
    num_agents = (len(header) - 1) // cols_per_agent

    T = len(rows)
    pos = np.zeros((T, num_agents, 3))
    goal = np.zeros((T, num_agents, 3)) if has_goals else None

    for t, row in enumerate(rows):
        for a in range(num_agents):
            base = 1 + a * cols_per_agent
            pos[t, a] = [float(row[base]), float(row[base + 1]), float(row[base + 2])]
            if has_goals:
                goal[t, a] = [float(row[base + 3]), float(row[base + 4]), float(row[base + 5])]

    return {
        "pos": pos,
        "goal": goal,
        "num_agents": num_agents,
        "num_steps": T,
    }


def compute_formation_error(pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Per-step mean formation error (distance to goal slot).

    Args:
        pos: [T, A, 3]
        goal: [T, A, 3]

    Returns:
        [T] array of mean formation errors.
    """
    # shape: [T, A]
    dist = np.linalg.norm(pos - goal, axis=2)
    return dist.mean(axis=1)  # shape: [T]


def compute_velocity_jitter(pos: np.ndarray, dt: float = 0.02) -> dict:
    """Compute velocity statistics from finite differences.

    Args:
        pos: [T, A, 3]
        dt: timestep duration

    Returns:
        dict with mean_speed, std_speed, mean_jitter (std of velocity magnitude per agent).
    """
    # shape: [T-1, A, 3]
    vel = np.diff(pos, axis=0) / dt
    # shape: [T-1, A]
    speed = np.linalg.norm(vel, axis=2)

    # Per-agent jitter = std of speed over time
    per_agent_jitter = speed.std(axis=0)  # shape: [A]

    return {
        "mean_speed": float(speed.mean()),
        "std_speed": float(speed.std()),
        "mean_jitter": float(per_agent_jitter.mean()),
        "per_agent_jitter": per_agent_jitter,
    }


def compute_convergence_time(
    formation_error: np.ndarray, threshold: float = 0.3, dt: float = 0.02
) -> float | None:
    """Steps until formation error stays below threshold for 50 consecutive steps.

    Returns time in seconds, or None if never converged.
    """
    window = 50
    below = formation_error < threshold
    for t in range(len(below) - window):
        if below[t : t + window].all():
            return t * dt
    return None


def compute_min_inter_agent_distance(pos: np.ndarray) -> np.ndarray:
    """Per-step minimum pairwise distance between agents.

    Args:
        pos: [T, A, 3]

    Returns:
        [T] array of min pairwise distances.
    """
    T, A, _ = pos.shape
    min_dists = np.full(T, np.inf)
    for i in range(A):
        for j in range(i + 1, A):
            d = np.linalg.norm(pos[:, i] - pos[:, j], axis=1)  # shape: [T]
            min_dists = np.minimum(min_dists, d)
    return min_dists


def analyze(csv_path: str, steady_start: int = 200, threshold: float = 0.3) -> dict:
    """Full analysis of a single trajectory CSV."""
    data = load_trajectory(csv_path)
    pos = data["pos"]
    goal = data["goal"]
    A = data["num_agents"]
    T = data["num_steps"]

    results = {"file": csv_path, "num_agents": A, "num_steps": T}

    # Formation error
    if goal is not None:
        fe = compute_formation_error(pos, goal)
        results["formation_error_mean"] = float(fe.mean())
        results["formation_error_steady"] = float(fe[steady_start:].mean()) if T > steady_start else float(fe.mean())
        results["formation_error_final_50"] = float(fe[-50:].mean()) if T >= 50 else float(fe.mean())
        results["convergence_time_s"] = compute_convergence_time(fe, threshold)
    else:
        results["formation_error_mean"] = None

    # Velocity jitter
    vel_stats = compute_velocity_jitter(pos)
    results["mean_speed_m_s"] = vel_stats["mean_speed"]
    results["velocity_jitter"] = vel_stats["mean_jitter"]

    # Steady-state velocity jitter (after convergence)
    if T > steady_start:
        ss_vel = compute_velocity_jitter(pos[steady_start:])
        results["steady_velocity_jitter"] = ss_vel["mean_jitter"]
        results["steady_mean_speed"] = ss_vel["mean_speed"]
    else:
        results["steady_velocity_jitter"] = vel_stats["mean_jitter"]
        results["steady_mean_speed"] = vel_stats["mean_speed"]

    # Inter-agent distances
    min_dists = compute_min_inter_agent_distance(pos)
    results["min_inter_agent_dist"] = float(min_dists.min())
    results["mean_min_inter_agent_dist"] = float(min_dists.mean())
    results["collision_steps"] = int((min_dists < 0.10).sum())  # collision radius

    return results


def print_results(results: dict):
    """Pretty-print analysis results."""
    print(f"\n{'=' * 60}")
    print(f"  Phase 4 Eval: {Path(results['file']).name}")
    print(f"{'=' * 60}")
    print(f"  Agents: {results['num_agents']}  |  Steps: {results['num_steps']}")
    print(f"{'-' * 60}")

    if results.get("formation_error_mean") is not None:
        print(f"  Formation error (full):       {results['formation_error_mean']:.4f} m")
        print(f"  Formation error (steady):     {results['formation_error_steady']:.4f} m")
        print(f"  Formation error (final 50):   {results['formation_error_final_50']:.4f} m")
        conv = results["convergence_time_s"]
        print(f"  Convergence time:             {conv:.2f} s" if conv else "  Convergence time:             DID NOT CONVERGE")

    print(f"{'-' * 60}")
    print(f"  Mean speed:                   {results['mean_speed_m_s']:.4f} m/s")
    print(f"  Velocity jitter (full):       {results['velocity_jitter']:.4f} m/s")
    print(f"  Velocity jitter (steady):     {results['steady_velocity_jitter']:.4f} m/s")
    print(f"  Steady mean speed:            {results['steady_mean_speed']:.4f} m/s")
    print(f"{'-' * 60}")
    print(f"  Min inter-agent distance:     {results['min_inter_agent_dist']:.4f} m")
    print(f"  Mean min inter-agent dist:    {results['mean_min_inter_agent_dist']:.4f} m")
    print(f"  Collision steps (< 0.10m):    {results['collision_steps']}")
    print(f"{'=' * 60}\n")


def compare_minco(csv_with: str, csv_without: str, steady_start: int = 200):
    """MINCO A/B comparison."""
    r_on = analyze(csv_with, steady_start)
    r_off = analyze(csv_without, steady_start)

    print(f"\n{'=' * 60}")
    print(f"  MINCO A/B Comparison (O2 Validation)")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<30} {'MINCO ON':>12} {'MINCO OFF':>12} {'Delta%':>8}")
    print(f"  {'-' * 62}")

    for label, key in [
        ("Velocity jitter (full)", "velocity_jitter"),
        ("Velocity jitter (steady)", "steady_velocity_jitter"),
        ("Steady mean speed", "steady_mean_speed"),
        ("Formation error (steady)", "formation_error_steady"),
    ]:
        v_on = r_on.get(key)
        v_off = r_off.get(key)
        if v_on is not None and v_off is not None and v_off > 0:
            pct = (v_off - v_on) / v_off * 100
            print(f"  {label:<30} {v_on:>12.4f} {v_off:>12.4f} {pct:>+7.1f}%")
        elif v_on is not None and v_off is not None:
            print(f"  {label:<30} {v_on:>12.4f} {v_off:>12.4f}     {'N/A':>4}")

    jitter_on = r_on["steady_velocity_jitter"]
    jitter_off = r_off["steady_velocity_jitter"]
    if jitter_off > 0:
        reduction = (jitter_off - jitter_on) / jitter_off * 100
        print(f"\n  O2 target: >= 20% jitter reduction")
        print(f"  Measured:  {reduction:.1f}% reduction")
        print(f"  Verdict:   {'PASS' if reduction >= 20 else 'FAIL'}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 trajectory evaluation")
    parser.add_argument("csv", nargs="?", help="Trajectory CSV file")
    parser.add_argument("--compare", nargs=2, metavar=("MINCO_ON", "MINCO_OFF"),
                        help="Compare two CSVs for MINCO A/B validation")
    parser.add_argument("--steady_start", type=int, default=200,
                        help="Step index where steady-state begins (default: 200)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Formation error convergence threshold in meters (default: 0.3)")
    parser.add_argument("--batch", nargs="+", metavar="CSV",
                        help="Analyze multiple CSVs and print summary table")
    args = parser.parse_args()

    if args.compare:
        compare_minco(args.compare[0], args.compare[1], args.steady_start)
    elif args.batch:
        print(f"\n{'=' * 90}")
        print(f"  Phase 4 Scale Testing Summary")
        print(f"{'=' * 90}")
        print(f"  {'File':<35} {'A':>3} {'FE_ss':>7} {'Conv_s':>7} {'Jitter':>8} {'MinDist':>8} {'Collis':>6}")
        print(f"  {'-' * 85}")
        for csv_path in args.batch:
            r = analyze(csv_path, args.steady_start, args.threshold)
            name = Path(csv_path).name[:34]
            fe = r.get("formation_error_steady")
            conv = r.get("convergence_time_s")
            print(f"  {name:<35} {r['num_agents']:>3} "
                  f"{fe:>7.4f} " if fe else f"  {name:<35} {r['num_agents']:>3}     N/A ",
                  end="")
            print(f"{conv:>6.2f}s " if conv else "    N/C ",
                  end="")
            print(f"{r['steady_velocity_jitter']:>8.4f} "
                  f"{r['min_inter_agent_dist']:>8.4f} "
                  f"{r['collision_steps']:>6}")
        print(f"{'=' * 90}\n")
    elif args.csv:
        results = analyze(args.csv, args.steady_start, args.threshold)
        print_results(results)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
