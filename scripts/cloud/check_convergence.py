"""Analyze training convergence from TFEvents logs."""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def analyze_convergence(log_dir: str) -> dict:
    """
    Analyze convergence of a training run from TFEvents files.

    The primary convergence signal is entropy collapse: when the policy's standard deviation
    stops changing, it has stopped exploring and is fully converged.

    Args:
        log_dir: Path to the log directory containing events.out.tfevents files.

    Returns:
        Dictionary with convergence metrics:
        - peak_reward_step: Step at which reward peaked
        - peak_reward_value: Maximum reward value
        - entropy_collapse_step: Step where policy entropy locked (std dev stopped changing)
        - plateau_reward_value: Reward value at entropy collapse
        - final_reward: Final reward value
        - final_step: Total steps logged
        - recommended_training_length: Suggested max iterations (entropy_collapse_step * 1.15)
    """
    ea = EventAccumulator(log_dir)
    ea.Reload()

    # Get reward and policy std dev curves
    reward_events = ea.Scalars("Reward / Total reward (mean)")
    std_events = ea.Scalars("Policy / Standard deviation (drone_0)")

    if not reward_events:
        raise ValueError(f"No reward scalars found in {log_dir}")

    # Extract steps and values
    reward_steps = [e.step for e in reward_events]
    reward_values = [e.value for e in reward_events]
    std_steps = [e.step for e in std_events]
    std_values = [e.value for e in std_events]

    # Find peak reward
    peak_idx = reward_values.index(max(reward_values))
    peak_reward_step = reward_steps[peak_idx]
    peak_reward_value = reward_values[peak_idx]

    # Detect entropy collapse: where policy std dev stops changing
    # This is the real convergence signal — when the policy stops exploring.
    entropy_collapse_step = None
    if len(std_values) > 2:
        for i in range(1, len(std_values)):
            if abs(std_values[i] - std_values[i - 1]) < 1e-6:
                entropy_collapse_step = std_steps[i]
                break

    # Find reward at entropy collapse step
    plateau_reward_value = None
    if entropy_collapse_step is not None:
        plateau_step = entropy_collapse_step
        for j, step in enumerate(reward_steps):
            if step >= entropy_collapse_step:
                plateau_reward_value = reward_values[j]
                break
    else:
        # Fallback: use final step
        plateau_step = reward_steps[-1]
        plateau_reward_value = reward_values[-1]

    # Recommended training length: entropy collapse step + 15% buffer
    recommended_training_length = int(plateau_step * 1.15)

    return {
        "peak_reward_step": peak_reward_step,
        "peak_reward_value": peak_reward_value,
        "entropy_collapse_step": entropy_collapse_step,
        "plateau_step": plateau_step,
        "plateau_reward_value": plateau_reward_value,
        "final_reward": reward_values[-1],
        "final_step": reward_steps[-1],
        "recommended_training_length": recommended_training_length,
        "total_events": len(reward_events),
    }


# Tags to extract for the assess report TB diagnostics section.
TB_DIAGNOSTIC_TAGS: list[str] = [
    "Policy / Standard deviation (drone_0)",
    "Info / mean_world_z",
    "Info / rew_pos",
    "Info / rew_ang_vel",
    "Info / rew_low_clearance",
    "Info / rew_terminated",
]


def extract_scalar_summary(
    log_dir: str,
    tags: list[str] | None = None,
) -> list[dict]:
    """Extract first/last scalar values for requested TensorBoard tags.

    Args:
        log_dir: Path to the log directory containing events.out.tfevents files.
        tags: List of scalar tag names. Defaults to TB_DIAGNOSTIC_TAGS.

    Returns:
        List of dicts with keys: tag, first_value, last_value, first_step, last_step.
        Tags not found in the event file are silently skipped.
    """
    if tags is None:
        tags = TB_DIAGNOSTIC_TAGS

    ea = EventAccumulator(log_dir)
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    results = []
    for tag in tags:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        if len(events) < 2:
            continue
        results.append({
            "tag": tag,
            "first_value": events[0].value,
            "last_value": events[-1].value,
            "first_step": events[0].step,
            "last_step": events[-1].step,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze training convergence from TFEvents logs."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (containing events.out.tfevents files).",
    )

    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return

    print(f"\n{'='*70}")
    print(f"Convergence Analysis: {run_dir.name}")
    print(f"{'='*70}\n")

    results = analyze_convergence(str(run_dir))

    print(f"Peak Reward:                {results['peak_reward_value']:>10.2f}  @ step {results['peak_reward_step']:>9,}")

    if results["entropy_collapse_step"] is not None:
        print(
            f"Entropy Collapse Step:      {results['entropy_collapse_step']:>10,}  "
            f"(policy stopped exploring, reward={results['plateau_reward_value']:.2f})"
        )
    else:
        print(f"Entropy Collapse Step:      {'Not detected':>10}")

    print(f"Final Reward:               {results['final_reward']:>10.2f}  @ step {results['final_step']:>9,}")

    print(f"\nRecommended Training Length: {results['recommended_training_length']:>10,}  steps")
    print(
        f"  (convergence @ {results['plateau_step']:,} steps, plus 15% buffer)\n"
        f"  Current run used:          {results['final_step']:>10,}  steps\n"
        f"  Wasted compute:            {max(0, results['final_step'] - results['recommended_training_length']):>10,}  steps\n"
    )

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
