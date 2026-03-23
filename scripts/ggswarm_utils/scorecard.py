"""Pass/warn/fail scorecard for post-training assessment.

Centralises the thresholds, verdict logic, scorecard printing, and markdown
report generation that were previously duplicated in post_train_assess.py.

Public API:
    assess_verdict(metric, value)      -> (verdict, threshold_hint)
    print_scorecard(run_name, metrics) -> overall_verdict str
    write_report(...)                  -> writes assess_report.md to run_dir
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Thresholds (single source of truth for the whole project)
# ---------------------------------------------------------------------------

# Metrics where HIGHER is better: (fail_below, warn_below, pass_at_or_above)
THRESHOLDS_HIGHER_BETTER: dict[str, tuple[float, float, float]] = {
    "survival_steps": (10.0, 200.0, 500.0),
    "airborne_ratio": (0.5, 0.85, 0.9),
}

# Metrics where LOWER is better: (pass_below, warn_below, fail_at_or_above)
THRESHOLDS_LOWER_BETTER: dict[str, tuple[float, float, float]] = {
    "ground_hit_rate": (0.05, 0.5, 0.5),
    "mean_roll_deg": (15.0, 60.0, 60.0),
    "orientation_violation_rate": (0.1, 0.5, 0.5),
    "mean_formation_error_m": (0.5, 1.5, 1.5),
}

# Ordered display sequence for the scorecard table
SCORECARD_METRICS: list[str] = [
    "survival_steps",
    "airborne_ratio",
    "ground_hit_rate",
    "mean_roll_deg",
    "orientation_violation_rate",
    "mean_formation_error_m",
]

# Decision guidance keyed on (verdict, metric).
# Phase 2A hover-stability uses a PD inner loop + 3-term Isaac-style reward
# (no upright/alive/terminated terms). Hints must not reference retired knobs.
DECISION_HINTS: dict[tuple[str, str], str] = {
    ("FAIL", "survival_steps"): (
        "Short survival before first batch ground hit — check TensorBoard "
        "(rew_low_clearance, low_clearance_frac, mean_world_z); tune "
        "rew_scale_low_clearance / rew_scale_pos / vel penalties or PD (kp_att, max_moment). "
        "Ignore pre-2026-03-22 survival numbers in run_history (collector bug)."
    ),
    ("FAIL", "airborne_ratio"): (
        "Low time above safe altitude — tune thrust/altitude: thrust_to_weight, "
        "PD kp_att/max_moment; strengthen rew_scale_pos or rebalance vel penalties "
        "in hover-stability cfg (not rew_scale_upright; it is disabled there)."
    ),
    ("FAIL", "mean_roll_deg"): (
        "Tilt still high — raise PD kp_att or max_moment (Rule 6 cfg); if roll is "
        "already moderate, prioritize altitude/position (rew_scale_pos) over adding "
        "upright terms unless you re-enable them explicitly in cfg."
    ),
    ("FAIL", "orientation_violation_rate"): (
        "Frequent large tilt — same lever as mean_roll_deg: PD gains first, then "
        "rew_scale_ang_vel in the 3-term hover reward if needed."
    ),
    ("FAIL", "ground_hit_rate"): (
        "Ground contact — verify spawn_z / min_height; increase hover thrust authority "
        "(thrust_to_weight, max_moment) and/or rew_scale_pos; PD should keep attitude "
        "recoverable once thrust supports altitude."
    ),
    ("FAIL", "mean_formation_error_m"): (
        "Phase 2A: formation reward is off — treat this as a rough position proxy, "
        "not a formation gate. For Phase 2B only: tighten rew_scale_formation / "
        "curriculum when stability metrics already pass."
    ),
}


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def assess_verdict(metric: str, value: float) -> tuple[str, str]:
    """Return (verdict, threshold_hint) for a single metric value.

    Args:
        metric: Metric key (must be in THRESHOLDS_HIGHER_BETTER or THRESHOLDS_LOWER_BETTER).
        value:  Observed value.

    Returns:
        Tuple of (verdict_str, hint_str) where verdict is 'PASS', 'WARN', 'FAIL', or 'UNKNOWN'.
    """
    if metric in THRESHOLDS_LOWER_BETTER:
        pass_t, warn_t, _ = THRESHOLDS_LOWER_BETTER[metric]
        if value < pass_t:
            return "PASS", f"< {pass_t}"
        if value < warn_t:
            return "WARN", f"< {warn_t}"
        return "FAIL", f"< {pass_t}"
    if metric in THRESHOLDS_HIGHER_BETTER:
        fail_t, warn_t, pass_t = THRESHOLDS_HIGHER_BETTER[metric]
        if value >= pass_t:
            return "PASS", f"> {pass_t}"
        if value >= warn_t:
            return "WARN", f"> {pass_t}"
        return "FAIL", f"> {pass_t}"
    return "UNKNOWN", "n/a"


# ---------------------------------------------------------------------------
# Scorecard printing
# ---------------------------------------------------------------------------


def print_scorecard(run_name: str, metrics: dict[str, float]) -> str:
    """Print PASS/WARN/FAIL scorecard to stdout.

    Args:
        run_name: Display name for the run (e.g. timestamp directory name).
        metrics:  Dict of metric_name -> float value.

    Returns:
        Overall verdict string: 'PASS', 'WARN', or 'FAIL'.
    """
    verdicts: list[str] = []
    lines: list[str] = []

    for key in SCORECARD_METRICS:
        if key not in metrics:
            lines.append(f"  {key:<35} {'n/a':>8}     (not reported)")
            continue
        val = metrics[key]
        verdict, hint = assess_verdict(key, val)
        verdicts.append(verdict)
        marker = "<" if verdict == "FAIL" else (" " if verdict == "PASS" else "~")
        lines.append(f"  {key:<35} {val:>8.3f}  {marker} {verdict:<4}  (threshold: {hint})")

    fail_count = verdicts.count("FAIL")
    warn_count = verdicts.count("WARN")
    pass_count = verdicts.count("PASS")
    overall = "FAIL" if fail_count > 0 else ("PASS" if pass_count == len(verdicts) else "WARN")

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  SCORECARD: {run_name}")
    print(sep)
    for line in lines:
        print(line)
    print(f"\n{sep}")
    print(f"  VERDICT: {overall}  ({fail_count} FAIL, {warn_count} WARN, {pass_count} PASS)")

    if overall == "FAIL":
        print("  ACTION REQUIRED: Do NOT retrain until reward config is adjusted.")
        for key in SCORECARD_METRICS:
            if key in metrics:
                v, _ = assess_verdict(key, metrics[key])
                if v == "FAIL" and (v, key) in DECISION_HINTS:
                    print(f"  [{key}] {DECISION_HINTS[(v, key)]}")
        print("  Log this scorecard to docs/status/changelog.md before any changes (Rule 20).")
    elif overall == "WARN":
        print("  Review WARN metrics before advancing to next phase.")
    else:
        print("  All metrics PASS -- ready to advance.")
    print(f"{sep}\n")

    return overall


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------


def format_tb_diagnostics(scalars: list[dict]) -> list[str]:
    """Format TB scalar summaries as markdown table lines for the assess report.

    Args:
        scalars: List of dicts from extract_scalar_summary() with keys:
                 tag, first_value, last_value, first_step, last_step.

    Returns:
        List of markdown lines (header + rows). Empty list if no scalars.
    """
    if not scalars:
        return []
    lines = [
        "## Training Diagnostics",
        "",
        "| Scalar | First | Last | Steps |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for s in scalars:
        first_k = f"{s['first_step'] // 1000}k"
        last_k = f"{s['last_step'] // 1000}k"
        lines.append(
            f"| `{s['tag']}` | {s['first_value']:.4f} | {s['last_value']:.4f} "
            f"| {first_k} -> {last_k} |"
        )
    return lines


def format_run_history_row(
    run_dir_name: str,
    metrics: dict[str, float],
    overall: str,
) -> str:
    """Format a ready-to-paste markdown table row for run_history.md.

    Args:
        run_dir_name: Run directory name (e.g. '2026-03-23_06-23-47_mappo_torch').
        metrics:      Dict of metric_name -> float value.
        overall:      Overall verdict string ('PASS'/'WARN'/'FAIL').

    Returns:
        Formatted markdown row string.
    """
    # Extract timestamp from dir name (first 19 chars = YYYY-MM-DD_HH-MM-SS)
    timestamp = run_dir_name[:19] if len(run_dir_name) >= 19 else run_dir_name
    surv = metrics.get("survival_steps", 0)
    airb = metrics.get("airborne_ratio", 0)
    ghit = metrics.get("ground_hit_rate", 0)
    roll = metrics.get("mean_roll_deg", 0)
    ovio = metrics.get("orientation_violation_rate", 0)
    return (
        f"| Run PD_ | {timestamp} | Phase 2A | {surv:.1f} | {airb:.3f} "
        f"| {ghit:.3f} | {roll:.1f}° | {ovio:.3f} | {overall} |"
    )


def write_report(
    run_dir: Path,
    metrics: dict[str, float],
    convergence: dict,
    overall: str,
    task: str,
    num_episodes: int,
    checkpoint_name: str,
    tb_diagnostics: list[dict] | None = None,
    trajectory_dir: Path | None = None,
) -> None:
    """Write assess_report.md into the run directory.

    Args:
        run_dir:         Path to the training run directory.
        metrics:         Flat dict of all eval metric values.
        convergence:     Dict returned by analyze_convergence().
        overall:         Overall scorecard verdict ('PASS'/'WARN'/'FAIL').
        task:            Gym task ID used for evaluation.
        num_episodes:    Number of evaluation episodes run.
        checkpoint_name: Filename of the evaluated checkpoint.
        tb_diagnostics:  List of dicts from extract_scalar_summary() (optional).
        trajectory_dir:  Path to trajectory plot directory (optional).
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    run_name = run_dir.name

    # Safe formatting helpers — convergence dict may be empty on failure
    peak_val = convergence.get("peak_reward_value", float("nan"))
    peak_step = convergence.get("peak_reward_step", 0)
    final_val = convergence.get("final_reward", float("nan"))
    final_step = convergence.get("final_step", 0)
    rec_budget = convergence.get("recommended_training_length", 0)
    collapse = convergence.get("entropy_collapse_step")
    collapse_str = f"{collapse:,}" if collapse else "Not detected"

    lines: list[str] = [
        f"# Assessment Report: {run_name}",
        "",
        f"Generated: {now}  ",
        f"Task: `{task}`  ",
        f"Checkpoint: `{checkpoint_name}`  ",
        f"Episodes: {num_episodes}",
        "",
        "## Convergence",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Peak reward | {peak_val:.2f} @ step {peak_step:,} |",
        f"| Final reward | {final_val:.2f} @ step {final_step:,} |",
        f"| Entropy collapse step | {collapse_str} |",
        f"| Recommended budget | {rec_budget:,} steps |",
    ]

    # TB scalar diagnostics (inserted between Convergence and Scorecard)
    tb_lines = format_tb_diagnostics(tb_diagnostics or [])
    if tb_lines:
        lines += ["", *tb_lines]

    lines += [
        "",
        "## Scorecard",
        "",
        "`survival_steps` = mean over eval episodes of steps until the first batch ground hit "
        "(`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a "
        "broken collector — see `docs/ops/post_train_analysis.md`.)",
        "",
        "| Metric | Value | Verdict | Threshold |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for key in SCORECARD_METRICS:
        if key not in metrics:
            lines.append(f"| `{key}` | n/a | n/a | n/a |")
            continue
        val = metrics[key]
        verdict, hint = assess_verdict(key, val)
        lines.append(f"| `{key}` | {val:.4f} | {verdict} | {hint} |")

    lines += [
        "",
        f"**Overall: {overall}**",
        "",
        "## All Metrics",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
    ]
    for k, v in sorted(metrics.items()):
        lines.append(f"| `{k}` | {v:.6f} |")

    # Trajectory plots section (if generated)
    if trajectory_dir is not None and trajectory_dir.exists():
        pngs = sorted(trajectory_dir.glob("*.png"))
        if pngs:
            rel = trajectory_dir.name  # e.g. "trajectories"
            lines += [
                "",
                "## Trajectory Plots",
                "",
            ]
            for png in pngs:
                lines.append(f"- `{rel}/{png.name}`")

    report_path = run_dir / "assess_report.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[REPORT] Written to: {report_path}")
