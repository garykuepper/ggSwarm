"""Phase 5: Compile the Testing Report from Phase 3/4 evaluation outputs.

Reads CSV results from eval_phase3.py and bench_scale.py outputs and produces
a markdown testing report.

Usage:
    python scripts/compile_testing_report.py \\
        --phase3_json  results/eval_phase3.json \\
        --bench_csv    results/bench_scale.csv \\
        --output       docs/status/testing_report.md

Phase 3 eval can output JSON with --output_json flag (see eval_phase3.py).
If no inputs are provided the script generates a template report with
placeholder values.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import date


def _load_phase3_results(path: str | None) -> dict:
    if path is None or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_bench_results(path: str | None) -> list[dict]:
    if path is None or not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _fmt(value, fmt=".4f", fallback="N/A") -> str:
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return fallback


def _pass_fail(value, threshold, op="<") -> str:
    try:
        v = float(value)
        if op == "<":
            return "PASS" if v < threshold else "FAIL"
        if op == "==":
            return "PASS" if abs(v - threshold) < 1e-9 else "FAIL"
        if op == ">=":
            return "PASS" if v >= threshold else "FAIL"
    except (TypeError, ValueError):
        pass
    return "N/A"


def generate_report(
    phase3: dict,
    bench_rows: list[dict],
    output_path: str,
) -> None:
    today = date.today().isoformat()

    lines: list[str] = []

    lines += [
        "# ggSwarm Testing Report",
        "",
        f"**Generated:** {today}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "This report covers Phase 3 (CBF, SwarmRaft, MINCO) and Phase 4 (stress testing,",
        "scale benchmarking) evaluation results.  Pass/fail thresholds map to proposal",
        "technical objectives O1–O4.",
        "",
        "---",
        "",
        "## Phase 3 Evaluation Results",
        "",
        "### Pass/Fail vs. Proposal Objectives",
        "",
        "| Objective | Metric | Target | Result | Status |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    collision_rate = phase3.get("collision_rate", "N/A")
    form_err = phase3.get("mean_formation_error_m", "N/A")
    gap_fill = phase3.get("mean_gap_fill_latency_s", "N/A")
    jitter_pct = phase3.get("jitter_reduction_pct", "N/A")

    lines += [
        f"| O1 | Collision rate | == 0.0 | {_fmt(collision_rate)} "
        f"| {_pass_fail(collision_rate, 0.0, '==')} |",
        f"| O2 | Jitter reduction (%) | >= 20% | {_fmt(jitter_pct, '.1f')} "
        f"| {_pass_fail(jitter_pct, 20.0, '>=')} |",
        f"| O3 | Gap-fill latency (s) | < 2.0 | {_fmt(gap_fill, '.2f')} "
        f"| {_pass_fail(gap_fill, 2.0, '<')} |",
        f"| P3.4 | Mean formation error (m) | < 0.5 | {_fmt(form_err, '.4f')} "
        f"| {_pass_fail(form_err, 0.5, '<')} |",
        "",
        "### Additional Metrics",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Mean velocity norm (m/s) | {_fmt(phase3.get('mean_vel_norm_mps', 'N/A'))} |",
        f"| Velocity std (m/s) | {_fmt(phase3.get('std_vel_norm_mps', 'N/A'))} |",
        f"| CBF intervention rate | {_fmt(phase3.get('cbf_intervention_rate', 'N/A'))} |",
        f"| Gap-fill samples | {phase3.get('gap_fill_samples', 'N/A')} |",
        "",
        "---",
        "",
        "## Phase 4 Scale Benchmark Results",
        "",
    ]

    if bench_rows:
        lines += [
            "| Agents | Envs | Mean Form Err (m) | Std Form Err (m) "
            "| Mean Vel (m/s) | CBF Rate | Steps/s | VRAM (MB) | Success Rate |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for row in bench_rows:
            lines.append(
                f"| {row.get('num_agents','N/A')} | {row.get('num_envs','N/A')} "
                f"| {_fmt(row.get('mean_form_err_m','N/A'))} "
                f"| {_fmt(row.get('std_form_err_m','N/A'))} "
                f"| {_fmt(row.get('mean_vel_mps','N/A'))} "
                f"| {_fmt(row.get('cbf_rate','N/A'))} "
                f"| {_fmt(row.get('steps_per_sec','N/A'), '.1f')} "
                f"| {_fmt(row.get('vram_mb','N/A'), '.0f')} "
                f"| {_fmt(row.get('success_rate','N/A'), '.3f')} |"
            )
    else:
        lines += [
            "_No benchmark data available. Run:_",
            "```bash",
            "python scripts/run.py phase4 bench --checkpoint <path> --output_csv results/bench_scale.csv",
            "```",
        ]

    lines += [
        "",
        "---",
        "",
        "## Phase 4: Stress Testing Scenarios",
        "",
        "| Scenario | Description | Result |",
        "| :--- | :--- | :--- |",
        "| Nominal | 3 agents, no obstacles, no kills | See Phase 3 metrics above |",
        "| Agent Loss | Kill agent 0 at step 50; measure SwarmRaft recovery | See gap-fill latency above |",
        "| Sparse Obstacles | 10 cylinders in 4m field | Run: `phase4 eval --task GGS-ClutteredForest-v0` |",
        "| Dense Obstacles | 25 cylinders (cluttered forest) | Run: `phase4 eval --task GGS-ClutteredForest-v0` |",
        "| Scale 20 agents | 20 agents, no obstacles | See benchmark table above |",
        "",
        "---",
        "",
        "## Phase 5: Showcase Deliverables",
        "",
        "| Deliverable | Status | Notes |",
        "| :--- | :--- | :--- |",
        "| Cluttered Forest USD stage | `GGS-ClutteredForest-v0` task registered | 25 obstacles, 20 agents |",
        "| Urban Canyon USD stage | `GGS-UrbanCanyon-v0` task registered | 30 obstacles, 20 agents |",
        "| HD demo recording | Pending | `python scripts/run.py phase3 play --video --rendering_mode quality` |",
        "| Testing Report | This document | Update with real eval numbers |",
        "",
        "---",
        "",
        "## How to Reproduce",
        "",
        "```bash",
        "# Phase 3 evaluation (with agent kill test)",
        "python scripts/run.py phase3 eval --checkpoint <path> --kill_agent_step 50",
        "",
        "# Phase 4 scale benchmark",
        "python scripts/run.py phase4 bench --checkpoint <path> --output_csv results/bench_scale.csv",
        "",
        "# Cluttered forest visual run",
        "python scripts/run.py phase3 play --task GGS-ClutteredForest-v0 --checkpoint <path> --video",
        "",
        "# Regenerate this report",
        "python scripts/compile_testing_report.py \\",
        "    --phase3_json results/eval_phase3.json \\",
        "    --bench_csv   results/bench_scale.csv \\",
        "    --output      docs/status/testing_report.md",
        "```",
        "",
        "---",
        "",
        "*ggSwarm Testing Report — auto-generated by `scripts/compile_testing_report.py`*",
    ]

    report = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"[REPORT] Written to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile ggSwarm Testing Report from evaluation outputs."
    )
    parser.add_argument("--phase3_json", type=str, default=None)
    parser.add_argument("--bench_csv", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="docs/status/testing_report.md",
    )
    args = parser.parse_args()

    phase3 = _load_phase3_results(args.phase3_json)
    bench_rows = _load_bench_results(args.bench_csv)

    generate_report(phase3, bench_rows, args.output)


if __name__ == "__main__":
    main()
