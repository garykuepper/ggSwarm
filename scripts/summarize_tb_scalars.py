"""Print TensorBoard scalar first/last values for a training run directory.

Usage:
    python scripts/summarize_tb_scalars.py --run_dir logs/skrl/ggswarm_marl/<timestamp>_mappo_torch
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

KEYWORDS = (
    "reward",
    "rew_",
    "loss",
    "entropy",
    "policy",
    "value",
    "learning",
    "std",
    "clearance",
    "world_z",
    "low_clear",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory containing events.out.tfevents.* files.",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    print(f"{run_dir.name}: {len(tags)} scalar tag(s)\n")
    for t in sorted(tags):
        tl = t.lower()
        if not any(k in tl for k in KEYWORDS):
            continue
        series = ea.Scalars(t)
        if not series:
            continue
        a, b = series[0], series[-1]
        print(
            f"{t}\tsteps {a.step}->{b.step}\t{a.value:.6g} -> {b.value:.6g}\t(n={len(series)})"
        )


if __name__ == "__main__":
    main()
