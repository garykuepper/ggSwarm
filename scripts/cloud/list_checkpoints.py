#!/usr/bin/env python3
"""List available checkpoints and their metadata."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="List available training checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python list_checkpoints.py
  python list_checkpoints.py --family hover
  python list_checkpoints.py --latest 20
        """,
    )
    parser.add_argument(
        "--family",
        choices=["marl", "hover"],
        default="marl",
        help="Training family (default: marl)",
    )
    parser.add_argument(
        "--latest",
        type=int,
        default=10,
        help="Show only latest N runs (default: 10)",
    )

    args = parser.parse_args()

    log_dir = Path(f"logs/skrl/ggswarm_{args.family}")

    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all checkpoint files
    checkpoints = []

    # Walk through run directories (sorted by modification time, newest first)
    run_dirs = sorted(
        [d for d in log_dir.iterdir() if d.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    for run_dir in run_dirs:
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.exists():
            for ckpt_file in sorted(checkpoints_dir.glob("*.pt")):
                is_best = ckpt_file.name == "best_agent.pt"
                size_mb = ckpt_file.stat().st_size / (1024 * 1024)
                checkpoints.append(
                    {
                        "path": f"{run_dir.name}/checkpoints/{ckpt_file.name}",
                        "name": ckpt_file.name,
                        "run_dir": run_dir.name,
                        "mtime": run_dir.stat().st_mtime,
                        "size_mb": size_mb,
                        "is_best": is_best,
                    }
                )

    if not checkpoints:
        print(f"No checkpoints found in {log_dir}")
        sys.exit(0)

    # Sort by date (newest first) and limit to latest N
    checkpoints = sorted(
        checkpoints,
        key=lambda x: x["mtime"],
        reverse=True,
    )[: args.latest]

    print(f"\nAvailable checkpoints (newest first):\n")

    for i, ckpt in enumerate(checkpoints, 1):
        best_flag = " [BEST]" if ckpt["is_best"] else ""
        print(
            f"[{i}] {ckpt['path']} ({ckpt['size_mb']:.1f} MB){best_flag}"
        )

    print()
    print("To use a checkpoint, run:")
    print(
        f"  python scripts/run.py phase2 play --checkpoint \"logs/skrl/ggswarm_{args.family}/<selected_checkpoint>\""
    )
    print()

    # Print the latest checkpoint path for easy use
    latest_path = log_dir / checkpoints[0]["path"]
    print("Latest checkpoint:")
    print(f"  {latest_path}")
    print()


if __name__ == "__main__":
    main()
