#!/usr/bin/env python3
"""Pull ggSwarm training results from GCS to local machine."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Repo root: scripts/cloud/ -> scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ggswarm_utils.gcloud_exec import (  # noqa: E402
    gcloud_not_found_message,
    resolve_gcloud_executable,
)


def _list_gcs_run_directories(gcloud: str, remote_prefix: str) -> list[str]:
    """Return run folder names under ``remote_prefix``, newest first.

    ``remote_prefix`` is e.g. ``gs://bucket/logs/skrl/ggswarm_marl`` (no trailing slash).
    Folder names use ISO-like timestamps so lexicographic reverse order ≈ newest first.
    """
    proc = subprocess.run(
        [gcloud, "storage", "ls", f"{remote_prefix.rstrip('/')}/"],
        capture_output=True,
        text=True,
        check=True,
    )
    names: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip().rstrip("/")
        if not line:
            continue
        base = line.split("/")[-1]
        # Training runs are ``<timestamp>_mappo_torch`` (MAPPO); skip stray prefixes.
        if base and "_mappo_torch" in base:
            names.append(base)
    # Newest first: string sort matches time order for ``YYYY-MM-DD_HH-MM-SS_...``.
    return sorted(set(names), reverse=True)


def _pull_latest_runs(
    *,
    gcloud: str,
    remote_path: str,
    local_log_dir: str,
    latest: int,
    dry_run: bool,
) -> None:
    """Copy only the N newest run directories using ``gcloud storage cp`` (Windows-safe)."""
    runs = _list_gcs_run_directories(gcloud, remote_path)
    selected = runs[:latest]
    if not selected:
        print(
            f"Error: No run directories (*_mappo_torch) found under {remote_path}/",
            file=sys.stderr,
        )
        sys.exit(1)

    dest = Path(local_log_dir)
    dest.mkdir(parents=True, exist_ok=True)
    print(
        f"Pulling {len(selected)} newest run(s) to {local_log_dir}: "
        f"{', '.join(selected)}"
    )
    for run in selected:
        src = f"{remote_path.rstrip('/')}/{run}"
        if dry_run:
            print(f"[DRY-RUN] {gcloud} storage cp -r {src} {local_log_dir}")
            continue
        subprocess.run(
            [gcloud, "storage", "cp", "-r", src, str(dest)],
            check=True,
            shell=False,
        )
    if dry_run:
        print("[DRY-RUN] No files were transferred.")
    else:
        print("Pull complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull training results from GCS (excluding videos from rsync mirror)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pull_results_from_gcs.py
  python pull_results_from_gcs.py --family hover
  python pull_results_from_gcs.py --latest 1
  python pull_results_from_gcs.py --family marl --latest 3 --dry-run
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
        default=None,
        metavar="N",
        help=(
            "If set, copy only the N newest run directories via "
            "'gcloud storage cp -r' (recommended on Windows). "
            "If omitted, mirror the whole family prefix via 'gcloud storage rsync'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without transferring files",
    )

    args = parser.parse_args()

    gcloud_exe = resolve_gcloud_executable()
    if gcloud_exe is None:
        print(f"Error: {gcloud_not_found_message()}", file=sys.stderr)
        sys.exit(1)

    gcs_bucket = os.environ.get("GGSWARM_GCS_BUCKET", "gs://gg-swarm-training-logs")
    local_log_dir = f"logs/skrl/ggswarm_{args.family}"
    remote_path = f"{gcs_bucket}/logs/skrl/ggswarm_{args.family}"

    if args.latest is not None:
        if args.latest < 1:
            print("Error: --latest must be >= 1", file=sys.stderr)
            sys.exit(2)
        try:
            _pull_latest_runs(
                gcloud=gcloud_exe,
                remote_path=remote_path,
                local_log_dir=local_log_dir,
                latest=args.latest,
                dry_run=args.dry_run,
            )
        except FileNotFoundError:
            print(f"Error: {gcloud_not_found_message()}", file=sys.stderr)
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Error: Pull failed with exit code {e.returncode}", file=sys.stderr)
            sys.exit(e.returncode)
        return

    cmd = [
        gcloud_exe,
        "storage",
        "rsync",
        "--recursive",
        "--exclude=videos/.*",
    ]

    if args.dry_run:
        cmd.append("--dry-run")
        print("[DRY-RUN] No files will be modified.")

    cmd.extend([remote_path, local_log_dir])

    print(
        f"Pulling {remote_path} to {local_log_dir} (rsync mirror; excluding videos). "
        "On Windows, prefer --latest N instead of full rsync."
    )
    try:
        subprocess.run(cmd, check=True, shell=sys.platform == "win32")
        print("Pull complete.")
    except FileNotFoundError:
        print(f"Error: {gcloud_not_found_message()}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Pull failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
