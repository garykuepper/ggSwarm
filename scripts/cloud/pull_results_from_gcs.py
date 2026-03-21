#!/usr/bin/env python3
"""Pull ggSwarm training results from GCS to local machine."""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Pull training results from GCS (excluding videos)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pull_results_from_gcs.py
  python pull_results_from_gcs.py --family hover
  python pull_results_from_gcs.py --dry-run
        """,
    )
    parser.add_argument(
        "--family",
        choices=["marl", "hover"],
        default="marl",
        help="Training family (default: marl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without transferring files",
    )

    args = parser.parse_args()

    gcs_bucket = os.environ.get("GGSWARM_GCS_BUCKET", "gs://gg-swarm-training-logs")
    local_log_dir = f"logs/skrl/ggswarm_{args.family}"
    remote_path = f"{gcs_bucket}/logs/skrl/ggswarm_{args.family}"

    cmd = [
        "gcloud",
        "storage",
        "rsync",
        "--recursive",
        "--exclude=videos/.*",
    ]

    if args.dry_run:
        cmd.append("--dry-run")
        print("[DRY-RUN] No files will be modified.")

    cmd.extend([remote_path, local_log_dir])

    print(f"Pulling {remote_path} to {local_log_dir} (excluding videos)")
    try:
        subprocess.run(cmd, check=True, shell=sys.platform == "win32")
        print("Pull complete.")
    except FileNotFoundError:
        print("Error: gcloud command not found. Install Google Cloud SDK.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: Pull failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
