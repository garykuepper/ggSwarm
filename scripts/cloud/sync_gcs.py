#!/usr/bin/env python3
"""Sync ggSwarm logs to/from GCS (local cross-platform replacement for sync_gcs.sh)."""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Sync training logs to/from GCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sync_gcs.py push
  python sync_gcs.py pull --family hover
  python sync_gcs.py pull --include-videos
  python sync_gcs.py pull --dry-run
        """,
    )
    parser.add_argument(
        "action",
        choices=["push", "pull"],
        help="Push to GCS or pull from GCS",
    )
    parser.add_argument(
        "--family",
        choices=["marl", "hover"],
        default="marl",
        help="Training family (default: marl)",
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Include videos in sync (default: exclude)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without transferring files",
    )

    args = parser.parse_args()

    # Read GCS URI from environment
    gcs_uri = os.environ.get("GGSWARM_GCS_URI")
    if not gcs_uri:
        print(
            "Error: GGSWARM_GCS_URI environment variable not set.",
            file=sys.stderr,
        )
        print(
            "Set it to gs://bucket-name/prefix and retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Determine local and remote paths
    local_log_dir = f"logs/skrl/ggswarm_{args.family}"
    remote_path = f"{gcs_uri}/logs/skrl/ggswarm_{args.family}"

    # Build rsync arguments
    cmd = [
        "gcloud",
        "storage",
        "rsync",
        "--recursive",
    ]

    # Add video exclusion unless --include-videos was passed
    if not args.include_videos:
        cmd.append("--exclude=videos/.*")

    # Add dry-run flag if requested
    if args.dry_run:
        cmd.append("--dry-run")
        print("[DRY-RUN] No files will be modified.")

    # Add source and destination
    if args.action == "push":
        print(f"Pushing {local_log_dir} to {remote_path}")
        cmd.extend([local_log_dir, remote_path])
    else:
        print(f"Pulling {remote_path} to {local_log_dir}")
        cmd.extend([remote_path, local_log_dir])

    try:
        subprocess.run(cmd, check=True, shell=sys.platform == "win32")
        print(f"{args.action.capitalize()} complete.")
    except FileNotFoundError:
        print("Error: gcloud command not found. Install Google Cloud SDK.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(
            f"Error: {args.action.capitalize()} failed with exit code {e.returncode}",
            file=sys.stderr,
        )
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
