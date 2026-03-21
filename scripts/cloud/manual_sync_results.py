#!/usr/bin/env python3
"""Manually sync training results when automatic sync fails (fallback to gcloud compute scp)."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Manually sync training results from VM when GCS sync fails",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manual_sync_results.py
  python manual_sync_results.py --family hover
  python manual_sync_results.py --force
        """,
    )
    parser.add_argument(
        "--family",
        choices=["marl", "hover"],
        default="marl",
        help="Training family (default: marl)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force sync even if checkpoints already exist locally",
    )
    parser.add_argument(
        "--zone",
        default="us-central1-a",
        help="GCP zone (default: us-central1-a)",
    )
    parser.add_argument(
        "--instance",
        default="isaacsim",
        help="GCE instance name (default: isaacsim)",
    )

    args = parser.parse_args()

    print("\n=== Manual Training Results Sync ===\n")

    gcs_bucket = "gs://gg-swarm-training-logs"
    local_log_dir = f"logs/skrl/ggswarm_{args.family}"
    remote_path = f"{gcs_bucket}/logs/skrl/ggswarm_{args.family}"

    # Check if local logs already synced
    local_log_path = Path(local_log_dir)
    checkpoint_count = 0
    if local_log_path.exists():
        checkpoint_count = len(list(local_log_path.glob("**/checkpoints/*.pt")))

    if checkpoint_count > 0 and not args.force:
        print(f"✓ Checkpoints already synced locally ({checkpoint_count} files found)")
        print(f"\nTo pull latest from GCS, run:")
        print(f"  python scripts/cloud/pull_results_from_gcs.py --family {args.family}\n")
        sys.exit(0)

    print("Falling back to direct VM copy via gcloud compute scp...")
    print("(This bypasses GCS but gets the job done quickly)\n")

    print(f"Copying logs from VM ({args.instance}) to local PC...")

    vm_source_path = f"/home/gkuep/ggSwarm/logs/skrl/ggswarm_{args.family}"

    scp_cmd = [
        "gcloud",
        "compute",
        "scp",
        "--zone",
        args.zone,
        "--recurse",
        f"{args.instance}:{vm_source_path}",
        local_log_dir,
    ]

    try:
        result = subprocess.run(
            scp_cmd,
            capture_output=True,
            text=True,
            shell=sys.platform == "win32",
        )

        # Show last 10 lines of output
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            for line in lines[-10:]:
                print(line)

        if result.returncode != 0:
            print(f"\n❌ Copy failed!", file=sys.stderr)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            sys.exit(result.returncode)

        print(f"\n✓ Sync complete!")
        print(f"\nCheckpoints are now available locally in: {local_log_dir}\n")

        print("Next steps:")
        print(f"  1. List checkpoints: python scripts/cloud/list_checkpoints.py --family {args.family}")
        print(f"  2. Evaluate: python scripts/run.py {args.family} eval --checkpoint <path>")
        print(f"  3. Play: python scripts/run.py {args.family} play --checkpoint <path> --video\n")

    except FileNotFoundError:
        print("Error: gcloud command not found. Install Google Cloud SDK.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
