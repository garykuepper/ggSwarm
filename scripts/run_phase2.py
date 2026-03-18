"""
Unified utility to run Phase 2 Brain Development tasks for ggSwarm.
Usage:
    python scripts/run_phase2.py train
    python scripts/run_phase2.py play
    python scripts/run_phase2.py monitor
"""

import argparse
import os
import subprocess
import sys

def run_command(command: list[str]):
    """Run a system command and pipe output."""
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)

def train(extra_args: list[str] = None):
    """Start Phase 2 training."""
    print("Starting Phase 2 Training (MAPPO + GATv2)...")
    cmd = [
        sys.executable, "scripts/skrl/train.py",
        "--task=Template-GGSwarm-Marl-Direct-v0",
        "--algorithm=MAPPO",
        "--headless",
        "--ml_framework", "torch",
        "--gnn"
    ]
    if extra_args:
        cmd.extend(extra_args)
    run_command(cmd)

def play(extra_args: list[str] = None):
    """Start Phase 2 playback."""
    print("Starting Phase 2 Playback (MAPPO + GATv2)...")
    cmd = [
        sys.executable, "scripts/skrl/play.py",
        "--task=Template-GGSwarm-Marl-Direct-v0",
        "--algorithm=MAPPO",
        "--ml_framework", "torch",
        "--gnn"
    ]
    if extra_args:
        cmd.extend(extra_args)
    run_command(cmd)

def monitor(extra_args: list[str] = None):
    """Start TensorBoard monitor."""
    print("Launching TensorBoard for Phase 2 logs...")
    log_dir = os.path.join("logs", "skrl", "ggswarm_marl")
    cmd = ["tensorboard", "--logdir", log_dir]
    if extra_args:
        cmd.extend(extra_args)
    run_command(cmd)

def main():
    parser = argparse.ArgumentParser(description="ggSwarm Phase 2 Helper Utility")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("train", help="Start training the GATv2 policy")
    subparsers.add_parser("play", help="Play back the latest trained policy")
    subparsers.add_parser("monitor", help="Launch TensorBoard for log monitoring")

    args, extra_args = parser.parse_known_args()

    if args.command == "train":
        train(extra_args)
    elif args.command == "play":
        play(extra_args)
    elif args.command == "monitor":
        monitor(extra_args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
