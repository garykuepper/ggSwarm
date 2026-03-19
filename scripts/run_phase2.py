"""
Unified utility to run Phase 2 Brain Development tasks for ggSwarm.
Usage:
    python scripts/run_phase2.py train
    python scripts/run_phase2.py train --max_iterations 5000
    python scripts/run_phase2.py train --num_envs 4 --num_agents 4
    python scripts/run_phase2.py play
    python scripts/run_phase2.py monitor
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _find_latest_checkpoint(
    log_root: Path,
    *,
    prefer_best: bool = True,
) -> str:
    """Find the newest checkpoint under the Phase 2 log root.

    Returns an absolute path string suitable for `--checkpoint`.
    """
    if not log_root.exists():
        raise FileNotFoundError(f"Log root not found: {log_root}")

    candidates: list[Path] = []
    for ckpt_dir in log_root.glob("**/checkpoints"):
        if not ckpt_dir.is_dir():
            continue
        if prefer_best:
            best = ckpt_dir / "best_agent.pt"
            if best.exists():
                candidates.append(best)
        candidates.extend(sorted(ckpt_dir.glob("agent_*.pt")))

    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under: {log_root}")

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(newest.resolve())

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

def train(extra_args: list[str] | None = None):
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

def play(extra_args: list[str] | None = None):
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

def monitor(extra_args: list[str] | None = None):
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

    train_parser = subparsers.add_parser("train", help="Start training the GATv2 policy")
    train_parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Training iterations (timesteps = max_iterations * 32). E.g. 5000 -> 160k timesteps.",
    )
    train_parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    train_parser.add_argument("--num_agents", type=int, default=None, help="Number of agents per environment.")
    train_parser.add_argument(
        "--resume_latest",
        action="store_true",
        default=False,
        help=(
            "Resume training from the newest checkpoint found under "
            "logs/skrl/ggswarm_marl/**/checkpoints/*.pt (prefers best_agent.pt)."
        ),
    )
    subparsers.add_parser("play", help="Play back the latest trained policy")
    subparsers.add_parser("monitor", help="Launch TensorBoard for log monitoring")

    args, extra_args = parser.parse_known_args()

    if args.command == "train":
        train_args = []
        if getattr(args, "max_iterations", None) is not None:
            train_args.extend(["--max_iterations", str(args.max_iterations)])
        if getattr(args, "num_envs", None) is not None:
            train_args.extend(["--num_envs", str(args.num_envs)])
        if getattr(args, "num_agents", None) is not None:
            train_args.extend(["--num_agents", str(args.num_agents)])
        if getattr(args, "resume_latest", False):
            log_root = Path("logs") / "skrl" / "ggswarm_marl"
            ckpt = _find_latest_checkpoint(log_root, prefer_best=True)
            print(f"[INFO] Resuming from latest checkpoint: {ckpt}")
            train_args.extend(["--checkpoint", ckpt])
        train(train_args + extra_args)
    elif args.command == "play":
        play(extra_args)
    elif args.command == "monitor":
        monitor(extra_args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
