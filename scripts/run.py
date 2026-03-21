"""Unified helper for hover, phase2, and debug workflows."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HOVER_TASK = "GGS-Hover-v0"
PHASE2_TASK = "Template-GGSwarm-Marl-Direct-v0"

# Use absolute paths so subprocesses spawned with a different CWD
# still share the same lock file and log roots.
REPO_ROOT = Path(__file__).resolve().parents[1]
HOVER_LOG_DIR = REPO_ROOT / "logs" / "skrl" / "ggswarm_hover"
PHASE2_LOG_DIR = REPO_ROOT / "logs" / "skrl" / "ggswarm_marl"
LOCK_DIR = REPO_ROOT / "logs" / "skrl" / "locks"
PHASE2_TRAIN_LOCK_PATH = LOCK_DIR / "phase2_train.lock"
# Stale lock tolerance (in seconds). If a previous run crashed, we don't want to block forever.
LOCK_TTL_S = 24 * 60 * 60


def _acquire_single_instance_lock(lock_path: Path) -> None:
    """Prevent launching the same long-running training job multiple times.

    This avoids accidental duplicate IsaacSim spawns when the wrapper CLI is invoked repeatedly.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    now_s = time.time()

    for _attempt in range(2):
        try:
            # O_EXCL makes lock creation atomic across processes.
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                lock_contents = f"pid={pid} time_s={int(now_s)}\n"
                os.write(fd, lock_contents.encode("utf-8"))
            finally:
                os.close(fd)
            return
        except FileExistsError:
            try:
                age_s = now_s - lock_path.stat().st_mtime
            except OSError:
                age_s = LOCK_TTL_S + 1
            if age_s <= LOCK_TTL_S:
                raise SystemExit(
                    f"[ERROR] Another phase2 training run appears active (lock: {lock_path}). "
                    "If this is unexpected, stop the old run and rerun."
                )
            # Lock is stale (previous run likely crashed). Remove and retry once.
            try:
                lock_path.unlink()
            except OSError:
                pass

    raise SystemExit(f"[ERROR] Failed to acquire lock: {lock_path}")


def _release_lock(lock_path: Path) -> None:
    """Remove the single-instance lock file if it exists."""
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _run_command(command: list[str]) -> None:
    """Run a command and stream output."""
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Command failed: {exc}")
        sys.exit(exc.returncode)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(130)


def _find_latest_checkpoint(log_root: Path, prefer_best: bool = True) -> str:
    """Return the newest checkpoint path under a log root."""
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
    return str(max(candidates, key=lambda p: p.stat().st_mtime).resolve())


def _add_common_sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_agents", type=int, default=None)


def _cmd_train(args: argparse.Namespace, *, task: str, gnn_default: bool) -> None:
    cmd = [
        sys.executable,
        "scripts/skrl/train.py",
        "--task",
        task,
        "--algorithm",
        "MAPPO",
        "--ml_framework",
        "torch",
    ]
    if args.headless:
        cmd.append("--headless")
    if args.device:
        cmd.extend(["--device", args.device])
    if args.num_envs is not None:
        cmd.extend(["--num_envs", str(args.num_envs)])
    if args.num_agents is not None:
        cmd.extend(["--num_agents", str(args.num_agents)])
    if args.max_iterations is not None:
        cmd.extend(["--max_iterations", str(args.max_iterations)])
    if args.no_progress:
        cmd.append("--no_progress")
    if args.progress_interval_s is not None:
        cmd.extend(["--progress_interval_s", str(args.progress_interval_s)])
    if args.eta_window_s is not None:
        cmd.extend(["--eta_window_s", str(args.eta_window_s)])
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if gnn_default and not args.no_gnn:
        cmd.append("--gnn")

    # Phase 2 training is long-running; prevent accidental duplicate spawns.
    lock_held = False
    try:
        if task == PHASE2_TASK:
            _acquire_single_instance_lock(PHASE2_TRAIN_LOCK_PATH)
            lock_held = True
        _run_command(cmd)
    finally:
        if lock_held:
            _release_lock(PHASE2_TRAIN_LOCK_PATH)


def _cmd_play(args: argparse.Namespace, *, task: str, gnn_default: bool) -> None:
    cmd = [
        sys.executable,
        "scripts/skrl/play.py",
        "--task",
        task,
        "--algorithm",
        "MAPPO",
        "--ml_framework",
        "torch",
    ]
    if args.headless:
        cmd.append("--headless")
    if args.device:
        cmd.extend(["--device", args.device])
    if args.num_envs is not None:
        cmd.extend(["--num_envs", str(args.num_envs)])
    if args.num_agents is not None:
        cmd.extend(["--num_agents", str(args.num_agents)])
    if args.max_steps is not None:
        cmd.extend(["--max_steps", str(args.max_steps)])
    if args.video:
        cmd.append("--video")
    if args.video_length is not None:
        cmd.extend(["--video_length", str(args.video_length)])
    if args.video_codec:
        cmd.extend(["--video_codec", args.video_codec])
    if args.video_bitrate:
        cmd.extend(["--video_bitrate", args.video_bitrate])
    if args.video_preset:
        cmd.extend(["--video_preset", args.video_preset])
    if args.video_ffmpeg_params:
        cmd.extend(["--video_ffmpeg_params", args.video_ffmpeg_params])
    rendering_mode = args.rendering_mode
    if args.video and rendering_mode is None:
        rendering_mode = "quality"
    if rendering_mode is not None:
        cmd.extend(["--rendering_mode", rendering_mode])
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        try:
            log_root = PHASE2_LOG_DIR if task == PHASE2_TASK else HOVER_LOG_DIR
            checkpoint_path = _find_latest_checkpoint(log_root=log_root, prefer_best=True)
            print(f"[INFO] Auto-selected checkpoint: {checkpoint_path}")
        except FileNotFoundError:
            checkpoint_path = None
    if checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])
    if gnn_default and not args.no_gnn:
        cmd.append("--gnn")
    if args.hover_debug:
        cmd.append("--hover_debug")
    _run_command(cmd)


def _cmd_eval_hover(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/eval_hover.py",
        "--task",
        HOVER_TASK,
        "--algorithm",
        "MAPPO",
        "--ml_framework",
        "torch",
    ]
    if args.headless:
        cmd.append("--headless")
    if args.device:
        cmd.extend(["--device", args.device])
    if args.num_envs is not None:
        cmd.extend(["--num_envs", str(args.num_envs)])
    if args.num_episodes is not None:
        cmd.extend(["--num_episodes", str(args.num_episodes)])
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    _run_command(cmd)


def _cmd_eval_phase2(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/eval_phase2.py",
        "--task",
        PHASE2_TASK,
        "--algorithm",
        "MAPPO",
        "--ml_framework",
        "torch",
        "--gnn",
        "--headless",
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.num_envs is not None:
        cmd.extend(["--num_envs", str(args.num_envs)])
    if args.num_agents is not None:
        cmd.extend(["--num_agents", str(args.num_agents)])
    if args.num_episodes is not None:
        cmd.extend(["--num_episodes", str(args.num_episodes)])
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    _run_command(cmd)


def _cmd_monitor(log_dir: Path) -> None:
    _run_command(["tensorboard", "--logdir", str(log_dir)])


def _cmd_debug_latest_checkpoint(args: argparse.Namespace) -> None:
    log_root = HOVER_LOG_DIR if args.family == "hover" else PHASE2_LOG_DIR
    checkpoint = _find_latest_checkpoint(log_root=log_root, prefer_best=not args.no_prefer_best)
    print(checkpoint)


def _cmd_debug_smoke(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/skrl/train.py",
        "--task",
        args.task,
        "--algorithm",
        "MAPPO",
        "--ml_framework",
        "torch",
        "--max_iterations",
        str(args.iterations),
    ]
    if args.headless:
        cmd.append("--headless")
    if args.device:
        cmd.extend(["--device", args.device])
    if args.num_envs is not None:
        cmd.extend(["--num_envs", str(args.num_envs)])
    if args.num_agents is not None:
        cmd.extend(["--num_agents", str(args.num_agents)])
    if args.gnn:
        cmd.append("--gnn")
    _run_command(cmd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified ggSwarm run helper")
    families = parser.add_subparsers(dest="family")

    hover = families.add_parser("hover", help="Hover baseline workflows")
    hover_cmds = hover.add_subparsers(dest="command")

    hover_train = hover_cmds.add_parser("train", help="Train hover policy")
    _add_common_sim_args(hover_train)
    hover_train.add_argument("--max_iterations", type=int, default=None)
    hover_train.add_argument("--no_progress", action="store_true", default=False)
    hover_train.add_argument("--progress_interval_s", type=float, default=10.0)
    hover_train.add_argument("--eta_window_s", type=float, default=120.0)
    hover_train.add_argument("--checkpoint", type=str, default=None)
    hover_train.add_argument("--no_gnn", action="store_true", default=False)
    hover_train.set_defaults(handler=lambda a: _cmd_train(a, task=HOVER_TASK, gnn_default=False))

    hover_play = hover_cmds.add_parser("play", help="Play hover policy")
    _add_common_sim_args(hover_play)
    hover_play.add_argument("--checkpoint", type=str, default=None)
    hover_play.add_argument("--max_steps", type=int, default=None)
    hover_play.add_argument("--video", action="store_true", default=False)
    hover_play.add_argument("--video_length", type=int, default=None)
    hover_play.add_argument(
        "--rendering_mode",
        type=str,
        choices=["performance", "balanced", "quality"],
        default=None,
    )
    hover_play.add_argument("--video_codec", type=str, default=None)
    hover_play.add_argument("--video_bitrate", type=str, default=None)
    hover_play.add_argument("--video_preset", type=str, default=None)
    hover_play.add_argument("--video_ffmpeg_params", type=str, default=None)
    hover_play.add_argument("--hover_debug", action="store_true", default=False)
    hover_play.add_argument("--no_gnn", action="store_true", default=False)
    hover_play.set_defaults(handler=lambda a: _cmd_play(a, task=HOVER_TASK, gnn_default=False))

    hover_eval = hover_cmds.add_parser("eval", help="Evaluate hover checkpoint")
    _add_common_sim_args(hover_eval)
    hover_eval.add_argument("--checkpoint", type=str, default=None)
    hover_eval.add_argument("--num_episodes", type=int, default=10)
    hover_eval.set_defaults(handler=_cmd_eval_hover)

    hover_monitor = hover_cmds.add_parser("monitor", help="Monitor hover TensorBoard logs")
    hover_monitor.set_defaults(handler=lambda _a: _cmd_monitor(HOVER_LOG_DIR))

    phase2 = families.add_parser("phase2", help="Phase 2 formation workflows")
    phase2_cmds = phase2.add_subparsers(dest="command")

    phase2_train = phase2_cmds.add_parser("train", help="Train phase2 policy")
    _add_common_sim_args(phase2_train)
    phase2_train.add_argument("--max_iterations", type=int, default=None)
    phase2_train.add_argument("--no_progress", action="store_true", default=False)
    phase2_train.add_argument("--progress_interval_s", type=float, default=10.0)
    phase2_train.add_argument("--eta_window_s", type=float, default=120.0)
    phase2_train.add_argument("--checkpoint", type=str, default=None)
    phase2_train.add_argument("--no_gnn", action="store_true", default=False)
    phase2_train.set_defaults(handler=lambda a: _cmd_train(a, task=PHASE2_TASK, gnn_default=True))

    phase2_play = phase2_cmds.add_parser("play", help="Play phase2 policy")
    _add_common_sim_args(phase2_play)
    phase2_play.add_argument("--checkpoint", type=str, default=None)
    phase2_play.add_argument("--max_steps", type=int, default=None)
    phase2_play.add_argument("--video", action="store_true", default=False)
    phase2_play.add_argument("--video_length", type=int, default=None)
    phase2_play.add_argument(
        "--rendering_mode",
        type=str,
        choices=["performance", "balanced", "quality"],
        default=None,
    )
    phase2_play.add_argument("--video_codec", type=str, default=None)
    phase2_play.add_argument("--video_bitrate", type=str, default=None)
    phase2_play.add_argument("--video_preset", type=str, default=None)
    phase2_play.add_argument("--video_ffmpeg_params", type=str, default=None)
    phase2_play.add_argument("--hover_debug", action="store_true", default=False)
    phase2_play.add_argument("--no_gnn", action="store_true", default=False)
    phase2_play.set_defaults(handler=lambda a: _cmd_play(a, task=PHASE2_TASK, gnn_default=True))

    phase2_eval = phase2_cmds.add_parser("eval", help="Evaluate phase2 checkpoint")
    _add_common_sim_args(phase2_eval)
    phase2_eval.add_argument("--checkpoint", type=str, default=None)
    phase2_eval.add_argument("--num_episodes", type=int, default=5)
    phase2_eval.set_defaults(handler=_cmd_eval_phase2)

    phase2_monitor = phase2_cmds.add_parser(
        "monitor", help="Monitor phase2 TensorBoard logs"
    )
    phase2_monitor.set_defaults(handler=lambda _a: _cmd_monitor(PHASE2_LOG_DIR))

    debug = families.add_parser("debug", help="Debug utilities")
    debug_cmds = debug.add_subparsers(dest="command")

    smoke = debug_cmds.add_parser("smoke", help="Run a very short training smoke test")
    _add_common_sim_args(smoke)
    smoke.add_argument("--task", type=str, default=HOVER_TASK)
    smoke.add_argument("--iterations", type=int, default=1)
    smoke.add_argument("--gnn", action="store_true", default=False)
    smoke.set_defaults(handler=_cmd_debug_smoke)

    latest = debug_cmds.add_parser(
        "latest-checkpoint", help="Print latest checkpoint path"
    )
    latest.add_argument("--family", choices=["hover", "phase2"], default="hover")
    latest.add_argument("--no_prefer_best", action="store_true", default=False)
    latest.set_defaults(handler=_cmd_debug_latest_checkpoint)

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return
    handler(args)


if __name__ == "__main__":
    main()
