"""Command-building helpers and process utilities for run.py.

Extracted from run.py to make the CLI layer thin and testable.
All ``build_*_cmd`` functions return a ``list[str]`` suitable for
``subprocess.run`` — they do NOT call sys.exit or spawn processes.

Public API:
    build_train_cmd(task, args, *, gnn_default) -> list[str]
    build_play_cmd(task, args, *, gnn_default, log_root)  -> list[str]
    build_eval_cmd(phase, args)                           -> list[str]
    build_assess_cmd(assess_script, task, args)            -> list[str]

When ``args.video`` is true, ``build_eval_cmd`` and ``build_assess_cmd`` append
``--video`` and related ffmpeg/rendering flags for subprocess entry points.
    find_latest_checkpoint(log_root, prefer_best)          -> str
    acquire_single_instance_lock(lock_path, ttl_s)         -> None
    release_lock(lock_path)                                -> None
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def find_latest_checkpoint(log_root: Path, prefer_best: bool = True) -> str:
    """Return the path of the newest checkpoint under ``log_root``.

    Searches all ``**/checkpoints/`` subdirectories.  If ``prefer_best`` is
    True (default) and a ``best_agent.pt`` exists in any run, it is preferred
    over numbered ``agent_*.pt`` files.

    Args:
        log_root:    Root directory to search (e.g. ``logs/skrl/ggswarm_marl``).
        prefer_best: When True, prefer ``best_agent.pt`` over numbered files.

    Returns:
        Absolute path string of the most recent checkpoint file.

    Raises:
        FileNotFoundError: If ``log_root`` does not exist or no checkpoints
                           are found beneath it.
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
    return str(max(candidates, key=lambda p: p.stat().st_mtime).resolve())


# ---------------------------------------------------------------------------
# Single-instance lock (prevents duplicate training spawns)
# ---------------------------------------------------------------------------

_DEFAULT_LOCK_TTL_S: int = 24 * 60 * 60  # 24 hours


def acquire_single_instance_lock(
    lock_path: Path, ttl_s: int = _DEFAULT_LOCK_TTL_S
) -> None:
    """Prevent launching the same long-running training job twice.

    Creates a lock file atomically.  If the file already exists and is
    younger than ``ttl_s`` seconds, raises ``SystemExit``.  Stale locks
    (older than ``ttl_s``) are removed and a single retry is attempted.

    Args:
        lock_path: Path to the lock file.
        ttl_s:     Maximum age of a valid lock in seconds.

    Raises:
        SystemExit: If an active lock exists, or if acquisition fails.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    now_s = time.time()

    for _attempt in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"pid={pid} time_s={int(now_s)}\n".encode())
            finally:
                os.close(fd)
            return
        except FileExistsError:
            try:
                age_s = now_s - lock_path.stat().st_mtime
            except OSError:
                age_s = ttl_s + 1
            if age_s <= ttl_s:
                raise SystemExit(
                    f"[ERROR] Another training run appears active (lock: {lock_path}). "
                    "If unexpected, stop the old run and rerun."
                )
            try:
                lock_path.unlink()
            except OSError:
                pass

    raise SystemExit(f"[ERROR] Failed to acquire lock: {lock_path}")


def release_lock(lock_path: Path) -> None:
    """Remove the single-instance lock file if it exists.

    Silently ignores ``FileNotFoundError`` (idempotent).
    """
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _append_if(cmd: list[str], condition: object, *flags: str) -> None:
    """Append ``flags`` to ``cmd`` when ``condition`` is truthy."""
    if condition:
        cmd.extend(flags)


def _append_opt(cmd: list[str], flag: str, value: object) -> None:
    """Append ``[flag, str(value)]`` to ``cmd`` when ``value`` is not None/False."""
    if value is not None and value is not False:
        cmd.extend([flag, str(value)])


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------


def build_train_cmd(
    task: str,
    args: object,
    *,
    gnn_default: bool,
) -> list[str]:
    """Build the ``scripts/skrl/train.py`` command list.

    Args:
        task:        Gym task ID.
        args:        Parsed ``argparse.Namespace`` from the train subparser.
        gnn_default: Whether to pass ``--gnn`` by default (suppressed by
                     ``args.no_gnn``).

    Returns:
        Command list ready for ``subprocess.run``.
    """
    cmd = [
        sys.executable,
        "scripts/skrl/train.py",
        "--task", task,
        "--algorithm", "MAPPO",
        "--ml_framework", "torch",
    ]
    _append_if(cmd, getattr(args, "headless", False), "--headless")
    _append_opt(cmd, "--device", getattr(args, "device", None))
    _append_opt(cmd, "--num_envs", getattr(args, "num_envs", None))
    _append_opt(cmd, "--num_agents", getattr(args, "num_agents", None))
    _append_opt(cmd, "--max_iterations", getattr(args, "max_iterations", None))
    _append_opt(
        cmd,
        "--action_telemetry_steps",
        getattr(args, "action_telemetry_steps", None),
    )
    _append_if(cmd, getattr(args, "no_progress", False), "--no_progress")
    _append_opt(cmd, "--progress_interval_s", getattr(args, "progress_interval_s", None))
    _append_opt(cmd, "--eta_window_s", getattr(args, "eta_window_s", None))
    _append_opt(cmd, "--checkpoint", getattr(args, "checkpoint", None))
    _append_opt(cmd, "--early_stop_step", getattr(args, "early_stop_step", None))
    _append_if(cmd, getattr(args, "no_early_stop", False), "--no_early_stop")
    _append_opt(cmd, "--log_subdir", getattr(args, "log_subdir", None))
    want_gnn = (
        gnn_default or getattr(args, "gnn", False)
    ) and not getattr(args, "no_gnn", False)
    if want_gnn:
        cmd.append("--gnn")
    return cmd


def build_play_cmd(
    task: str,
    args: object,
    *,
    gnn_default: bool,
    log_root: Path | None = None,
) -> list[str]:
    """Build the ``scripts/skrl/play.py`` command list.

    Auto-discovers the latest checkpoint when ``args.checkpoint`` is None and
    ``log_root`` is provided.

    Args:
        task:        Gym task ID.
        args:        Parsed ``argparse.Namespace`` from the play subparser.
        gnn_default: Whether to pass ``--gnn`` by default.
        log_root:    Root to search for checkpoints when none is given.

    Returns:
        Command list ready for ``subprocess.run``.
    """
    cmd = [
        sys.executable,
        "scripts/skrl/play.py",
        "--task", task,
        "--algorithm", "MAPPO",
        "--ml_framework", "torch",
    ]
    _append_if(cmd, getattr(args, "headless", False), "--headless")
    _append_opt(cmd, "--device", getattr(args, "device", None))
    _append_opt(cmd, "--num_envs", getattr(args, "num_envs", None))
    _append_opt(cmd, "--num_agents", getattr(args, "num_agents", None))
    _append_opt(cmd, "--max_steps", getattr(args, "max_steps", None))

    if getattr(args, "video", False):
        cmd.append("--video")
    _append_opt(cmd, "--video_length", getattr(args, "video_length", None))
    _append_opt(cmd, "--video_codec", getattr(args, "video_codec", None))
    _append_opt(cmd, "--video_bitrate", getattr(args, "video_bitrate", None))
    _append_opt(cmd, "--video_preset", getattr(args, "video_preset", None))
    _append_opt(cmd, "--video_ffmpeg_params", getattr(args, "video_ffmpeg_params", None))

    rendering_mode = getattr(args, "rendering_mode", None)
    if getattr(args, "video", False) and rendering_mode is None:
        rendering_mode = "quality"
    _append_opt(cmd, "--rendering_mode", rendering_mode)

    checkpoint_path = getattr(args, "checkpoint", None)
    if checkpoint_path is None and log_root is not None:
        try:
            checkpoint_path = find_latest_checkpoint(log_root, prefer_best=True)
            print(f"[INFO] Auto-selected checkpoint: {checkpoint_path}")
        except FileNotFoundError:
            checkpoint_path = None
    _append_opt(cmd, "--checkpoint", checkpoint_path)

    want_gnn = (
        gnn_default or getattr(args, "gnn", False)
    ) and not getattr(args, "no_gnn", False)
    if want_gnn:
        cmd.append("--gnn")
    _append_if(cmd, getattr(args, "hover_debug", False), "--hover_debug")
    return cmd


def build_eval_cmd(
    phase: str,
    args: object,
    *,
    extra_flags: list[str] | None = None,
) -> list[str]:
    """Build a ``scripts/eval.py`` command for the given phase.

    The phase key (e.g. ``"2a"``, ``"hover"``) is resolved to a task ID and
    GNN default inside ``eval.py`` via ``PHASE_REGISTRY`` — no task ID or
    script path is needed here.

    Args:
        phase:       Phase key (one of the keys in PHASE_REGISTRY).
        args:        Parsed ``argparse.Namespace`` from the eval subparser.
        extra_flags: Additional literal flags to append (e.g.
                     ``["--kill_agent_step", "50"]``).

    Returns:
        Command list ready for ``subprocess.run``.
    """
    cmd = [
        sys.executable,
        "scripts/eval.py",
        "--phase", phase,
        "--headless",
    ]
    # --gnn / --no-gnn: forward if explicitly set; eval.py defaults to registry
    no_gnn = getattr(args, "no_gnn", False)
    if no_gnn:
        cmd.append("--no-gnn")
    _append_opt(cmd, "--device", getattr(args, "device", None))
    _append_opt(cmd, "--num_envs", getattr(args, "num_envs", None))
    _append_opt(cmd, "--num_agents", getattr(args, "num_agents", None))
    _append_opt(cmd, "--num_episodes", getattr(args, "num_episodes", None))
    _append_opt(cmd, "--seed", getattr(args, "seed", None))
    _append_opt(cmd, "--checkpoint", getattr(args, "checkpoint", None))
    if extra_flags:
        cmd.extend(extra_flags)

    if getattr(args, "video", False):
        cmd.append("--video")
        _append_opt(cmd, "--video_length", getattr(args, "video_length", None))
        _append_opt(cmd, "--video_codec", getattr(args, "video_codec", None))
        _append_opt(cmd, "--video_bitrate", getattr(args, "video_bitrate", None))
        _append_opt(cmd, "--video_preset", getattr(args, "video_preset", None))
        _append_opt(cmd, "--video_ffmpeg_params", getattr(args, "video_ffmpeg_params", None))
        rendering_mode = getattr(args, "rendering_mode", None)
        if rendering_mode is None:
            rendering_mode = "quality"
        _append_opt(cmd, "--rendering_mode", rendering_mode)

    return cmd


def build_assess_cmd(
    assess_script: str | Path,
    task: str,
    args: object,
) -> list[str]:
    """Build the ``post_train_assess.py`` command.

    Args:
        assess_script: Path to the assess script.
        task:          Gym task ID.
        args:          Parsed ``argparse.Namespace`` from the assess subparser.

    Returns:
        Command list ready for ``subprocess.run``.
    """
    cmd = [
        sys.executable,
        str(assess_script),
        "--run_dir", getattr(args, "run_dir", ""),
        "--task", task,
        "--num_episodes", str(getattr(args, "num_episodes", 5) or 5),
        "--headless",
    ]
    if getattr(args, "no_sync", False):
        cmd.append("--no_sync")
    if getattr(args, "trajectories", False):
        cmd.append("--trajectories")
    _append_opt(cmd, "--device", getattr(args, "device", None))
    _append_opt(cmd, "--seed", getattr(args, "seed", None))
    if getattr(args, "video", False):
        cmd.append("--video")
        _append_opt(cmd, "--video_length", getattr(args, "video_length", None))
        _append_opt(cmd, "--video_codec", getattr(args, "video_codec", None))
        _append_opt(cmd, "--video_bitrate", getattr(args, "video_bitrate", None))
        _append_opt(cmd, "--video_preset", getattr(args, "video_preset", None))
        _append_opt(cmd, "--video_ffmpeg_params", getattr(args, "video_ffmpeg_params", None))
        rendering_mode = getattr(args, "rendering_mode", None)
        if rendering_mode is None:
            rendering_mode = "quality"
        _append_opt(cmd, "--rendering_mode", rendering_mode)
    return cmd
