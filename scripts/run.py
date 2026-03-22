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
HOVER_STABILITY_TASK = "Template-GGSwarm-Marl-HoverStability-v0"
PHASE2B_TASK = "Template-GGSwarm-Marl-Formation-v0"
PHASE3_TASK = "Template-GGSwarm-Marl-Phase3-v0"
PHASE4_TASK = "Template-GGSwarm-Marl-Phase4-v0"

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


def _cmd_eval_phase3(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/eval_phase3.py",
        "--task",
        args.task or PHASE3_TASK,
        "--algorithm",
        "MAPPO",
        "--ml_framework",
        "torch",
        "--headless",
    ]
    # --no_gnn is an explicit CLI arg (Rule 16); default is GNN on for Phase 3
    if not getattr(args, "no_gnn", False):
        cmd.append("--gnn")
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
    if args.kill_agent_step is not None:
        cmd.extend(["--kill_agent_step", str(args.kill_agent_step)])
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
        "--headless",
    ]
    # --no_gnn is an explicit CLI arg (Rule 16); default is GNN on for Phase 2
    if not getattr(args, "no_gnn", False):
        cmd.append("--gnn")
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


def _cmd_eval_phase2b(args: argparse.Namespace) -> None:
    """Evaluate a Phase B formation checkpoint using Phase 2 metrics."""
    cmd = [
        sys.executable,
        "scripts/eval_phase2.py",
        "--task",
        PHASE2B_TASK,
        "--algorithm",
        "MAPPO",
        "--ml_framework",
        "torch",
        "--headless",
    ]
    if not getattr(args, "no_gnn", False):
        cmd.append("--gnn")
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


def _cmd_eval_hover_stability(args: argparse.Namespace) -> None:
    """Evaluate a hover-stability checkpoint using Phase 2 formation metrics.

    Uses eval_phase2.py with the HoverStability task ID so formation metrics
    are measured even though formation rewards were disabled during training.
    This reveals whether multi-agent hover generalises to the formation context.
    """
    cmd = [
        sys.executable,
        "scripts/eval_phase2.py",
        "--task",
        HOVER_STABILITY_TASK,
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


# Pass/Fail thresholds for the post-training scorecard (Rule 20).
_ASSESS_THRESHOLDS: dict[str, tuple[float | None, float | None, float | None]] = {
    # metric: (fail_above or None, warn_above or None, pass_above or None)
    # For metrics where lower is better, invert logic in _assess_verdict.
    "survival_steps":               (10.0,    200.0,  500.0),   # FAIL < 10, PASS > 500
    "airborne_ratio":               (0.5,     0.85,   0.9),     # FAIL < 0.5, PASS > 0.9
    "ground_hit_rate":              (None,    None,   None),     # handled separately (lower better)
    "mean_roll_deg":                (None,    None,   None),     # handled separately (lower better)
    "orientation_violation_rate":   (None,    None,   None),     # handled separately (lower better)
    "mean_formation_error_m":       (None,    None,   None),     # handled separately (lower better)
}

_THRESHOLDS_LOWER_BETTER: dict[str, tuple[float, float, float]] = {
    # metric: (pass_below, warn_below, fail_above)
    "ground_hit_rate":              (0.05, 0.5,  0.5),
    "mean_roll_deg":                (15.0, 60.0, 60.0),
    "orientation_violation_rate":   (0.1,  0.5,  0.5),
    "mean_formation_error_m":       (0.5,  1.5,  1.5),
}


def _assess_verdict(
    metric: str, value: float
) -> tuple[str, str]:
    """Return (verdict, threshold_hint) for a single metric value."""
    if metric in _THRESHOLDS_LOWER_BETTER:
        pass_t, warn_t, fail_t = _THRESHOLDS_LOWER_BETTER[metric]
        if value < pass_t:
            return "PASS", f"< {pass_t}"
        if value < warn_t:
            return "WARN", f"< {warn_t}"
        return "FAIL", f"< {pass_t}"
    thresholds = _ASSESS_THRESHOLDS.get(metric)
    if thresholds is None:
        return "UNKNOWN", "n/a"
    fail_t, warn_t, pass_t = thresholds
    if pass_t is not None and value >= pass_t:
        return "PASS", f"> {pass_t}"
    if warn_t is not None and value >= warn_t:
        return "WARN", f"> {pass_t}"
    return "FAIL", f"> {pass_t}"


def _cmd_assess(args: argparse.Namespace, *, task: str) -> None:
    """Run the full post-training assessment sequence and print a scorecard.

    Steps (all local, no GCE required):
    1. Convergence check — reads TFEvents, prints entropy collapse step.
    2. Best checkpoint eval — runs eval_phase2.py for N episodes, captures JSON output.
    3. Scorecard — prints PASS/WARN/FAIL per metric, overall verdict.

    Exits with code 0 (all PASS/WARN) or 1 (any FAIL).
    """
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[ERROR] --run_dir not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    run_name = run_dir.name
    print(f"\n{'='*70}")
    print(f"  POST-TRAINING ASSESSMENT: {run_name}")
    print(f"{'='*70}\n")

    # --- Step 1: Convergence check (no Isaac Lab needed) ---
    print("CONVERGENCE CHECK")
    print("-" * 40)
    conv_script = REPO_ROOT / "scripts" / "cloud" / "check_convergence.py"
    _run_command([sys.executable, str(conv_script), "--log_dir", str(run_dir)])
    print()

    # --- Step 2: Checkpoint progression (optional; skipped if no interval set) ---
    if getattr(args, "progression", False):
        print("CHECKPOINT PROGRESSION")
        print("-" * 40)
        prog_script = REPO_ROOT / "scripts" / "analyze_checkpoints.py"
        _run_command(
            [
                sys.executable,
                str(prog_script),
                "--run_dir",
                str(run_dir),
                "--interval",
                "10000",
                "--num_episodes",
                "3",
                "--task",
                task,
                "--headless",
            ]
        )
        print()

    # --- Step 3: Best checkpoint eval ---
    best_ckpt = run_dir / "checkpoints" / "best_agent.pt"
    if not best_ckpt.exists():
        # Fallback: latest numbered checkpoint
        ckpts = sorted((run_dir / "checkpoints").glob("agent_*.pt"))
        if ckpts:
            best_ckpt = ckpts[-1]
            print(f"[WARN] best_agent.pt not found; using {best_ckpt.name}")
        else:
            print("[ERROR] No checkpoints found in run_dir/checkpoints/", file=sys.stderr)
            sys.exit(1)

    num_episodes = getattr(args, "num_episodes", 5) or 5
    json_out = run_dir / "assess_metrics.json"

    print("STABILITY + FORMATION EVAL")
    print(f"  checkpoint : {best_ckpt.name}")
    print(f"  episodes   : {num_episodes}")
    print("-" * 40)

    eval_cmd = [
        sys.executable,
        "scripts/eval_phase2.py",
        "--task",
        task,
        "--algorithm",
        "MAPPO",
        "--ml_framework",
        "torch",
        "--gnn",
        "--headless",
        "--checkpoint",
        str(best_ckpt),
        "--num_episodes",
        str(num_episodes),
        "--output_json",
        str(json_out),
    ]
    if args.device:
        eval_cmd.extend(["--device", args.device])
    _run_command(eval_cmd)

    # --- Step 4: Scorecard ---
    import json  # noqa: PLC0415 — deferred; not needed unless assess is run

    if not json_out.exists():
        print("\n[WARN] eval_phase2.py did not produce assess_metrics.json; "
              "scorecard skipped.\n       Re-run with a version of eval_phase2.py "
              "that supports --output_json.")
        return

    with json_out.open() as fh:
        metrics: dict[str, float] = json.load(fh)

    ordered_metrics = [
        "survival_steps",
        "airborne_ratio",
        "ground_hit_rate",
        "mean_roll_deg",
        "orientation_violation_rate",
        "mean_formation_error_m",
    ]

    print(f"\n{'='*70}")
    print(f"  SCORECARD: {run_name}")
    print(f"{'='*70}")

    verdicts: list[str] = []
    for key in ordered_metrics:
        if key not in metrics:
            print(f"  {key:<35} n/a    (not reported)")
            continue
        val = metrics[key]
        verdict, hint = _assess_verdict(key, val)
        verdicts.append(verdict)
        marker = "←" if verdict == "FAIL" else (" " if verdict == "PASS" else "~")
        print(f"  {key:<35} {val:>8.3f}  {marker} {verdict:<4}  (threshold: {hint})")

    fail_count = verdicts.count("FAIL")
    pass_count = verdicts.count("PASS")
    overall = "FAIL" if fail_count > 0 else ("PASS" if pass_count == len(verdicts) else "WARN")

    print(f"\n{'='*70}")
    print(f"  VERDICT: {overall}  ({fail_count} FAIL, "
          f"{verdicts.count('WARN')} WARN, {pass_count} PASS out of {len(verdicts)})")
    if overall == "FAIL":
        print("  ACTION REQUIRED: Do NOT retrain until reward config is adjusted.")
        print("  Log this scorecard to docs/status/changelog.md before any changes.")
    elif overall == "WARN":
        print("  Continue monitoring; review WARN metrics before Phase 3 gate.")
    else:
        print("  All metrics PASS — ready to advance to Phase 3 or re-enable formation.")
    print(f"{'='*70}\n")

    sys.exit(0 if overall != "FAIL" else 1)


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


def _build_parser() -> argparse.ArgumentParser:  # noqa: C901  # grandfathered — do not add branches
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
    phase2_eval.add_argument(
        "--no_gnn",
        action="store_true",
        default=False,
        help="Disable GNN policy (use MLP). Required when checkpoint was trained without --gnn.",
    )
    phase2_eval.set_defaults(handler=_cmd_eval_phase2)

    phase2_assess = phase2_cmds.add_parser(
        "assess",
        help="Run full post-training assessment (convergence + eval + scorecard)",
    )
    phase2_assess.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (logs/skrl/ggswarm_marl/<timestamp>)",
    )
    phase2_assess.add_argument("--num_episodes", type=int, default=5)
    phase2_assess.add_argument(
        "--progression",
        action="store_true",
        default=False,
        help="Also run checkpoint progression sweep (adds ~3 min).",
    )
    phase2_assess.add_argument("--device", type=str, default=None)
    phase2_assess.set_defaults(handler=lambda a: _cmd_assess(a, task=PHASE2_TASK))

    phase2_monitor = phase2_cmds.add_parser(
        "monitor", help="Monitor phase2 TensorBoard logs"
    )
    phase2_monitor.set_defaults(handler=lambda _a: _cmd_monitor(PHASE2_LOG_DIR))

    hover_stability = families.add_parser(
        "hover-stability", help="Phase A hover-stability training (formation disabled)"
    )
    hs_cmds = hover_stability.add_subparsers(dest="command")

    hs_train = hs_cmds.add_parser("train", help="Train hover-stability policy (80k iters)")
    _add_common_sim_args(hs_train)
    hs_train.add_argument("--max_iterations", type=int, default=None)
    hs_train.add_argument("--no_progress", action="store_true", default=False)
    hs_train.add_argument("--progress_interval_s", type=float, default=10.0)
    hs_train.add_argument("--eta_window_s", type=float, default=120.0)
    hs_train.add_argument("--checkpoint", type=str, default=None)
    hs_train.add_argument("--no_gnn", action="store_true", default=False)
    hs_train.set_defaults(
        handler=lambda a: _cmd_train(a, task=HOVER_STABILITY_TASK, gnn_default=True)
    )

    hs_eval = hs_cmds.add_parser(
        "eval", help="Evaluate hover-stability checkpoint (Phase 2 metrics)"
    )
    _add_common_sim_args(hs_eval)
    hs_eval.add_argument("--checkpoint", type=str, default=None)
    hs_eval.add_argument("--num_episodes", type=int, default=5)
    hs_eval.add_argument("--no_gnn", action="store_true", default=False)
    hs_eval.set_defaults(handler=_cmd_eval_hover_stability)

    hs_assess = hs_cmds.add_parser(
        "assess",
        help="Run full post-training assessment for hover-stability run",
    )
    hs_assess.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (logs/skrl/ggswarm_marl/<timestamp>)",
    )
    hs_assess.add_argument("--num_episodes", type=int, default=5)
    hs_assess.add_argument(
        "--progression",
        action="store_true",
        default=False,
        help="Also run checkpoint progression sweep (adds ~3 min).",
    )
    hs_assess.add_argument("--device", type=str, default=None)
    hs_assess.set_defaults(
        handler=lambda a: _cmd_assess(a, task=HOVER_STABILITY_TASK)
    )

    hs_monitor = hs_cmds.add_parser(
        "monitor", help="Monitor hover-stability TensorBoard logs"
    )
    hs_monitor.set_defaults(handler=lambda _a: _cmd_monitor(PHASE2_LOG_DIR))

    phase2b = families.add_parser(
        "phase2b",
        help="Phase B formation training (resume from Phase A hover-stability checkpoint)",
    )
    phase2b_cmds = phase2b.add_subparsers(dest="command")

    phase2b_train = phase2b_cmds.add_parser(
        "train",
        help="Train Phase B formation policy (120k iters, resume from Phase A best_agent.pt)",
    )
    _add_common_sim_args(phase2b_train)
    phase2b_train.add_argument("--max_iterations", type=int, default=None)
    phase2b_train.add_argument("--no_progress", action="store_true", default=False)
    phase2b_train.add_argument("--progress_interval_s", type=float, default=10.0)
    phase2b_train.add_argument("--eta_window_s", type=float, default=120.0)
    phase2b_train.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Phase A best_agent.pt to resume from (Rule 21: curriculum resets to 0).",
    )
    phase2b_train.add_argument("--no_gnn", action="store_true", default=False)
    phase2b_train.set_defaults(
        handler=lambda a: _cmd_train(a, task=PHASE2B_TASK, gnn_default=True)
    )

    phase2b_play = phase2b_cmds.add_parser("play", help="Play Phase B formation policy")
    _add_common_sim_args(phase2b_play)
    phase2b_play.add_argument("--checkpoint", type=str, default=None)
    phase2b_play.add_argument("--max_steps", type=int, default=None)
    phase2b_play.add_argument("--video", action="store_true", default=False)
    phase2b_play.add_argument("--video_length", type=int, default=None)
    phase2b_play.add_argument(
        "--rendering_mode",
        type=str,
        choices=["performance", "balanced", "quality"],
        default=None,
    )
    phase2b_play.add_argument("--video_codec", type=str, default=None)
    phase2b_play.add_argument("--video_bitrate", type=str, default=None)
    phase2b_play.add_argument("--video_preset", type=str, default=None)
    phase2b_play.add_argument("--video_ffmpeg_params", type=str, default=None)
    phase2b_play.add_argument("--hover_debug", action="store_true", default=False)
    phase2b_play.add_argument("--no_gnn", action="store_true", default=False)
    phase2b_play.set_defaults(
        handler=lambda a: _cmd_play(a, task=PHASE2B_TASK, gnn_default=True)
    )

    phase2b_eval = phase2b_cmds.add_parser(
        "eval", help="Evaluate Phase B formation checkpoint"
    )
    _add_common_sim_args(phase2b_eval)
    phase2b_eval.add_argument("--checkpoint", type=str, default=None)
    phase2b_eval.add_argument("--num_episodes", type=int, default=5)
    phase2b_eval.add_argument(
        "--no_gnn",
        action="store_true",
        default=False,
        help="Disable GNN policy (use MLP). Required when checkpoint was trained without --gnn.",
    )
    phase2b_eval.set_defaults(handler=_cmd_eval_phase2b)

    phase2b_assess = phase2b_cmds.add_parser(
        "assess",
        help="Run full post-training assessment for Phase B run",
    )
    phase2b_assess.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (logs/skrl/ggswarm_marl/<timestamp>)",
    )
    phase2b_assess.add_argument("--num_episodes", type=int, default=5)
    phase2b_assess.add_argument(
        "--progression",
        action="store_true",
        default=False,
        help="Also run checkpoint progression sweep (adds ~3 min).",
    )
    phase2b_assess.add_argument("--device", type=str, default=None)
    phase2b_assess.set_defaults(handler=lambda a: _cmd_assess(a, task=PHASE2B_TASK))

    phase2b_monitor = phase2b_cmds.add_parser(
        "monitor", help="Monitor Phase B TensorBoard logs"
    )
    phase2b_monitor.set_defaults(handler=lambda _a: _cmd_monitor(PHASE2_LOG_DIR))

    phase3 = families.add_parser("phase3", help="Phase 3 muscle refinement workflows")
    phase3_cmds = phase3.add_subparsers(dest="command")

    phase3_train = phase3_cmds.add_parser("train", help="Train phase3 policy (14-dim obs)")
    _add_common_sim_args(phase3_train)
    phase3_train.add_argument("--max_iterations", type=int, default=None)
    phase3_train.add_argument("--no_progress", action="store_true", default=False)
    phase3_train.add_argument("--progress_interval_s", type=float, default=10.0)
    phase3_train.add_argument("--eta_window_s", type=float, default=120.0)
    phase3_train.add_argument("--checkpoint", type=str, default=None)
    phase3_train.add_argument("--no_gnn", action="store_true", default=False)
    phase3_train.set_defaults(handler=lambda a: _cmd_train(a, task=PHASE3_TASK, gnn_default=True))

    phase3_play = phase3_cmds.add_parser("play", help="Play phase3 policy")
    _add_common_sim_args(phase3_play)
    phase3_play.add_argument("--checkpoint", type=str, default=None)
    phase3_play.add_argument("--max_steps", type=int, default=None)
    phase3_play.add_argument("--video", action="store_true", default=False)
    phase3_play.add_argument("--video_length", type=int, default=None)
    phase3_play.add_argument(
        "--rendering_mode",
        type=str,
        choices=["performance", "balanced", "quality"],
        default=None,
    )
    phase3_play.add_argument("--video_codec", type=str, default=None)
    phase3_play.add_argument("--video_bitrate", type=str, default=None)
    phase3_play.add_argument("--video_preset", type=str, default=None)
    phase3_play.add_argument("--video_ffmpeg_params", type=str, default=None)
    phase3_play.add_argument("--hover_debug", action="store_true", default=False)
    phase3_play.add_argument("--no_gnn", action="store_true", default=False)
    phase3_play.set_defaults(handler=lambda a: _cmd_play(a, task=PHASE3_TASK, gnn_default=True))

    phase3_eval = phase3_cmds.add_parser("eval", help="Evaluate phase3 checkpoint (P3 metrics)")
    _add_common_sim_args(phase3_eval)
    phase3_eval.add_argument("--task", type=str, default=None)
    phase3_eval.add_argument("--checkpoint", type=str, default=None)
    phase3_eval.add_argument("--num_episodes", type=int, default=10)
    phase3_eval.add_argument(
        "--kill_agent_step",
        type=int,
        default=50,
        help="Episode step to kill agent 0 for SwarmRaft gap-fill test (0 = disabled).",
    )
    phase3_eval.add_argument(
        "--no_gnn",
        action="store_true",
        default=False,
        help="Disable GNN policy (use MLP). Required when checkpoint was trained without --gnn.",
    )
    phase3_eval.set_defaults(handler=_cmd_eval_phase3)

    phase3_monitor = phase3_cmds.add_parser(
        "monitor", help="Monitor phase3 TensorBoard logs"
    )
    phase3_monitor.set_defaults(handler=lambda _a: _cmd_monitor(PHASE2_LOG_DIR))

    phase4 = families.add_parser("phase4", help="Phase 4 stress testing workflows")
    phase4_cmds = phase4.add_subparsers(dest="command")

    phase4_train = phase4_cmds.add_parser("train", help="Train phase4 policy (with agent loss)")
    _add_common_sim_args(phase4_train)
    phase4_train.add_argument("--max_iterations", type=int, default=None)
    phase4_train.add_argument("--no_progress", action="store_true", default=False)
    phase4_train.add_argument("--progress_interval_s", type=float, default=10.0)
    phase4_train.add_argument("--eta_window_s", type=float, default=120.0)
    phase4_train.add_argument("--checkpoint", type=str, default=None)
    phase4_train.add_argument("--no_gnn", action="store_true", default=False)
    phase4_train.set_defaults(handler=lambda a: _cmd_train(a, task=PHASE4_TASK, gnn_default=True))

    phase4_eval = phase4_cmds.add_parser(
        "eval", help="Evaluate phase4 checkpoint (Phase 3 metrics + kill test)"
    )
    _add_common_sim_args(phase4_eval)
    # Default to PHASE4_TASK so phase4 eval uses the correct task (Rule 16).
    # eval_phase3.py is intentionally reused here: Phase 4 uses the same metric
    # set (formation + survival + kill-test) as Phase 3 — only the env config differs.
    # intentional: phase4 eval reuses eval_phase3.py; Phase 4 task passed via default.
    phase4_eval.add_argument("--task", type=str, default=PHASE4_TASK)
    phase4_eval.add_argument("--checkpoint", type=str, default=None)
    phase4_eval.add_argument("--num_episodes", type=int, default=10)
    phase4_eval.add_argument("--kill_agent_step", type=int, default=50)
    phase4_eval.add_argument(
        "--no_gnn",
        action="store_true",
        default=False,
        help="Disable GNN policy (use MLP). Required when checkpoint was trained without --gnn.",
    )
    phase4_eval.set_defaults(handler=lambda a: _cmd_eval_phase3(a))

    phase4_bench = phase4_cmds.add_parser(
        "bench", help="Run scale benchmark (3→20 agents)"
    )
    _add_common_sim_args(phase4_bench)
    phase4_bench.add_argument("--checkpoint", type=str, required=True)
    phase4_bench.add_argument(
        "--agent_counts", nargs="+", type=int, default=[3, 5, 10, 15, 20]
    )
    phase4_bench.add_argument("--steps_per_run", type=int, default=500)
    phase4_bench.add_argument("--output_csv", type=str, default=None)
    phase4_bench.set_defaults(
        handler=lambda a: _run_command(
            [
                sys.executable,
                "scripts/bench_scale.py",
                "--checkpoint",
                a.checkpoint,
                "--headless",
                "--agent_counts",
                *[str(n) for n in a.agent_counts],
                "--steps_per_run",
                str(a.steps_per_run),
            ]
            + (["--output_csv", a.output_csv] if a.output_csv else [])
            + (["--device", a.device] if a.device else [])
            + (["--num_envs", str(a.num_envs)] if a.num_envs else [])
        )
    )

    phase4_monitor = phase4_cmds.add_parser("monitor", help="Monitor phase4 TensorBoard logs")
    phase4_monitor.set_defaults(handler=lambda _a: _cmd_monitor(PHASE2_LOG_DIR))

    phase5 = families.add_parser("phase5", help="Phase 5 showcase workflows")
    phase5_cmds = phase5.add_subparsers(dest="command")

    showcase_tasks = ["GGS-Showcase-v0", "GGS-ClutteredForest-v0", "GGS-UrbanCanyon-v0"]

    phase5_play = phase5_cmds.add_parser("play", help="Play showcase policy (HD rendering)")
    _add_common_sim_args(phase5_play)
    phase5_play.add_argument(
        "--task",
        type=str,
        default="GGS-ClutteredForest-v0",
        choices=showcase_tasks,
    )
    phase5_play.add_argument("--checkpoint", type=str, default=None)
    phase5_play.add_argument("--max_steps", type=int, default=None)
    phase5_play.add_argument("--video", action="store_true", default=False)
    phase5_play.add_argument("--video_length", type=int, default=None)
    phase5_play.add_argument(
        "--rendering_mode",
        type=str,
        choices=["performance", "balanced", "quality"],
        default="quality",
    )
    phase5_play.add_argument("--video_codec", type=str, default=None)
    phase5_play.add_argument("--video_bitrate", type=str, default=None)
    phase5_play.add_argument("--video_preset", type=str, default=None)
    phase5_play.add_argument("--video_ffmpeg_params", type=str, default=None)
    phase5_play.add_argument("--hover_debug", action="store_true", default=False)
    phase5_play.add_argument("--no_gnn", action="store_true", default=False)
    phase5_play.set_defaults(
        handler=lambda a: _cmd_play(a, task=a.task, gnn_default=True)
    )

    phase5_report = phase5_cmds.add_parser(
        "report", help="Compile Testing Report from Phase 3/4 results"
    )
    phase5_report.add_argument("--phase3_json", type=str, default=None)
    phase5_report.add_argument("--bench_csv", type=str, default=None)
    phase5_report.add_argument(
        "--output", type=str, default="docs/status/testing_report.md"
    )
    phase5_report.set_defaults(
        handler=lambda a: _run_command(
            [
                sys.executable,
                "scripts/compile_testing_report.py",
                "--output",
                a.output,
            ]
            + (["--phase3_json", a.phase3_json] if a.phase3_json else [])
            + (["--bench_csv", a.bench_csv] if a.bench_csv else [])
        )
    )

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
