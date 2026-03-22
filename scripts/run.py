"""Unified helper for hover, phase2, and debug workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure ggswarm_utils is importable when run.py is invoked from the repo root
# (scripts/ is not a package, so we add it explicitly).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ggswarm_utils.cli as cli  # noqa: E402
from ggswarm_utils.sim_helpers import PHASE_REGISTRY  # noqa: E402

# Task ID constants derived from PHASE_REGISTRY — single source of truth.
HOVER_TASK = PHASE_REGISTRY["hover"].task
PHASE2_TASK = PHASE_REGISTRY["2"].task
HOVER_STABILITY_TASK = PHASE_REGISTRY["2a"].task
PHASE2B_TASK = PHASE_REGISTRY["2b"].task
PHASE3_TASK = PHASE_REGISTRY["3"].task
PHASE4_TASK = PHASE_REGISTRY["4"].task

REPO_ROOT = Path(__file__).resolve().parents[1]
HOVER_LOG_DIR = REPO_ROOT / "logs" / "skrl" / "ggswarm_hover"
PHASE2_LOG_DIR = REPO_ROOT / "logs" / "skrl" / "ggswarm_marl"
LOCK_DIR = REPO_ROOT / "logs" / "skrl" / "locks"
PHASE2_TRAIN_LOCK_PATH = LOCK_DIR / "phase2_train.lock"


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Command handlers (thin wrappers — logic lives in ggswarm_utils.cli)
# ---------------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace, *, task: str, gnn_default: bool) -> None:
    lock_held = False
    try:
        if task == PHASE2_TASK:
            cli.acquire_single_instance_lock(PHASE2_TRAIN_LOCK_PATH)
            lock_held = True
        _run_command(cli.build_train_cmd(task, args, gnn_default=gnn_default))
    finally:
        if lock_held:
            cli.release_lock(PHASE2_TRAIN_LOCK_PATH)


def _cmd_play(args: argparse.Namespace, *, task: str, gnn_default: bool) -> None:
    log_root = PHASE2_LOG_DIR if task != HOVER_TASK else HOVER_LOG_DIR
    _run_command(cli.build_play_cmd(task, args, gnn_default=gnn_default, log_root=log_root))


def _cmd_eval(args: argparse.Namespace, *, phase: str) -> None:
    """Dispatch to scripts/eval.py with the given phase key.

    This is the single eval handler used by all family eval subcommands and
    the top-level ``run.py eval --phase`` command.  Phase 3/4 also forward
    ``--kill_agent_step`` if present.
    """
    kill_step = getattr(args, "kill_agent_step", None)
    extra: list[str] = []
    if kill_step is not None:
        extra = ["--kill_agent_step", str(kill_step)]
    _run_command(cli.build_eval_cmd(phase, args, extra_flags=extra or None))


def _cmd_assess(args: argparse.Namespace, *, task: str) -> None:
    assess_script = REPO_ROOT / "scripts" / "post_train_assess.py"
    _run_command(cli.build_assess_cmd(assess_script, task, args))


def _cmd_monitor(log_dir: Path) -> None:
    _run_command(["tensorboard", "--logdir", str(log_dir)])


def _cmd_debug_latest_checkpoint(args: argparse.Namespace) -> None:
    log_root = HOVER_LOG_DIR if args.family == "hover" else PHASE2_LOG_DIR
    checkpoint = cli.find_latest_checkpoint(log_root, prefer_best=not args.no_prefer_best)
    print(checkpoint)


def _cmd_debug_smoke(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/skrl/train.py",
        "--task", args.task,
        "--algorithm", "MAPPO",
        "--ml_framework", "torch",
        "--max_iterations", str(args.iterations),
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


# ---------------------------------------------------------------------------
# Argument builders
# ---------------------------------------------------------------------------


def _add_common_sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_agents", type=int, default=None)


def _add_train_args(p: argparse.ArgumentParser) -> None:
    """Common arguments for all train subparsers."""
    _add_common_sim_args(p)
    p.add_argument("--max_iterations", type=int, default=None)
    p.add_argument(
        "--action_telemetry_steps",
        type=int,
        default=None,
        help=(
            "Forward to train.py: log act_clamp / moment telemetry in TB for first N env steps "
            "(local diag only; omit on GCE)."
        ),
    )
    p.add_argument("--no_progress", action="store_true", default=False)
    p.add_argument("--progress_interval_s", type=float, default=10.0)
    p.add_argument("--eta_window_s", type=float, default=120.0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--gnn",
        action="store_true",
        default=False,
        help=(
            "Forward --gnn to train.py. Hover-stability / phase2 already default to GNN "
            "(this flag is a no-op then); use for clarity or with tasks that default to MLP."
        ),
    )
    p.add_argument("--no_gnn", action="store_true", default=False)


def _add_play_args(p: argparse.ArgumentParser) -> None:
    """Common arguments for all play subparsers."""
    _add_common_sim_args(p)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--video", action="store_true", default=False)
    p.add_argument("--video_length", type=int, default=None)
    p.add_argument(
        "--rendering_mode",
        type=str,
        choices=["performance", "balanced", "quality"],
        default=None,
    )
    p.add_argument("--video_codec", type=str, default=None)
    p.add_argument("--video_bitrate", type=str, default=None)
    p.add_argument("--video_preset", type=str, default=None)
    p.add_argument("--video_ffmpeg_params", type=str, default=None)
    p.add_argument("--hover_debug", action="store_true", default=False)
    p.add_argument(
        "--gnn",
        action="store_true",
        default=False,
        help="Forward --gnn to play.py (default-on families: no-op unless combined with --no_gnn).",
    )
    p.add_argument("--no_gnn", action="store_true", default=False)


def _add_eval_video_args(p: argparse.ArgumentParser) -> None:
    """Optional rgb_array recording for eval/assess (headless-safe)."""
    p.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record eval clip under run_dir/videos/eval (eval) or run_dir (assess).",
    )
    p.add_argument("--video_length", type=int, default=200)
    p.add_argument(
        "--rendering_mode",
        type=str,
        choices=["performance", "balanced", "quality"],
        default=None,
        help="Isaac rendering quality (default with --video: quality).",
    )
    p.add_argument("--video_codec", type=str, default=None)
    p.add_argument("--video_bitrate", type=str, default=None)
    p.add_argument("--video_preset", type=str, default=None)
    p.add_argument("--video_ffmpeg_params", type=str, default=None)


def _add_eval_args(p: argparse.ArgumentParser, default_episodes: int = 5) -> None:
    """Common arguments for all eval subparsers."""
    _add_common_sim_args(p)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--num_episodes", type=int, default=default_episodes)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Env RNG seed (default matches skrl_mappo_cfg.yaml).",
    )
    p.add_argument(
        "--no_gnn",
        action="store_true",
        default=False,
        help="Disable GNN policy (use MLP). Required when checkpoint was trained without --gnn.",
    )
    _add_eval_video_args(p)


def _add_assess_args(p: argparse.ArgumentParser) -> None:
    """Common arguments for all assess subparsers."""
    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (logs/skrl/ggswarm_marl/<timestamp>)",
    )
    p.add_argument("--num_episodes", type=int, default=5)
    p.add_argument(
        "--no_sync",
        action="store_true",
        default=False,
        help=(
            "Skip GCS pull when local checkpoints are missing. Default: pull that run "
            "from gs://gg-swarm-training-logs via gcloud (requires auth + gcloud on PATH "
            "or GGSWARM_GCLOUD)."
        ),
    )
    p.add_argument(
        "--progression",
        action="store_true",
        default=False,
        help="Also run checkpoint progression sweep (adds ~3 min).",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Eval seed forwarded to post_train_assess (default matches skrl_mappo_cfg.yaml).",
    )
    _add_eval_video_args(p)


# ---------------------------------------------------------------------------
# Family registration helper
# ---------------------------------------------------------------------------


def _register_standard_family(
    families: argparse._SubParsersAction,
    name: str,
    help_text: str,
    task: str,
    gnn_default: bool,
    log_dir: Path,
    eval_handler,
    eval_default_episodes: int = 5,
    assess_task: str | None = None,
) -> None:
    """Register train / play / eval / assess / monitor subcommands for a family."""
    fam = families.add_parser(name, help=help_text)
    cmds = fam.add_subparsers(dest="command")
    _task = task
    _gnn = gnn_default
    _assess_task = assess_task or task

    train = cmds.add_parser("train", help=f"Train {name} policy")
    _add_train_args(train)
    train.set_defaults(handler=lambda a, t=_task, g=_gnn: _cmd_train(a, task=t, gnn_default=g))

    play = cmds.add_parser("play", help=f"Play {name} policy")
    _add_play_args(play)
    play.set_defaults(handler=lambda a, t=_task, g=_gnn: _cmd_play(a, task=t, gnn_default=g))

    ev = cmds.add_parser("eval", help=f"Evaluate {name} checkpoint")
    _add_eval_args(ev, default_episodes=eval_default_episodes)
    ev.set_defaults(handler=eval_handler)

    assess = cmds.add_parser("assess", help=f"Full post-training assessment for {name}")
    _add_assess_args(assess)
    assess.set_defaults(handler=lambda a, t=_assess_task: _cmd_assess(a, task=t))

    monitor = cmds.add_parser("monitor", help=f"Monitor {name} TensorBoard logs")
    monitor.set_defaults(handler=lambda _a, ld=log_dir: _cmd_monitor(ld))


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:  # noqa: C901  # grandfathered — do not add branches
    parser = argparse.ArgumentParser(description="Unified ggSwarm run helper")
    families = parser.add_subparsers(dest="family")

    # --- top-level eval (single unified entry point) -------------------------
    eval_top = families.add_parser(
        "eval",
        help=(
            "Evaluate any checkpoint via scripts/eval.py --phase <X>. "
            "Aliases: hover eval, phase2 eval, hover-stability eval, etc."
        ),
    )
    eval_top.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=sorted(PHASE_REGISTRY),
        help="Phase to evaluate (see PHASE_REGISTRY for all options).",
    )
    _add_eval_args(eval_top)
    eval_top.add_argument(
        "--kill_agent_step",
        type=int,
        default=0,
        help="[Phase 3/4] Episode step at which to kill agent 0 (0 = disabled).",
    )
    eval_top.set_defaults(handler=lambda a: _cmd_eval(a, phase=a.phase))

    _register_standard_family(
        families,
        name="hover",
        help_text="Hover baseline workflows",
        task=HOVER_TASK,
        gnn_default=False,
        log_dir=HOVER_LOG_DIR,
        eval_handler=lambda a: _cmd_eval(a, phase="hover"),
        eval_default_episodes=10,
    )

    _register_standard_family(
        families,
        name="phase2",
        help_text="Phase 2 formation workflows",
        task=PHASE2_TASK,
        gnn_default=True,
        log_dir=PHASE2_LOG_DIR,
        eval_handler=lambda a: _cmd_eval(a, phase="2"),
    )

    _register_standard_family(
        families,
        name="hover-stability",
        help_text="Phase 2A hover-stability training (formation disabled)",
        task=HOVER_STABILITY_TASK,
        gnn_default=True,
        log_dir=PHASE2_LOG_DIR,
        eval_handler=lambda a: _cmd_eval(a, phase="2a"),
    )

    _register_standard_family(
        families,
        name="phase2b",
        help_text="Phase 2B formation training (resume from Phase 2A checkpoint)",
        task=PHASE2B_TASK,
        gnn_default=True,
        log_dir=PHASE2_LOG_DIR,
        eval_handler=lambda a: _cmd_eval(a, phase="2b"),
    )

    # --- phase3 ---------------------------------------------------------------
    phase3 = families.add_parser("phase3", help="Phase 3 muscle refinement workflows")
    phase3_cmds = phase3.add_subparsers(dest="command")

    p3_train = phase3_cmds.add_parser("train", help="Train phase3 policy (14-dim obs)")
    _add_train_args(p3_train)
    p3_train.set_defaults(handler=lambda a: _cmd_train(a, task=PHASE3_TASK, gnn_default=True))

    p3_play = phase3_cmds.add_parser("play", help="Play phase3 policy")
    _add_play_args(p3_play)
    p3_play.set_defaults(handler=lambda a: _cmd_play(a, task=PHASE3_TASK, gnn_default=True))

    p3_eval = phase3_cmds.add_parser("eval", help="Evaluate phase3 checkpoint (P3 metrics)")
    _add_common_sim_args(p3_eval)
    p3_eval.add_argument("--checkpoint", type=str, default=None)
    p3_eval.add_argument("--num_episodes", type=int, default=10)
    p3_eval.add_argument(
        "--kill_agent_step",
        type=int,
        default=50,
        help="Episode step to kill agent 0 for SwarmRaft gap-fill test (0 = disabled).",
    )
    p3_eval.add_argument("--no_gnn", action="store_true", default=False)
    _add_eval_video_args(p3_eval)
    p3_eval.set_defaults(handler=lambda a: _cmd_eval(a, phase="3"))

    p3_monitor = phase3_cmds.add_parser("monitor", help="Monitor phase3 TensorBoard logs")
    p3_monitor.set_defaults(handler=lambda _a: _cmd_monitor(PHASE2_LOG_DIR))

    # --- phase4 ---------------------------------------------------------------
    phase4 = families.add_parser("phase4", help="Phase 4 stress testing workflows")
    phase4_cmds = phase4.add_subparsers(dest="command")

    p4_train = phase4_cmds.add_parser("train", help="Train phase4 policy (with agent loss)")
    _add_train_args(p4_train)
    p4_train.set_defaults(handler=lambda a: _cmd_train(a, task=PHASE4_TASK, gnn_default=True))

    p4_eval = phase4_cmds.add_parser(
        "eval",
        help="Evaluate phase4 checkpoint (Phase 3 metrics + kill test)",
    )
    _add_common_sim_args(p4_eval)
    # intentional: phase4 eval reuses Phase3Collector via PHASE_REGISTRY (Rule 16).
    p4_eval.add_argument("--checkpoint", type=str, default=None)
    p4_eval.add_argument("--num_episodes", type=int, default=10)
    p4_eval.add_argument("--kill_agent_step", type=int, default=50)
    p4_eval.add_argument("--no_gnn", action="store_true", default=False)
    _add_eval_video_args(p4_eval)
    p4_eval.set_defaults(handler=lambda a: _cmd_eval(a, phase="4"))

    p4_bench = phase4_cmds.add_parser("bench", help="Run scale benchmark (3→20 agents)")
    _add_common_sim_args(p4_bench)
    p4_bench.add_argument("--checkpoint", type=str, required=True)
    p4_bench.add_argument("--agent_counts", nargs="+", type=int, default=[3, 5, 10, 15, 20])
    p4_bench.add_argument("--steps_per_run", type=int, default=500)
    p4_bench.add_argument("--output_csv", type=str, default=None)
    p4_bench.set_defaults(
        handler=lambda a: _run_command(
            [
                sys.executable, "scripts/bench_scale.py",
                "--checkpoint", a.checkpoint,
                "--headless",
                "--agent_counts", *[str(n) for n in a.agent_counts],
                "--steps_per_run", str(a.steps_per_run),
            ]
            + (["--output_csv", a.output_csv] if a.output_csv else [])
            + (["--device", a.device] if a.device else [])
            + (["--num_envs", str(a.num_envs)] if a.num_envs else [])
        )
    )

    p4_monitor = phase4_cmds.add_parser("monitor", help="Monitor phase4 TensorBoard logs")
    p4_monitor.set_defaults(handler=lambda _a: _cmd_monitor(PHASE2_LOG_DIR))

    # --- phase5 ---------------------------------------------------------------
    phase5 = families.add_parser("phase5", help="Phase 5 showcase workflows")
    phase5_cmds = phase5.add_subparsers(dest="command")

    showcase_tasks = ["GGS-Showcase-v0", "GGS-ClutteredForest-v0", "GGS-UrbanCanyon-v0"]

    p5_play = phase5_cmds.add_parser("play", help="Play showcase policy (HD rendering)")
    _add_play_args(p5_play)
    p5_play.add_argument(
        "--task", type=str, default="GGS-ClutteredForest-v0", choices=showcase_tasks
    )
    p5_play.set_defaults(
        rendering_mode="quality",
        handler=lambda a: _cmd_play(a, task=a.task, gnn_default=True),
    )

    p5_report = phase5_cmds.add_parser(
        "report", help="Compile Testing Report from Phase 3/4 results"
    )
    p5_report.add_argument("--phase3_json", type=str, default=None)
    p5_report.add_argument("--bench_csv", type=str, default=None)
    p5_report.add_argument("--output", type=str, default="docs/status/testing_report.md")
    p5_report.set_defaults(
        handler=lambda a: _run_command(
            [sys.executable, "scripts/compile_testing_report.py", "--output", a.output]
            + (["--phase3_json", a.phase3_json] if a.phase3_json else [])
            + (["--bench_csv", a.bench_csv] if a.bench_csv else [])
        )
    )

    # --- debug ----------------------------------------------------------------
    debug = families.add_parser("debug", help="Debug utilities")
    debug_cmds = debug.add_subparsers(dest="command")

    smoke = debug_cmds.add_parser("smoke", help="Run a very short training smoke test")
    _add_common_sim_args(smoke)
    smoke.add_argument("--task", type=str, default=HOVER_TASK)
    smoke.add_argument("--iterations", type=int, default=1)
    smoke.add_argument("--gnn", action="store_true", default=False)
    smoke.set_defaults(handler=_cmd_debug_smoke)

    latest = debug_cmds.add_parser("latest-checkpoint", help="Print latest checkpoint path")
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
