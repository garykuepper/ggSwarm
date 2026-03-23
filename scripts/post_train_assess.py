"""Single-script post-training assessment pipeline.

Runs in one Python process (no subprocess spawning):
  Step 0 - GCS sync       : pull run artifacts from GCS bucket (optional)
  Step 1 - Convergence    : TensorBoard analysis (no Isaac Lab needed)
  Step 2 - Eval           : Isaac Lab simulation via run_eval() from eval_runner
  Step 3 - Scorecard      : PASS / WARN / FAIL per metric
  Step 4 - Report         : write assess_metrics.json + assess_report.md to run_dir

Usage (run from repo root):
    .\\env_isaaclab\\Scripts\\python.exe scripts/post_train_assess.py ^
        --run_dir logs/skrl/ggswarm_marl/<timestamp>_mappo_torch ^
        --task Template-GGSwarm-Marl-HoverStability-v0 ^
        --num_episodes 5 ^
        --headless

To skip GCS sync when data is already local:
    ... --no_sync

With optional headless video (MP4 under run_dir/videos/eval/):
    ... --video [--video_length 200]

Exits 0 if overall verdict is PASS or WARN, 1 if FAIL.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path

# Ensure scripts/ is on the path so ggswarm_utils resolves correctly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ggswarm_utils.gcs_sync import DEFAULT_GCS_BUCKET, has_local_data, sync_from_gcs
from ggswarm_utils.scorecard import format_run_history_row, print_scorecard, write_report
from ggswarm_utils.sim_helpers import PHASE_REGISTRY, phase_from_task


# ---------------------------------------------------------------------------
# Argparse (must come before AppLauncher to avoid conflicts)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single-script post-training assessment: sync + convergence + eval + scorecard."
        )
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Local path to training run directory (logs/skrl/ggswarm_marl/<timestamp>_mappo_torch).",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Gym task ID (e.g. Template-GGSwarm-Marl-HoverStability-v0).",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Eval episodes (default 5).",
    )
    parser.add_argument(
        "--gnn",
        action="store_true",
        default=True,
        help="Use GGSwarmGNNPolicy (GATv2). Default: True.",
    )
    parser.add_argument(
        "--no_sync",
        action="store_true",
        default=False,
        help="Skip GCS sync (use when data is already local).",
    )
    parser.add_argument(
        "--gcs_bucket",
        type=str,
        default=DEFAULT_GCS_BUCKET,
        help=f"GCS bucket URI (default: {DEFAULT_GCS_BUCKET}).",
    )
    parser.add_argument(
        "--airborne_height_margin",
        type=float,
        default=0.2,
        help="Height margin above min_height to count as airborne (default 0.2 m).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Eval RNG seed (default matches skrl_mappo_cfg.yaml for train–eval parity).",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help=(
            "Record eval clip under run_dir/videos/eval (headless rgb_array; "
            "enables cameras)."
        ),
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=200,
        help="Eval video length in environment steps (default 200).",
    )
    parser.add_argument("--video_codec", type=str, default=None)
    parser.add_argument("--video_bitrate", type=str, default=None)
    parser.add_argument("--video_preset", type=str, default=None)
    parser.add_argument(
        "--video_ffmpeg_params",
        type=str,
        default=None,
        help='Extra ffmpeg params as one string, e.g. "-cq 18 -rc vbr".',
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Full post-training assessment pipeline."""
    from isaaclab.app import AppLauncher  # noqa: PLC0415

    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    # Hydra reads sys.argv directly — strip our custom args so it only sees overrides.
    sys.argv = [sys.argv[0]] + hydra_args

    run_dir = Path(args_cli.run_dir).resolve()

    print(f"\n{'='*70}")
    print(f"  POST-TRAINING ASSESSMENT: {run_dir.name}")
    print(f"  Task: {args_cli.task}")
    print(f"  Episodes: {args_cli.num_episodes}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Step 0: GCS Sync
    # ------------------------------------------------------------------
    if args_cli.no_sync:
        print("[SYNC] --no_sync specified; skipping GCS sync.")
    elif has_local_data(run_dir):
        print(f"[SYNC] Local data found at {run_dir}; skipping GCS sync.")
        print("[SYNC]   Pass --no_sync to suppress this check, or delete the directory to force re-sync.")
    else:
        print("[SYNC] No local data found. Syncing from GCS...")
        try:
            sync_from_gcs(run_dir, args_cli.gcs_bucket)
        except Exception as exc:
            print(f"[SYNC] ERROR: GCS sync failed: {exc}", file=sys.stderr)
            if not has_local_data(run_dir):
                print(
                    "[SYNC] No local checkpoints after failed sync. "
                    "Fix gcloud auth / PATH (or run pull_results_from_gcs.py --latest 1), "
                    "then retry.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print("[SYNC] Partial local data present; continuing.")

    if not run_dir.exists():
        print(f"[ERROR] --run_dir not found after sync attempt: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: Convergence check (pure TensorBoard — before Isaac Lab boot)
    # ------------------------------------------------------------------
    print("\nCONVERGENCE CHECK")
    print("-" * 40)
    convergence: dict = {}
    try:
        from cloud.check_convergence import analyze_convergence  # noqa: PLC0415

        convergence = analyze_convergence(str(run_dir))
        print(
            f"  Peak reward:     {convergence['peak_reward_value']:>10.2f}"
            f"  @ step {convergence['peak_reward_step']:>9,}"
        )
        collapse = convergence.get("entropy_collapse_step")
        if collapse:
            print(
                f"  Entropy collapse:{collapse:>10,}"
                f"  (reward={convergence.get('plateau_reward_value', 0):.2f})"
            )
        else:
            print("  Entropy collapse: Not detected (policy may still be exploring)")
        print(
            f"  Final reward:    {convergence['final_reward']:>10.2f}"
            f"  @ step {convergence['final_step']:>9,}"
        )
        print(f"  Rec. budget:     {convergence['recommended_training_length']:>10,}  steps")
    except Exception as exc:
        print(f"  [WARN] Convergence check failed: {exc}")
        convergence = {}

    # ------------------------------------------------------------------
    # Step 1b: TB scalar diagnostics (same TFEvents, no Isaac Lab needed)
    # ------------------------------------------------------------------
    tb_diagnostics: list[dict] = []
    try:
        from cloud.check_convergence import extract_scalar_summary  # noqa: PLC0415

        tb_diagnostics = extract_scalar_summary(str(run_dir))
        if tb_diagnostics:
            print(f"\n  TB diagnostics: {len(tb_diagnostics)} scalar(s) extracted")
            for s in tb_diagnostics:
                print(
                    f"    {s['tag']}: {s['first_value']:.4f} → {s['last_value']:.4f}"
                    f"  ({s['first_step'] // 1000}k → {s['last_step'] // 1000}k)"
                )
    except Exception as exc:
        print(f"  [WARN] TB diagnostics extraction failed: {exc}")

    # ------------------------------------------------------------------
    # Boot Isaac Lab (heavy — after convergence so output is visible early)
    # ------------------------------------------------------------------
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Register task modules (must happen after AppLauncher)
    import ggSwarm.tasks  # noqa: F401, PLC0415
    import isaaclab_tasks  # noqa: F401, PLC0415

    # ------------------------------------------------------------------
    # Step 2: Eval via unified run_eval()
    # ------------------------------------------------------------------
    print("\nEVALUATION")
    print("-" * 40)

    best_ckpt = run_dir / "checkpoints" / "best_agent.pt"
    if not best_ckpt.exists():
        ckpts = sorted((run_dir / "checkpoints").glob("agent_*.pt"))
        if ckpts:
            best_ckpt = ckpts[-1]
            print(f"  [WARN] best_agent.pt not found; using {best_ckpt.name}")
        else:
            print("[ERROR] No checkpoints found.", file=sys.stderr)
            sys.exit(1)

    print(f"  checkpoint : {best_ckpt.name}")
    print(f"  task       : {args_cli.task}")

    # Resolve phase from task ID for collector dispatch
    phase_key = phase_from_task(args_cli.task)
    if phase_key is None:
        # Unknown task — fall back to Phase2Collector (most general)
        print(
            f"  [WARN] Task '{args_cli.task}' not in PHASE_REGISTRY; "
            "using Phase2Collector metrics."
        )
        phase_key = "2"
    phase_cfg = PHASE_REGISTRY[phase_key]

    # Instantiate the correct collector
    module_path, cls_name = phase_cfg.collector_cls.rsplit(".", 1)
    collector_module = importlib.import_module(module_path)
    collector_cls = getattr(collector_module, cls_name)

    # Phase2Collector accepts airborne_height_margin
    sig = inspect.signature(collector_cls.__init__)
    if "airborne_height_margin" in sig.parameters:
        collector = collector_cls(airborne_height_margin=args_cli.airborne_height_margin)
    else:
        collector = collector_cls()

    from ggswarm_utils.eval_runner import run_eval  # noqa: PLC0415

    metrics = run_eval(
        task=args_cli.task,
        simulation_app=simulation_app,
        collector=collector,
        checkpoint=str(best_ckpt),
        num_episodes=args_cli.num_episodes,
        gnn=args_cli.gnn,
        seed=args_cli.seed,
        device=getattr(args_cli, "device", None),
        num_envs=None,
        num_agents=None,
        run_dir=run_dir,
        video=args_cli.video,
        video_length=args_cli.video_length,
        video_codec=args_cli.video_codec,
        video_bitrate=args_cli.video_bitrate,
        video_preset=args_cli.video_preset,
        video_ffmpeg_params=args_cli.video_ffmpeg_params,
    )

    if not metrics:
        print("[ERROR] Eval produced no metrics — simulation may have ended early.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Write assess_metrics.json
    # ------------------------------------------------------------------
    json_out = run_dir / "assess_metrics.json"
    with json_out.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\n[METRICS] Written to: {json_out}")

    # ------------------------------------------------------------------
    # Step 3: Scorecard
    # ------------------------------------------------------------------
    overall = print_scorecard(run_dir.name, metrics)

    # ------------------------------------------------------------------
    # Step 4: Markdown report
    # ------------------------------------------------------------------
    write_report(
        run_dir=run_dir,
        metrics=metrics,
        convergence=convergence,
        overall=overall,
        task=args_cli.task,
        num_episodes=args_cli.num_episodes,
        checkpoint_name=best_ckpt.name,
        tb_diagnostics=tb_diagnostics,
    )

    # ------------------------------------------------------------------
    # Ready-to-paste run_history row
    # ------------------------------------------------------------------
    row = format_run_history_row(run_dir.name, metrics, overall)
    print(f"\n{'─' * 70}")
    print("  Copy to docs/status/run_history.md:")
    print(f"  {row}")
    print(f"{'─' * 70}\n")

    simulation_app.close()
    sys.exit(0 if overall != "FAIL" else 1)


if __name__ == "__main__":
    main()
