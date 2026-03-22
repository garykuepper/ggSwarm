"""Unified evaluation entry point for all ggSwarm phases.

Dispatches to the correct phase-specific collector via PHASE_REGISTRY.

Usage:
    python scripts/eval.py --phase 2a --checkpoint logs/.../best_agent.pt
    python scripts/eval.py --phase hover --num_episodes 10 --headless
    python scripts/eval.py --phase 3 --kill_agent_step 50 --headless
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# Ensure ggswarm_utils is importable when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ggswarm_utils.sim_helpers import PHASE_REGISTRY, resolve_phase  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified eval entry point for all ggSwarm phases."
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=sorted(PHASE_REGISTRY),
        help=(
            "Phase to evaluate.  One of: "
            + ", ".join(sorted(PHASE_REGISTRY))
            + ".  Maps to a task ID and collector via PHASE_REGISTRY."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (.pt).  Auto-discovers latest if omitted.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes (defaults to phase default).",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Override number of parallel environments.",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=None,
        help="Override number of agents per environment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="RNG seed for environment resets.",
    )
    parser.add_argument(
        "--gnn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable/disable GGSwarmGNNPolicy.  Defaults to the phase's "
            "gnn_default from PHASE_REGISTRY."
        ),
    )
    parser.add_argument(
        "--kill_agent_step",
        type=int,
        default=0,
        help="[Phase 3/4 only] Episode step at which to kill agent 0 (0 = disabled).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="If set, write the final metrics dict as JSON to this path.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to the training run directory (used for log naming).",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    """Parse arguments, boot Isaac Lab, dispatch to phase collector, print results."""
    parser = _build_parser()
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    phase_cfg = resolve_phase(args_cli.phase)

    # Respect explicit --gnn / --no-gnn; fall back to registry default
    use_gnn: bool = phase_cfg.gnn_default if args_cli.gnn is None else args_cli.gnn
    num_episodes: int = (
        phase_cfg.default_episodes
        if args_cli.num_episodes is None
        else args_cli.num_episodes
    )

    # Boot AppLauncher *before* any Isaac Lab imports
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Register task modules (must happen after AppLauncher)
    import ggSwarm.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401

    # Lazy-load the collector class from its dotted path
    module_path, cls_name = phase_cfg.collector_cls.rsplit(".", 1)
    collector_module = importlib.import_module(module_path)
    collector_cls = getattr(collector_module, cls_name)

    # Phase 3/4 collectors accept kill_agent_step
    if args_cli.kill_agent_step > 0 and hasattr(collector_cls, "__init__"):
        import inspect  # noqa: PLC0415
        sig = inspect.signature(collector_cls.__init__)
        if "kill_agent_step" in sig.parameters:
            collector = collector_cls(kill_agent_step=args_cli.kill_agent_step)
        else:
            collector = collector_cls()
    else:
        collector = collector_cls()

    from ggswarm_utils.eval_runner import run_eval  # noqa: PLC0415

    metrics = run_eval(
        task=phase_cfg.task,
        simulation_app=simulation_app,
        collector=collector,
        checkpoint=args_cli.checkpoint,
        num_episodes=num_episodes,
        gnn=use_gnn,
        seed=args_cli.seed,
        device=getattr(args_cli, "device", None),
        num_envs=args_cli.num_envs,
        num_agents=args_cli.num_agents,
        run_dir=args_cli.run_dir,
    )

    # Write JSON output if requested
    if args_cli.output_json is not None:
        out_path = Path(args_cli.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"[EVAL] Metrics written to: {out_path}")

    # Print summary
    print(f"\n=== Phase {args_cli.phase.upper()} Evaluation Summary ===")
    print(f"checkpoint : {args_cli.checkpoint or '(auto-discovered)'}")
    print(f"task       : {phase_cfg.task}")
    print(f"episodes   : {num_episodes}")
    print()
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.6f}")
        else:
            print(f"  {key}: {val}")

    simulation_app.close()


if __name__ == "__main__":
    main()
