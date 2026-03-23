"""Trajectory diagnostic: run a checkpoint eval and save altitude, XY, and attitude plots.

Generates per-episode plots saved to <run_dir>/trajectories/:
  - altitude_trace_ep{N}.png : Z position vs time; shows crash-reset saw-tooth vs genuine hover
  - xy_trace_ep{N}.png       : top-down XY path; shows horizontal drift vs hover-in-place
  - attitude_trace_ep{N}.png : roll and pitch in degrees vs time (if quaternion util available)

Usage (PowerShell, repo root):
    .\\env_isaaclab\\Scripts\\python.exe scripts\\plot_trajectories.py `
        --run_dir logs\\skrl\\ggswarm_marl\\<timestamp>_mappo_torch `
        --num_episodes 3 `
        --num_envs 1 `
        --headless

Optional flags:
    --checkpoint best_agent.pt   (default; filename under <run_dir>/checkpoints/)
    --seed 42                    (default; matches training seed)
    --task Template-GGSwarm-Marl-HoverStability-v0   (default)
    --no_gnn                     (omit GATv2 policy injection)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# matplotlib must be available before Isaac Lab boot so we can fail fast.
try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive, headless-safe backend
    import matplotlib.pyplot as plt
except ImportError:
    print(
        "[plot_trajectories] ERROR: matplotlib is required. "
        "Install it with:  pip install matplotlib",
        file=sys.stderr,
    )
    sys.exit(1)

import torch

# Ensure scripts/ is on sys.path so ggswarm_utils resolves without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from isaaclab.app import AppLauncher  # noqa: E402


# ---------------------------------------------------------------------------
# Argparse (before AppLauncher — AppLauncher.add_app_launcher_args appends to it)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (logs/skrl/ggswarm_marl/<timestamp>_mappo_torch).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint filename under <run_dir>/checkpoints/ (default: best_agent.pt).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Template-GGSwarm-Marl-HoverStability-v0",
        help="Gym task ID (default: Template-GGSwarm-Marl-HoverStability-v0).",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=3,
        help="Number of eval episodes to record (default: 3).",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Parallel envs to run (default: 1; keep small to bound memory).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Eval RNG seed (default: 42, matches training seed).",
    )
    parser.add_argument(
        "--no_gnn",
        action="store_true",
        default=False,
        help="Disable GATv2 GNN policy injection (use MLP policy instead).",
    )
    return parser


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_altitude(
    traj_steps: list[torch.Tensor],
    ep: int,
    out_dir: Path,
    min_height: float,
    max_height: float,
    spawn_z_min: float,
    spawn_z_max: float,
) -> None:
    """Z altitude vs episode step for each agent in env 0."""
    z = torch.stack([s[0, :, 2] for s in traj_steps])  # [T, num_agents]
    T, A = z.shape
    steps = list(range(T))
    airborne_z = min_height + 0.2

    fig, ax = plt.subplots(figsize=(13, 4))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for a in range(A):
        ax.plot(steps, z[:, a].numpy(), color=colors[a % len(colors)], linewidth=1.2, label=f"drone_{a}")

    ax.axhline(min_height, color="red", linestyle="--", linewidth=1.0, label=f"min_height {min_height} m (crash)")
    ax.axhline(airborne_z, color="darkorange", linestyle="--", linewidth=1.0, label=f"airborne {airborne_z} m (gate)")
    ax.axhspan(spawn_z_min, spawn_z_max, alpha=0.10, color="green", label=f"spawn band {spawn_z_min}–{spawn_z_max} m")
    ax.axhline(max_height, color="grey", linestyle="--", linewidth=0.8, label=f"max_height {max_height} m (ceiling)")

    ax.set_xlabel("Episode step")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Altitude trace — Episode {ep}  ({T} steps, {A} agents)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(-0.2, max_height + 0.3)

    out = out_dir / f"altitude_trace_ep{ep}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  [altitude] {out.name}")


def _plot_xy(
    traj_steps: list[torch.Tensor],
    ep: int,
    out_dir: Path,
) -> None:
    """Top-down XY trajectory for each agent in env 0; cross marks spawn position."""
    xy = torch.stack([s[0, :, :2] for s in traj_steps])  # [T, num_agents, 2]
    T, A, _ = xy.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for a in range(A):
        x = xy[:, a, 0].numpy()
        y = xy[:, a, 1].numpy()
        ax.plot(x, y, color=colors[a % len(colors)], alpha=0.75, linewidth=1.1, label=f"drone_{a}")
        # Mark spawn position (first recorded step)
        ax.plot(x[0], y[0], "x", color=colors[a % len(colors)], markersize=11, markeredgewidth=2.0)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"XY trace — Episode {ep}  (cross = spawn)  [{T} steps]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    out = out_dir / f"xy_trace_ep{ep}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  [xy]       {out.name}")


def _plot_attitude(
    roll_steps: list[torch.Tensor],
    pitch_steps: list[torch.Tensor],
    ep: int,
    out_dir: Path,
) -> None:
    """Roll and pitch in degrees vs episode step for each agent in env 0."""
    roll = torch.stack([s[0] for s in roll_steps])   # [T, num_agents]
    pitch = torch.stack([s[0] for s in pitch_steps])  # [T, num_agents]
    T, A = roll.shape
    steps = list(range(T))

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    gate_deg = 15.0

    for a in range(A):
        axes[0].plot(steps, roll[:, a].numpy(), color=colors[a % len(colors)], linewidth=1.1, label=f"drone_{a}")
        axes[1].plot(steps, pitch[:, a].numpy(), color=colors[a % len(colors)], linewidth=1.1, label=f"drone_{a}")

    for ax, name in [(axes[0], "Roll (°)"), (axes[1], "Pitch (°)")]:
        ax.axhline(gate_deg, color="darkorange", linestyle="--", linewidth=0.9, label=f"gate ±{gate_deg}°")
        ax.axhline(-gate_deg, color="darkorange", linestyle="--", linewidth=0.9)
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.6)
        ax.set_ylabel(name)
        ax.legend(loc="upper right", fontsize=8)

    axes[1].set_xlabel("Episode step")
    fig.suptitle(f"Attitude trace — Episode {ep}  ({T} steps, {A} agents)")

    out = out_dir / f"attitude_trace_ep{ep}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  [attitude] {out.name}")


# ---------------------------------------------------------------------------
# TrajectoryCollector
# ---------------------------------------------------------------------------


def _make_trajectory_collector(phase2_cls, track_envs: int, euler_fn):
    """Return a TrajectoryCollector class (Phase2Collector subclass)."""

    class TrajectoryCollector(phase2_cls):
        """Phase2Collector that also accumulates per-step pos_w and attitude."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._track = track_envs
            # episode_num → list of [track_envs, num_agents, 3] cpu tensors
            self.pos_traj: dict[int, list[torch.Tensor]] = {}
            # episode_num → list of [track_envs, num_agents] cpu tensors (degrees)
            self.roll_traj: dict[int, list[torch.Tensor]] = {}
            self.pitch_traj: dict[int, list[torch.Tensor]] = {}

        def on_step(self, *, base_env, obs, step, episode_step, episode_num):
            super().on_step(
                base_env=base_env,
                obs=obs,
                step=step,
                episode_step=episode_step,
                episode_num=episode_num,
            )
            n_env = base_env.num_envs
            n_ag = base_env.cfg.num_agents
            t = min(self._track, n_env)

            pos_w = base_env.robot.data.root_pos_w.view(n_env, n_ag, 3)

            if episode_num not in self.pos_traj:
                self.pos_traj[episode_num] = []
                self.roll_traj[episode_num] = []
                self.pitch_traj[episode_num] = []

            self.pos_traj[episode_num].append(pos_w[:t].detach().cpu().clone())

            if euler_fn is not None:
                quat_w = base_env.robot.data.root_quat_w.view(n_env, n_ag, 4)
                # euler_xyz_from_quat returns (roll, pitch, yaw) tuple of [N] tensors
                roll_rad, pitch_rad, _ = euler_fn(quat_w[:t].reshape(-1, 4))
                self.roll_traj[episode_num].append(
                    torch.rad2deg(roll_rad.reshape(t, n_ag)).detach().cpu().clone()
                )
                self.pitch_traj[episode_num].append(
                    torch.rad2deg(pitch_rad.reshape(t, n_ag)).detach().cpu().clone()
                )

    return TrajectoryCollector


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    run_dir = Path(args_cli.run_dir).resolve()
    if not run_dir.exists():
        print(f"[ERROR] --run_dir not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = run_dir / "trajectories"
    out_dir.mkdir(exist_ok=True)

    # Resolve checkpoint -------------------------------------------------------
    ckpt_name = args_cli.checkpoint or "best_agent.pt"
    checkpoint = run_dir / "checkpoints" / ckpt_name
    if not checkpoint.exists():
        ckpts = sorted((run_dir / "checkpoints").glob("agent_*.pt"))
        if not ckpts:
            print(f"[ERROR] No checkpoints found in {run_dir / 'checkpoints'}", file=sys.stderr)
            sys.exit(1)
        checkpoint = ckpts[-1]
        print(f"[WARN] {ckpt_name} not found; using {checkpoint.name}")

    print(f"\n{'='*65}")
    print(f"  plot_trajectories")
    print(f"  run_dir   : {run_dir.name}")
    print(f"  checkpoint: {checkpoint.name}")
    print(f"  episodes  : {args_cli.num_episodes}")
    print(f"  num_envs  : {args_cli.num_envs}")
    print(f"  output    : {out_dir}")
    print(f"{'='*65}\n")

    # Boot Isaac Lab -----------------------------------------------------------
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import ggSwarm.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401

    from ggswarm_utils.collector_factory import make_collector
    from ggswarm_utils.composite_collector import CompositeCollector
    from ggswarm_utils.eval_runner import run_eval
    from ggswarm_utils.sim_helpers import PHASE_REGISTRY
    from ggswarm_utils.trajectory_collector import TrajectoryDataCollector
    from ggswarm_utils.trajectory_plots import generate_all_trajectory_plots

    euler_fn = None
    try:
        from isaaclab.utils.math import euler_xyz_from_quat

        euler_fn = euler_xyz_from_quat
    except Exception:
        print("[WARN] euler_xyz_from_quat not available; attitude plots will be skipped.")

    # Build and run collector --------------------------------------------------
    phase_cfg = PHASE_REGISTRY["2"]
    primary = make_collector(phase_cfg, airborne_height_margin=0.2)
    traj_collector = TrajectoryDataCollector(track_envs=args_cli.num_envs, euler_fn=euler_fn)
    collector = CompositeCollector(primary, traj_collector)

    metrics = run_eval(
        task=args_cli.task,
        simulation_app=simulation_app,
        collector=collector,
        checkpoint=str(checkpoint),
        num_episodes=args_cli.num_episodes,
        gnn=not args_cli.no_gnn,
        seed=args_cli.seed,
        device=None,
        num_envs=args_cli.num_envs,
        num_agents=None,
        run_dir=run_dir,
    )

    # Print metrics summary ----------------------------------------------------
    print("\n[plot_trajectories] Eval metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:<38} {v:.4f}")

    # Generate plots -----------------------------------------------------------
    MIN_HEIGHT = 0.1
    MAX_HEIGHT = 3.0
    SPAWN_Z_MIN = 0.65
    SPAWN_Z_MAX = 1.65

    print(f"\n[plot_trajectories] Writing plots to {out_dir} ...")
    created = generate_all_trajectory_plots(
        traj_collector,
        out_dir,
        min_height=MIN_HEIGHT,
        max_height=MAX_HEIGHT,
        spawn_z_min=SPAWN_Z_MIN,
        spawn_z_max=SPAWN_Z_MAX,
    )

    print(f"\n[plot_trajectories] Done. {len(created)} plot(s) saved to:")
    print(f"  {out_dir}")

    simulation_app.close()


if __name__ == "__main__":
    main()
