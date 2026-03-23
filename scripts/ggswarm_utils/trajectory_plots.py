"""Pure plotting functions for trajectory diagnostic visualisation.

Extracted from ``plot_trajectories.py`` so that ``post_train_assess.py`` and
any other caller can generate the same plots without duplicating code.

Public API:
    plot_altitude(traj_steps, ep, out_dir, min_height, max_height, spawn_z_min, spawn_z_max)
    plot_xy(traj_steps, ep, out_dir)
    plot_attitude(roll_steps, pitch_steps, ep, out_dir)
    generate_all_trajectory_plots(collector, out_dir, ...)
"""

from __future__ import annotations

from pathlib import Path

import torch


def _ensure_mpl():
    """Import matplotlib with Agg backend; deferred to avoid top-level import."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    return plt


def plot_altitude(
    traj_steps: list[torch.Tensor],
    ep: int,
    out_dir: Path,
    min_height: float,
    max_height: float,
    spawn_z_min: float,
    spawn_z_max: float,
) -> Path:
    """Z altitude vs episode step for each agent in env 0.

    Returns the path to the saved PNG.
    """
    plt = _ensure_mpl()

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
    ax.axhspan(spawn_z_min, spawn_z_max, alpha=0.10, color="green", label=f"spawn band {spawn_z_min}-{spawn_z_max} m")
    ax.axhline(max_height, color="grey", linestyle="--", linewidth=0.8, label=f"max_height {max_height} m (ceiling)")

    ax.set_xlabel("Episode step")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Altitude trace - Episode {ep}  ({T} steps, {A} agents)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(-0.2, max_height + 0.3)

    out = out_dir / f"altitude_trace_ep{ep}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_xy(
    traj_steps: list[torch.Tensor],
    ep: int,
    out_dir: Path,
) -> Path:
    """Top-down XY trajectory for each agent in env 0; cross marks spawn.

    Returns the path to the saved PNG.
    """
    plt = _ensure_mpl()

    xy = torch.stack([s[0, :, :2] for s in traj_steps])  # [T, num_agents, 2]
    T, A, _ = xy.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for a in range(A):
        x = xy[:, a, 0].numpy()
        y = xy[:, a, 1].numpy()
        ax.plot(x, y, color=colors[a % len(colors)], alpha=0.75, linewidth=1.1, label=f"drone_{a}")
        ax.plot(x[0], y[0], "x", color=colors[a % len(colors)], markersize=11, markeredgewidth=2.0)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"XY trace - Episode {ep}  (cross = spawn)  [{T} steps]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")

    out = out_dir / f"xy_trace_ep{ep}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_attitude(
    roll_steps: list[torch.Tensor],
    pitch_steps: list[torch.Tensor],
    ep: int,
    out_dir: Path,
) -> Path:
    """Roll and pitch in degrees vs episode step for each agent in env 0.

    Returns the path to the saved PNG.
    """
    plt = _ensure_mpl()

    roll = torch.stack([s[0] for s in roll_steps])    # [T, num_agents]
    pitch = torch.stack([s[0] for s in pitch_steps])   # [T, num_agents]
    T, A = roll.shape
    steps = list(range(T))

    fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    gate_deg = 15.0

    for a in range(A):
        axes[0].plot(steps, roll[:, a].numpy(), color=colors[a % len(colors)], linewidth=1.1, label=f"drone_{a}")
        axes[1].plot(steps, pitch[:, a].numpy(), color=colors[a % len(colors)], linewidth=1.1, label=f"drone_{a}")

    for ax, name in [(axes[0], "Roll (deg)"), (axes[1], "Pitch (deg)")]:
        ax.axhline(gate_deg, color="darkorange", linestyle="--", linewidth=0.9, label=f"gate +/-{gate_deg} deg")
        ax.axhline(-gate_deg, color="darkorange", linestyle="--", linewidth=0.9)
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.6)
        ax.set_ylabel(name)
        ax.legend(loc="upper right", fontsize=8)

    axes[1].set_xlabel("Episode step")
    fig.suptitle(f"Attitude trace - Episode {ep}  ({T} steps, {A} agents)")

    out = out_dir / f"attitude_trace_ep{ep}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def generate_all_trajectory_plots(
    collector: object,
    out_dir: Path,
    min_height: float = 0.3,
    max_height: float = 2.5,
    spawn_z_min: float = 0.5,
    spawn_z_max: float = 1.0,
) -> list[Path]:
    """Generate altitude, XY, and (optionally) attitude plots for every episode.

    Args:
        collector: A ``TrajectoryDataCollector`` instance with populated
                   ``pos_traj``, ``roll_traj``, and ``pitch_traj`` attributes.
        out_dir:   Directory where PNGs are saved.
        min_height, max_height, spawn_z_min, spawn_z_max: Altitude reference
                   lines for the altitude plot.

    Returns:
        List of paths to generated PNGs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for ep, traj_steps in sorted(collector.pos_traj.items()):
        if not traj_steps:
            continue

        created.append(plot_altitude(
            traj_steps, ep, out_dir,
            min_height=min_height,
            max_height=max_height,
            spawn_z_min=spawn_z_min,
            spawn_z_max=spawn_z_max,
        ))
        created.append(plot_xy(traj_steps, ep, out_dir))

        roll_steps = collector.roll_traj.get(ep, [])
        pitch_steps = collector.pitch_traj.get(ep, [])
        if roll_steps and pitch_steps:
            created.append(plot_attitude(roll_steps, pitch_steps, ep, out_dir))

    return created
