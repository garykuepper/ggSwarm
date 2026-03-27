"""Generate a 2×2 trajectory summary plot from recorded drone state data."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


DRONE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def generate_trajectory_plots(
    pos_data: list[torch.Tensor],
    quat_data: list[torch.Tensor],
    out_dir: str | Path,
    agent_names: list[str] | None = None,
    euler_fn=None,
    min_height: float = 0.1,
    max_height: float = 2.0,
) -> Path:
    """Create a 2×2 trajectory summary and save as PNG.

    Args:
        pos_data: List of [num_agents, 3] tensors (one per step).
        quat_data: List of [num_agents, 4] tensors (one per step).
        out_dir: Directory to save the plot.
        agent_names: Agent labels (default: drone_0, drone_1, ...).
        euler_fn: Callable(quat) → (roll, pitch, yaw) in radians. If None, attitude subplot is skipped.
        min_height: Minimum height line for altitude plot.
        max_height: Maximum height line for altitude plot.

    Returns:
        Path to the saved PNG file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = torch.stack(pos_data)  # [T, num_agents, 3]
    T, A, _ = pos.shape
    steps = list(range(T))

    if agent_names is None:
        agent_names = [f"drone_{i}" for i in range(A)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- Top-left: Altitude ---
    ax = axes[0, 0]
    for a in range(A):
        ax.plot(steps, pos[:, a, 2].numpy(), color=DRONE_COLORS[a % len(DRONE_COLORS)],
                linewidth=1.2, label=agent_names[a])
    ax.axhline(min_height, color="red", linestyle="--", linewidth=0.9, label=f"min {min_height}m")
    ax.axhline(max_height, color="grey", linestyle="--", linewidth=0.8, label=f"max {max_height}m")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Altitude")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(-0.2, max_height + 0.3)

    # --- Top-right: XY trace ---
    ax = axes[0, 1]
    for a in range(A):
        x = pos[:, a, 0].numpy()
        y = pos[:, a, 1].numpy()
        ax.plot(x, y, color=DRONE_COLORS[a % len(DRONE_COLORS)], alpha=0.75,
                linewidth=1.1, label=agent_names[a])
        ax.plot(x[0], y[0], "x", color=DRONE_COLORS[a % len(DRONE_COLORS)],
                markersize=10, markeredgewidth=2.0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trace (× = spawn)")
    ax.legend(fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")

    # --- Bottom-left: Attitude ---
    ax = axes[1, 0]
    if euler_fn is not None and len(quat_data) > 0:
        quat = torch.stack(quat_data)  # [T, num_agents, 4]
        roll_all, pitch_all, _ = euler_fn(quat.reshape(-1, 4))
        roll_deg = torch.rad2deg(roll_all).reshape(T, A)
        pitch_deg = torch.rad2deg(pitch_all).reshape(T, A)
        for a in range(A):
            ax.plot(steps, roll_deg[:, a].numpy(), color=DRONE_COLORS[a % len(DRONE_COLORS)],
                    linewidth=1.0, label=f"{agent_names[a]} roll")
            ax.plot(steps, pitch_deg[:, a].numpy(), color=DRONE_COLORS[a % len(DRONE_COLORS)],
                    linewidth=1.0, linestyle="--", alpha=0.7, label=f"{agent_names[a]} pitch")
        ax.axhline(15.0, color="darkorange", linestyle="--", linewidth=0.8, label="±15° gate")
        ax.axhline(-15.0, color="darkorange", linestyle="--", linewidth=0.8)
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.5)
        ax.legend(fontsize=6, ncol=2)
    else:
        ax.text(0.5, 0.5, "Attitude data unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="grey")
    ax.set_xlabel("Step")
    ax.set_ylabel("Degrees")
    ax.set_title("Attitude (roll solid, pitch dashed)")

    # --- Bottom-right: Inter-drone distances ---
    ax = axes[1, 1]
    if A >= 2:
        pair_idx = 0
        pair_colors = ["tab:purple", "tab:brown", "tab:pink", "tab:cyan"]
        for i in range(A):
            for j in range(i + 1, A):
                dist = torch.linalg.norm(pos[:, i, :] - pos[:, j, :], dim=1).numpy()
                ax.plot(steps, dist, color=pair_colors[pair_idx % len(pair_colors)],
                        linewidth=1.1, label=f"{agent_names[i]}↔{agent_names[j]}")
                pair_idx += 1
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.5)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Need ≥2 agents for distance", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="grey")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Inter-Drone Distance")

    fig.suptitle(f"Trajectory Summary ({T} steps, {A} agents)", fontsize=14)
    fig.tight_layout()

    out_path = out_dir / "trajectory_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Trajectory plot saved: {out_path}")
    return out_path
