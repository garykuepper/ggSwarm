"""Generate a 2x2 trajectory summary plot from recorded drone state data."""

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
    min_height: float = 0.05,
    max_height: float = 2.0,
    goal_data: list[torch.Tensor] | None = None,
    env_origins: torch.Tensor | None = None,
    target_spacing: float | None = None,
    centroid: tuple[float, float, float] | None = None,
    collision_radius: float | None = None,
    prefix: str | None = None,
) -> Path:
    """Create a 2x2 trajectory summary and save as PNG.

    Args:
        pos_data: List of [num_agents, 3] tensors (one per step).
        quat_data: List of [num_agents, 4] tensors (one per step).
        out_dir: Directory to save the plot.
        agent_names: Agent labels (default: drone_0, drone_1, ...).
        euler_fn: Callable(quat) -> (roll, pitch, yaw) in radians.
        min_height: Minimum height line for altitude plot.
        max_height: Maximum height line for altitude plot.
        goal_data: List of [num_agents, 3] tensors (goal position per step).
        env_origins: [num_agents, 3] tensor — subtracted to get local-frame positions.
        target_spacing: Target inter-drone spacing (m) — shown as horizontal line.
        collision_radius: Body collision radius (m) — shown as red danger line.

    Returns:
        Path to the saved PNG file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = torch.stack(pos_data)  # [T, num_agents, 3]
    T, A, _ = pos.shape
    steps = list(range(T))

    goal = torch.stack(goal_data) if goal_data else None

    # Convert to local frame (subtract env origins)
    if env_origins is not None:
        pos = pos - env_origins.unsqueeze(0)  # [T, A, 3] - [1, A, 3]
        if goal is not None:
            goal = goal - env_origins.unsqueeze(0)

    if agent_names is None:
        agent_names = [f"drone_{i}" for i in range(A)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- Top-left: Altitude ---
    ax = axes[0, 0]
    for a in range(A):
        ax.plot(steps, pos[:, a, 2].numpy(), color=DRONE_COLORS[a % len(DRONE_COLORS)],
                linewidth=1.2, label=agent_names[a])
    if goal is not None:
        for a in range(A):
            ax.plot(steps, goal[:, a, 2].numpy(), color=DRONE_COLORS[a % len(DRONE_COLORS)],
                    linewidth=1.0, linestyle=":", alpha=0.6, label=f"{agent_names[a]} goal Z")
    if centroid is not None:
        ax.axhline(centroid[2], color="green", linestyle="-.", linewidth=1.2,
                   label=f"centroid Z={centroid[2]}m")
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
    # In cloud mode all goals are identical (group centroid) — show final
    # drone positions instead, which are more informative.
    _goals_identical = (
        goal is not None
        and A > 1
        and (goal[-1, :, :2] - goal[-1, 0:1, :2]).abs().max().item() < 0.01
    )
    if _goals_identical:
        for a in range(A):
            fx = pos[-1, a, 0].item()
            fy = pos[-1, a, 1].item()
            ax.plot(fx, fy, "o", color=DRONE_COLORS[a % len(DRONE_COLORS)],
                    markersize=10, markeredgewidth=1.5, markerfacecolor="none",
                    label=f"{agent_names[a]} final pos")
    elif goal is not None:
        for a in range(A):
            gx = goal[-1, a, 0].item()
            gy = goal[-1, a, 1].item()
            ax.plot(gx, gy, "*", color=DRONE_COLORS[a % len(DRONE_COLORS)],
                    markersize=15, markeredgewidth=1.0, label=f"{agent_names[a]} goal")
    if centroid is not None:
        ax.plot(centroid[0], centroid[1], "D", color="green",
                markersize=12, markeredgewidth=2.0, markerfacecolor="none",
                label="centroid")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trace (x = spawn, * = goal / o = final, D = centroid)")
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
        ax.axhline(15.0, color="darkorange", linestyle="--", linewidth=0.8, label="+/-15 gate")
        ax.axhline(-15.0, color="darkorange", linestyle="--", linewidth=0.8)
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.5)
        ax.legend(fontsize=6, ncol=2)
    else:
        ax.text(0.5, 0.5, "Attitude data unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="grey")
    ax.set_xlabel("Step")
    ax.set_ylabel("Degrees")
    ax.set_title("Attitude (roll solid, pitch dashed)")

    # --- Bottom-right: K-nearest neighbor distance per drone ---
    ax = axes[1, 1]
    K = min(2, A - 1)  # K-nearest (match training K)
    if A >= 2:
        for a in range(A):
            # Compute distance from drone a to all others at each step
            diffs = pos[:, a:a+1, :] - pos  # [T, A, 3]
            dists = torch.linalg.norm(diffs, dim=2)  # [T, A]
            dists[:, a] = float("inf")  # exclude self

            # Get K nearest distances per step
            knn_dists, _ = torch.topk(dists, K, dim=1, largest=False)  # [T, K]

            # Plot mean of K-nearest as single line per drone
            mean_knn = knn_dists.mean(dim=1).numpy()
            ax.plot(steps, mean_knn, color=DRONE_COLORS[a % len(DRONE_COLORS)],
                    linewidth=1.0, alpha=0.8, label=f"{agent_names[a]} K={K}nn")

        if target_spacing is not None:
            ax.axhline(target_spacing, color="green", linestyle="--", linewidth=1.2,
                       label=f"target {target_spacing}m")
        if collision_radius is not None:
            ax.axhline(collision_radius, color="red", linestyle="-", linewidth=1.5,
                       alpha=0.8, label=f"collision {collision_radius}m")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.5)
        ax.set_ylabel("Distance (m)")
        ax.set_title(f"K={K} Nearest Neighbor Distance")
        ax.legend(fontsize=6, ncol=2)
    elif goal is not None:
        for a in range(A):
            dist = torch.linalg.norm(pos[:, a, :] - goal[:, a, :], dim=1).numpy()
            ax.plot(steps, dist, color=DRONE_COLORS[a % len(DRONE_COLORS)],
                    linewidth=1.1, label=f"{agent_names[a]}")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.5)
        ax.set_ylabel("Distance (m)")
        ax.set_title("Distance to Goal")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No distance data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="grey")
        ax.set_ylabel("Distance (m)")
        ax.set_title("Distance")
    ax.set_xlabel("Step")

    fig.suptitle(f"Trajectory Summary ({T} steps, {A} agents)", fontsize=14)
    fig.tight_layout()

    fname = f"{prefix}-trajectory_summary.png" if prefix else "trajectory_summary.png"
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Trajectory plot saved: {out_path}")
    return out_path
