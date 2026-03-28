"""Control Barrier Function (CBF) safety shield for swarm collision avoidance.

Projects unsafe actions onto the safe set defined by pairwise minimum
separation constraints. Operates within each swarm group — drones in
different groups don't interact.

Post-policy filter: no retraining needed. Enable via cfg.cbf_enabled.

Lateral repulsion: when drones are approaching too closely, injects
roll/pitch moments to tilt the drone away from its neighbor, not just
clamp thrust. This is critical because thrust-only CBF cannot prevent
horizontal convergence.
"""

from __future__ import annotations

import torch


def apply_cbf(
    actions: torch.Tensor,
    pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    vel_w: torch.Tensor,
    num_agents: int,
    d_safe: float,
    gamma: float,
    lateral_scale: float = 0.5,
) -> torch.Tensor:
    """Apply CBF safety projection to actions within each swarm group.

    For each pair (i,j) in a group, checks the barrier constraint:
        h_ij = ||p_i - p_j||^2 - d_safe^2
        h_dot_ij = 2 * (p_i - p_j) . (v_i - v_j)
        Safe: h_dot_ij + gamma * h_ij >= 0

    When violated, injects lateral moments to tilt the drone away from
    its neighbor AND clamps thrust to prevent further closure.

    Args:
        actions: [N, 4] raw actions in [-1, 1]
        pos_w: [N, 3] world positions
        env_origins: [N, 3] env origins (subtracted for local positions)
        vel_w: [N, 3] world velocities
        num_agents: agents per swarm group
        d_safe: minimum safe distance (m)
        gamma: barrier decay rate
        lateral_scale: strength of lateral moment injection (0-1)

    Returns:
        [N, 4] safe actions
    """
    if num_agents <= 1:
        return actions

    N = actions.shape[0]
    A = num_agents
    G = N // A

    # Local positions (env_origins subtracted)
    pos_local = pos_w - env_origins  # shape: [N, 3]
    pos_g = pos_local.reshape(G, A, 3)
    vel_g = vel_w.reshape(G, A, 3)
    act_safe = actions.reshape(G, A, 4).clone()

    d_safe_sq = d_safe * d_safe

    for i in range(A):
        for j in range(A):
            if i == j:
                continue

            # Pairwise barrier computation
            diff = pos_g[:, i] - pos_g[:, j]  # [G, 3]
            dist_sq = (diff * diff).sum(dim=1)  # [G]
            h = dist_sq - d_safe_sq  # [G]

            vel_diff = vel_g[:, i] - vel_g[:, j]  # [G, 3]
            h_dot = 2.0 * (diff * vel_diff).sum(dim=1)  # [G]

            # Check barrier constraint
            constraint = h_dot + gamma * h  # [G]
            unsafe = constraint < 0  # [G] bool

            if unsafe.any():
                # Violation severity: 0 at boundary, 1 at max violation
                strength = (-constraint[unsafe] / (gamma * d_safe_sq + 1e-6)).clamp(0.0, 1.0)  # [U]

                # Escape direction in XY plane (from j toward i)
                escape_xy = diff[unsafe, :2]  # [U, 2]
                escape_norm = escape_xy.norm(dim=1, keepdim=True).clamp(min=1e-6)  # [U, 1]
                escape_dir = escape_xy / escape_norm  # [U, 2] unit vector

                # Inject lateral moments to tilt drone i away from drone j
                # act[:, 2] = pitch moment → X-axis lateral movement
                # act[:, 1] = roll moment → Y-axis lateral movement
                act_safe[unsafe, i, 2] += strength * escape_dir[:, 0] * lateral_scale
                act_safe[unsafe, i, 1] += strength * escape_dir[:, 1] * lateral_scale

                # Also reduce thrust to prevent aggressive vertical closure
                act_safe[unsafe, i, 0] = act_safe[unsafe, i, 0].clamp(min=-0.5)

    # Clamp all actions back to valid range
    return act_safe.reshape(N, 4).clamp(-1.0, 1.0)
