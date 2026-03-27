"""Control Barrier Function (CBF) safety shield for swarm collision avoidance.

Projects unsafe actions onto the safe set defined by pairwise minimum
separation constraints. Operates within each swarm group — drones in
different groups don't interact.

Post-policy filter: no retraining needed. Enable via cfg.cbf_enabled.
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
) -> torch.Tensor:
    """Apply CBF safety projection to actions within each swarm group.

    For each pair (i,j) in a group, checks the barrier constraint:
        h_ij = ||p_i - p_j||^2 - d_safe^2
        h_dot_ij = 2 * (p_i - p_j) . (v_i - v_j)
        Safe: h_dot_ij + gamma * h_ij >= 0

    When violated, reduces the approaching drone's thrust to prevent
    further closure.

    Args:
        actions: [N, 4] raw actions in [-1, 1]
        pos_w: [N, 3] world positions
        env_origins: [N, 3] env origins (subtracted for local positions)
        vel_w: [N, 3] world velocities
        num_agents: agents per swarm group
        d_safe: minimum safe distance (m)
        gamma: barrier decay rate

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
                # Reduce thrust for drone i where unsafe
                # Clamp toward hover thrust (action=0 → 50% max thrust)
                act_safe[unsafe, i, 0] = act_safe[unsafe, i, 0].clamp(min=-0.5)

    return act_safe.reshape(N, 4)
