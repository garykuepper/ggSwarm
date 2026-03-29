"""Control Barrier Function (CBF) safety shield for swarm collision avoidance.

Minimally invasive QP-inspired safety filter: for each pair (i,j), checks the
barrier constraint h_dot + gamma * h >= 0. When violated, applies a small
gradient-direction correction to nudge both drones apart. The correction
magnitude is clamped to avoid destabilizing the policy.

Operates within each swarm group — drones in different groups don't interact.
Post-policy filter: no retraining needed. Enable via cfg.cbf_enabled.
"""

from __future__ import annotations

import torch

# Maximum per-pair correction magnitude per action channel.
# Kept small to avoid destabilizing hover — the QP only nudges.
_MAX_CORRECTION = 0.15


def apply_cbf(
    actions: torch.Tensor,
    pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    vel_w: torch.Tensor,
    num_agents: int,
    d_safe: float,
    gamma: float,
    alive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply CBF safety projection to actions within each swarm group.

    For each pair (i,j) in a group, enforces the barrier constraint:
        h_ij = ||p_i - p_j||^2 - d_safe^2
        h_dot_ij = 2 * (p_i - p_j) . (v_i - v_j)
        Safe: h_dot_ij + gamma * h_ij >= 0

    When violated, computes a correction along the normalized spatial
    separation direction, mapped to action channels. The correction is
    proportional to the violation severity but clamped to _MAX_CORRECTION
    to avoid flipping drones.

    Args:
        actions: [N, 4] raw actions in [-1, 1]
        pos_w: [N, 3] world positions
        env_origins: [N, 3] env origins (subtracted for local positions)
        vel_w: [N, 3] world velocities
        num_agents: agents per swarm group
        d_safe: minimum safe distance (m)
        gamma: barrier decay rate
        alive_mask: [N] bool — if provided, skip pairs involving dead drones

    Returns:
        [N, 4] safe actions
    """
    if num_agents <= 1:
        return actions

    N = actions.shape[0]  # shape: [N, 4]
    A = num_agents
    G = N // A

    # Local positions (env_origins subtracted)
    pos_local = pos_w - env_origins  # shape: [N, 3]
    pos_g = pos_local.reshape(G, A, 3)  # shape: [G, A, 3]
    vel_g = vel_w.reshape(G, A, 3)  # shape: [G, A, 3]
    act_g = actions.reshape(G, A, 4).clone()  # shape: [G, A, 4]

    d_safe_sq = d_safe * d_safe

    # Pre-compute alive mask per group if provided
    alive_g = alive_mask.reshape(G, A) if alive_mask is not None else None  # shape: [G, A]

    for i in range(A):
        for j in range(i + 1, A):
            # Skip pairs involving dead drones
            if alive_g is not None:
                both_alive = alive_g[:, i] & alive_g[:, j]  # shape: [G]
                if not both_alive.any():
                    continue

            # Pairwise barrier computation
            diff = pos_g[:, i] - pos_g[:, j]  # shape: [G, 3]
            dist_sq = (diff * diff).sum(dim=1)  # shape: [G]
            h = dist_sq - d_safe_sq  # shape: [G]

            vel_diff = vel_g[:, i] - vel_g[:, j]  # shape: [G, 3]
            h_dot = 2.0 * (diff * vel_diff).sum(dim=1)  # shape: [G]

            # Barrier constraint: h_dot + gamma * h >= 0
            constraint = h_dot + gamma * h  # shape: [G]
            unsafe = constraint < 0  # shape: [G] bool
            if alive_g is not None:
                unsafe = unsafe & both_alive  # shape: [G]

            if not unsafe.any():
                continue

            # Violation severity: 0 at boundary, 1 at max violation
            # Normalized by gamma * d_safe^2 so strength is scale-invariant
            strength = (-constraint[unsafe] / (gamma * d_safe_sq + 1e-6)).clamp(0.0, 1.0)  # shape: [U]

            # Normalized escape direction (from j toward i in XYZ)
            diff_unsafe = diff[unsafe]  # shape: [U, 3]
            dist_unsafe = diff_unsafe.norm(dim=1, keepdim=True).clamp(min=1e-6)  # shape: [U, 1]
            escape_dir = diff_unsafe / dist_unsafe  # shape: [U, 3] unit vector

            # Map spatial escape direction to action-space correction:
            #   diff_z -> thrust (act[:,0])
            #   diff_y -> roll   (act[:,1])
            #   diff_x -> pitch  (act[:,2])
            #   yaw    -> no correction (act[:,3])
            # Correction = strength * _MAX_CORRECTION * escape_dir component
            s = (strength * _MAX_CORRECTION).unsqueeze(1)  # shape: [U, 1]

            # Drone i: push away from j (positive escape direction)
            act_g[unsafe, i, 0] = act_g[unsafe, i, 0] + s.squeeze(1) * escape_dir[:, 2]
            act_g[unsafe, i, 1] = act_g[unsafe, i, 1] + s.squeeze(1) * escape_dir[:, 1]
            act_g[unsafe, i, 2] = act_g[unsafe, i, 2] + s.squeeze(1) * escape_dir[:, 0]

            # Drone j: push away from i (opposite direction)
            act_g[unsafe, j, 0] = act_g[unsafe, j, 0] - s.squeeze(1) * escape_dir[:, 2]
            act_g[unsafe, j, 1] = act_g[unsafe, j, 1] - s.squeeze(1) * escape_dir[:, 1]
            act_g[unsafe, j, 2] = act_g[unsafe, j, 2] - s.squeeze(1) * escape_dir[:, 0]

    # Clamp all actions back to valid range
    return act_g.reshape(N, 4).clamp(-1.0, 1.0)
