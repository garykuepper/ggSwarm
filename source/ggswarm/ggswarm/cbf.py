"""Control Barrier Function (CBF) safety shield for swarm collision avoidance.

Minimally invasive QP safety filter: solves min ||u - u_nom||^2 subject to
pairwise barrier constraints h_dot_ij + gamma * h_ij >= 0. Only modifies
actions that would violate safety, and modifies them as little as possible.

Operates within each swarm group — drones in different groups don't interact.
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
    """Apply CBF-QP safety projection to actions within each swarm group.

    For each pair (i,j) in a group, enforces the barrier constraint:
        h_ij = ||p_i - p_j||^2 - d_safe^2
        h_dot_ij = 2 * (p_i - p_j) . (v_i - v_j)
        Safe: h_dot_ij + gamma * h_ij >= 0

    When violated, projects u_nom onto the constraint boundary using the
    analytical QP solution (gradient projection). For multiple simultaneous
    violations, applies sequential projections (converges in 2-3 iterations
    for sparse constraints).

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

    N = actions.shape[0]  # shape: [N, 4]
    A = num_agents
    G = N // A

    # Local positions (env_origins subtracted)
    pos_local = pos_w - env_origins  # shape: [N, 3]
    pos_g = pos_local.reshape(G, A, 3)  # shape: [G, A, 3]
    vel_g = vel_w.reshape(G, A, 3)  # shape: [G, A, 3]
    act_g = actions.reshape(G, A, 4).clone()  # shape: [G, A, 4]

    d_safe_sq = d_safe * d_safe

    # Sequential projection: iterate pairs, project each drone's action
    # onto the barrier constraint boundary when violated.
    # 2 passes for convergence with overlapping constraints.
    for _pass in range(2):
        for i in range(A):
            for j in range(i + 1, A):
                # Pairwise barrier computation
                diff = pos_g[:, i] - pos_g[:, j]  # shape: [G, 3]
                dist_sq = (diff * diff).sum(dim=1)  # shape: [G]
                h = dist_sq - d_safe_sq  # shape: [G]

                vel_diff = vel_g[:, i] - vel_g[:, j]  # shape: [G, 3]
                h_dot = 2.0 * (diff * vel_diff).sum(dim=1)  # shape: [G]

                # Barrier constraint: h_dot + gamma * h >= 0
                constraint = h_dot + gamma * h  # shape: [G]
                unsafe = constraint < 0  # shape: [G] bool

                if not unsafe.any():
                    continue

                # Violation magnitude
                violation = -constraint[unsafe]  # shape: [U], positive

                # The barrier constraint is linear in accelerations, which are
                # proportional to actions. The gradient of h_dot w.r.t. action
                # of drone i is along the relative position direction projected
                # onto action space. For a quadrotor with [thrust, roll, pitch, yaw]:
                #   - thrust (act[:,0]) affects Z acceleration
                #   - roll   (act[:,1]) affects Y lateral
                #   - pitch  (act[:,2]) affects X lateral
                #   - yaw    (act[:,3]) affects heading (minimal barrier effect)
                #
                # Map spatial diff to action-space gradient:
                #   diff_x -> pitch correction (act[:,2])
                #   diff_y -> roll correction  (act[:,1])
                #   diff_z -> thrust correction (act[:,0])
                diff_unsafe = diff[unsafe]  # shape: [U, 3]

                # Action-space gradient for drone i (maps spatial barrier to action dims)
                # grad_a[k] = direction in action space that increases h_dot
                grad_a = torch.zeros(unsafe.sum(), 4, device=actions.device)  # shape: [U, 4]
                grad_a[:, 0] = diff_unsafe[:, 2]  # thrust <- Z separation
                grad_a[:, 1] = diff_unsafe[:, 1]  # roll <- Y separation
                grad_a[:, 2] = diff_unsafe[:, 0]  # pitch <- X separation
                # yaw (act[:,3]) has negligible barrier effect, leave at 0

                grad_norm_sq = (grad_a * grad_a).sum(dim=1).clamp(min=1e-8)  # shape: [U]

                # Project: u* = u + (violation / ||grad||^2) * grad
                # This is the minimal correction to satisfy the constraint
                correction_scale = (violation / grad_norm_sq).unsqueeze(1)  # shape: [U, 1]

                # Apply correction to drone i (push away from j)
                act_g[unsafe, i] = act_g[unsafe, i] + correction_scale * grad_a * 0.5
                # Apply opposite correction to drone j (push away from i)
                act_g[unsafe, j] = act_g[unsafe, j] - correction_scale * grad_a * 0.5

    # Clamp all actions back to valid range
    return act_g.reshape(N, 4).clamp(-1.0, 1.0)
