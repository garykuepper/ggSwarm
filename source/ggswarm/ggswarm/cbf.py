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
    max_correction: float = _MAX_CORRECTION,
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
            # Correction = strength * max_correction * escape_dir component
            s = (strength * max_correction).unsqueeze(1)  # shape: [U, 1]

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


def apply_cbf_obstacles(
    actions: torch.Tensor,
    pos_w: torch.Tensor,
    env_origins: torch.Tensor,
    vel_w: torch.Tensor,
    obstacle_pos: torch.Tensor,
    d_safe: float,
    gamma: float,
    max_correction: float = 0.25,
    lateral_blend: float = 0.7,
    dampen_floor: float = 0.5,
    dampen_strength: float = 0.5,
) -> torch.Tensor:
    """Apply CBF safety projection for drone-obstacle avoidance.

    Treats each static cylinder as an immovable virtual agent. Only the
    drone receives a correction (obstacles don't move). d_safe should
    include the obstacle radius: effective_d = d_safe + obstacle_radius.

    Uses a stronger correction than drone-drone CBF since obstacles are
    immovable and the drone must do all the work.

    Args:
        actions: [N, 4] raw actions in [-1, 1]
        pos_w: [N, 3] drone world positions
        env_origins: [N, 3] env origins
        vel_w: [N, 3] drone world velocities
        obstacle_pos: [K, 3] obstacle positions in LOCAL frame (env_origins=0)
        d_safe: minimum safe distance (m), should include obstacle radius
        gamma: barrier decay rate
        max_correction: max action-space correction per step toward escape
        lateral_blend: lateral vs radial mix in escape direction (1.0 = pure lateral)
        dampen_floor: min retained policy magnitude when fully suppressing
            toward-obstacle component (1.0 = no dampening)
        dampen_strength: how aggressively to dampen toward-obstacle policy
            components (0.0 = no dampening, 1.0 = full at max urgency)

    Returns:
        [N, 4] safe actions
    """
    N = actions.shape[0]  # shape: [N, 4]
    K = obstacle_pos.shape[0]  # shape: [K, 3]
    if K == 0:
        return actions

    # Local drone positions  # shape: [N, 3]
    pos_local = pos_w - env_origins

    act = actions.clone()  # shape: [N, 4]
    d_safe_sq = d_safe * d_safe

    for k in range(K):
        # Barrier: drone XY position vs obstacle k (ignore Z for cylinders)
        obs_k = obstacle_pos[k]  # shape: [3]
        diff = pos_local - obs_k.unsqueeze(0)  # shape: [N, 3]

        # Use XY distance only for cylindrical obstacles (infinite height)
        diff_xy = diff.clone()  # shape: [N, 3]
        diff_xy[:, 2] = 0.0  # zero out Z for 2D barrier
        dist_sq_xy = (diff_xy * diff_xy).sum(dim=1)  # shape: [N]
        h = dist_sq_xy - d_safe_sq  # shape: [N]

        # h_dot: obstacle is static, so v_obstacle = 0 (XY only)
        vel_xy = vel_w.clone()  # shape: [N, 3]
        vel_xy[:, 2] = 0.0
        h_dot = 2.0 * (diff_xy * vel_xy).sum(dim=1)  # shape: [N]

        # Barrier constraint
        constraint = h_dot + gamma * h  # shape: [N]
        unsafe = constraint < 0  # shape: [N] bool

        if not unsafe.any():
            continue

        # Violation severity — stronger scaling for obstacles
        strength = (-constraint[unsafe] / (gamma * d_safe_sq + 1e-6)).clamp(0.0, 1.0)  # shape: [U]

        # Compute escape direction: lateral (perpendicular to velocity)
        # so drones steer AROUND obstacles instead of fighting backwards
        diff_unsafe = diff_xy[unsafe]  # shape: [U, 3]
        dist_unsafe = diff_unsafe.norm(dim=1, keepdim=True).clamp(min=1e-6)  # shape: [U, 1]
        radial_dir = diff_unsafe / dist_unsafe  # shape: [U, 3] unit radial (away from obstacle)

        # Lateral direction: rotate radial 90 degrees in XY plane
        # Choose the side that aligns with existing velocity (go WITH momentum)
        lateral_dir = torch.zeros_like(radial_dir)  # shape: [U, 3]
        lateral_dir[:, 0] = -radial_dir[:, 1]  # rotate 90 deg: (x,y) -> (-y,x)
        lateral_dir[:, 1] = radial_dir[:, 0]

        # Pick sign: go with velocity (dot product with drone velocity)
        vel_unsafe = vel_xy[unsafe]  # shape: [U, 3]
        dot = (lateral_dir * vel_unsafe).sum(dim=1)  # shape: [U]
        sign = torch.where(dot >= 0, torch.ones_like(dot), -torch.ones_like(dot))
        lateral_dir = lateral_dir * sign.unsqueeze(1)

        # Blend: mostly lateral (steer around) + some radial (push away)
        escape_dir = lateral_blend * lateral_dir + (1.0 - lateral_blend) * radial_dir  # shape: [U, 3]
        escape_norm = escape_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        escape_dir = escape_dir / escape_norm

        # Dampen policy action components pulling TOWARD the obstacle
        # proportional to CBF urgency (strength 0-1). dampen_strength controls
        # how aggressive the dampening gets; dampen_floor is the minimum
        # retained policy magnitude (so we never fully suppress the policy).
        toward_roll = -escape_dir[:, 1]   # roll direction toward obstacle
        toward_pitch = -escape_dir[:, 0]  # pitch direction toward obstacle
        roll_toward = (act[unsafe, 1] * toward_roll) > 0  # shape: [U] bool
        pitch_toward = (act[unsafe, 2] * toward_pitch) > 0  # shape: [U] bool
        dampen = (1.0 - dampen_strength * strength).clamp(min=dampen_floor)  # shape: [U]
        act[unsafe, 1] = torch.where(roll_toward, act[unsafe, 1] * dampen, act[unsafe, 1])
        act[unsafe, 2] = torch.where(pitch_toward, act[unsafe, 2] * dampen, act[unsafe, 2])

        # Add lateral correction on top
        s = (strength * max_correction).unsqueeze(1)  # shape: [U, 1]
        act[unsafe, 0] = act[unsafe, 0] + s.squeeze(1) * escape_dir[:, 2]  # thrust
        act[unsafe, 1] = act[unsafe, 1] + s.squeeze(1) * escape_dir[:, 1]  # roll
        act[unsafe, 2] = act[unsafe, 2] + s.squeeze(1) * escape_dir[:, 0]  # pitch

    return act.clamp(-1.0, 1.0)
