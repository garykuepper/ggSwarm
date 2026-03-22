"""L4 Safety Shield: Control Barrier Function for inter-agent collision avoidance.

Pure-torch, no Isaac imports -- safe for unit testing in isolation.

Barrier function (pairwise spherical):
    h_ij(x) = ||p_i - p_j||^2 - d_safe^2

When h_ij approaches zero the raw action for agent i is projected onto the safe
half-space via the analytical gradient:

    a_safe = a_raw - min(0, (grad_h . a_raw + gamma * h) / ||grad_h||^2) * grad_h

Only the thrust channel (action dim 0, mapped to body-Z force) participates in the
barrier correction since it dominates XY velocity; moment channels (dims 1-3) are
scaled proportionally when the barrier is critically violated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CbfParams:
    """CBF configuration parameters."""

    d_safe: float
    gamma: float
    # Fraction of d_safe at which the barrier starts activating (hysteresis).
    activation_margin: float = 1.5


def apply_cbf_safety(
    actions: torch.Tensor,
    pos_w: torch.Tensor,
    lin_vel_w: torch.Tensor,
    params: CbfParams,
) -> tuple[torch.Tensor, float]:
    """Apply CBF safety projection to raw policy actions.

    Projects each agent's action onto the safe half-space for every pairwise
    constraint that is close to or violating the safety barrier.  Constraints
    are evaluated symmetrically but correction is applied to both agents in a
    pair so the net effect is cooperative avoidance.

    Args:
        actions: Raw policy actions. Shape [num_envs, num_agents, 4].
        pos_w:   Agent world positions. Shape [num_envs, num_agents, 3].
        lin_vel_w: Agent world-frame linear velocities. Shape [num_envs, num_agents, 3].
        params:  CBF configuration.

    Returns:
        safe_actions: Corrected actions. Shape [num_envs, num_agents, 4].
        intervention_rate: Fraction of (env, agent) pairs that were corrected.
    """

    # shape: [num_envs, num_agents, 4]
    if actions.dim() != 3 or actions.shape[-1] != 4:
        raise ValueError(
            f"actions must be [num_envs, num_agents, 4]; got {tuple(actions.shape)}"
        )
    if pos_w.shape != actions.shape[:2] + (3,):
        raise ValueError(
            f"pos_w must be [num_envs, num_agents, 3]; got {tuple(pos_w.shape)}"
        )
    if lin_vel_w.shape != pos_w.shape:
        raise ValueError(
            f"lin_vel_w must match pos_w shape; got {tuple(lin_vel_w.shape)}"
        )

    num_envs, num_agents, _ = pos_w.shape
    safe_actions = actions.clone()
    # shape: [num_envs, num_agents] -- tracks which agents had corrections applied
    intervention_mask = torch.zeros(
        num_envs, num_agents, dtype=torch.bool, device=actions.device
    )

    d_safe_sq = params.d_safe**2
    activation_radius_sq = (params.d_safe * params.activation_margin) ** 2

    # Evaluate all pairs (i, j) with i < j
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            # shape: [num_envs, 3]
            diff_pos = pos_w[:, i, :] - pos_w[:, j, :]
            # shape: [num_envs]
            dist_sq = (diff_pos**2).sum(dim=-1)

            # Barrier value h_ij = dist^2 - d_safe^2
            h = dist_sq - d_safe_sq  # shape: [num_envs]

            # Only process pairs that are within the activation radius
            active = dist_sq < activation_radius_sq  # shape: [num_envs]
            if not active.any():
                continue

            # Gradient of h w.r.t. position of agent i: grad_h_i = 2*(p_i - p_j)
            # Mapped into action space: the thrust action (dim 0) changes body-Z
            # force which directly influences Z velocity.  For horizontal safety
            # we use the full velocity dot product as the Lie derivative.
            # grad_h mapped to action space via velocity direction:
            # Lf_h = 2 * (p_i - p_j) . (v_i - v_j)   (time derivative of h along f)
            diff_vel = lin_vel_w[:, i, :] - lin_vel_w[:, j, :]  # [num_envs, 3]
            # Lie derivative of h along the drift (velocity)
            lf_h = 2.0 * (diff_pos * diff_vel).sum(dim=-1)  # [num_envs]

            # CBF condition requires: Lf_h + gamma * h >= 0
            # If violated, compute correction magnitude
            violation = -(lf_h + params.gamma * h)  # positive when CBF violated
            needs_correction = active & (violation > 0.0)
            if not needs_correction.any():
                continue

            # Analytical projection: correct agent i's thrust (action dim 0)
            # by projecting along the barrier gradient direction in velocity space.
            # grad_h in velocity space (unit direction of p_i - p_j)
            dist = torch.sqrt(dist_sq.clamp(min=1e-6))  # [num_envs]
            grad_dir = diff_pos / dist.unsqueeze(-1)  # [num_envs, 3], unit vector

            # The thrust correction needed to satisfy CBF:
            # delta_a = violation / (2 * dist)   (from chain rule)
            # Only apply to active and violating pairs
            correction_mag = violation / (2.0 * dist.clamp(min=1e-6))

            # Map correction magnitude to action dim 0 (thrust).
            # The sign convention: positive correction pushes agents apart.
            # For agent i: increase thrust if below j; decrease if above.
            # Simplified: project correction onto body-Z (Z component of grad_dir)
            thrust_correction_i = correction_mag * grad_dir[:, 2]
            thrust_correction_j = -correction_mag * grad_dir[:, 2]

            mask = needs_correction.float()  # [num_envs]

            # Apply thrust corrections to action dim 0
            safe_actions[:, i, 0] = safe_actions[:, i, 0] + mask * thrust_correction_i
            safe_actions[:, j, 0] = safe_actions[:, j, 0] + mask * thrust_correction_j

            # Proportional correction to moment channels (dims 1-3) for horizontal avoidance
            for dim in range(1, 4):
                dir_component = grad_dir[:, dim - 1] if dim <= 3 else torch.zeros_like(mask)
                safe_actions[:, i, dim] = (
                    safe_actions[:, i, dim] + mask * correction_mag * dir_component * 0.5
                )
                safe_actions[:, j, dim] = (
                    safe_actions[:, j, dim] - mask * correction_mag * dir_component * 0.5
                )

            intervention_mask[:, i] |= needs_correction
            intervention_mask[:, j] |= needs_correction

    # Clamp corrected actions to valid range
    safe_actions = safe_actions.clamp(-1.0, 1.0)

    intervention_rate = intervention_mask.float().mean().item()
    return safe_actions, intervention_rate


def apply_cbf_obstacle_safety(
    actions: torch.Tensor,
    pos_w: torch.Tensor,
    lin_vel_w: torch.Tensor,
    obstacle_positions: torch.Tensor,
    obstacle_d_safe: float,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Apply CBF safety projection for static obstacle avoidance.

    Barrier function per obstacle k for agent i:
        h_ik(x) = ||p_i - p_k||^2 - d_safe^2

    Args:
        actions: Shape [num_envs, num_agents, 4].
        pos_w:   Agent positions. Shape [num_envs, num_agents, 3].
        lin_vel_w: Agent velocities. Shape [num_envs, num_agents, 3].
        obstacle_positions: XYZ centres of obstacles. Shape [num_obstacles, 3].
        obstacle_d_safe: Safety radius around each obstacle (m).
        gamma: Barrier decay rate.

    Returns:
        safe_actions: Shape [num_envs, num_agents, 4].
        intervention_rate: Fraction of (env, agent) pairs corrected.
    """

    if actions.dim() != 3 or actions.shape[-1] != 4:
        raise ValueError(
            f"actions must be [num_envs, num_agents, 4]; got {tuple(actions.shape)}"
        )

    num_envs, num_agents, _ = pos_w.shape
    num_obstacles = obstacle_positions.shape[0]
    if num_obstacles == 0:
        return actions, 0.0

    safe_actions = actions.clone()
    intervention_mask = torch.zeros(
        num_envs, num_agents, dtype=torch.bool, device=actions.device
    )
    d_safe_sq = obstacle_d_safe**2
    activation_radius_sq = (obstacle_d_safe * 2.0) ** 2

    obs_pos = obstacle_positions.to(actions.device)  # shape: [num_obstacles, 3]

    for i in range(num_agents):
        for k in range(num_obstacles):
            # shape: [num_envs, 3]
            diff_pos = pos_w[:, i, :] - obs_pos[k].unsqueeze(0)
            dist_sq = (diff_pos**2).sum(dim=-1)  # [num_envs]
            h = dist_sq - d_safe_sq

            active = dist_sq < activation_radius_sq
            if not active.any():
                continue

            # Lie derivative: velocity component moving toward obstacle
            # Obstacles are static so diff_vel = lin_vel_w[:,i,:]
            lf_h = 2.0 * (diff_pos * lin_vel_w[:, i, :]).sum(dim=-1)

            violation = -(lf_h + gamma * h)
            needs_correction = active & (violation > 0.0)
            if not needs_correction.any():
                continue

            dist = torch.sqrt(dist_sq.clamp(min=1e-6))
            grad_dir = diff_pos / dist.unsqueeze(-1)
            correction_mag = violation / (2.0 * dist.clamp(min=1e-6))

            mask = needs_correction.float()
            thrust_correction = correction_mag * grad_dir[:, 2]
            safe_actions[:, i, 0] = safe_actions[:, i, 0] + mask * thrust_correction

            for dim in range(1, 4):
                dir_component = grad_dir[:, dim - 1]
                safe_actions[:, i, dim] = (
                    safe_actions[:, i, dim] + mask * correction_mag * dir_component * 0.5
                )

            intervention_mask[:, i] |= needs_correction

    safe_actions = safe_actions.clamp(-1.0, 1.0)
    intervention_rate = intervention_mask.float().mean().item()
    return safe_actions, intervention_rate


def compute_pairwise_distances(pos_w: torch.Tensor) -> torch.Tensor:
    """Compute all pairwise L2 distances between agents.

    Args:
        pos_w: Shape [num_envs, num_agents, 3].

    Returns:
        distances: Shape [num_envs, num_agents, num_agents]. Diagonal is zero.
    """

    # shape: [num_envs, num_agents, num_agents, 3]
    diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
    # shape: [num_envs, num_agents, num_agents]
    return torch.norm(diff, dim=-1)


def compute_min_pairwise_distance(pos_w: torch.Tensor) -> torch.Tensor:
    """Return the minimum inter-agent distance for each environment.

    Args:
        pos_w: Shape [num_envs, num_agents, 3].

    Returns:
        min_dist: Shape [num_envs].
    """

    num_envs, num_agents, _ = pos_w.shape
    distances = compute_pairwise_distances(pos_w)
    # Mask diagonal (self distances) with a large value
    eye = (
        torch.eye(num_agents, device=pos_w.device, dtype=torch.bool)
        .unsqueeze(0)
        .expand(num_envs, -1, -1)
    )
    distances = distances.masked_fill(eye, float("inf"))
    # shape: [num_envs]
    min_dist, _ = distances.view(num_envs, -1).min(dim=-1)
    return min_dist
