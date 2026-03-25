"""Pure torch helpers for ggSwarm swarm logic contracts.

These functions intentionally avoid Isaac Sim / Isaac Lab imports so they can be
unit-tested quickly and deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MarlRewardParams:
    """Reward and curriculum parameters for `GGSwarmMarlEnv`."""

    # Curriculum
    curriculum_start_step: int
    curriculum_end_step: int
    # Minimum position-reward multiplier so hover signal never fully disappears.
    curriculum_pos_floor: float

    # Reward scales
    rew_scale_pos: float
    rew_scale_formation: float
    rew_scale_cohesion: float
    rew_scale_separation: float
    rew_scale_vel: float
    rew_scale_ang_vel: float
    rew_scale_alive: float
    rew_scale_terminated: float
    # Uprightness: rewards the drone for staying level (projected_gravity_b[:,2] == -1).
    rew_scale_upright: float

    # Reward sigmas / thresholds
    rew_pos_sigma: float
    rew_formation_sigma: float
    target_formation_dist: float
    graph_connectivity_radius: float
    min_separation_dist: float

    # Termination bounds
    min_height: float
    max_height: float
    # Penalty depth below (min_height + low_clearance_margin_m); 0 scale disables.
    rew_scale_low_clearance: float
    low_clearance_margin_m: float


def compute_curriculum_alpha(
    common_step_counter: int,
    *,
    curriculum_start_step: int,
    curriculum_end_step: int,
) -> float:
    """Compute curriculum alpha in [0, 1], matching env gating behavior."""

    start_step = curriculum_start_step
    end_step = curriculum_end_step
    step_ratio = (common_step_counter - start_step) / max(1, end_step - start_step)
    alpha = max(0.0, min(1.0, float(step_ratio)))
    return alpha


def compute_adjacency_matrix(
    pos_w: torch.Tensor,
    *,
    graph_connectivity_radius: float,
) -> torch.Tensor:
    """Compute dense adjacency matrix from inter-agent L2 distances.

    Adjacency contract:
    - shape: [num_envs, num_agents, num_agents]
    - diagonal entries are zero (no self-edges)
    - edge exists when `dist < graph_connectivity_radius` (strictly less)
    """

    # shape: [num_envs, num_agents, 3]
    if pos_w.dim() != 3 or pos_w.shape[-1] != 3:
        raise ValueError(
            "pos_w must have shape [num_envs, num_agents, 3]; "
            f"got {tuple(pos_w.shape)}"
        )

    # shape: [num_envs, num_agents, num_agents]
    num_envs, num_agents, _ = pos_w.shape
    diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)
    adj_matrix = (dist < graph_connectivity_radius).float()

    eye = torch.eye(num_agents, device=pos_w.device, dtype=adj_matrix.dtype).unsqueeze(0)
    adj_matrix = adj_matrix * (1.0 - eye)

    # Lightweight invariants (helps catch subtle graph corruption).
    if adj_matrix.shape != (num_envs, num_agents, num_agents):
        raise AssertionError("Adjacency matrix shape mismatch.")
    diag = torch.diagonal(adj_matrix, dim1=1, dim2=2)
    if not torch.allclose(diag, torch.zeros_like(diag)):
        raise AssertionError("Adjacency diagonal must be zero.")

    return adj_matrix


def adjacency_to_edge_index(
    adj_matrix: torch.Tensor,
) -> torch.Tensor:
    """Convert dense batched adjacency matrix into `edge_index` for PyG.

    This uses the same indexing scheme as `GGSwarmGNNPolicy`:
    - Flatten graphs across the batch by shifting node indices by `env_idx * num_agents`.
    - Returns a tensor with shape [2, num_edges], dtype `torch.long`.
    """

    # shape: [num_envs, num_agents, num_agents]
    if adj_matrix.dim() != 3 or adj_matrix.shape[1] != adj_matrix.shape[2]:
        raise ValueError(
            "adj_matrix must have shape [num_envs, num_agents, num_agents]; "
            f"got {tuple(adj_matrix.shape)}"
        )

    num_envs, num_agents, _ = adj_matrix.shape
    del num_envs  # (kept for clarity; `num_agents` is used in shifting)

    indices = adj_matrix.nonzero(as_tuple=False)  # shape: [num_edges, 3]
    if indices.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=adj_matrix.device)

    env_idx = indices[:, 0]
    src = indices[:, 1] + (env_idx * num_agents)
    dst = indices[:, 2] + (env_idx * num_agents)
    edge_index = torch.stack([src, dst], dim=0)  # shape: [2, num_edges]
    return edge_index


def compute_marl_rewards(
    *,
    pos_w: torch.Tensor,
    desired_pos_w: torch.Tensor,
    lin_vel_b: torch.Tensor,
    ang_vel_b: torch.Tensor,
    proj_grav_b: torch.Tensor,
    common_step_counter: int,
    params: MarlRewardParams,
    return_terms: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute total MARL rewards per agent.

    Args:
        proj_grav_b: Gravity projected into each drone's body frame.
            Shape [num_envs, num_agents, 3]. When level, z-component is -1.0.
        return_terms: If True, return (total_rewards, terms_dict) for logging.

    Returns:
        If return_terms is False: rewards tensor of shape [num_envs, num_agents]
        If return_terms is True: (rewards, terms_dict) where terms_dict has keys
            for each reward component (all have shape [num_envs, num_agents])
    """

    # shape: [num_envs, num_agents, 3]
    if pos_w.dim() != 3 or pos_w.shape[-1] != 3:
        raise ValueError(f"pos_w must be [num_envs, num_agents, 3]; got {tuple(pos_w.shape)}")
    if desired_pos_w.shape != pos_w.shape:
        raise ValueError(
            "desired_pos_w must have the same shape as pos_w: "
            f"pos_w={tuple(pos_w.shape)}, desired_pos_w={tuple(desired_pos_w.shape)}"
        )
    if lin_vel_b.shape != pos_w.shape or ang_vel_b.shape != pos_w.shape:
        raise ValueError("lin_vel_b and ang_vel_b must have the same shape as pos_w.")
    if proj_grav_b.shape != pos_w.shape:
        raise ValueError("proj_grav_b must have the same shape as pos_w.")

    num_envs, num_agents, _ = pos_w.shape
    if num_agents < 2:
        raise ValueError("compute_marl_rewards expects num_agents >= 2 (formation term).")

    alpha = compute_curriculum_alpha(
        common_step_counter,
        curriculum_start_step=params.curriculum_start_step,
        curriculum_end_step=params.curriculum_end_step,
    )

    # shape: [num_envs, num_agents, num_agents]
    diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
    inter_agent_distances = torch.norm(diff, dim=-1)

    # shape: [num_envs, num_agents]
    dist_to_goal = torch.norm(desired_pos_w - pos_w, dim=-1)

    # Position (Hover) Reward.
    # Fades out as formation takes over, but never below `curriculum_pos_floor`
    # so the drone always has incentive to stay aloft and at its target position.
    pos_scale = max(params.curriculum_pos_floor, 1.0 - alpha)
    rew_pos = (
        torch.exp(-dist_to_goal / params.rew_pos_sigma)
        * params.rew_scale_pos
        * pos_scale
    )

    # Formation Reward (fades in)
    spacing_error = torch.abs(inter_agent_distances - params.target_formation_dist)
    eye = torch.eye(num_agents, device=pos_w.device, dtype=spacing_error.dtype).unsqueeze(0)
    spacing_error = spacing_error * (1.0 - eye)  # ignore self-distances
    mean_spacing_error = spacing_error.sum(dim=2) / (num_agents - 1)
    rew_formation = (
        torch.exp(-mean_spacing_error / params.rew_formation_sigma)
        * params.rew_scale_formation
        * alpha
    )

    # Cohesion Reward (fades in)
    max_neighbor_dist, _ = torch.max(inter_agent_distances, dim=2)
    rew_cohesion = (
        params.rew_scale_cohesion
        * torch.exp(-max_neighbor_dist / params.graph_connectivity_radius)
        * alpha
    )

    # Separation Penalty (always on)
    collision_mask = (inter_agent_distances < params.min_separation_dist) & (1.0 - eye).bool()
    rew_separation = params.rew_scale_separation * collision_mask.any(dim=2).float()

    # Uprightness reward: proj_grav_b[:, :, 2] is -1.0 when level, +1.0 when inverted.
    # Negating gives +1.0 when level and -1.0 when inverted.
    # This ensures thrust (applied in body frame) actually lifts the drone.
    # shape: [num_envs, num_agents]
    rew_upright = params.rew_scale_upright * (-proj_grav_b[:, :, 2])

    # Velocity penalties
    rew_vel = torch.norm(lin_vel_b, dim=-1) * params.rew_scale_vel
    rew_ang_vel = torch.norm(ang_vel_b, dim=-1) * params.rew_scale_ang_vel

    # Alive bonus
    rew_alive = torch.full_like(rew_pos, params.rew_scale_alive)

    # Termination penalty
    is_terminated = (pos_w[:, :, 2] < params.min_height) | (pos_w[:, :, 2] > params.max_height)
    rew_terminated = is_terminated.float() * params.rew_scale_terminated

    # Low-clearance shaping: penalize time spent below eval-aligned safe altitude.
    # shape: [num_envs, num_agents]
    clearance_z = params.min_height + params.low_clearance_margin_m
    depth_below_clearance = torch.clamp(clearance_z - pos_w[:, :, 2], min=0.0)
    rew_low_clearance = params.rew_scale_low_clearance * depth_below_clearance

    total_rewards = (
        rew_pos
        + rew_formation
        + rew_cohesion
        + rew_separation
        + rew_upright
        + rew_vel
        + rew_ang_vel
        + rew_alive
        + rew_terminated
        + rew_low_clearance
    )

    # shape: [num_envs, num_agents]
    if total_rewards.shape != (num_envs, num_agents):
        raise AssertionError("Reward tensor shape mismatch.")
    
    if return_terms:
        terms_dict = {
            "rew_pos": rew_pos,
            "rew_formation": rew_formation,
            "rew_cohesion": rew_cohesion,
            "rew_separation": rew_separation,
            "rew_upright": rew_upright,
            "rew_vel": rew_vel,
            "rew_ang_vel": rew_ang_vel,
            "rew_alive": rew_alive,
            "rew_terminated": rew_terminated,
            "rew_low_clearance": rew_low_clearance,
        }
        return total_rewards, terms_dict
    
    return total_rewards


@dataclass(frozen=True)
class FormationRewardParams:
    """Parameters for formation-only rewards (cohesion, separation, spacing).

    Used by ``compute_formation_rewards`` which is added on top of
    ``compute_stable_hover_rewards`` in hybrid mode (Phase 2B+).
    """

    curriculum_start_step: int
    curriculum_end_step: int
    rew_scale_formation: float
    rew_scale_cohesion: float
    rew_scale_separation: float
    rew_formation_sigma: float
    target_formation_dist: float
    graph_connectivity_radius: float
    min_separation_dist: float


def compute_formation_rewards(
    *,
    pos_w: torch.Tensor,
    common_step_counter: int,
    params: FormationRewardParams,
    return_terms: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute formation/cohesion/separation rewards only (no hover terms).

    Designed to be added on top of ``compute_stable_hover_rewards`` in hybrid
    mode.  Curriculum alpha gates formation and cohesion; separation is always on.

    Args:
        pos_w: Drone positions, shape ``[num_envs, num_agents, 3]``.
        common_step_counter: Current training step for curriculum.
        params: Formation reward scales and thresholds.
        return_terms: If True, return ``(total, terms_dict)``.

    Returns:
        Formation rewards of shape ``[num_envs, num_agents]``.
    """
    # shape: [num_envs, num_agents, 3]
    if pos_w.dim() != 3 or pos_w.shape[-1] != 3:
        raise ValueError(f"pos_w must be [num_envs, num_agents, 3]; got {tuple(pos_w.shape)}")

    num_envs, num_agents, _ = pos_w.shape
    if num_agents < 2:
        raise ValueError("compute_formation_rewards expects num_agents >= 2.")

    alpha = compute_curriculum_alpha(
        common_step_counter,
        curriculum_start_step=params.curriculum_start_step,
        curriculum_end_step=params.curriculum_end_step,
    )

    # shape: [num_envs, num_agents, num_agents]
    diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
    inter_agent_distances = torch.norm(diff, dim=-1)
    eye = torch.eye(num_agents, device=pos_w.device, dtype=inter_agent_distances.dtype).unsqueeze(0)

    # Formation reward (fades in with alpha)
    # shape: [num_envs, num_agents]
    spacing_error = torch.abs(inter_agent_distances - params.target_formation_dist)
    spacing_error = spacing_error * (1.0 - eye)
    mean_spacing_error = spacing_error.sum(dim=2) / (num_agents - 1)
    rew_formation = (
        torch.exp(-mean_spacing_error / params.rew_formation_sigma)
        * params.rew_scale_formation
        * alpha
    )

    # Cohesion reward (fades in with alpha)
    # shape: [num_envs, num_agents]
    max_neighbor_dist, _ = torch.max(inter_agent_distances, dim=2)
    rew_cohesion = (
        params.rew_scale_cohesion
        * torch.exp(-max_neighbor_dist / params.graph_connectivity_radius)
        * alpha
    )

    # Separation penalty (always on)
    # shape: [num_envs, num_agents]
    collision_mask = (inter_agent_distances < params.min_separation_dist) & (1.0 - eye).bool()
    rew_separation = params.rew_scale_separation * collision_mask.any(dim=2).float()

    total = rew_formation + rew_cohesion + rew_separation

    if return_terms:
        terms_dict = {
            "rew_formation": rew_formation,
            "rew_cohesion": rew_cohesion,
            "rew_separation": rew_separation,
        }
        return total, terms_dict

    return total


@dataclass(frozen=True)
class HoverRewardParams:
    """Reward parameters for `GGSwarmHoverEnv`."""

    rew_scale_pos: float
    rew_pos_sigma: float
    rew_scale_vel: float
    rew_scale_ang_vel: float
    rew_scale_alive: float
    rew_scale_ground_hit: float
    rew_scale_terminated: float
    hover_reward_min_height: float
    ground_hit_height: float


@dataclass(frozen=True)
class StableHoverRewardParams:
    """Reward parameters for PD-controller-based hover-stability training.

    Matches the 3-term structure of Isaac Lab's Isaac-Quadcopter-Direct-v0:
    - distance-to-goal (tanh-mapped, step_dt scaled)
    - linear velocity squared penalty (step_dt scaled)
    - angular velocity squared penalty (step_dt scaled)

    With the PD attitude controller providing inherent stability, no upright or
    alive terms are required for the hover phase. Optional ``rew_scale_terminated``
    adds a dense penalty whenever ``z < min_height`` (see ``compute_stable_hover_rewards``).

    Optional low-clearance shaping (same semantics as ``compute_marl_rewards``)
    aligns Phase 2A training with the post-train airborne scorecard band.
    """

    rew_scale_pos: float       # distance-to-goal scale (e.g. 15.0)
    rew_scale_vel: float       # linear velocity squared scale (e.g. -0.05)
    rew_scale_ang_vel: float   # angular velocity squared scale (e.g. -0.01)
    step_dt: float             # simulation step time (s) for dt-scaling
    min_height: float = 0.1
    low_clearance_margin_m: float = 0.2
    rew_scale_low_clearance: float = 0.0
    # Dense penalty each step while z < min_height; 0 disables. Often one step before reset.
    rew_scale_terminated: float = 0.0
    # Tanh distance scale (metres). Smaller = sharper: reward drops faster with distance.
    # PD10: 0.25 makes crash-reset ~4.5x worse than hover (was 1.1x at 0.8).
    pos_tanh_sigma: float = 0.8


def compute_stable_hover_rewards(
    *,
    pos_w: torch.Tensor,
    desired_pos_w: torch.Tensor,
    lin_vel_b: torch.Tensor,
    ang_vel_b: torch.Tensor,
    params: StableHoverRewardParams,
    return_terms: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute hover rewards using the Isaac Lab-style formulation (pos + vel + ang_vel).

    Position and velocity terms use ``step_dt`` so reward magnitude is independent of
    simulation frequency (matches Isaac-Quadcopter-Direct-v0 convention). The optional
    ground term ``rew_terminated`` is **not** dt-scaled: it applies ``scale`` on every
    step where ``z < min_height`` (dense while grounded; often a single step before
    episode reset in Isaac Lab, but not guaranteed).

    Args:
        pos_w: Current drone positions, shape ``[num_envs, num_agents, 3]``.
        desired_pos_w: Goal positions, same shape.
        lin_vel_b: Linear velocity in body frame, same shape.
        ang_vel_b: Angular velocity in body frame, same shape.
        params: Reward scales and step_dt.

    Returns:
        If ``return_terms`` is False: rewards tensor of shape ``[num_envs, num_agents]``.
        If True: ``(rewards, terms_dict)`` with per-term tensors for logging.
    """

    if pos_w.dim() != 3 or pos_w.shape[-1] != 3:
        raise ValueError(f"pos_w must be [num_envs, num_agents, 3]; got {tuple(pos_w.shape)}")
    if desired_pos_w.shape != pos_w.shape:
        raise ValueError(
            f"desired_pos_w shape {tuple(desired_pos_w.shape)} != pos_w shape {tuple(pos_w.shape)}"
        )
    if lin_vel_b.shape != pos_w.shape or ang_vel_b.shape != pos_w.shape:
        raise ValueError("lin_vel_b and ang_vel_b must match pos_w shape.")

    # shape: [num_envs, num_agents]
    dist_to_goal = torch.norm(desired_pos_w - pos_w, dim=-1)

    # Tanh-mapped position reward: smooth, bounded in [0, 1], maximised at goal.
    # Smaller sigma = sharper: reward drops faster with distance from goal.
    rew_pos = (1.0 - torch.tanh(dist_to_goal / params.pos_tanh_sigma)) * params.rew_scale_pos * params.step_dt

    # Squared velocity penalties: dt-scaled so total episode penalty is frequency-independent.
    # shape: [num_envs, num_agents]
    rew_vel = torch.sum(lin_vel_b ** 2, dim=-1) * params.rew_scale_vel * params.step_dt
    rew_ang_vel = torch.sum(ang_vel_b ** 2, dim=-1) * params.rew_scale_ang_vel * params.step_dt

    # Penalty depth below (min_height + low_clearance_margin_m); 0 scale skips.
    # shape: [num_envs, num_agents]
    clearance_z = params.min_height + params.low_clearance_margin_m
    depth_below_clearance = torch.clamp(clearance_z - pos_w[:, :, 2], min=0.0)
    rew_low_clearance = params.rew_scale_low_clearance * depth_below_clearance

    # Dense penalty each step while below min_height; 0 scale skips. Typically one step
    # before reset when out_of_bounds fires on the same step, but not guaranteed.
    # shape: [num_envs, num_agents]
    ground_hit = pos_w[:, :, 2] < params.min_height
    rew_terminated = ground_hit.float() * params.rew_scale_terminated

    total_rewards = rew_pos + rew_vel + rew_ang_vel + rew_low_clearance + rew_terminated

    if total_rewards.shape != pos_w.shape[:2]:
        raise AssertionError("Reward tensor shape mismatch.")

    if return_terms:
        z = torch.zeros_like(rew_pos)
        terms_dict = {
            "rew_pos": rew_pos,
            "rew_vel": rew_vel,
            "rew_ang_vel": rew_ang_vel,
            "rew_low_clearance": rew_low_clearance,
            "rew_terminated": rew_terminated,
            "rew_formation": z,
            "rew_cohesion": z,
            "rew_separation": z,
            "rew_upright": z,
            "rew_alive": z,
        }
        return total_rewards, terms_dict

    return total_rewards


def compute_hover_rewards(
    *,
    pos_w: torch.Tensor,
    desired_pos_w: torch.Tensor,
    lin_vel_b: torch.Tensor,
    ang_vel_b: torch.Tensor,
    params: HoverRewardParams,
) -> torch.Tensor:
    """Compute hover-only baseline rewards per agent.

    Returns:
        rewards: tensor of shape [num_envs, num_agents]
    """

    # shape: [num_envs, num_agents, 3]
    if pos_w.dim() != 3 or pos_w.shape[-1] != 3:
        raise ValueError(f"pos_w must be [num_envs, num_agents, 3]; got {tuple(pos_w.shape)}")
    if desired_pos_w.shape != pos_w.shape:
        raise ValueError(
            "desired_pos_w must have the same shape as pos_w: "
            f"pos_w={tuple(pos_w.shape)}, desired_pos_w={tuple(desired_pos_w.shape)}"
        )
    if lin_vel_b.shape != pos_w.shape or ang_vel_b.shape != pos_w.shape:
        raise ValueError("lin_vel_b and ang_vel_b must have the same shape as pos_w.")

    # shape: [num_envs, num_agents]
    dist_to_goal = torch.norm(desired_pos_w - pos_w, dim=-1)

    # Gate position reward near/above a minimal height.
    above_hover_floor = (pos_w[:, :, 2] >= params.hover_reward_min_height).float()
    rew_pos = (
        torch.exp(-dist_to_goal / params.rew_pos_sigma)
        * params.rew_scale_pos
        * above_hover_floor
    )

    # Velocity penalties (and alive bonus).
    rew_vel = torch.norm(lin_vel_b, dim=-1) * params.rew_scale_vel
    rew_ang_vel = torch.norm(ang_vel_b, dim=-1) * params.rew_scale_ang_vel
    rew_alive = torch.full_like(rew_pos, params.rew_scale_alive)

    # Crash penalty near/at ground.
    ground_hit = pos_w[:, :, 2] < params.ground_hit_height
    rew_ground_hit = ground_hit.float() * params.rew_scale_ground_hit
    rew_terminated = ground_hit.float() * params.rew_scale_terminated

    total_rewards = rew_pos + rew_vel + rew_ang_vel + rew_alive + rew_ground_hit + rew_terminated
    return total_rewards

