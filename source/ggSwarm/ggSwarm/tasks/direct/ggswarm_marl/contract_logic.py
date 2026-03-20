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

    # Reward scales
    rew_scale_pos: float
    rew_scale_formation: float
    rew_scale_cohesion: float
    rew_scale_separation: float
    rew_scale_vel: float
    rew_scale_ang_vel: float
    rew_scale_alive: float
    rew_scale_terminated: float

    # Reward sigmas / thresholds
    rew_pos_sigma: float
    rew_formation_sigma: float
    target_formation_dist: float
    graph_connectivity_radius: float
    min_separation_dist: float

    # Termination bounds
    min_height: float
    max_height: float


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
    common_step_counter: int,
    params: MarlRewardParams,
) -> torch.Tensor:
    """Compute total MARL rewards per agent.

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

    # Position (Hover) Reward (fades out)
    rew_pos = (
        torch.exp(-dist_to_goal / params.rew_pos_sigma)
        * params.rew_scale_pos
        * (1.0 - alpha)
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

    # Velocity penalties
    rew_vel = torch.norm(lin_vel_b, dim=-1) * params.rew_scale_vel
    rew_ang_vel = torch.norm(ang_vel_b, dim=-1) * params.rew_scale_ang_vel

    # Alive bonus
    rew_alive = torch.full_like(rew_pos, params.rew_scale_alive)

    # Termination penalty
    is_terminated = (pos_w[:, :, 2] < params.min_height) | (pos_w[:, :, 2] > params.max_height)
    rew_terminated = is_terminated.float() * params.rew_scale_terminated

    total_rewards = (
        rew_pos
        + rew_formation
        + rew_cohesion
        + rew_separation
        + rew_vel
        + rew_ang_vel
        + rew_alive
        + rew_terminated
    )

    # shape: [num_envs, num_agents]
    if total_rewards.shape != (num_envs, num_agents):
        raise AssertionError("Reward tensor shape mismatch.")
    return total_rewards


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

