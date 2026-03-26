"""Unit tests for contract_logic.py.

Covers: adjacency matrix contract, edge index conversion, curriculum alpha,
MARL reward shapes/signs, hover reward shapes/signs, and input validation.
No Isaac Sim required.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from contract_logic import (
    HoverRewardParams,
    MarlRewardParams,
    adjacency_to_edge_index,
    compute_adjacency_matrix,
    compute_curriculum_alpha,
    compute_hover_rewards,
    compute_marl_rewards,
)
from tests.conftest import NUM_AGENTS, NUM_ENVS, agents_far_apart, circular_formation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_marl_params() -> MarlRewardParams:
    return MarlRewardParams(
        curriculum_start_step=1000,
        curriculum_end_step=5000,
        curriculum_pos_floor=0.4,
        rew_scale_pos=3.0,
        rew_scale_formation=1.0,
        rew_scale_cohesion=0.2,
        rew_scale_separation=-5.0,
        rew_scale_vel=-0.15,
        rew_scale_ang_vel=-0.5,
        rew_scale_alive=1.0,
        rew_scale_terminated=-20.0,
        rew_scale_upright=5.0,
        rew_pos_sigma=0.5,
        rew_formation_sigma=0.3,
        target_formation_dist=0.20,
        graph_connectivity_radius=2.0,
        min_separation_dist=0.10,
        min_height=0.1,
        max_height=3.0,
        rew_scale_low_clearance=0.0,
        low_clearance_margin_m=0.2,
    )


def _default_hover_params() -> HoverRewardParams:
    return HoverRewardParams(
        rew_scale_pos=3.0,
        rew_pos_sigma=0.5,
        rew_scale_vel=-0.15,
        rew_scale_ang_vel=-0.5,
        rew_scale_alive=1.0,
        rew_scale_ground_hit=-5.0,
        rew_scale_terminated=-20.0,
        hover_reward_min_height=0.15,
        ground_hit_height=0.12,
    )


def _make_marl_inputs(num_envs, num_agents, device):
    """Return (pos_w, desired_pos_w, lin_vel_b, ang_vel_b, proj_grav_b) tensors."""
    torch.manual_seed(42)
    pos_w = torch.zeros(num_envs, num_agents, 3, device=device)
    pos_w[..., 2] = 1.0  # hover height
    desired_pos_w = pos_w.clone()
    lin_vel_b = torch.zeros(num_envs, num_agents, 3, device=device)
    ang_vel_b = torch.zeros(num_envs, num_agents, 3, device=device)
    proj_grav_b = torch.zeros(num_envs, num_agents, 3, device=device)
    proj_grav_b[..., 2] = -1.0  # perfectly level
    return pos_w, desired_pos_w, lin_vel_b, ang_vel_b, proj_grav_b


# ---------------------------------------------------------------------------
# Adjacency matrix
# ---------------------------------------------------------------------------


class TestComputeAdjacencyMatrix:
    def test_diagonal_zero(self, device):
        pos = agents_far_apart(NUM_ENVS, NUM_AGENTS, spacing=0.5, device=device)
        adj = compute_adjacency_matrix(pos, graph_connectivity_radius=2.0)
        diag = torch.diagonal(adj, dim1=1, dim2=2)
        assert torch.all(diag == 0.0), "Diagonal must be zero (no self-edges)"

    def test_shape(self, device):
        pos = agents_far_apart(NUM_ENVS, NUM_AGENTS, spacing=0.5, device=device)
        adj = compute_adjacency_matrix(pos, graph_connectivity_radius=2.0)
        assert adj.shape == (NUM_ENVS, NUM_AGENTS, NUM_AGENTS)

    def test_symmetric(self, device):
        pos = agents_far_apart(NUM_ENVS, NUM_AGENTS, spacing=0.5, device=device)
        adj = compute_adjacency_matrix(pos, graph_connectivity_radius=2.0)
        assert torch.allclose(adj, adj.transpose(1, 2)), "Adjacency must be symmetric"

    def test_agents_within_radius_connected(self, device):
        # Agents 0.3 m apart, radius 2.0 m -- all should be connected
        pos = agents_far_apart(NUM_ENVS, NUM_AGENTS, spacing=0.3, device=device)
        adj = compute_adjacency_matrix(pos, graph_connectivity_radius=2.0)
        # Off-diagonal entries (mask out diagonal) must all be 1.0
        eye_bool = torch.eye(NUM_AGENTS, dtype=torch.bool, device=device).unsqueeze(0)
        off_diag = adj[~eye_bool.expand_as(adj)]
        assert torch.all(off_diag == 1.0), "All in-radius pairs must be connected"

    def test_agents_outside_radius_disconnected(self, device):
        # Agents 5.0 m apart, radius 2.0 m -- no connections
        pos = agents_far_apart(NUM_ENVS, NUM_AGENTS, spacing=5.0, device=device)
        adj = compute_adjacency_matrix(pos, graph_connectivity_radius=2.0)
        eye_bool = torch.eye(NUM_AGENTS, dtype=torch.bool, device=device).unsqueeze(0)
        off_diag = adj[~eye_bool.expand_as(adj)]
        assert torch.all(off_diag == 0.0), "Out-of-radius pairs must not be connected"

    def test_strictly_less_than_radius(self, device):
        # Agent at exactly radius should NOT be connected (strictly < radius)
        pos = torch.zeros(1, 2, 3, device=device)
        pos[0, 0, 0] = 0.0
        pos[0, 1, 0] = 2.0  # exactly 2.0 m away
        pos[:, :, 2] = 1.0
        adj = compute_adjacency_matrix(pos, graph_connectivity_radius=2.0)
        assert adj[0, 0, 1] == 0.0, "Exactly-radius distance must NOT produce an edge"

    def test_invalid_shape_raises(self, device):
        bad_pos = torch.zeros(4, 3, device=device)  # missing agent dim
        with pytest.raises(ValueError):
            compute_adjacency_matrix(bad_pos, graph_connectivity_radius=1.0)

    def test_values_are_binary(self, device):
        pos = agents_far_apart(NUM_ENVS, NUM_AGENTS, spacing=0.5, device=device)
        adj = compute_adjacency_matrix(pos, graph_connectivity_radius=2.0)
        unique = adj.unique()
        assert set(unique.tolist()).issubset({0.0, 1.0}), "Adjacency values must be 0 or 1"


# ---------------------------------------------------------------------------
# Edge index conversion
# ---------------------------------------------------------------------------


class TestAdjacencyToEdgeIndex:
    def test_shape_first_dim(self, device):
        pos = agents_far_apart(NUM_ENVS, NUM_AGENTS, spacing=0.5, device=device)
        adj = compute_adjacency_matrix(pos, graph_connectivity_radius=2.0)
        edge_index = adjacency_to_edge_index(adj)
        assert edge_index.shape[0] == 2, "edge_index must have shape [2, E]"

    def test_dtype_long(self, device):
        pos = agents_far_apart(NUM_ENVS, NUM_AGENTS, spacing=0.5, device=device)
        adj = compute_adjacency_matrix(pos, graph_connectivity_radius=2.0)
        edge_index = adjacency_to_edge_index(adj)
        assert edge_index.dtype == torch.long

    def test_empty_graph_returns_zero_edges(self, device):
        # All agents far apart -- no edges
        pos = agents_far_apart(1, 2, spacing=100.0, device=device)
        adj = compute_adjacency_matrix(pos, graph_connectivity_radius=1.0)
        edge_index = adjacency_to_edge_index(adj)
        assert edge_index.shape == (2, 0), "Empty graph must return shape [2, 0]"

    def test_node_indices_within_bounds(self, device):
        pos = agents_far_apart(NUM_ENVS, NUM_AGENTS, spacing=0.5, device=device)
        adj = compute_adjacency_matrix(pos, graph_connectivity_radius=2.0)
        edge_index = adjacency_to_edge_index(adj)
        max_node = NUM_ENVS * NUM_AGENTS - 1
        assert edge_index.max().item() <= max_node
        assert edge_index.min().item() >= 0

    def test_invalid_shape_raises(self, device):
        bad = torch.zeros(4, 3, device=device)  # must be 3D
        with pytest.raises(ValueError):
            adjacency_to_edge_index(bad)


# ---------------------------------------------------------------------------
# Curriculum alpha
# ---------------------------------------------------------------------------


class TestComputeCurriculumAlpha:
    def test_zero_before_start(self):
        alpha = compute_curriculum_alpha(0, curriculum_start_step=1000, curriculum_end_step=5000)
        assert alpha == 0.0

    def test_zero_at_start(self):
        alpha = compute_curriculum_alpha(1000, curriculum_start_step=1000, curriculum_end_step=5000)
        assert alpha == 0.0

    def test_one_at_end(self):
        alpha = compute_curriculum_alpha(5000, curriculum_start_step=1000, curriculum_end_step=5000)
        assert alpha == 1.0

    def test_one_after_end(self):
        alpha = compute_curriculum_alpha(9999, curriculum_start_step=1000, curriculum_end_step=5000)
        assert alpha == 1.0

    def test_midpoint(self):
        alpha = compute_curriculum_alpha(3000, curriculum_start_step=1000, curriculum_end_step=5000)
        assert abs(alpha - 0.5) < 1e-5

    def test_monotone(self):
        steps = range(0, 6000, 100)
        alphas = [
            compute_curriculum_alpha(s, curriculum_start_step=1000, curriculum_end_step=5000)
            for s in steps
        ]
        for a, b in zip(alphas, alphas[1:]):
            assert a <= b, "Curriculum alpha must be non-decreasing"


# ---------------------------------------------------------------------------
# MARL rewards
# ---------------------------------------------------------------------------


class TestComputeMarlRewards:
    def test_output_shape(self, device):
        pos, des, lv, av, pg = _make_marl_inputs(NUM_ENVS, NUM_AGENTS, device)
        rewards = compute_marl_rewards(
            pos_w=pos,
            desired_pos_w=des,
            lin_vel_b=lv,
            ang_vel_b=av,
            proj_grav_b=pg,
            common_step_counter=0,
            params=_default_marl_params(),
        )
        assert rewards.shape == (NUM_ENVS, NUM_AGENTS)

    def test_return_terms_keys(self, device):
        pos, des, lv, av, pg = _make_marl_inputs(NUM_ENVS, NUM_AGENTS, device)
        _, terms = compute_marl_rewards(
            pos_w=pos,
            desired_pos_w=des,
            lin_vel_b=lv,
            ang_vel_b=av,
            proj_grav_b=pg,
            common_step_counter=0,
            params=_default_marl_params(),
            return_terms=True,
        )
        expected_keys = {
            "rew_pos",
            "rew_formation",
            "rew_cohesion",
            "rew_separation",
            "rew_upright",
            "rew_vel",
            "rew_ang_vel",
            "rew_alive",
            "rew_terminated",
            "rew_low_clearance",
        }
        assert set(terms.keys()) == expected_keys

    def test_terms_shapes(self, device):
        pos, des, lv, av, pg = _make_marl_inputs(NUM_ENVS, NUM_AGENTS, device)
        _, terms = compute_marl_rewards(
            pos_w=pos,
            desired_pos_w=des,
            lin_vel_b=lv,
            ang_vel_b=av,
            proj_grav_b=pg,
            common_step_counter=0,
            params=_default_marl_params(),
            return_terms=True,
        )
        for key, val in terms.items():
            assert val.shape == (NUM_ENVS, NUM_AGENTS), f"{key} has wrong shape"

    def test_alive_bonus_positive(self, device):
        pos, des, lv, av, pg = _make_marl_inputs(NUM_ENVS, NUM_AGENTS, device)
        _, terms = compute_marl_rewards(
            pos_w=pos,
            desired_pos_w=des,
            lin_vel_b=lv,
            ang_vel_b=av,
            proj_grav_b=pg,
            common_step_counter=0,
            params=_default_marl_params(),
            return_terms=True,
        )
        assert torch.all(terms["rew_alive"] > 0), "Alive bonus must always be positive"

    def test_termination_penalty_when_crashed(self, device):
        pos, des, lv, av, pg = _make_marl_inputs(NUM_ENVS, NUM_AGENTS, device)
        # Force agent 0 below min_height in all envs
        pos[:, 0, 2] = 0.0
        _, terms = compute_marl_rewards(
            pos_w=pos,
            desired_pos_w=des,
            lin_vel_b=lv,
            ang_vel_b=av,
            proj_grav_b=pg,
            common_step_counter=0,
            params=_default_marl_params(),
            return_terms=True,
        )
        assert torch.all(terms["rew_terminated"][:, 0] < 0), "Crashed agent must get negative termination penalty"

    def test_no_termination_when_airborne(self, device):
        pos, des, lv, av, pg = _make_marl_inputs(NUM_ENVS, NUM_AGENTS, device)
        _, terms = compute_marl_rewards(
            pos_w=pos,
            desired_pos_w=des,
            lin_vel_b=lv,
            ang_vel_b=av,
            proj_grav_b=pg,
            common_step_counter=0,
            params=_default_marl_params(),
            return_terms=True,
        )
        assert torch.all(terms["rew_terminated"] == 0), "No termination penalty when airborne"

    def test_upright_positive_when_level(self, device):
        pos, des, lv, av, pg = _make_marl_inputs(NUM_ENVS, NUM_AGENTS, device)
        # proj_grav_b[:,2] == -1.0 (level) -> rew_upright = scale * 1.0 > 0
        _, terms = compute_marl_rewards(
            pos_w=pos,
            desired_pos_w=des,
            lin_vel_b=lv,
            ang_vel_b=av,
            proj_grav_b=pg,
            common_step_counter=0,
            params=_default_marl_params(),
            return_terms=True,
        )
        assert torch.all(terms["rew_upright"] > 0), "Level drone must get positive upright reward"

    def test_upright_negative_when_inverted(self, device):
        pos, des, lv, av, pg = _make_marl_inputs(NUM_ENVS, NUM_AGENTS, device)
        pg[..., 2] = 1.0  # inverted: proj_grav_b[:,2] == +1.0
        _, terms = compute_marl_rewards(
            pos_w=pos,
            desired_pos_w=des,
            lin_vel_b=lv,
            ang_vel_b=av,
            proj_grav_b=pg,
            common_step_counter=0,
            params=_default_marl_params(),
            return_terms=True,
        )
        assert torch.all(terms["rew_upright"] < 0), "Inverted drone must get negative upright reward"

    def test_separation_penalty_when_too_close(self, device):
        pos, des, lv, av, pg = _make_marl_inputs(NUM_ENVS, NUM_AGENTS, device)
        # Put all agents at the same position (zero separation)
        pos[:, :, :] = 0.0
        pos[:, :, 2] = 1.0
        _, terms = compute_marl_rewards(
            pos_w=pos,
            desired_pos_w=des,
            lin_vel_b=lv,
            ang_vel_b=av,
            proj_grav_b=pg,
            common_step_counter=0,
            params=_default_marl_params(),
            return_terms=True,
        )
        assert torch.all(terms["rew_separation"] < 0), "Agents too close must get separation penalty"

    def test_no_nan_in_output(self, device):
        pos, des, lv, av, pg = _make_marl_inputs(NUM_ENVS, NUM_AGENTS, device)
        rewards = compute_marl_rewards(
            pos_w=pos,
            desired_pos_w=des,
            lin_vel_b=lv,
            ang_vel_b=av,
            proj_grav_b=pg,
            common_step_counter=2500,
            params=_default_marl_params(),
        )
        assert not torch.isnan(rewards).any(), "Rewards must not contain NaN"

    def test_requires_at_least_two_agents(self, device):
        pos = torch.zeros(1, 1, 3, device=device)
        pos[..., 2] = 1.0
        des = pos.clone()
        lv = torch.zeros_like(pos)
        av = torch.zeros_like(pos)
        pg = torch.zeros_like(pos)
        pg[..., 2] = -1.0
        with pytest.raises(ValueError):
            compute_marl_rewards(
                pos_w=pos,
                desired_pos_w=des,
                lin_vel_b=lv,
                ang_vel_b=av,
                proj_grav_b=pg,
                common_step_counter=0,
                params=_default_marl_params(),
            )

    def test_mismatched_shapes_raise(self, device):
        pos, des, lv, av, pg = _make_marl_inputs(NUM_ENVS, NUM_AGENTS, device)
        bad_des = torch.zeros(NUM_ENVS, NUM_AGENTS + 1, 3, device=device)
        with pytest.raises(ValueError):
            compute_marl_rewards(
                pos_w=pos,
                desired_pos_w=bad_des,
                lin_vel_b=lv,
                ang_vel_b=av,
                proj_grav_b=pg,
                common_step_counter=0,
                params=_default_marl_params(),
            )

    def test_low_clearance_penalty_below_plane(self, device):
        pos, des, lv, av, pg = _make_marl_inputs(NUM_ENVS, NUM_AGENTS, device)
        # Default min_height=0.1, margin=0.2 => clearance plane z=0.3
        pos[:, :, 2] = 0.25
        params = replace(_default_marl_params(), rew_scale_low_clearance=-10.0)
        _, terms = compute_marl_rewards(
            pos_w=pos,
            desired_pos_w=des,
            lin_vel_b=lv,
            ang_vel_b=av,
            proj_grav_b=pg,
            common_step_counter=0,
            params=params,
            return_terms=True,
        )
        assert torch.all(terms["rew_low_clearance"] < 0), "Below-clearance must be penalised"


# ---------------------------------------------------------------------------
# Hover rewards
# ---------------------------------------------------------------------------


class TestComputeHoverRewards:
    def _inputs(self, device):
        pos = torch.zeros(NUM_ENVS, 1, 3, device=device)
        pos[..., 2] = 1.0
        des = pos.clone()
        lv = torch.zeros_like(pos)
        av = torch.zeros_like(pos)
        return pos, des, lv, av

    def test_output_shape(self, device):
        pos, des, lv, av = self._inputs(device)
        rewards = compute_hover_rewards(
            pos_w=pos, desired_pos_w=des, lin_vel_b=lv, ang_vel_b=av,
            params=_default_hover_params(),
        )
        assert rewards.shape == (NUM_ENVS, 1)

    def test_no_nan(self, device):
        pos, des, lv, av = self._inputs(device)
        rewards = compute_hover_rewards(
            pos_w=pos, desired_pos_w=des, lin_vel_b=lv, ang_vel_b=av,
            params=_default_hover_params(),
        )
        assert not torch.isnan(rewards).any()

    def test_ground_hit_penalty(self, device):
        pos, des, lv, av = self._inputs(device)
        pos[..., 2] = 0.05  # below ground_hit_height (0.12)
        rewards = compute_hover_rewards(
            pos_w=pos, desired_pos_w=des, lin_vel_b=lv, ang_vel_b=av,
            params=_default_hover_params(),
        )
        # Ground hit + termination penalty must dominate
        assert torch.all(rewards < 0), "Crashed drone must receive net negative reward"

    def test_mismatched_shapes_raise(self, device):
        pos, des, lv, av = self._inputs(device)
        bad_des = torch.zeros(NUM_ENVS, 2, 3, device=device)
        with pytest.raises(ValueError):
            compute_hover_rewards(
                pos_w=pos, desired_pos_w=bad_des, lin_vel_b=lv, ang_vel_b=av,
                params=_default_hover_params(),
            )
