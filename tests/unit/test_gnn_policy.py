"""Unit tests for agents/skrl_gnn_policy.py (Phase 2 GATv2 policy).

Covers: model construction, forward pass shapes with and without adj_matrix,
fallback self-loop path, log_std parameter properties, gradient flow, and
NaN/Inf safety.

No Isaac Sim required -- omni.* and isaaclab_tasks are stubbed in conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import gymnasium as gym

# conftest.py stubs omni.* and adds source/ggSwarm to sys.path before this
# module loads, so the relative import inside skrl_gnn_policy resolves.
from ggSwarm.tasks.direct.ggswarm_marl.agents.skrl_gnn_policy import GGSwarmGNNPolicy


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

OBS_DIM = 12
ACTION_DIM = 4
NUM_ENVS = 4
NUM_AGENTS = 3
BATCH = NUM_ENVS * NUM_AGENTS  # 12 nodes in the batched graph


def _obs_space(obs_dim: int = OBS_DIM) -> gym.spaces.Box:
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)


def _act_space(action_dim: int = ACTION_DIM) -> gym.spaces.Box:
    return gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)


def _make_policy(device: str = "cpu", obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM) -> GGSwarmGNNPolicy:
    policy = GGSwarmGNNPolicy(
        observation_space=_obs_space(obs_dim),
        action_space=_act_space(action_dim),
        device=device,
    )
    return policy.to(device)


def _fully_connected_adj(num_envs: int, num_agents: int, device: str) -> torch.Tensor:
    """Fully-connected (no self-loops) adjacency. Shape [num_envs, num_agents, num_agents]."""
    eye = torch.eye(num_agents, device=device).unsqueeze(0)
    return (1.0 - eye).expand(num_envs, -1, -1)


def _inputs(batch: int, obs_dim: int, adj: torch.Tensor | None, device: str) -> dict:
    obs = torch.randn(batch, obs_dim, device=device)
    base: dict = {"states": obs}
    if adj is not None:
        base["extras"] = {"adj_matrix": adj}
    return base


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestModelConstruction:
    def test_has_conv1(self, device):
        policy = _make_policy(device)
        assert hasattr(policy, "conv1"), "GATv2 layer conv1 must exist"

    def test_has_conv2(self, device):
        policy = _make_policy(device)
        assert hasattr(policy, "conv2"), "GATv2 layer conv2 must exist"

    def test_has_action_head(self, device):
        policy = _make_policy(device)
        assert hasattr(policy, "action_head"), "Linear action head must exist"
        assert policy.action_head.out_features == ACTION_DIM

    def test_log_std_parameter_shape(self, device):
        policy = _make_policy(device)
        assert policy.log_std_parameter.shape == (ACTION_DIM,), (
            f"log_std_parameter must have shape ({ACTION_DIM},)"
        )

    def test_log_std_parameter_requires_grad(self, device):
        policy = _make_policy(device)
        assert policy.log_std_parameter.requires_grad, "log_std_parameter must be learnable"

    def test_parameter_count_nonzero(self, device):
        policy = _make_policy(device)
        total = sum(p.numel() for p in policy.parameters())
        assert total > 0, "Model must have trainable parameters"

    def test_initial_log_std_kwarg(self, device):
        """initial_log_std kwarg must set the starting value of log_std_parameter."""
        init_val = -1.5
        policy = GGSwarmGNNPolicy(
            observation_space=_obs_space(),
            action_space=_act_space(),
            device=device,
            initial_log_std=init_val,
        ).to(device)
        assert torch.allclose(
            policy.log_std_parameter,
            torch.full((ACTION_DIM,), init_val, device=device),
        ), "initial_log_std must initialise log_std_parameter to that value"


# ---------------------------------------------------------------------------
# Forward pass -- output shapes
# ---------------------------------------------------------------------------


class TestForwardShapes:
    def test_action_mean_shape_with_adj(self, device):
        policy = _make_policy(device)
        adj = _fully_connected_adj(NUM_ENVS, NUM_AGENTS, device)
        inputs = _inputs(BATCH, OBS_DIM, adj, device)
        action_mean, log_std, _ = policy.compute(inputs, role="policy")
        assert action_mean.shape == (BATCH, ACTION_DIM), (
            f"action_mean must be [{BATCH}, {ACTION_DIM}], got {tuple(action_mean.shape)}"
        )

    def test_log_std_shape_with_adj(self, device):
        policy = _make_policy(device)
        adj = _fully_connected_adj(NUM_ENVS, NUM_AGENTS, device)
        inputs = _inputs(BATCH, OBS_DIM, adj, device)
        _, log_std, _ = policy.compute(inputs, role="policy")
        assert log_std.shape == (ACTION_DIM,)

    def test_action_mean_shape_no_adj(self, device):
        """Fallback self-loop path must produce same output shape."""
        policy = _make_policy(device)
        inputs = _inputs(BATCH, OBS_DIM, adj=None, device=device)
        action_mean, _, _ = policy.compute(inputs, role="policy")
        assert action_mean.shape == (BATCH, ACTION_DIM)

    def test_returns_empty_dict_as_third_element(self, device):
        policy = _make_policy(device)
        inputs = _inputs(BATCH, OBS_DIM, adj=None, device=device)
        result = policy.compute(inputs, role="policy")
        assert isinstance(result[2], dict), "Third return value must be a dict"

    def test_3d_obs_input_flattened(self, device):
        """3D obs [num_envs, num_agents, obs_dim] must be auto-flattened."""
        policy = _make_policy(device)
        obs_3d = torch.randn(NUM_ENVS, NUM_AGENTS, OBS_DIM, device=device)
        adj = _fully_connected_adj(NUM_ENVS, NUM_AGENTS, device)
        inputs = {"states": obs_3d, "extras": {"adj_matrix": adj}}
        action_mean, _, _ = policy.compute(inputs, role="policy")
        assert action_mean.shape == (BATCH, ACTION_DIM)

    def test_single_env_single_agent(self, device):
        """Degenerate case: 1 env, 1 agent must not crash."""
        policy = _make_policy(device)
        obs = torch.randn(1, OBS_DIM, device=device)
        inputs = {"states": obs}  # no adj_matrix → self-loop fallback
        action_mean, _, _ = policy.compute(inputs, role="policy")
        assert action_mean.shape == (1, ACTION_DIM)

    def test_different_obs_dims(self, device):
        """Phase 3 obs space (14 dims) must work without code changes."""
        obs_dim_14 = 14
        policy = _make_policy(device, obs_dim=obs_dim_14)
        obs = torch.randn(BATCH, obs_dim_14, device=device)
        adj = _fully_connected_adj(NUM_ENVS, NUM_AGENTS, device)
        inputs = {"states": obs, "extras": {"adj_matrix": adj}}
        action_mean, _, _ = policy.compute(inputs, role="policy")
        assert action_mean.shape == (BATCH, ACTION_DIM)


# ---------------------------------------------------------------------------
# Adjacency matrix handling
# ---------------------------------------------------------------------------


class TestAdjacencyHandling:
    def test_fully_connected_produces_edges(self, device):
        """With a fully-connected graph every node should receive messages."""
        policy = _make_policy(device)
        adj = _fully_connected_adj(NUM_ENVS, NUM_AGENTS, device)
        inputs = _inputs(BATCH, OBS_DIM, adj, device)
        action_mean, _, _ = policy.compute(inputs, role="policy")
        # Two different random obs should give different outputs (GNN aggregates neighbours)
        torch.manual_seed(0)
        inputs2 = _inputs(BATCH, OBS_DIM, adj, device)
        inputs["states"] = torch.randn(BATCH, OBS_DIM, device=device)
        out1, _, _ = policy.compute(inputs, role="policy")
        out2, _, _ = policy.compute(inputs2, role="policy")
        assert not torch.allclose(out1, out2), "Different obs must produce different outputs"

    def test_no_adj_key_in_extras(self, device):
        """extras dict without 'adj_matrix' key must use self-loop fallback."""
        policy = _make_policy(device)
        inputs = {"states": torch.randn(BATCH, OBS_DIM, device=device), "extras": {}}
        action_mean, _, _ = policy.compute(inputs, role="policy")
        assert action_mean.shape == (BATCH, ACTION_DIM)

    def test_extras_not_a_dict(self, device):
        """extras as a non-dict (old SKRL behaviour) must not crash."""
        policy = _make_policy(device)
        inputs = {"states": torch.randn(BATCH, OBS_DIM, device=device), "extras": None}
        action_mean, _, _ = policy.compute(inputs, role="policy")
        assert action_mean.shape == (BATCH, ACTION_DIM)

    def test_zero_edge_adj_matrix(self, device):
        """Adjacency with no edges (isolated nodes) must fall through without error."""
        policy = _make_policy(device)
        # All-zero adjacency: no edges
        adj = torch.zeros(NUM_ENVS, NUM_AGENTS, NUM_AGENTS, device=device)
        inputs = _inputs(BATCH, OBS_DIM, adj, device)
        action_mean, _, _ = policy.compute(inputs, role="policy")
        assert action_mean.shape == (BATCH, ACTION_DIM)

    def test_adj_matrix_none_explicit(self, device):
        """adj_matrix=None explicitly in extras must use self-loop fallback."""
        policy = _make_policy(device)
        inputs = {
            "states": torch.randn(BATCH, OBS_DIM, device=device),
            "extras": {"adj_matrix": None},
        }
        action_mean, _, _ = policy.compute(inputs, role="policy")
        assert action_mean.shape == (BATCH, ACTION_DIM)


# ---------------------------------------------------------------------------
# NaN / Inf safety
# ---------------------------------------------------------------------------


class TestNaNSafety:
    def test_no_nan_with_adj_matrix(self, device):
        policy = _make_policy(device)
        adj = _fully_connected_adj(NUM_ENVS, NUM_AGENTS, device)
        torch.manual_seed(0)
        for _ in range(10):
            inputs = _inputs(BATCH, OBS_DIM, adj, device)
            action_mean, log_std, _ = policy.compute(inputs, role="policy")
            assert not torch.isnan(action_mean).any(), "NaN in action_mean with adj_matrix"
            assert not torch.isinf(action_mean).any(), "Inf in action_mean with adj_matrix"

    def test_no_nan_without_adj_matrix(self, device):
        policy = _make_policy(device)
        torch.manual_seed(1)
        for _ in range(10):
            inputs = _inputs(BATCH, OBS_DIM, adj=None, device=device)
            action_mean, _, _ = policy.compute(inputs, role="policy")
            assert not torch.isnan(action_mean).any()
            assert not torch.isinf(action_mean).any()

    def test_no_nan_for_near_zero_obs(self, device):
        policy = _make_policy(device)
        obs = torch.full((BATCH, OBS_DIM), 1e-38, device=device)
        inputs = {"states": obs}
        action_mean, _, _ = policy.compute(inputs, role="policy")
        assert not torch.isnan(action_mean).any()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    def test_gradient_flows_through_action_mean(self, device):
        policy = _make_policy(device)
        adj = _fully_connected_adj(NUM_ENVS, NUM_AGENTS, device)
        inputs = _inputs(BATCH, OBS_DIM, adj, device)
        action_mean, _, _ = policy.compute(inputs, role="policy")
        loss = action_mean.sum()
        loss.backward()
        for name, param in policy.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_log_std_gradient_flows(self, device):
        policy = _make_policy(device)
        inputs = _inputs(BATCH, OBS_DIM, adj=None, device=device)
        _, log_std, _ = policy.compute(inputs, role="policy")
        (log_std.sum()).backward()
        assert policy.log_std_parameter.grad is not None
        assert not torch.isnan(policy.log_std_parameter.grad).any()

    def test_gradients_differ_adj_vs_no_adj(self, device):
        """Gradients should differ between fully-connected and isolated graphs."""
        torch.manual_seed(42)
        obs = torch.randn(BATCH, OBS_DIM, device=device)
        adj = _fully_connected_adj(NUM_ENVS, NUM_AGENTS, device)

        policy_adj = _make_policy(device)
        policy_iso = _make_policy(device)
        # Copy weights so they start identical
        policy_iso.load_state_dict(policy_adj.state_dict())

        out_adj, _, _ = policy_adj.compute({"states": obs, "extras": {"adj_matrix": adj}}, "policy")
        out_iso, _, _ = policy_iso.compute({"states": obs}, "policy")

        out_adj.sum().backward()
        out_iso.sum().backward()

        # Outputs should differ due to different message-passing graphs
        assert not torch.allclose(out_adj, out_iso), (
            "Fully-connected and isolated graphs must produce different outputs"
        )
