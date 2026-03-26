"""Unit tests for minco_trajectory.py.

Covers: EMA correctness, alpha=1 passthrough, convergence to constant input,
variance reduction, reset isolation, buffer ordering, jitter metric, NaN safety,
and input validation.
No Isaac Sim required.
"""

from __future__ import annotations

import pytest
import torch

from minco_trajectory import MincoParams, MincoSmoother
from tests.conftest import ACTION_DIM, NUM_AGENTS, NUM_ENVS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _params(alpha: float = 0.5, buffer_size: int = 8) -> MincoParams:
    return MincoParams(smoothing_alpha=alpha, buffer_size=buffer_size)


def _make_smoother(
    num_envs: int = NUM_ENVS,
    num_agents: int = NUM_AGENTS,
    action_dim: int = ACTION_DIM,
    alpha: float = 0.5,
    buffer_size: int = 8,
    device: str = "cpu",
) -> MincoSmoother:
    return MincoSmoother(
        num_envs=num_envs,
        num_agents=num_agents,
        action_dim=action_dim,
        params=_params(alpha=alpha, buffer_size=buffer_size),
        device=device,
    )


def _rand_actions(num_envs: int = NUM_ENVS, num_agents: int = NUM_AGENTS,
                  action_dim: int = ACTION_DIM, device: str = "cpu") -> torch.Tensor:
    return (torch.rand(num_envs, num_agents, action_dim, device=device) * 2 - 1).clamp(-1, 1)


# ---------------------------------------------------------------------------
# Output shape and type
# ---------------------------------------------------------------------------


class TestSmoothOutputShape:
    def test_output_shape_matches_input(self, device):
        smoother = _make_smoother(device=device)
        actions = _rand_actions(device=device)
        out = smoother.smooth(actions)
        assert out.shape == actions.shape

    def test_output_dtype_matches_input(self, device):
        smoother = _make_smoother(device=device)
        actions = _rand_actions(device=device)
        out = smoother.smooth(actions)
        assert out.dtype == actions.dtype

    def test_wrong_shape_raises(self, device):
        smoother = _make_smoother(device=device)
        bad = torch.zeros(NUM_ENVS, NUM_AGENTS + 1, ACTION_DIM, device=device)
        with pytest.raises(ValueError):
            smoother.smooth(bad)

    def test_wrong_ndim_raises(self, device):
        smoother = _make_smoother(device=device)
        bad = torch.zeros(NUM_ENVS * NUM_AGENTS, ACTION_DIM, device=device)
        with pytest.raises(ValueError):
            smoother.smooth(bad)


# ---------------------------------------------------------------------------
# Alpha = 1.0 passthrough
# ---------------------------------------------------------------------------


class TestAlphaOnePassthrough:
    def test_passthrough_first_step(self, device):
        """With alpha=1.0, first step output must equal input exactly."""
        smoother = _make_smoother(alpha=1.0, device=device)
        actions = _rand_actions(device=device)
        out = smoother.smooth(actions)
        # First step: smooth = 1.0 * a + 0.0 * 0 = a
        assert torch.allclose(out, actions), "alpha=1.0 first step must be identity"

    def test_passthrough_all_steps(self, device):
        """With alpha=1.0, every step must return the raw actions unchanged."""
        smoother = _make_smoother(alpha=1.0, device=device)
        for _ in range(20):
            actions = _rand_actions(device=device)
            out = smoother.smooth(actions)
            assert torch.allclose(out, actions), "alpha=1.0 must always pass through actions"


# ---------------------------------------------------------------------------
# EMA correctness
# ---------------------------------------------------------------------------


class TestEMACorrectness:
    def test_first_step_zero_prev(self, device):
        """First step: smooth = alpha * action (prev state is 0)."""
        alpha = 0.3
        smoother = _make_smoother(alpha=alpha, device=device)
        actions = torch.ones(NUM_ENVS, NUM_AGENTS, ACTION_DIM, device=device)
        out = smoother.smooth(actions)
        expected = alpha * actions
        assert torch.allclose(out, expected, atol=1e-6), "First step EMA must equal alpha * action"

    def test_second_step_uses_prev(self, device):
        """Second step must use the smoothed output from the first step."""
        alpha = 0.5
        smoother = _make_smoother(alpha=alpha, device=device)
        a1 = torch.ones(NUM_ENVS, NUM_AGENTS, ACTION_DIM, device=device)
        a2 = torch.full_like(a1, 2.0)
        s1 = smoother.smooth(a1)  # alpha * 1.0 + (1-alpha) * 0 = alpha
        s2 = smoother.smooth(a2)  # alpha * 2.0 + (1-alpha) * s1
        expected = alpha * a2 + (1 - alpha) * s1
        assert torch.allclose(s2, expected, atol=1e-6)

    def test_ema_converges_to_constant_input(self, device):
        """Repeatedly feeding a constant action must converge to that action."""
        alpha = 0.3
        smoother = _make_smoother(alpha=alpha, device=device)
        constant = torch.full((NUM_ENVS, NUM_AGENTS, ACTION_DIM), 0.7, device=device)
        out = None
        for _ in range(200):
            out = smoother.smooth(constant)
        assert torch.allclose(out, constant, atol=1e-3), (
            "EMA must converge to constant input after enough steps"
        )

    def test_low_alpha_output_closer_to_prev(self, device):
        """Low alpha = more smoothing: output should be closer to previous state."""
        smoother_high = _make_smoother(alpha=0.9, device=device)
        smoother_low = _make_smoother(alpha=0.1, device=device)

        actions_1 = torch.zeros(NUM_ENVS, NUM_AGENTS, ACTION_DIM, device=device)
        actions_2 = torch.ones(NUM_ENVS, NUM_AGENTS, ACTION_DIM, device=device)

        # Warm up both to zero
        for _ in range(10):
            smoother_high.smooth(actions_1)
            smoother_low.smooth(actions_1)

        # Step to actions_2
        out_high = smoother_high.smooth(actions_2)
        out_low = smoother_low.smooth(actions_2)

        # Low alpha should be closer to 0 (previous), high alpha closer to 1 (new)
        dist_high_to_one = (out_high - actions_2).abs().mean()
        dist_low_to_one = (out_low - actions_2).abs().mean()
        assert dist_low_to_one > dist_high_to_one, (
            "Lower alpha must produce output further from the new action (more inertia)"
        )


# ---------------------------------------------------------------------------
# Variance / jitter reduction
# ---------------------------------------------------------------------------


class TestVarianceReduction:
    def test_smoothing_reduces_output_variance(self, device):
        """White noise input with alpha<1 must have lower output variance than input."""
        torch.manual_seed(7)
        alpha = 0.3
        n_steps = 200
        smoother = _make_smoother(alpha=alpha, buffer_size=n_steps, device=device)
        outputs = []
        for _ in range(n_steps):
            noise = torch.randn(NUM_ENVS, NUM_AGENTS, ACTION_DIM, device=device)
            out = smoother.smooth(noise)
            outputs.append(out)

        outputs_t = torch.stack(outputs, dim=0)   # [n_steps, E, A, D]
        noise_var = 1.0  # white noise variance ≈ 1
        smooth_var = outputs_t.var(dim=0).mean().item()
        assert smooth_var < noise_var * 0.9, (
            f"Smoothed variance ({smooth_var:.4f}) must be < white noise variance ({noise_var:.4f})"
        )

    def test_jitter_lower_after_smoothing(self, device):
        """Output of low-alpha smoother must have lower std than output of alpha=1.0 (raw)."""
        torch.manual_seed(42)
        n_steps = 200
        smoother_smooth = _make_smoother(alpha=0.2, buffer_size=n_steps, device=device)
        smoother_raw = _make_smoother(alpha=1.0, buffer_size=n_steps, device=device)

        out_smooth_list = []
        out_raw_list = []
        # Use same noise sequence for both smoothers
        noise_seq = [torch.randn(NUM_ENVS, NUM_AGENTS, ACTION_DIM, device=device)
                     for _ in range(n_steps)]
        for noise in noise_seq:
            out_smooth_list.append(smoother_smooth.smooth(noise))
            out_raw_list.append(smoother_raw.smooth(noise))

        # Compute std of outputs over time
        out_smooth = torch.stack(out_smooth_list, dim=0)  # [n_steps, E, A, D]
        out_raw = torch.stack(out_raw_list, dim=0)

        jitter_smooth = out_smooth.std(dim=0).mean().item()
        jitter_raw = out_raw.std(dim=0).mean().item()
        assert jitter_smooth < jitter_raw, (
            f"Smoothed output std ({jitter_smooth:.4f}) must be less than raw output std ({jitter_raw:.4f})"
        )


# ---------------------------------------------------------------------------
# Reset isolation
# ---------------------------------------------------------------------------


class TestResetEnvs:
    def test_reset_zeros_smooth_prev(self, device):
        smoother = _make_smoother(device=device)
        # Warm up
        for _ in range(5):
            smoother.smooth(_rand_actions(device=device))

        env_ids = torch.arange(NUM_ENVS, device=device)
        smoother.reset_envs(env_ids)
        assert torch.all(smoother._smooth_prev == 0.0), "reset_envs must zero the EMA state"

    def test_reset_zeros_buffer(self, device):
        smoother = _make_smoother(device=device)
        for _ in range(5):
            smoother.smooth(_rand_actions(device=device))
        env_ids = torch.arange(NUM_ENVS, device=device)
        smoother.reset_envs(env_ids)
        assert torch.all(smoother._buffer == 0.0), "reset_envs must zero the history buffer"

    def test_reset_does_not_affect_other_envs(self, device):
        smoother = _make_smoother(device=device)
        # Warm up all envs
        for _ in range(5):
            smoother.smooth(_rand_actions(device=device))

        prev_state_env1 = smoother._smooth_prev[1].clone()
        env_ids = torch.tensor([0], device=device)  # only env 0
        smoother.reset_envs(env_ids)

        assert torch.all(smoother._smooth_prev[0] == 0.0), "Env 0 state must be zeroed"
        if NUM_ENVS > 1:
            assert torch.allclose(smoother._smooth_prev[1], prev_state_env1), (
                "Env 1 state must not change when only env 0 is reset"
            )

    def test_after_reset_first_step_matches_fresh_smoother(self, device):
        """After reset, the first smooth() call must match a brand-new smoother."""
        smoother = _make_smoother(alpha=0.4, device=device)
        # Warm up
        for _ in range(10):
            smoother.smooth(_rand_actions(device=device))
        # Reset env 0
        smoother.reset_envs(torch.tensor([0], device=device))

        fresh = _make_smoother(alpha=0.4, device=device)
        actions = _rand_actions(device=device)
        out_reset = smoother.smooth(actions)[0]
        out_fresh = fresh.smooth(actions)[0]
        assert torch.allclose(out_reset, out_fresh, atol=1e-6), (
            "First step after reset must match output of a fresh smoother"
        )


# ---------------------------------------------------------------------------
# Action history buffer
# ---------------------------------------------------------------------------


class TestActionHistoryBuffer:
    def test_get_action_history_shape(self, device):
        buf = 8
        smoother = _make_smoother(buffer_size=buf, device=device)
        for _ in range(buf):
            smoother.smooth(_rand_actions(device=device))
        history = smoother.get_action_history()
        assert history.shape == (NUM_ENVS, NUM_AGENTS, buf, ACTION_DIM)

    def test_buffer_size_maintained_before_full(self, device):
        """History shape must be buffer_size even before the buffer is filled."""
        buf = 10
        smoother = _make_smoother(buffer_size=buf, device=device)
        # Feed only 3 steps
        for _ in range(3):
            smoother.smooth(_rand_actions(device=device))
        history = smoother.get_action_history()
        assert history.shape[2] == buf, "get_action_history must always return buffer_size entries"

    def test_buffer_contains_last_n_actions(self, device):
        """The history buffer must contain the most recently written actions."""
        buf = 4
        smoother = _make_smoother(buffer_size=buf, device=device)
        last_actions = []
        for i in range(buf):
            a = torch.full((NUM_ENVS, NUM_AGENTS, ACTION_DIM), float(i) * 0.1, device=device)
            smoother.smooth(a)
            last_actions.append(a.clone())

        history = smoother.get_action_history()
        # The history (chronological order) should contain exactly the last `buf` raw inputs
        for step_idx, expected in enumerate(last_actions):
            assert torch.allclose(history[:, :, step_idx, :], expected, atol=1e-6), (
                f"Buffer entry {step_idx} must match raw action from step {step_idx}"
            )


# ---------------------------------------------------------------------------
# NaN safety
# ---------------------------------------------------------------------------


class TestNaNSafety:
    def test_no_nan_for_random_inputs(self, device):
        smoother = _make_smoother(device=device)
        torch.manual_seed(0)
        for _ in range(50):
            actions = torch.randn(NUM_ENVS, NUM_AGENTS, ACTION_DIM, device=device)
            out = smoother.smooth(actions)
            assert not torch.isnan(out).any(), "NaN detected in smoother output"

    def test_no_nan_with_near_zero_inputs(self, device):
        smoother = _make_smoother(device=device)
        tiny = torch.full((NUM_ENVS, NUM_AGENTS, ACTION_DIM), 1e-38, device=device)
        out = smoother.smooth(tiny)
        assert not torch.isnan(out).any()

    def test_no_nan_after_reset(self, device):
        smoother = _make_smoother(device=device)
        for _ in range(5):
            smoother.smooth(_rand_actions(device=device))
        smoother.reset_envs(torch.arange(NUM_ENVS, device=device))
        out = smoother.smooth(_rand_actions(device=device))
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# compute_jitter
# ---------------------------------------------------------------------------


class TestComputeJitter:
    def test_jitter_shape(self, device):
        smoother = _make_smoother(device=device)
        for _ in range(8):
            smoother.smooth(_rand_actions(device=device))
        jitter = smoother.compute_jitter()
        assert jitter.shape == (NUM_ENVS, NUM_AGENTS)

    def test_jitter_non_negative(self, device):
        smoother = _make_smoother(device=device)
        torch.manual_seed(99)
        for _ in range(8):
            smoother.smooth(torch.randn(NUM_ENVS, NUM_AGENTS, ACTION_DIM, device=device))
        jitter = smoother.compute_jitter()
        assert torch.all(jitter >= 0.0), "Jitter (std) must be non-negative"

    def test_jitter_zero_for_constant_input(self, device):
        """Constant input → zero variance in buffer → zero jitter."""
        buf = 8
        smoother = _make_smoother(buffer_size=buf, device=device)
        constant = torch.full((NUM_ENVS, NUM_AGENTS, ACTION_DIM), 0.5, device=device)
        for _ in range(buf):
            smoother.smooth(constant)
        jitter = smoother.compute_jitter()
        assert torch.allclose(jitter, torch.zeros_like(jitter), atol=1e-5), (
            "Constant input should produce zero jitter in the buffer"
        )
