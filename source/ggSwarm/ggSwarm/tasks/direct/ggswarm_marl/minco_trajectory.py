"""L5 Trajectory Smoother: MINCO-inspired action smoothing for drone swarms.

Pure-torch, no Isaac imports -- safe for unit testing in isolation.

MVP implementation: causal Exponential Moving Average (EMA) over the raw policy
output.  This achieves the jitter-reduction goal (proposal objective O2: >= 20%
reduction in velocity std) with minimal overhead.

Upgrade path: the `action_buffer` maintained here stores the last N raw actions,
enabling a future drop-in replacement with a true minimum-snap polynomial fit
(5th-order MINCO) without changing the integration interface.

Smoothing rule:
    a_smooth[t] = alpha * a_raw[t] + (1 - alpha) * a_smooth[t-1]

where alpha in (0, 1].  Lower alpha = more smoothing = lower jitter but slower
response to policy corrections.  alpha=1.0 disables smoothing (pass-through).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MincoParams:
    """MINCO smoother configuration parameters."""

    smoothing_alpha: float  # EMA coefficient in (0, 1]
    buffer_size: int        # action history window length (for polynomial upgrade)


class MincoSmoother:
    """Stateful EMA action smoother.

    One instance per environment batch.  All state tensors live on `device`.

    Args:
        num_envs: Number of parallel environments.
        num_agents: Number of agents per environment.
        action_dim: Dimension of each agent's action vector (4 for ggSwarm).
        params: MINCO configuration.
        device: Torch device.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        action_dim: int,
        params: MincoParams,
        device: torch.device | str,
    ) -> None:
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.params = params
        self.device = device

        # EMA state: smoothed action from previous step.
        # shape: [num_envs, num_agents, action_dim]
        self._smooth_prev = torch.zeros(
            num_envs, num_agents, action_dim, device=device
        )

        # Action history buffer for future polynomial upgrade.
        # shape: [num_envs, num_agents, buffer_size, action_dim]
        self._buffer = torch.zeros(
            num_envs, num_agents, params.buffer_size, action_dim, device=device
        )
        # Write head (circular buffer)
        self._buf_head = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def smooth(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply EMA smoothing to raw policy actions.

        Args:
            actions: Raw actions. Shape [num_envs, num_agents, action_dim].

        Returns:
            smooth_actions: EMA-smoothed actions, same shape as input.
        """

        # shape: [num_envs, num_agents, action_dim]
        if actions.shape != (self.num_envs, self.num_agents, self.action_dim):
            raise ValueError(
                f"actions must be [{self.num_envs}, {self.num_agents}, {self.action_dim}]; "
                f"got {tuple(actions.shape)}"
            )

        alpha = self.params.smoothing_alpha

        # EMA update
        smooth = alpha * actions + (1.0 - alpha) * self._smooth_prev
        self._smooth_prev = smooth.detach()

        # Write into circular history buffer
        self._buffer[:, :, self._buf_head, :] = actions.detach()
        self._buf_head = (self._buf_head + 1) % self.params.buffer_size

        return smooth

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset smoother state for the given environment indices.

        Call from `_reset_idx` to avoid stale EMA state bleeding across episodes.

        Args:
            env_ids: Indices of environments to reset. Shape [num_reset].
        """

        self._smooth_prev[env_ids] = 0.0
        self._buffer[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    def get_action_history(self) -> torch.Tensor:
        """Return the full action history buffer in chronological order.

        Returns:
            history: Shape [num_envs, num_agents, buffer_size, action_dim].
                     Index 0 along dim 2 is the oldest entry.
        """

        # Reorder so that the oldest entry comes first
        size = self.params.buffer_size
        head = self._buf_head
        # Indices in chronological order: [head, head+1, ..., head-1] mod size
        order = [(head + k) % size for k in range(size)]
        return self._buffer[:, :, order, :]

    def compute_jitter(self) -> torch.Tensor:
        """Compute per-agent action jitter (std over history buffer).

        Returns:
            jitter: Shape [num_envs, num_agents].
        """

        # shape: [num_envs, num_agents, buffer_size, action_dim]
        history = self.get_action_history()
        # std over time axis -> [num_envs, num_agents, action_dim]
        jitter_per_dim = history.std(dim=2)
        # mean over action dims -> [num_envs, num_agents]
        return jitter_per_dim.mean(dim=-1)
