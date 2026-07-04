"""Simulated UWB peer-ranging channel. Pure torch — no Isaac imports.

Model per undirected link (i, j): d = ||x_i - x_j|| + bias + N(0, sigma),
symmetric per-link Bernoulli dropout, fixed latency via ring buffer, and a
persistent per-drone fault bias (inject_fault) for FN/recovery evaluation.
Noise defaults calibrated to Crazyflie LPS/TWR literature (see
docs/ggswarm_live/decentralization_plan.md §3).

All buffers are preallocated in __init__ and mutated in place — measure() is
called from the env step path, which bans explicit tensor construction.
"""

from __future__ import annotations

import torch


class UwbRangingSim:
    """Vectorized UWB ranging simulator over [num_envs, num_agents] drones."""

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        device: torch.device,
        *,
        noise_std: float,
        bias: float,
        dropout_prob: float,
        latency_steps: int,
    ) -> None:
        E, A = num_envs, num_agents
        L = max(0, int(latency_steps))
        self._E, self._A, self._L = E, A, L
        self.noise_std = float(noise_std)
        self.bias = float(bias)
        self.dropout_prob = float(dropout_prob)
        self._t = 0

        self._noise = torch.zeros(E, A, A, device=device)  # shape: [E, A, A]
        self._keep = torch.zeros(E, A, A, device=device)  # shape: [E, A, A]
        self._ring = torch.zeros(L + 1, E, A, A, device=device)  # shape: [L+1, E, A, A]
        self._ring_valid = torch.zeros(L + 1, E, A, A, dtype=torch.bool, device=device)
        self._held = torch.zeros(E, A, A, device=device)  # last valid range per link
        self._fault_bias = torch.zeros(E, A, device=device)  # shape: [E, A]
        self._eye = torch.eye(A, dtype=torch.bool, device=device).unsqueeze(0)  # [1, A, A]

    def measure(self, pos_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pos_g: [E, A, 3] true env-local positions -> (ranges [E, A, A], valid [E, A, A])."""
        true_d = torch.cdist(pos_g, pos_g)  # shape: [E, A, A]

        # Symmetric Gaussian noise: draw full matrix, mirror the upper triangle.
        self._noise.normal_(0.0, self.noise_std)
        n_upper = torch.triu(self._noise, diagonal=1)
        sym_noise = n_upper + n_upper.transpose(1, 2)

        # Per-drone fault bias applies to every link touching a faulted drone.
        fb = self._fault_bias.unsqueeze(2) + self._fault_bias.unsqueeze(1)  # [E, A, A]

        ranges_now = true_d + self.bias + sym_noise + fb

        # Symmetric per-link Bernoulli dropout (upper triangle mirrored).
        self._keep.bernoulli_(1.0 - self.dropout_prob)
        k_upper = torch.triu(self._keep, diagonal=1)
        valid_now = (k_upper + k_upper.transpose(1, 2)) > 0.5
        valid_now = valid_now & ~self._eye

        # Ring buffer: write t, read t - L.
        w = self._t % (self._L + 1)
        r = (self._t - self._L) % (self._L + 1)
        self._ring[w].copy_(ranges_now)
        self._ring_valid[w].copy_(valid_now)
        ranges_d = self._ring[r]
        valid_d = self._ring_valid[r]
        self._t += 1

        # Hold-last-valid so returned ranges are always finite.
        self._held.copy_(torch.where(valid_d, ranges_d, self._held))
        return self._held, valid_d

    def inject_fault(self, fault_mask: torch.Tensor, bias_m: float) -> None:
        """fault_mask: [E, A] bool — persistent range bias until reset_idx."""
        self._fault_bias[fault_mask] = float(bias_m)

    def reset_idx(self, env_ids: torch.Tensor, pos_g: torch.Tensor) -> None:
        """Seed ring/held with honest spawn ranges; clear faults. pos_g: [n_reset, A, 3]."""
        true_d = torch.cdist(pos_g, pos_g) + self.bias  # honest reading at spawn
        self._ring[:, env_ids] = true_d.unsqueeze(0)
        self._ring_valid[:, env_ids] = ~self._eye
        self._held[env_ids] = true_d
        self._fault_bias[env_ids] = 0.0
