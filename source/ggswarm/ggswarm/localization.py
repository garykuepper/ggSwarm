"""Per-drone decentralized localization. Pure torch — no Isaac imports.

Odometry-anchored gauge (docs/ggswarm_live/decentralization_plan.md §3):
estimates seed from spawn truth, dead-reckon on noisy velocity odometry, and
each drone refines only its own estimate with damped Gauss-Newton steps
against peers' latency-delayed broadcast estimates and noisy UWB ranges.
No joint solve, no leader — the correct/test/recover math is the
companion-computer algorithm.

Called from the env step path: no explicit tensor construction here; scratch
buffers preallocated in __init__, RNG in-place.
"""

from __future__ import annotations

import torch


class DecentralizedLocalizer:
    """Vectorized per-drone estimator over [num_envs, num_agents] drones."""

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        device: torch.device,
        *,
        correct_iters: int,
        damping: float,
        odom_noise_std: float,
        recovery_irls_iters: int,
        recovery_huber_delta: float,
        min_recovery_peers: int,
        known_bias: float = 0.0,
    ) -> None:
        E, A = num_envs, num_agents
        self._E, self._A = E, A
        self.correct_iters = int(correct_iters)
        self.damping = float(damping)
        self.odom_noise_std = float(odom_noise_std)
        self.recovery_irls_iters = int(recovery_irls_iters)
        self.recovery_huber_delta = float(recovery_huber_delta)
        self.min_recovery_peers = int(min_recovery_peers)
        self.known_bias = float(known_bias)

        self.p_hat = torch.zeros(E, A, 3, device=device)  # shape: [E, A, 3]
        self._p_broadcast = torch.zeros(E, A, 3, device=device)  # peers' delayed view
        self._v_broadcast = torch.zeros(E, A, 3, device=device)  # peers' delayed odom velocity
        self._v_last = torch.zeros(E, A, 3, device=device)  # own last-used odom velocity
        self._b_pred = torch.zeros(E, A, 3, device=device)  # forward-predicted peer positions
        self._odom_noise = torch.zeros(E * A, 3, device=device)  # shape: [E*A, 3]
        self.residual = torch.zeros(E, A, device=device)  # shape: [E, A]
        self.flags = torch.zeros(E, A, dtype=torch.bool, device=device)

    # ------------------------------------------------------------ core steps

    def propagate(self, lin_vel_w: torch.Tensor, dt: float, noise_scale: float = 1.0) -> None:
        """lin_vel_w: [E*A, 3] world-frame velocity (sim-perfect odom before noise)."""
        self._odom_noise.normal_(0.0, self.odom_noise_std * noise_scale)
        # Remember the (noisy) odom velocity actually integrated — it is
        # published alongside p_hat so peers can forward-predict our stale
        # broadcast by one tick (locally computable, hardware-legal).
        self._v_last.copy_((lin_vel_w + self._odom_noise).reshape(self._E, self._A, 3))
        self.p_hat += self._v_last * dt

    def _link_weights(self, valid: torch.Tensor, alive_g: torch.Tensor) -> torch.Tensor:
        """Valid link AND both endpoints alive. -> float [E, A, A]."""
        both_alive = alive_g.unsqueeze(1) & alive_g.unsqueeze(2)
        return (valid & both_alive).float()

    def _gn_step(self, ranges: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """One scaled-gradient Gauss-Newton step for every drone at once.

        Each drone i minimizes sum_j w_ij (||p_i - b_j|| - d_ij)^2 over its own
        p_i, holding forward-predicted peer broadcasts b_j fixed.
        Returns step [E, A, 3].
        """
        diff = self.p_hat.unsqueeze(2) - self._b_pred.unsqueeze(1)  # [E, A, A, 3]
        dist = diff.norm(dim=3).clamp(min=1e-6)  # shape: [E, A, A]
        r = dist - ranges
        u = diff / dist.unsqueeze(3)  # unit vectors i -> j
        g = ((w * r).unsqueeze(3) * u).sum(dim=2)  # [E, A, 3]
        wsum = w.sum(dim=2, keepdim=True).clamp(min=1e-6)  # [E, A, 1]
        return g / wsum

    def correct(
        self, ranges: torch.Tensor, valid: torch.Tensor, alive_g: torch.Tensor, dt: float
    ) -> None:
        """Refine own estimates against forward-predicted peer broadcasts.

        Broadcasts are one control tick stale (publish-then-read), and
        pairwise ranges are translation-invariant, so GN against raw stale
        broadcasts reads the swarm's real one-tick translation as a coherent
        residual and drags the gauge backwards every step. Fix, fully local
        per drone: peers broadcast (position, velocity), and each receiver
        dead-reckons every peer forward by dt before ranging against it.
        dt: control tick used to forward-predict (env passes step_dt).
        """
        # Subtract known (calibrated) bias from measured ranges.
        ranges = ranges - self.known_bias
        # Forward-predict stale peer broadcasts one tick using their own
        # broadcast odometry velocity (locally computable on hardware).
        torch.add(self._p_broadcast, self._v_broadcast, alpha=dt, out=self._b_pred)
        w = self._link_weights(valid, alive_g)
        for _ in range(self.correct_iters):
            self.p_hat -= self.damping * self._gn_step(ranges, w)

        # Per-drone residual: mean |range inconsistency| over usable links.
        diff = self.p_hat.unsqueeze(2) - self._b_pred.unsqueeze(1)
        r_abs = (diff.norm(dim=3) - ranges).abs() * w
        self.residual.copy_(r_abs.sum(dim=2) / w.sum(dim=2).clamp(min=1.0))

        # Publish (position, velocity) for the next step's peers
        # (1-step broadcast latency).
        self._p_broadcast.copy_(self.p_hat)
        self._v_broadcast.copy_(self._v_last)

    # ---------------------------------------------------------- diagnostics

    def rmse(self, pos_true_g: torch.Tensor, alive_g: torch.Tensor) -> torch.Tensor:
        """Position RMSE vs truth over alive drones, per env -> [E]."""
        sq = (self.p_hat - pos_true_g).square().sum(dim=2) * alive_g.float()
        n = alive_g.float().sum(dim=1).clamp(min=1.0)
        return (sq.sum(dim=1) / n).sqrt()

    def gauge_drift(self, pos_true_g: torch.Tensor, alive_g: torch.Tensor) -> torch.Tensor:
        """Norm of the common-mode (mean) estimate error over alive drones -> [E]."""
        err = (self.p_hat - pos_true_g) * alive_g.unsqueeze(2).float()
        n = alive_g.float().sum(dim=1).clamp(min=1.0).unsqueeze(1)
        return (err.sum(dim=1) / n).norm(dim=1)

    def reset_idx(self, env_ids: torch.Tensor, pos_true_g: torch.Tensor) -> None:
        """Seed estimates from spawn truth (takeoff-layout datum). pos_true_g: [n_reset, A, 3]."""
        self.p_hat[env_ids] = pos_true_g
        self._p_broadcast[env_ids] = pos_true_g
        self._v_broadcast[env_ids] = 0.0
        self._v_last[env_ids] = 0.0
        self.residual[env_ids] = 0.0
        self.flags[env_ids] = False
