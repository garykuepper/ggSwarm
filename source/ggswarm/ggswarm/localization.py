"""Per-drone decentralized localization. Pure torch — no Isaac imports.

Odometry-anchored gauge (docs/ggswarm_live/decentralization_plan.md §3):
estimates seed from spawn truth, dead-reckon on noisy velocity odometry, and
each drone refines only its own estimate with damped Gauss-Newton steps
against peers' latency-delayed broadcast estimates and noisy UWB ranges.
No joint solve, no leader — the correct/test/recover math is the
companion-computer algorithm.

Per-control-tick call order (innovation gating — flags are decided on the
PRE-FIT residual, before any GN iteration, so a faulted tick never
contaminates estimates):

    loc.propagate(vel, dt)                          # dead-reckon on odometry
    ranges, valid = uwb.measure(pos)                # env-side ranging
    loc.update_residuals(ranges, valid, alive, dt)  # pre-fit residual + _b_pred
    loc.run_fault_test(mu, sigma, k)                # flags from pre-fit residual
    loc.correct(ranges, valid, alive, dt)           # GN, gated by ~flags (both ends)
    loc.recover(ranges, valid, alive, dt)           # IRLS, accept-if-consistent

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
        recovery_jump_gate: float = 2.0,
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
        # Recovery candidates may not jump more than this multiple of the
        # fault threshold away from the dead-reckoned estimate (see recover()).
        self.recovery_jump_gate = float(recovery_jump_gate)

        self.p_hat = torch.zeros(E, A, 3, device=device)  # shape: [E, A, 3]
        self._p_broadcast = torch.zeros(E, A, 3, device=device)  # peers' delayed view
        self._v_broadcast = torch.zeros(E, A, 3, device=device)  # peers' delayed odom velocity
        self._v_last = torch.zeros(E, A, 3, device=device)  # own last-used odom velocity
        self._b_pred = torch.zeros(E, A, 3, device=device)  # forward-predicted peer positions
        self._odom_noise = torch.zeros(E * A, 3, device=device)  # shape: [E*A, 3]
        self.residual = torch.zeros(E, A, device=device)  # shape: [E, A]
        self.flags = torch.zeros(E, A, dtype=torch.bool, device=device)
        self._p_snap = torch.zeros(E, A, 3, device=device)  # pre-recovery snapshot
        # Acceptance threshold for recovery candidates; set by run_fault_test.
        self._fault_threshold = float("inf")

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
        return self._gn_step_with(ranges, w, diff, dist, r)

    def _gn_step_with(
        self,
        ranges: torch.Tensor,
        w: torch.Tensor,
        diff: torch.Tensor,
        dist: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """GN step from precomputed geometry (avoids recomputing diff/dist/r).

        `ranges` is unused directly (folded into `r` by the caller) but kept
        in the signature so `_gn_step` and `recover` share one call shape.
        """
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

        Innovation gating: links touching a flagged drone (either end) carry
        zero weight, so flagged drones neither corrupt honest peers nor
        self-correct against their own faulted ranges — they dead-reckon on
        odometry (clean under a ranging fault) until recover() re-fixes them.
        """
        # Subtract known (calibrated) bias from measured ranges.
        ranges = ranges - self.known_bias
        # Forward-predict stale peer broadcasts one tick using their own
        # broadcast odometry velocity (locally computable on hardware).
        # (Recomputed even if update_residuals already ran this tick, so
        # correct() also works standalone in honest-only call sites.)
        torch.add(self._p_broadcast, self._v_broadcast, alpha=dt, out=self._b_pred)
        ok = (~self.flags).float()  # shape: [E, A]
        w = self._link_weights(valid, alive_g) * ok.unsqueeze(1) * ok.unsqueeze(2)
        for _ in range(self.correct_iters):
            self.p_hat -= self.damping * self._gn_step(ranges, w)

        # Publish (position, velocity) for the next step's peers
        # (1-step broadcast latency).
        self._p_broadcast.copy_(self.p_hat)
        self._v_broadcast.copy_(self._v_last)

    # ------------------------------------------------- fault test + recovery

    def update_residuals(
        self, ranges: torch.Tensor, valid: torch.Tensor, alive_g: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """PRE-FIT residual per drone, before any GN iteration this tick.

        residual_i = median_j |dist(p_hat_i, b_pred_j) - (d_ij - known_bias)|
        over usable links. Median, not mean: an honest drone sees at most a
        few faulted links among its peers, and the median rejects them, so
        one faulted drone does not push honest peers over the flag threshold
        — while a drone whose OWN ranging is faulted sees the bias on every
        link and its median residual jumps. Already-flagged targets are also
        excluded (flag state is broadcast alongside position/velocity, so
        this stays fully local). A drone with zero usable links keeps its
        previous residual — no evidence never clears (or raises) a verdict,
        which prevents flag flapping when links vanish.
        Also builds this tick's _b_pred, which correct()/recover() reuse.
        Call BEFORE run_fault_test/correct/recover. Returns residual [E, A].
        """
        ranges = ranges - self.known_bias
        torch.add(self._p_broadcast, self._v_broadcast, alpha=dt, out=self._b_pred)
        w = self._link_weights(valid, alive_g) * (~self.flags).unsqueeze(1).float()
        diff = self.p_hat.unsqueeze(2) - self._b_pred.unsqueeze(1)  # [E, A, A, 3]
        r_abs = (diff.norm(dim=3) - ranges).abs()  # shape: [E, A, A]
        r_abs = r_abs.masked_fill(w <= 0.0, float("nan"))
        med = r_abs.nanmedian(dim=2).values  # NaN where a drone has no usable links
        self.residual.copy_(torch.where(torch.isnan(med), self.residual, med))
        return self.residual

    def run_fault_test(self, mu: float, sigma: float, k: float) -> torch.Tensor:
        """Flag drones whose pre-fit residual exceeds mu + k*sigma (local verdict)."""
        self._fault_threshold = float(mu + k * sigma)
        self.flags.copy_(self.residual > self._fault_threshold)
        return self.flags

    def recover(
        self, ranges: torch.Tensor, valid: torch.Tensor, alive_g: torch.Tensor, dt: float
    ) -> None:
        """Accept-if-consistent IRLS/Huber re-multilateration of flagged drones.

        Mirrors the SwarmRaft paper's Stage-2 recovery without its Raft
        transport. Drones with < min_recovery_peers usable non-flagged peers
        keep their dead-reckoned estimate (the paper's INS-fallback note).

        The IRLS candidate is only ACCEPTED where BOTH hold; otherwise the
        drone reverts to its dead-reckoned estimate:

        1. Range-consistent: the candidate's residual against this tick's
           forward-predicted targets falls back below the run_fault_test
           threshold.
        2. Odometry-consistent: the candidate jump from the dead-reckoned
           estimate is <= recovery_jump_gate * threshold. Needed because a
           common range bias is nearly geometrically consistent with a
           displaced position (measured on the r=1 octagon, +1.0 m bias: the
           converged wrong fit has mean residual ~0.11 vs honest floor
           ~0.077 — range residuals alone cannot separate). Odometry can:
           the pre-fit test flags divergence the tick it crosses threshold,
           so a legitimate recovery correction is threshold-sized, while a
           biased-ranging fit demands a jump far beyond it.

        Under a ranging fault the candidate demands a ~bias-sized jump ->
        revert -> error stays at odometry-drift level. Under estimate
        divergence with clean ranging the candidate is consistent and
        threshold-sized -> accepted (the paper-faithful path).

        Targets reuse `_b_pred` built earlier this tick by update_residuals/
        correct (`_b_pred = _p_broadcast + _v_broadcast * dt`); recover() is
        always called after them in the tick order, so it is not recomputed.
        `dt` is accepted for signature symmetry with correct().
        """
        if not self.flags.any():
            return
        # Subtract known (calibrated) bias, mirroring correct()'s treatment.
        ranges = ranges - self.known_bias
        # Usable target links: valid, both alive, target NOT flagged.
        w_ok = self._link_weights(valid, alive_g) * (~self.flags).unsqueeze(1).float()
        enough = w_ok.sum(dim=2) >= float(self.min_recovery_peers)  # [E, A]
        upd = self.flags & enough & alive_g  # shape: [E, A]
        upd_mask = upd.unsqueeze(2).float()  # [E, A, 1]
        if upd_mask.sum() == 0:
            return
        # Snapshot the dead-reckoned estimates as the revert fallback.
        self._p_snap.copy_(self.p_hat)
        for _ in range(self.recovery_irls_iters):
            diff = self.p_hat.unsqueeze(2) - self._b_pred.unsqueeze(1)
            dist = diff.norm(dim=3).clamp(min=1e-6)
            r = dist - ranges
            huber = torch.clamp(self.recovery_huber_delta / r.abs().clamp(min=1e-6), max=1.0)
            w = w_ok * huber
            step = self._gn_step_with(ranges, w, diff, dist, r)
            self.p_hat -= self.damping * step * upd_mask
        # Accept-if-consistent: candidate residual vs the fault threshold,
        # AND candidate jump vs the odometry-drift envelope (see docstring).
        diff = self.p_hat.unsqueeze(2) - self._b_pred.unsqueeze(1)
        r_abs = (diff.norm(dim=3) - ranges).abs() * w_ok
        cand_res = r_abs.sum(dim=2) / w_ok.sum(dim=2).clamp(min=1.0)  # [E, A]
        jump = (self.p_hat - self._p_snap).norm(dim=2)  # shape: [E, A]
        consistent = (cand_res <= self._fault_threshold) & (
            jump <= self.recovery_jump_gate * self._fault_threshold
        )
        accept = (upd & consistent).unsqueeze(2)  # [E, A, 1]
        self.p_hat.copy_(torch.where(accept, self.p_hat, self._p_snap))
        self._p_broadcast.copy_(self.p_hat)

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
