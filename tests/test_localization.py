"""Synthetic-trajectory tests for DecentralizedLocalizer (pure torch, no Isaac)."""
import math

import torch

from ggswarm.localization import DecentralizedLocalizer
from ggswarm.ranging import UwbRangingSim

E, A, DT = 32, 8, 0.02
DEV = torch.device("cpu")


def make_stack(noise_std=0.10, bias=0.05, dropout=0.05, latency=1, odom_std=0.02, known_bias=None):
    if known_bias is None:
        known_bias = bias
    rng = UwbRangingSim(E, A, DEV, noise_std=noise_std, bias=bias,
                        dropout_prob=dropout, latency_steps=latency)
    loc = DecentralizedLocalizer(E, A, DEV, correct_iters=3, damping=0.5,
                                 odom_noise_std=odom_std, recovery_irls_iters=5,
                                 recovery_huber_delta=0.10, min_recovery_peers=4,
                                 known_bias=known_bias)
    return rng, loc


def octagon_traj(t):
    """Octagon of radius 1 whose centroid orbits slowly — pos [E, A, 3], vel [E*A, 3]."""
    theta = torch.arange(A) * (2 * math.pi / A)
    base = torch.stack([theta.cos(), theta.sin(), torch.ones(A)], dim=1)  # [A, 3]
    cx, cy = 0.5 * math.sin(0.2 * t), 0.5 * math.cos(0.2 * t)
    vx, vy = 0.5 * 0.2 * math.cos(0.2 * t), -0.5 * 0.2 * math.sin(0.2 * t)
    pos = base.unsqueeze(0).expand(E, A, 3).contiguous()
    pos = pos + torch.tensor([cx, cy, 0.0])
    vel = torch.tensor([vx, vy, 0.0]).expand(E * A, 3).contiguous()
    return pos, vel


def run_honest(steps=500, **kw):
    torch.manual_seed(0)
    rng, loc = make_stack(**kw)
    pos0, _ = octagon_traj(0.0)
    loc.reset_idx(torch.arange(E), pos0)
    rng.reset_idx(torch.arange(E), pos0)
    alive = torch.ones(E, A, dtype=torch.bool)
    for s in range(steps):
        pos, vel = octagon_traj(s * DT)
        loc.propagate(vel, DT)
        ranges, valid = rng.measure(pos)
        loc.correct(ranges, valid, alive, DT)
    return rng, loc, pos, alive


def test_steady_state_rmse():
    _, loc, pos, alive = run_honest()
    rmse = loc.rmse(pos, alive)
    assert rmse.mean().item() <= 0.10, f"RMSE {rmse.mean():.3f} > 0.10 m"


def test_no_reflection_flips():
    _, loc, pos, alive = run_honest()
    per_drone_err = (loc.p_hat - pos).norm(dim=2)  # [E, A]
    assert per_drone_err.max().item() < 0.5  # a mirror flip would be ~2 m on an r=1 octagon


def test_gauge_drift_within_random_walk_envelope():
    _, loc, pos, alive = run_honest()
    drift = loc.gauge_drift(pos, alive)
    # Common-mode random walk: odom_std * dt * sqrt(steps / A) ~ 0.003 m; allow 20x slack.
    assert drift.mean().item() < 0.06


def test_dead_reckon_fallback_no_valid_links():
    torch.manual_seed(0)
    rng, loc = make_stack(dropout=1.0, latency=0, odom_std=0.0)
    pos0, _ = octagon_traj(0.0)
    loc.reset_idx(torch.arange(E), pos0)
    rng.reset_idx(torch.arange(E), pos0)
    alive = torch.ones(E, A, dtype=torch.bool)
    for s in range(50):
        pos, vel = octagon_traj(s * DT)
        loc.propagate(vel, DT)
        ranges, valid = rng.measure(pos)
        loc.correct(ranges, valid, alive, DT)  # all links invalid -> pure dead reckoning
    assert loc.rmse(pos, alive).mean().item() < 0.02  # noiseless odom integrates cleanly


def test_nan_free_at_10x_noise():
    _, loc, pos, alive = run_honest(steps=300, noise_std=1.0, bias=0.5, odom_std=0.2)
    assert torch.isfinite(loc.p_hat).all()


def run_with_fault(fault_drone=3, fault_bias=1.0, steps=300, fault_at=150,
                   calib_from=50, recover=True):
    """Full tick order with mu/sigma calibrated from the honest pre-fault window."""
    torch.manual_seed(0)
    rng, loc = make_stack()
    pos0, _ = octagon_traj(0.0)
    loc.reset_idx(torch.arange(E), pos0)
    rng.reset_idx(torch.arange(E), pos0)
    alive = torch.ones(E, A, dtype=torch.bool)
    mask = torch.zeros(E, A, dtype=torch.bool)
    mask[:, fault_drone] = True
    calib, mu, sigma, flagged_at = [], None, None, None
    for s in range(steps):
        if s == fault_at:
            rng.inject_fault(mask, fault_bias)
            cal = torch.stack(calib)
            mu, sigma = cal.mean().item(), cal.std().item()
        pos, vel = octagon_traj(s * DT)
        loc.propagate(vel, DT)
        ranges, valid = rng.measure(pos)
        res = loc.update_residuals(ranges, valid, alive, DT)
        if mu is None:
            if s >= calib_from:
                calib.append(res.clone())
        else:
            loc.run_fault_test(mu, sigma, 3.0)
            if flagged_at is None and loc.flags[:, fault_drone].float().mean() > 0.5:
                flagged_at = s
        loc.correct(ranges, valid, alive, DT)
        if recover and mu is not None:
            loc.recover(ranges, valid, alive, DT)
    return loc, pos, alive, flagged_at


def test_fault_is_flagged():
    loc, pos, alive, flagged_at = run_with_fault(recover=False)
    assert flagged_at is not None and flagged_at - 150 <= 25  # flagged within 0.5 s


def test_false_positive_rate_honest():
    """Calibrate mu/sigma from an honest run, then assert flag rate <= 1%."""
    torch.manual_seed(0)
    rng, loc = make_stack()
    pos0, _ = octagon_traj(0.0)
    loc.reset_idx(torch.arange(E), pos0)
    rng.reset_idx(torch.arange(E), pos0)
    alive = torch.ones(E, A, dtype=torch.bool)
    calib = []
    for s in range(200):
        pos, vel = octagon_traj(s * DT)
        loc.propagate(vel, DT)
        ranges, valid = rng.measure(pos)
        res = loc.update_residuals(ranges, valid, alive, DT)
        if s >= 50:  # skip the settling transient
            calib.append(res.clone())
        loc.correct(ranges, valid, alive, DT)
    cal = torch.stack(calib)
    mu, sigma = cal.mean().item(), cal.std().item()
    rates = []
    for s in range(200, 300):
        pos, vel = octagon_traj(s * DT)
        loc.propagate(vel, DT)
        ranges, valid = rng.measure(pos)
        loc.update_residuals(ranges, valid, alive, DT)
        loc.run_fault_test(mu, sigma, 3.0)
        rates.append(loc.flags.float().mean().item())
        loc.correct(ranges, valid, alive, DT)
    assert sum(rates) / len(rates) <= 0.01


def test_recovery_bounds_error():
    loc, pos, alive, _ = run_with_fault(recover=True)
    err_faulted = (loc.p_hat[:, 3] - pos[:, 3]).norm(dim=1)
    assert err_faulted.mean().item() < 0.20  # recovered despite 1.0 m ranging bias


def test_recovery_skipped_below_min_peers():
    torch.manual_seed(0)
    rng, loc = make_stack(dropout=1.0, latency=0)  # no usable links at all
    pos0, _ = octagon_traj(0.0)
    loc.reset_idx(torch.arange(E), pos0)
    rng.reset_idx(torch.arange(E), pos0)
    alive = torch.ones(E, A, dtype=torch.bool)
    loc.flags[:, 2] = True
    before = loc.p_hat.clone()
    ranges, valid = rng.measure(pos0)
    loc.recover(ranges, valid, alive, DT)
    assert torch.allclose(loc.p_hat, before)  # dead-reckon hold, no update
