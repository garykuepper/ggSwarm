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
        loc.correct(ranges, valid, alive)
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
        loc.correct(ranges, valid, alive)  # all links invalid -> pure dead reckoning
    assert loc.rmse(pos, alive).mean().item() < 0.02  # noiseless odom integrates cleanly


def test_nan_free_at_10x_noise():
    _, loc, pos, alive = run_honest(steps=300, noise_std=1.0, bias=0.5, odom_std=0.2)
    assert torch.isfinite(loc.p_hat).all()
