"""MINCO minimum-jerk trajectory filter (L3 layer).

Single-segment minimum-jerk (s=3) trajectory from current state (x, v, a) to
target (x_target, 0, 0) over planning horizon T. Evaluates the closed-form
5th-order polynomial at t=dt each step, producing C2-continuous actions.

Replaces EMA smoothing with strictly superior higher-order smoothing.
Post-policy filter: no retraining needed. Enable via cfg.minco_enabled.

Reference: Wang et al. (2022), "Geometrically Constrained Trajectory
Optimization for Multicopters", IEEE T-RO — simplified to single segment.
"""

from __future__ import annotations

import torch


def apply_minco(
    target: torch.Tensor,
    pos: torch.Tensor,
    vel: torch.Tensor,
    acc: torch.Tensor,
    dt: float,
    horizon: float,
    max_vel: float,
    max_acc: float,
) -> torch.Tensor:
    """Advance the minimum-jerk filter one step toward the target action.

    Computes the unique 5th-order polynomial minimizing ∫|jerk|²dt from
    (pos, vel, acc) to (target, 0, 0) over horizon T, then evaluates at t=dt.

    The basis functions for normalized time tau = dt/T are:
        p(tau) = x0 + delta * h_p + vel*T * h_v + acc*T²/2 * h_a
    where delta = target - x0 and:
        h_p = 10τ³ - 15τ⁴ + 6τ⁵
        h_v = τ - 6τ³ + 8τ⁴ - 3τ⁵
        h_a = τ² - 3τ³ + 3τ⁴ - τ⁵

    Args:
        target: [N, 4] raw GNN actions (the waypoint to track)
        pos: [N, 4] current smoothed action (mutated in-place)
        vel: [N, 4] action velocity state (mutated in-place)
        acc: [N, 4] action acceleration state (mutated in-place)
        dt: env step time (seconds)
        horizon: planning horizon T (seconds)
        max_vel: velocity clamp (action-units/s)
        max_acc: acceleration clamp (action-units/s²)

    Returns:
        [N, 4] smoothed action (same tensor as pos)
    """
    # Normalized time
    tau = dt / horizon
    tau2 = tau * tau
    tau3 = tau2 * tau
    tau4 = tau3 * tau
    tau5 = tau4 * tau

    # Position basis functions (min-jerk polynomial)
    h_p = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
    h_v = tau - 6.0 * tau3 + 8.0 * tau4 - 3.0 * tau5
    h_a = tau2 - 3.0 * tau3 + 3.0 * tau4 - tau5

    # Velocity basis functions (dp/dt, divided by horizon for real-time vel)
    inv_T = 1.0 / horizon
    dh_p = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) * inv_T
    dh_v = (1.0 - 18.0 * tau2 + 32.0 * tau3 - 15.0 * tau4) * inv_T
    dh_a = (2.0 * tau - 9.0 * tau2 + 12.0 * tau3 - 5.0 * tau4) * 0.5

    # Acceleration basis functions (d²p/dt²)
    inv_T2 = inv_T * inv_T
    ddh_p = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) * inv_T2
    ddh_v = (-36.0 * tau + 96.0 * tau2 - 60.0 * tau3) * inv_T2
    ddh_a = (2.0 - 18.0 * tau + 36.0 * tau2 - 20.0 * tau3) * inv_T * 0.5

    # Displacement to target
    delta = target - pos  # shape: [N, 4]

    # Scaled state contributions
    vel_T = vel * horizon  # shape: [N, 4]
    acc_T2h = acc * (horizon * horizon * 0.5)  # shape: [N, 4]

    # Evaluate polynomial — new state
    new_pos = pos + delta * h_p + vel_T * h_v + acc_T2h * h_a  # shape: [N, 4]
    new_vel = delta * dh_p + vel_T * dh_v + acc_T2h * dh_a  # shape: [N, 4]
    new_acc = delta * ddh_p + vel_T * ddh_v + acc_T2h * ddh_a  # shape: [N, 4]

    # Clamp for stability
    new_vel.clamp_(-max_vel, max_vel)
    new_acc.clamp_(-max_acc, max_acc)

    # Update state in-place
    pos.copy_(new_pos)
    vel.copy_(new_vel)
    acc.copy_(new_acc)

    return pos.clamp(-1.0, 1.0)
