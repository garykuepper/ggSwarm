"""Open-loop PD attitude sanity checks without Isaac Sim."""

from __future__ import annotations

import math

import torch

from attitude_controller import AttitudeControllerParams, compute_attitude_control


_DEFAULT_PARAMS = AttitudeControllerParams(
    kp_att=0.045,
    kd_att=0.005,
    kp_yaw=0.01,
    max_tilt_angle=0.52,
    max_yaw_rate=3.14159,
    max_moment=0.03,
    thrust_to_weight=2.0,
)

# Crazyflie sim inertia (from scripts/extract_crazyflie_inertia.py)
_IXX = 1.657e-5  # kg*m^2


def test_zero_actions_level_hover_wrench_is_stable_over_repeated_calls() -> None:
    """Constant level state + zero policy command → identical thrust/moment each step."""
    params = _DEFAULT_PARAMS
    proj = torch.tensor([[0.0, 0.0, -1.0]])
    ang_vel = torch.zeros(1, 3)
    actions = torch.zeros(1, 4)
    thrust0 = torch.zeros(1, 1, 3)
    moment0 = torch.zeros(1, 1, 3)
    compute_attitude_control(
        actions,
        proj,
        ang_vel,
        0.274,
        params,
        thrust0,
        moment0,
        None,
    )
    for _ in range(30):
        thrust = torch.zeros(1, 1, 3)
        moment = torch.zeros(1, 1, 3)
        compute_attitude_control(
            actions,
            proj,
            ang_vel,
            0.274,
            params,
            thrust,
            moment,
            None,
        )
        assert torch.allclose(thrust, thrust0)
        assert torch.allclose(moment, moment0)


def test_restoring_roll_moment_opposes_actual_roll_at_zero_desired() -> None:
    """Tilted gravity + zero attitude command → roll torque opposes roll error (restoring)."""
    params = _DEFAULT_PARAMS
    robot_weight = 0.274
    for phi in (0.15, -0.22, 0.4):
        grav_y = math.sin(phi)
        grav_z = -math.cos(phi)
        proj = torch.tensor([[0.0, grav_y, grav_z]])
        ang_vel = torch.zeros(1, 3)
        actions = torch.zeros(1, 4)
        thrust_buf = torch.zeros(1, 1, 3)
        moment_buf = torch.zeros(1, 1, 3)
        compute_attitude_control(
            actions,
            proj,
            ang_vel,
            robot_weight,
            params,
            thrust_buf,
            moment_buf,
            None,
        )
        mx = float(moment_buf[0, 0, 0])
        actual_roll = math.atan2(-grav_y, -grav_z)
        assert actual_roll != 0.0
        assert mx * actual_roll < 0.0


def test_p_term_fits_at_zero_angular_velocity() -> None:
    """PD P-term at max tilt fits within max_moment when angular velocity is zero.

    With overdamped gains (kd=0.005, zeta=2.9), saturation at high angular velocity
    is expected and intentional — the heavy damping acts as a safety limiter during
    early RL exploration (PD13/PD14 proved critical damping causes ballistic flips).
    But the pure P-term at max tilt should fit within the moment budget.
    """
    params = _DEFAULT_PARAMS
    robot_weight = 0.276

    # P-term only (zero angular velocity): should NOT saturate
    grav_y = math.sin(0.52)  # 30 deg tilt
    grav_z = -math.cos(0.52)
    proj = torch.tensor([[0.0, grav_y, grav_z]])
    ang_vel = torch.zeros(1, 3)
    actions = torch.zeros(1, 4)

    thrust_buf = torch.zeros(1, 1, 3)
    moment_buf = torch.zeros(1, 1, 3)
    pre_clamp = torch.zeros(1, 1, 3)

    compute_attitude_control(
        actions, proj, ang_vel, robot_weight, params,
        thrust_buf, moment_buf, pre_clamp,
    )

    pre_moment = float(pre_clamp[0, 0, 0])
    assert abs(pre_moment) < params.max_moment, (
        f"P-term alone saturates at max tilt: {pre_moment:.6f} >= {params.max_moment}"
    )


def test_damping_ratio_is_overdamped() -> None:
    """Verify the gain/inertia combination is intentionally overdamped (zeta > 1.0).

    PD13/PD14 proved that critical damping (zeta=1.0) causes ballistic flips during
    early RL exploration. The overdamped response (zeta ~2.9) acts as a natural safety
    limiter. The train-eval gap from saturation is addressed via eval_noise_std instead.
    """
    kp = _DEFAULT_PARAMS.kp_att
    kd = _DEFAULT_PARAMS.kd_att
    I = _IXX
    zeta = kd / (2.0 * math.sqrt(kp * I))
    assert zeta > 1.5, (
        f"Damping ratio {zeta:.3f} is not sufficiently overdamped — "
        f"PD13/PD14 showed zeta < 1.5 causes ballistic flips during RL exploration"
    )
