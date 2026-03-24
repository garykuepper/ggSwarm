"""Open-loop PD attitude sanity checks without Isaac Sim."""

from __future__ import annotations

import math

import torch

from attitude_controller import AttitudeControllerParams, compute_attitude_control


_DEFAULT_PARAMS = AttitudeControllerParams(
    kp_att=0.045,
    kd_att=0.00173,
    kp_yaw=0.01,
    max_tilt_angle=0.52,
    max_yaw_rate=3.14159,
    max_moment=0.08,
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


def test_no_saturation_at_max_tilt_and_moderate_ang_vel() -> None:
    """PD moment stays within max_moment at worst-case tilt and moderate angular velocity.

    With critically-damped gains (kd=0.00173, derived from Ixx=1.66e-5), the combined
    P+D moment at 30 deg tilt + 5 rad/s angular velocity must fit within max_moment=0.08.
    Previous overdamped gains (kd=0.005) caused the D-term to consume 107% of the P budget.
    """
    params = _DEFAULT_PARAMS
    robot_weight = 0.276

    # Test at several (tilt, angular velocity) combinations
    test_cases = [
        (0.52, 0.0, "max tilt, zero omega"),
        (0.52, 5.0, "max tilt, 5 rad/s"),
        (0.26, 10.0, "15 deg, 10 rad/s"),
        (0.10, 20.0, "6 deg, 20 rad/s"),
    ]

    for tilt_rad, ang_vel_val, label in test_cases:
        grav_y = math.sin(tilt_rad)
        grav_z = -math.cos(tilt_rad)
        proj = torch.tensor([[0.0, grav_y, grav_z]])
        ang_vel = torch.tensor([[ang_vel_val, 0.0, 0.0]])
        actions = torch.zeros(1, 4)

        thrust_buf = torch.zeros(1, 1, 3)
        moment_buf = torch.zeros(1, 1, 3)
        pre_clamp = torch.zeros(1, 1, 3)

        compute_attitude_control(
            actions, proj, ang_vel, robot_weight, params,
            thrust_buf, moment_buf, pre_clamp,
        )

        pre_moment = float(pre_clamp[0, 0, 0])
        clamped_moment = float(moment_buf[0, 0, 0])

        # Pre-clamp should not exceed max_moment (no saturation)
        assert abs(pre_moment) < params.max_moment, (
            f"[{label}] Saturated: pre-clamp={pre_moment:.6f} >= max_moment={params.max_moment}"
        )
        # Clamped and pre-clamp should be equal (no clipping occurred)
        assert abs(pre_moment - clamped_moment) < 1e-6, (
            f"[{label}] Moment was clipped: pre={pre_moment:.6f} vs clamped={clamped_moment:.6f}"
        )


def test_damping_ratio_is_near_critical() -> None:
    """Verify the gain/inertia combination produces near-critical damping (zeta ~ 1.0)."""
    kp = _DEFAULT_PARAMS.kp_att
    kd = _DEFAULT_PARAMS.kd_att
    I = _IXX
    zeta = kd / (2.0 * math.sqrt(kp * I))
    assert 0.8 <= zeta <= 1.3, (
        f"Damping ratio {zeta:.3f} is outside [0.8, 1.3] — "
        f"gains may need recalibration for Ixx={I:.2e}"
    )
