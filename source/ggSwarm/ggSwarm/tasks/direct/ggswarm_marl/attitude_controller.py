"""Pure-torch PD attitude controller for the Crazyflie drone inner loop.

The RL policy outputs high-level attitude commands
``[thrust_cmd, desired_roll, desired_pitch, desired_yaw_rate]``.
This module converts those commands into physical thrust force and body-frame
moments using a PD controller for roll/pitch and a P controller for yaw rate.

Reference implementation: OmniDrones ``AttitudeController``
  (https://github.com/btx0424/OmniDrones/blob/main/omni_drones/controllers/lee_position_controller.py)
Real-hardware gains verified against Crazyflie 2.x firmware cascaded PID.
No Isaac Lab / Isaac Sim imports -- fully unit-testable with plain PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttitudeControllerParams:
    """Configurable gains for the inner-loop PD attitude controller.

    All fields correspond to ``GGSwarmMarlEnvCfg`` parameters (Rule 6).
    """

    # Attitude proportional gain (Nm/rad). Controls how aggressively the
    # controller corrects attitude error. OmniDrones reference: ~3.0 (pre-inertia).
    kp_att: float
    # Attitude derivative/damping gain (Nm/(rad/s)). Damps oscillation.
    # OmniDrones reference: ~0.52 (pre-inertia).
    kd_att: float
    # Yaw rate proportional gain (Nm/(rad/s)). Pure P control on yaw.
    kp_yaw: float
    # Maximum tilt angle commanded by the policy (rad). 0.52 rad ≈ 30 deg.
    max_tilt_angle: float
    # Maximum yaw rate commanded by the policy (rad/s). 3.14 rad/s ≈ 180 deg/s.
    max_yaw_rate: float
    # Moment output clamp (Nm). Prevents runaway on large errors at episode start.
    max_moment: float
    # Thrust mapping (same semantics as before: action=0 → hover).
    thrust_to_weight: float


def compute_attitude_control(
    actions: torch.Tensor,
    proj_grav_b: torch.Tensor,
    ang_vel_b: torch.Tensor,
    robot_weight: float,
    params: AttitudeControllerParams,
    thrust_buf: torch.Tensor,
    moment_buf: torch.Tensor,
) -> None:
    """Compute thrust and moments from attitude commands, writing into pre-allocated buffers.

    The caller (``GGSwarmMarlEnv._pre_physics_step``) passes ``thrust_buf`` and
    ``moment_buf`` pre-allocated in ``__init__`` so this function performs zero
    heap allocations per call (Rule 15).

    Args:
        actions: Policy actions, shape ``[N, 4]``, each in ``[-1, 1]``.
            ``actions[:, 0]`` -- collective thrust command.
            ``actions[:, 1]`` -- desired roll (scaled to ``±max_tilt_angle``).
            ``actions[:, 2]`` -- desired pitch (scaled to ``±max_tilt_angle``).
            ``actions[:, 3]`` -- desired yaw rate (scaled to ``±max_yaw_rate``).
        proj_grav_b: Gravity vector projected into body frame, shape ``[N, 3]``.
            When level: ``[0, 0, -1]``.
        ang_vel_b: Body-frame angular velocity, shape ``[N, 3]``.
        robot_weight: ``mass * gravity`` (scalar, Newtons).
        params: Controller gains (``AttitudeControllerParams``).
        thrust_buf: Pre-allocated output buffer, shape ``[N, 1, 3]``. Written in-place.
        moment_buf: Pre-allocated output buffer, shape ``[N, 1, 3]``. Written in-place.
    """
    # shape: [N]
    thrust_val = (actions[:, 0] + 1.0) * 0.5  # map [-1,1] -> [0,1]; 0 -> hover
    total_thrust = params.thrust_to_weight * robot_weight * thrust_val

    # Desired attitude from policy output (scaled to physical ranges)
    # shape: [N]
    desired_roll = actions[:, 1] * params.max_tilt_angle
    desired_pitch = actions[:, 2] * params.max_tilt_angle
    desired_yaw_rate = actions[:, 3] * params.max_yaw_rate

    # Current roll and pitch estimated from projected gravity.
    # When level, proj_grav_b = [0, 0, -1]:
    #   actual_roll  = atan2(-grav_y, -grav_z) = atan2(0, 1) = 0
    #   actual_pitch = asin(grav_x)            = asin(0)     = 0
    # shape: [N]
    actual_roll = torch.atan2(-proj_grav_b[:, 1], -proj_grav_b[:, 2])
    actual_pitch = torch.asin(proj_grav_b[:, 0].clamp(-1.0, 1.0))

    # PD control for roll/pitch, P control for yaw rate
    # shape: [N]
    moment_roll = (
        params.kp_att * (desired_roll - actual_roll)
        - params.kd_att * ang_vel_b[:, 0]
    )
    moment_pitch = (
        params.kp_att * (desired_pitch - actual_pitch)
        - params.kd_att * ang_vel_b[:, 1]
    )
    moment_yaw = params.kp_yaw * (desired_yaw_rate - ang_vel_b[:, 2])

    # Write thrust into pre-allocated buffer (Z axis in body frame)
    thrust_buf.zero_()
    thrust_buf[:, 0, 2] = total_thrust

    # Write moments into pre-allocated buffer; clamp to prevent runaway
    moment_buf[:, 0, 0] = moment_roll.clamp(-params.max_moment, params.max_moment)
    moment_buf[:, 0, 1] = moment_pitch.clamp(-params.max_moment, params.max_moment)
    moment_buf[:, 0, 2] = moment_yaw.clamp(-params.max_moment, params.max_moment)
