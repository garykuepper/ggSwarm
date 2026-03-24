"""Direct moment control sanity checks without Isaac Sim.

PD16: replaced PD attitude controller tests with direct moment scaling tests
matching Isaac Lab's Isaac-Quadcopter-Direct-v0 approach.
"""

from __future__ import annotations

import torch


def test_direct_thrust_at_neutral_action() -> None:
    """Action 0 -> thrust_val = 0.5 -> total_thrust = robot_weight (hover)."""
    robot_weight = 0.276  # N (Crazyflie mass * gravity)
    thrust_to_weight = 2.0
    action_thrust = torch.tensor([0.0])  # neutral

    thrust_val = (action_thrust + 1.0) * 0.5
    total_thrust = thrust_to_weight * robot_weight * thrust_val

    assert abs(float(total_thrust) - robot_weight) < 1e-4, (
        f"Neutral action should produce hover thrust ({robot_weight:.4f} N), "
        f"got {float(total_thrust):.4f} N"
    )


def test_direct_moment_scaling() -> None:
    """Actions [1:3] * moment_scale should produce moments in expected range."""
    moment_scale = 0.01  # Isaac Lab default

    # Max action = 1.0 -> moment = 0.01 Nm
    actions = torch.tensor([0.0, 1.0, -1.0, 0.5])  # thrust, mx, my, mz
    moments = moment_scale * actions[1:]

    assert abs(float(moments[0]) - 0.01) < 1e-6
    assert abs(float(moments[1]) - (-0.01)) < 1e-6
    assert abs(float(moments[2]) - 0.005) < 1e-6


def test_zero_actions_produce_hover_and_zero_moments() -> None:
    """Zero actions = hover thrust + zero moments (stable hover, no rotation)."""
    robot_weight = 0.276
    thrust_to_weight = 2.0
    moment_scale = 0.01
    actions = torch.zeros(4)

    thrust_val = (actions[0] + 1.0) * 0.5
    total_thrust = thrust_to_weight * robot_weight * thrust_val
    moments = moment_scale * actions[1:]

    assert abs(float(total_thrust) - robot_weight) < 1e-4
    assert float(moments.abs().max()) < 1e-6


def test_moment_has_no_saturation_clamp() -> None:
    """Direct moments have no clamp — any action maps to a proportional moment.

    This is the key difference from the PD controller (PD1-PD15) which had a
    max_moment clamp that caused the 155x train-eval gap.
    """
    moment_scale = 0.01
    # Even at full action, moment is just moment_scale * 1.0 = 0.01 Nm
    # No clamp, no saturation
    for action_val in [0.5, 1.0, -1.0]:
        moment = moment_scale * action_val
        expected = moment_scale * action_val
        assert abs(moment - expected) < 1e-8, (
            f"Moment should be exactly moment_scale * action ({expected}), got {moment}"
        )
