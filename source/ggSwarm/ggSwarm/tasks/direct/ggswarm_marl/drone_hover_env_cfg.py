# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Configuration for the single-drone hover baseline environment."""

from isaaclab.utils import configclass

from .drone_swarm_env_cfg import GGSwarmMarlEnvCfg


@configclass
class GGSwarmHoverEnvCfg(GGSwarmMarlEnvCfg):
    """Hover-only environment config for `GGS-Hover-v0`."""

    # Single-agent hover baseline.
    num_agents = 1

    # Keep a clean log namespace for hover-only runs.
    experiment_name: str = "ggswarm_hover"

    # Hover reward scales.
    rew_scale_pos = 2.0
    rew_scale_vel = -0.1
    rew_scale_ang_vel = -0.02
    rew_scale_alive = 0.1
    rew_scale_terminated = -2.0
    rew_scale_ground_hit = -25.0

    # Disable formation-related terms in hover baseline.
    rew_scale_formation = 0.0
    rew_scale_cohesion = 0.0
    rew_scale_separation = 0.0

    # Height thresholds for reward gating and crash detection.
    hover_reward_min_height: float = 0.20
    ground_hit_height: float = 0.10

    # No curriculum needed for pure hover learning.
    curriculum_start_step: int = 0
    curriculum_end_step: int = 1
