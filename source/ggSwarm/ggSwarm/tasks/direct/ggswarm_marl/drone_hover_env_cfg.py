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

    # Simplified 3-term reward (Isaac Lab style, dt-scaled).
    rew_scale_pos: float = 15.0
    rew_scale_vel: float = -0.05
    rew_scale_ang_vel: float = -0.01

    # Disable formation-related terms in hover baseline.
    rew_scale_formation: float = 0.0
    rew_scale_cohesion: float = 0.0
    rew_scale_separation: float = 0.0

    # No curriculum needed for pure hover learning.
    curriculum_start_step: int = 0
    curriculum_end_step: int = 1
