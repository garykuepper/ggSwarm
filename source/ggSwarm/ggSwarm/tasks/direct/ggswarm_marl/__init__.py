# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-GGSwarm-Marl-Direct-v0",
    entry_point=f"{__name__}.drone_swarm_env:GGSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_swarm_env_cfg:GGSwarmMarlEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

# IsaacLab's generic "play" helper sometimes expects an `Isaac-*-Play` task id.
# We provide a compatibility alias that maps to the same underlying env/cfg.
gym.register(
    id="Isaac-GGSwarm-Marl-Play",
    entry_point=f"{__name__}.drone_swarm_env:GGSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_swarm_env_cfg:GGSwarmMarlEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

# Hover-stability task: 12-dim obs (same as Phase 2), formation rewards disabled.
# Use for Phase A stability training before re-introducing formation curriculum.
gym.register(
    id="Template-GGSwarm-Marl-HoverStability-v0",
    entry_point=f"{__name__}.drone_swarm_env:GGSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_swarm_env_cfg:GGSwarmMarlHoverStabilityCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

# Phase B: formation training resumed from Phase A hover-stability checkpoint.
# Per Rule 21, curriculum_start_step=0 in GGSwarmMarlFormationCfg because
# common_step_counter resets to 0 on env re-init even with --checkpoint.
gym.register(
    id="Template-GGSwarm-Marl-Formation-v0",
    entry_point=f"{__name__}.drone_swarm_env:GGSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_swarm_env_cfg:GGSwarmMarlFormationCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

# Phase 2C: perturbation robustness testing with random impulse pushes.
# Inherits all formation + hybrid rewards from Phase 2B; enables push_enabled=True.
gym.register(
    id="Template-GGSwarm-Marl-Perturbation-v0",
    entry_point=f"{__name__}.drone_swarm_env:GGSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_swarm_env_cfg:GGSwarmMarlPerturbationCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

# Phase 3 task: 14-dim obs (12 base + is_leader + num_alive_frac).
# Uses the same env class but with raft_enabled=True in cfg overrides.
# Phase 2 task (Template-GGSwarm-Marl-Direct-v0) is unchanged.
gym.register(
    id="Template-GGSwarm-Marl-Phase3-v0",
    entry_point=f"{__name__}.drone_swarm_env:GGSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_swarm_env_cfg:GGSwarmMarlEnvCfgPhase3",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

# Phase 4 task: Phase 3 stack + agent-loss stress testing.
gym.register(
    id="Template-GGSwarm-Marl-Phase4-v0",
    entry_point=f"{__name__}.drone_swarm_env:GGSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_swarm_env_cfg:GGSwarmMarlEnvCfgPhase4",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

# Phase 5 showcase tasks: 20-agent + full GNSC stack.
gym.register(
    id="GGS-Showcase-v0",
    entry_point=f"{__name__}.drone_swarm_env:GGSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_swarm_env_cfg_showcase:GGSwarmShowcaseCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="GGS-ClutteredForest-v0",
    entry_point=f"{__name__}.drone_swarm_env:GGSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_swarm_env_cfg_showcase:GGSwarmClutteredForestCfg"
        ),
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="GGS-UrbanCanyon-v0",
    entry_point=f"{__name__}.drone_swarm_env:GGSwarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.drone_swarm_env_cfg_showcase:GGSwarmUrbanCanyonCfg"
        ),
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="GGS-Hover-v0",
    entry_point=f"{__name__}.drone_hover_env:GGSwarmHoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_hover_env_cfg:GGSwarmHoverEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_hover_cfg.yaml",
    },
)
