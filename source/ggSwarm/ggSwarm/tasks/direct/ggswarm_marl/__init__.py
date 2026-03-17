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
    id="Template-Ggswarm-Marl-Direct-v0",
    entry_point=f"{__name__}.ggswarm_marl_env:GgswarmMarlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ggswarm_marl_env_cfg:GgswarmMarlEnvCfg",
        "skrl_MAPPO_cfg_entry_point": f"{agents.__name__}:skrl_MAPPO_cfg.yaml",
    },
)