"""Phase 5 showcase environment configurations.

Provides ready-to-use configs for:
  - ``GGSwarmClutteredForestCfg``: dense obstacle forest for the HD demo (20 agents)
  - ``GGSwarmUrbanCanyonCfg``: urban-canyon variant with denser obstacle grid
  - ``GGSwarmShowcaseCfg``: 20-agent Phase 3 stack tuned for visual quality

All configs inherit from ``GGSwarmMarlEnvCfgPhase4`` so the full GNSC stack
(CBF, SwarmRaft, MINCO, agent-loss) is active.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .drone_swarm_env_cfg import GGSwarmMarlEnvCfgPhase4


@configclass
class GGSwarmShowcaseCfg(GGSwarmMarlEnvCfgPhase4):
    """20-agent showcase: Phase 3 full stack, no obstacles, optimised for visual output.

    Use with ``Template-GGSwarm-Marl-Phase4-v0`` or a dedicated showcase task.
    Reduce ``scene.num_envs`` to 1 when recording to maximise render quality.
    """

    def __post_init__(self):
        super().__post_init__()
        self.possible_agents = [f"drone_{i}" for i in range(self.num_agents)]
        self.action_spaces = {agent: 4 for agent in self.possible_agents}
        self.observation_spaces = {agent: 14 for agent in self.possible_agents}

    num_agents: int = 20
    episode_length_s: float = 30.0
    # Single env for clean video capture
    scene_num_envs: int = 1

    # Tighter formation for visual impact
    target_formation_dist: float = 0.25

    # Phase 3 features
    cbf_enabled: bool = True
    raft_enabled: bool = True
    minco_enabled: bool = True
    minco_smoothing_alpha: float = 0.4

    # Agent loss disabled for stable showcase; enable for fault-recovery demo
    agent_loss_enabled: bool = False


@configclass
class GGSwarmClutteredForestCfg(GGSwarmShowcaseCfg):
    """20-agent cluttered forest showcase.

    Spawns dense cylindrical obstacles simulating forest trees.
    CBF obstacle avoidance must be active (cbf_enabled=True).

    Proposal deliverable: O4 -- "HD demonstration video of 20+ agents
    navigating a cluttered forest."
    """

    obstacle_enabled: bool = True
    obstacle_count: int = 25       # dense forest
    obstacle_radius: float = 0.15  # tree trunk radius
    obstacle_height: float = 3.0   # tall trees
    obstacle_field_size: float = 3.0

    cbf_obstacle_d_safe: float = 0.35  # wider margin for visual safety


@configclass
class GGSwarmUrbanCanyonCfg(GGSwarmShowcaseCfg):
    """20-agent urban canyon: denser, taller obstacles in a grid pattern.

    Obstacle positions are configured as a grid in _setup_scene when
    ``obstacle_enabled=True``.  This config sets parameters to produce
    a corridor-like environment.
    """

    obstacle_enabled: bool = True
    obstacle_count: int = 30
    obstacle_radius: float = 0.25
    obstacle_height: float = 4.0
    obstacle_field_size: float = 4.0
    cbf_obstacle_d_safe: float = 0.40
