# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

try:
    # Common Isaac Lab layout
    from isaaclab_assets.robots import CRAZYFLIE_CFG
except ImportError:  # pragma: no cover
    try:
        # Alternative layout used by some releases
        from isaaclab_assets.robots.crazyflie import CRAZYFLIE_CFG
    except ImportError:  # pragma: no cover
        # Backwards-compat for older asset layouts
        from isaaclab_assets import CRAZYFLIE_CFG


@configclass
class GGSwarmMarlEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # multi-agent specification and spaces definition
    num_agents = 3

    possible_agents: list[str] = []
    action_spaces: dict = {}
    observation_spaces: dict = {}
    state_space = -1

    def __post_init__(self):
        self.possible_agents = [f"drone_{i}" for i in range(self.num_agents)]
        self.action_spaces = {agent: 4 for agent in self.possible_agents}
        self.observation_spaces = {agent: 12 for agent in self.possible_agents}

    # viewer
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.2, 1.2, 1.4),  # closer camera for small drones
        lookat=(0.0, 0.0, 0.9),  # look-at near typical hover/formation altitude
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
    )

    # robot(s)
    robot_cfg: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/drone_.*"
    )

    # swarm specific
    # With the current thrust mapping in `drone_swarm_env.py`,
    # `action_z = 0.0 -> thrust_val = 0.5`, so `thrust_to_weight = 1.9`
    # makes the neutral action hover (total thrust ~= 1.0 * weight).
    # Aligned with Isaac Lab's Isaac-Quadcopter-Direct-v0 reference baseline.
    thrust_to_weight: float = 1.9
    # Restored to 0.01 (Isaac Lab reference for Crazyflie).
    # Previous reduction to 0.001 was 10x overcorrection; tumbling was caused by
    # wide spawn yaw and lack of upright reward, both now addressed.
    moment_scale: float = 0.01
    graph_connectivity_radius: float = 2.0  # (metres) for L2 adjacency matrix
    # Tighter yaw range (0.1 vs 0.3) keeps drones more level at spawn, reducing early tumble pressure.
    spawn_yaw_range: float = 0.1  # ± range for random yaw (rad)
    target_formation_dist: float = 0.20  # desired inter-agent spacing (m)
    drone_radius: float = 0.05  # (metres) approximate collision radius
    min_separation_dist: float = 0.10  # (metres) minimum allowed inter-agent distance

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=5.0, replicate_physics=True
    )

    # reward scales
    rew_scale_pos = 3.0
    rew_scale_vel = -0.15
    rew_scale_ang_vel = -0.5
    # Stronger alive bonus to reinforce staying airborne.
    rew_scale_alive = 1.0
    # Aggressive crash penalty to prioritize recovery over all else.
    rew_scale_terminated = -20.0
    # Uprightness reward: now exceeds position reward (5.0 vs 3.0) to make "stay level" the top priority.
    # This ensures drones prioritize orientation control over reaching the goal while tumbling.
    rew_scale_upright: float = 5.0
    # Phase 2 rewards
    rew_scale_formation = 1.0
    rew_scale_cohesion = 0.2
    rew_scale_separation = -5.0

    # reward sigmas
    rew_pos_sigma = 0.5
    rew_formation_sigma = 0.3

    # reset states/conditions
    min_height = 0.1
    max_height = 3.0
    # Smaller spawn radius so drones start close to goals for cleaner gradients.
    spawn_dist = 0.5  # max distance from origin for spawning drones
    # Z range for random spawn altitude. These values are used directly in _reset_idx;
    # changing min_height/max_height alone does NOT affect spawn altitude (Rule 6).
    spawn_z_min: float = 0.5  # metres above env origin floor
    spawn_z_max: float = 1.5  # metres above env origin floor

    # curriculum: formation rewards fade in later to give hover more time to stabilize.
    # Start at 80k (later than before), reach full strength by 250k.
    # curriculum_pos_floor ensures hover signal never fully disappears.
    curriculum_start_step: int = 80000
    curriculum_end_step: int = 250000
    curriculum_pos_floor: float = 0.4

    # --- Phase 3: CBF Safety Shield (L4) ---
    # All False by default so Phase 2 train/play is unaffected.
    cbf_enabled: bool = False
    cbf_d_safe: float = 0.12          # safety radius (m); slightly above min_separation_dist
    cbf_gamma: float = 1.0             # barrier decay rate
    cbf_activation_margin: float = 1.5  # fraction of d_safe at which barrier activates
    # Set negative to penalise reliance on CBF (0.0 = log only, no reward signal)
    rew_scale_cbf_intervention: float = 0.0

    # --- Phase 3: SwarmRaft Consensus (L3) ---
    raft_enabled: bool = False
    raft_tick_interval: int = 10       # physics steps between SwarmRaft ticks
    raft_heartbeat_timeout: int = 50   # steps before declaring an agent lost
    raft_replan_cooldown: int = 100    # min steps between formation replans
    # Target hover altitude used by SwarmRaft when replanning formation slots.
    # Must be a cfg parameter (Rule 6) — do not hardcode 1.0 in SwarmRaftParams construction.
    raft_formation_height: float = 1.0  # metres above env origin floor

    # --- Phase 3: MINCO Trajectory Smoother (L5) ---
    minco_enabled: bool = False
    minco_buffer_size: int = 5         # action history window
    minco_smoothing_alpha: float = 0.3  # EMA coefficient (lower = smoother)


    # --- Phase 4: Stress Testing ---
    # Agent loss simulation: kill a random agent at a random step each episode.
    agent_loss_enabled: bool = False
    # (min_step, max_step) within episode to kill a random agent; 0 = disable
    agent_loss_interval_min: int = 30
    agent_loss_interval_max: int = 150

    # Obstacle environment (Phase 4B -- obstacles added in _setup_scene)
    obstacle_enabled: bool = False
    obstacle_count: int = 10
    # Obstacle half-extents: radius (m) for cylindrical obstacles
    obstacle_radius: float = 0.2
    obstacle_height: float = 2.0
    # Obstacle field extent (m from env origin)
    obstacle_field_size: float = 2.0

    # CBF obstacle extension (Phase 4B): safety radius around obstacle centres
    cbf_obstacle_d_safe: float = 0.30


@configclass
class GGSwarmMarlHoverStabilityCfg(GGSwarmMarlEnvCfg):
    """Hover-stability training mode: formation rewards disabled, pure stability objective.

    Use task id ``Template-GGSwarm-Marl-HoverStability-v0``.
    Compatible with Phase 2 GNN policy (same 12-dim obs space).
    After passing the stability assessment, use best_agent.pt as --checkpoint
    for subsequent formation training with the standard Phase 2 task.

    Two-phase training strategy:
    - Phase A: train with this config (80k iters); target survival_steps > 500,
      airborne_ratio > 0.9, mean_roll < 15 degrees.
    - Phase B: resume checkpoint with Template-GGSwarm-Marl-Direct-v0 and
      curriculum_start_step=0 for immediate formation signal.
    """

    # Disable all formation/coordination signals — each agent trains independently
    rew_scale_formation: float = 0.0
    rew_scale_cohesion: float = 0.0
    rew_scale_separation: float = 0.0

    # Lock curriculum alpha=0 for entire run: pos reward at 100%, formation at 0%
    curriculum_start_step: int = 999999
    curriculum_end_step: int = 1000000
    curriculum_pos_floor: float = 1.0  # pos reward never fades; always full signal

    # Rebalanced stability rewards — Run 1 levels that showed improvement (63.5° roll)
    # vs. Run 4 (75.8° roll) which used 5.0/-0.5/-20.0 and destabilized the policy
    rew_scale_upright: float = 3.0      # Run 1 level; not 5.0 which destabilized Run 4
    rew_scale_ang_vel: float = -0.25    # Run 1 level; not -0.5
    rew_scale_pos: float = 2.0          # slightly lower than 3.0; stability > position
    rew_scale_alive: float = 1.5        # stronger survival incentive
    rew_scale_terminated: float = -10.0  # reduced from -20.0 (Run 4 crash was too harsh)

    # Wider spawn yaw so policy learns diverse attitude recovery scenarios
    spawn_yaw_range: float = 0.3  # ± rad (up from default 0.1)


@configclass
class GGSwarmMarlFormationCfg(GGSwarmMarlEnvCfg):
    """Phase B formation training: resume from Phase A hover-stability checkpoint.

    Use task id ``Template-GGSwarm-Marl-Formation-v0``.

    Per Rule 21: ``curriculum_start_step=0`` because ``common_step_counter`` resets
    to 0 on env re-init, even when ``--checkpoint`` is passed.  Formation signal is
    active from step 1 so no GPU time is wasted re-waiting for the curriculum ramp.

    Workflow:
    1. Train Phase A (hover-stability) to stability gate.
    2. Resume with this config: ``phase2b train --checkpoint <PhaseA>/best_agent.pt``
    3. Assess with ``phase2b assess --run_dir <PhaseB_run>``.
    4. Advance to Phase 3 when formation_error < 0.5 m and stability metrics hold.

    # TODO (Phase C): once Phase B assess gate passes, add perturbation disturbance
    # to force robust attitude recovery under external forces.  See Rule 2 —
    # do not implement until Phase B is complete.
    """

    # Curriculum active from step 1 (Rule 21 — counter resets to 0 on resume)
    curriculum_start_step: int = 0
    curriculum_end_step: int = 80000   # full formation strength by 80k steps of Phase B
    curriculum_pos_floor: float = 0.3  # hover signal never drops below 30%

    # Stability rewards carried over from Phase A (Run 1 levels that passed gate)
    rew_scale_upright: float = 3.0
    rew_scale_ang_vel: float = -0.25
    rew_scale_terminated: float = -10.0
    rew_scale_alive: float = 1.5

    # Formation rewards re-enabled
    rew_scale_formation: float = 1.0
    rew_scale_cohesion: float = 0.2
    rew_scale_separation: float = -5.0

    # Keep wider spawn yaw from Phase A so policy handles diverse attitudes
    spawn_yaw_range: float = 0.3  # ± rad


@configclass
class GGSwarmMarlEnvCfgPhase3(GGSwarmMarlEnvCfg):
    """Phase 3 configuration: enables CBF, SwarmRaft, and MINCO.

    Obs space expands from 12 to 14 dims when raft_enabled=True.
    Use task id ``Template-GGSwarm-Marl-Phase3-v0`` for training runs that
    require the consensus observation features.  Phase 2 checkpoints are NOT
    compatible with this config due to the obs space change.
    """

    def __post_init__(self):
        super().__post_init__()
        # Update observation spaces to 14-dim (base 12 + is_leader + num_alive_frac)
        self.observation_spaces = {agent: 14 for agent in self.possible_agents}

    cbf_enabled: bool = True
    raft_enabled: bool = True
    minco_enabled: bool = True


@configclass
class GGSwarmMarlEnvCfgPhase4(GGSwarmMarlEnvCfgPhase3):
    """Phase 4 configuration: Phase 3 stack + agent-loss stress testing + obstacles.

    Inherits all Phase 3 features.  Enable ``agent_loss_enabled`` and/or
    ``obstacle_enabled`` to stress-test the full GNSC pipeline.
    """

    agent_loss_enabled: bool = True
    obstacle_enabled: bool = False   # start with clear env; enable for obstacle benchmark
