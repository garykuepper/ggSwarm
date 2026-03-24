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
    # action[0] = 0.0 -> thrust_val = 0.5 -> total_thrust = weight (hover).
    # thrust_to_weight * 0.5 == 1.0 => nominal hover at neutral collective (Isaac-style).
    # Aligned with Isaac Lab's Isaac-Quadcopter-Direct-v0 reference baseline.
    thrust_to_weight: float = 2.0

    # --- Direct Moment Control (matching Isaac Lab Isaac-Quadcopter-Direct-v0) ---
    # PD16: dropped the PD attitude controller. The RL policy now directly outputs
    # [thrust_cmd, moment_x, moment_y, moment_z] scaled by moment_scale.
    # Isaac Lab's reference uses this exact approach (quadcopter_env.py line 153).
    # The PD controller (PD1-PD15) introduced a saturation clamp (max_moment) that
    # created a 155x train-eval gap — policy learned hover WITH noise dithering past
    # saturation, but deterministic eval couldn't replicate. Direct moments have no
    # saturation, so train and eval behavior should match.
    # Moment scaling factor (Nm per unit action). At action=1.0, moment = 0.01 Nm.
    # Matches Isaac Lab's QuadcopterEnvCfg.moment_scale exactly.
    moment_scale: float = 0.01

    # --- Deprecated PD params (kept for backward compat with Phase 3 CBF configs) ---
    kp_att: float = 0.045
    kd_att: float = 0.005
    kp_yaw: float = 0.01
    max_tilt_angle: float = 0.52
    max_yaw_rate: float = 3.14159
    max_moment: float = 0.03

    # When > 0, log action stats to extras["log"] for TensorBoard. 0 = disabled (default).
    action_telemetry_max_env_steps: int = 0

    # Use ``compute_stable_hover_rewards`` (tanh position, squared vel, dt-scaled) instead
    # of ``compute_marl_rewards``. Phase 2A hover-stability enables this; formation does not.
    use_stable_hover_rewards: bool = False

    # When True, goal XY is overridden to each agent's spawn XY in _reset_idx.
    # Makes Phase 2A a pure hover-in-place task: zero initial position error, no lateral gradient.
    # Must be False for Phase 2B+ so agents navigate to formation circle slots.
    hover_in_place: bool = False

    graph_connectivity_radius: float = 2.0  # (metres) for L2 adjacency matrix
    # Tighter yaw range (0.1 vs 0.3) keeps drones more level at spawn, reducing early tumble pressure.
    spawn_yaw_range: float = 0.1  # ± range for random yaw (rad)
    target_formation_dist: float = 0.20  # desired inter-agent spacing (m)
    drone_radius: float = 0.05  # (metres) approximate collision radius
    min_separation_dist: float = 0.10  # (metres) minimum allowed inter-agent distance

    # scene (base default; HoverStability overrides to 512 for L4 GPU throughput)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=5.0, replicate_physics=True
    )

    # reward scales (Phase 2B formation training defaults)
    # Formation training uses ``compute_marl_rewards`` (Gaussian position, L2 vel norms).
    # Phase 2A hover-stability sets ``use_stable_hover_rewards=True`` for tanh + squared
    # vel terms with ``step_dt`` scaling (Isaac-Quadcopter-Direct-v0 style).
    rew_scale_pos: float = 15.0          # marl: exp(-dist/sigma); stable-hover: tanh, dt-scaled
    rew_scale_vel: float = -0.05         # marl: L2 norm; stable-hover: squared L2, dt-scaled
    rew_scale_ang_vel: float = -0.01     # marl: L2 norm; stable-hover: squared L2, dt-scaled
    rew_scale_alive: float = 0.0         # disabled; PD controller makes alive bonus unnecessary
    rew_scale_terminated: float = 0.0    # disabled; dones handled by height bounds
    rew_scale_upright: float = 0.0       # disabled; PD controller maintains upright
    # Phase 2B formation rewards
    rew_scale_formation: float = 1.0
    rew_scale_cohesion: float = 0.2
    rew_scale_separation: float = -5.0

    # reward sigmas
    rew_pos_sigma = 0.5
    rew_formation_sigma = 0.3
    # Tanh distance scale for stable-hover position reward (metres). Smaller = sharper.
    pos_tanh_sigma: float = 0.8

    # reset states/conditions
    min_height = 0.1
    max_height = 3.0
    # Shapes reward below (min_height + low_clearance_margin_m); 0 = disabled (Phase 2B default).
    rew_scale_low_clearance: float = 0.0
    # Metres above crash floor; align with eval airborne margin (Phase2Collector default 0.2).
    low_clearance_margin_m: float = 0.2
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
    """Hover-stability training mode: PD attitude controller + Isaac-style stable hover rewards.

    Use task id ``Template-GGSwarm-Marl-HoverStability-v0``.
    Compatible with Phase 2 GNN policy (same 12-dim obs space).

    Rewards are computed via ``compute_stable_hover_rewards`` (``use_stable_hover_rewards``)
    — tanh position, squared velocity penalties, ``step_dt`` scaling — plus optional
    low-clearance shaping (``rew_scale_low_clearance``) for scorecard alignment.

    The inner-loop PD controller (kp_att/kd_att/kp_yaw) tracks policy attitude setpoints.

    Two-phase training strategy:
    - Phase A: train with this config (80k steps); target survival_steps > 500,
      airborne_ratio > 0.9, mean_roll < 15 degrees.
    - Phase B: resume checkpoint with Template-GGSwarm-Marl-Formation-v0 and
      curriculum_start_step=0 for immediate formation signal.
    """

    # Disable all formation/coordination signals — each agent trains independently
    rew_scale_formation: float = 0.0
    rew_scale_cohesion: float = 0.0
    rew_scale_separation: float = 0.0

    # Isaac-Quadcopter-Direct-v0 style rewards (see ``contract_logic.compute_stable_hover_rewards``).
    use_stable_hover_rewards: bool = True

    # Pure hover-in-place: goal XY = spawn XY so the position reward has no lateral component.
    # Root cause fix for PD1-PD6 persistent 21°/24° tilt (formation circle XY gradient).
    hover_in_place: bool = True

    # Lock curriculum: hover signal at 100%, formation at 0% for full run
    curriculum_start_step: int = 999999
    curriculum_end_step: int = 1000000
    curriculum_pos_floor: float = 1.0

    # Stable-hover reward: tanh distance (dt-scaled), squared vel/ang_vel penalties (dt-scaled),
    # plus low-clearance shaping (see ``compute_stable_hover_rewards``).
    rew_scale_pos: float = 18.0
    # PD10: sharpen position discriminator — at sigma=0.25 reward drops to 5% at 0.5m drift
    # (was 45% at sigma=0.8). Crash-reset strategy yields ~40 vs hover ~180 per episode (4.5x ratio).
    pos_tanh_sigma: float = 0.25
    # PD10: 5.5x increase so velocity penalty is visible in TB (still 60x below pos peak at hover).
    # PD13: reverted to PD10 value. PD11/PD12 reduced these to work around overdamped PD;
    # with critically-damped gains (kd=0.00173), corrections no longer produce excessive ang_vel.
    rew_scale_vel: float = -0.3
    # PD10: 5x increase so angular velocity penalty is visible in TB (was flat across PD1-PD9).
    rew_scale_ang_vel: float = -0.06

    # Disable unused reward terms (PD controller makes these redundant)
    rew_scale_upright: float = 0.0
    rew_scale_alive: float = 0.0
    # Dense penalty each step while z < min_height (see compute_stable_hover_rewards).
    # PD8: -5.0 caused ceiling escape (generous sigma=0.8 let policy drift up without losing
    # position reward). PD10: re-enabled at -2.0 because tight sigma=0.25 suppresses ceiling
    # escape — drifting up immediately loses position reward, countering any "flee floor" gradient.
    rew_scale_terminated: float = -2.0
    # Penalty per metre below (min_height + low_clearance_margin_m); aligns MDP with scorecard airborne gate.
    rew_scale_low_clearance: float = -8.0

    # Moderate spawn yaw; PD controller recovers from wider perturbations
    spawn_yaw_range: float = 0.3  # ± rad

    # Phase 2A-only: more vertical margin above min_height; goal Z still tracks spawn Z in _reset_idx.
    spawn_z_min: float = 0.65
    spawn_z_max: float = 1.65

    # PD10: enable action telemetry for full run — thrust_val_mean, moment saturation in TB.
    action_telemetry_max_env_steps: int = 999999

    # Scale up parallel envs for GCE L4 GPU (24 GB VRAM).
    # PD17: 512 -> 2048 envs. At 512 envs, training used only 2.7 GB of 23 GB VRAM (12%).
    # 2048 envs * 3 agents = 6144 parallel rollouts (vs Isaac Lab's 4096 single-agent).
    # Should improve gradient diversity and convergence speed.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=5.0, replicate_physics=True
    )


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
