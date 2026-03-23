# Changelog: ggSwarm Development

This document tracks major technical changes and milestone completions for each project phase.

## Phase 1: Foundation (Weeks 5-6)

- [2026-03-16] Initialized project repository and structure.
- [2026-03-16] Codified project rules and architecture documentation.
- [2026-03-16] Established status reporting framework.
- [2026-03-16] Implemented `GGSwarmMarlEnv` (DirectMARLEnv) with multi-agent Crazyflie spawning, environment cloning, and Articulation initialization.
- [2026-03-16] Implemented `GGSwarmMarlEnvCfg` with dynamic agent/space registration via `__post_init__()`.
- [2026-03-16] Built 12-dim observation space (body-frame velocities, gravity, relative goal position).
- [2026-03-16] Built 4-dim action space (thrust + 3-axis moments) mapped to propeller forces/torques.
- [2026-03-16] Implemented distance-based adjacency matrix (L2 graph connectivity, threshold 2.0m).
- [2026-03-16] Implemented reward function (Gaussian position reward, velocity penalties, alive bonus).
- [2026-03-16] Implemented reset logic with random spawn positions and yaw.
- [2026-03-16] Added an early Phase 1 verification script for visual adjacency evidence
  (superseded; use `scripts/run.py` for current workflows).
- [2026-03-16] Registered environment as `Template-GGSwarm-Marl-Direct-v0`.
- [2026-03-16] Set up SKRL MAPPO training pipeline (`train.py`, `play.py`, `skrl_mappo_cfg.yaml`).
- [2026-03-16] Renamed env files to `drone_swarm_env.py` / `drone_swarm_env_cfg.py` for clarity.
- [2026-03-16] Finalized comprehensive Phase 1 technical documentation.

## Phase 2: Brain Development (Weeks 7-8)

- [2026-03-18] Parameterized magic numbers in `drone_swarm_env.py` and `drone_swarm_env_cfg.py` (Rule 12).
- [2026-03-18] Implemented curriculum reward shaping with alpha-scaled formation and cohesion rewards.
- [2026-03-18] Added `rew_scale_formation`, `rew_scale_cohesion`, and `rew_scale_separation` to config (Rule 9).
- [2026-03-18] Created `skrl_gnn_policy.py` bridging PyTorch Geometric GATv2 with SKRL Gaussian models.
- [2026-03-18] Integrated `--gnn` flag in `train.py` for optional GNN policy activation.
- [2026-03-18] Optimized MAPPO hyperparameters (`rollouts: 32`, `mini_batches: 4`, `lr: 1e-4`).
- [2026-03-19] Added a Phase 2 evaluation script (`scripts/eval_phase2.py`) to measure mean formation error, separation events, and speed for trained checkpoints.
- [2026-03-19] Updated reset goal logic to assign deterministic formation slots (circle) per agent to provide a stable reference for `rel_pos_to_goal` during Phase 2 coordination training.
- [2026-03-19] Added a dedicated hover baseline task `GGS-Hover-v0` with
  `GGSwarmHoverEnv` / `GGSwarmHoverEnvCfg` (single-agent spawn-hold), plus
  hover-specific MAPPO config and metrics script (`scripts/eval_hover.py`) to
  validate airborne stability before formation training.
- [2026-03-19] Added explicit major ground-hit penalty (`rew_scale_ground_hit`) and immediate crash termination in hover rewards/dones to prevent false improvement while grounded.
- [2026-03-20] Aligned `hover_reward_min_height` with `ground_hit_height` to reduce reward gating discontinuity during early dip events, improving the chance to recover altitude after brief ground touches.
- [2026-03-19] Added a unified CLI helper `scripts/run.py` with consistent subcommands for `hover`, `phase2`, and `debug` workflows (train/play/eval/monitor, smoke test, latest-checkpoint), while keeping older scripts for backward compatibility.
- [2026-03-20] Fixed MAPPO training for single-agent MARL (e.g. `GGS-Hover-v0`): SKRL
  `single_agent_train` omitted `infos['shared_states']` (`KeyError`). `train.py` now
  wraps the vec env to inject shared states from `env.state()` when `MAPPO` and
  `num_agents==1`.
- [2026-03-20] Tuned Phase 2 triangle baseline visibility and spacing: set closer
  viewer camera in `drone_swarm_env_cfg.py` and reduced
  `target_formation_dist` to `0.20m` for tighter 3-drone formation training.
- [2026-03-20] Shifted Phase 2 to altitude-first recovery tuning:
  increased early hover stabilization influence, delayed curriculum onset for
  formation pressure, set deterministic `write_interval` /
  `checkpoint_interval`, and set a 50k-step corrective training budget.
- [2026-03-20] Extended `scripts/eval_phase2.py` with altitude stability metrics:
  `mean_altitude_error_m`, `ground_hit_rate`, and `airborne_ratio`.
- [2026-03-20] Phase 2 recovery tuning: increased `rew_scale_pos` and `rew_scale_alive` and
  moved `curriculum_start_step`/`curriculum_end_step` later so formation pressure ramps up
  after altitude stabilization. Increased Phase 2 default MAPPO `trainer.timesteps` to
  `100000` so the curriculum reaches the full formation regime within a single run.
- [2026-03-20] Documented GCE training and monitoring (`docs/gce_training_and_monitoring.md`):
  VM train commands, `nohup` log tail via `gcloud compute ssh`, TensorBoard over SSH `-L`,
  and `tensorboard --inspect` sanity check on the VM log root.
- [2026-03-20] Implemented GCS sync workflow for results transfer between GCE VM and Windows PC:
  - Added `docs/gce_results_sync.md` with bucket setup, `gsutil rsync` push/pull examples, auth config, and dry-run workflow.
  - Added `scripts/cloud/sync_gcs.ps1` (PowerShell) and `scripts/cloud/sync_gcs.sh` (bash) with env-based URI, `--family marl|hover`, `--include-videos`, and dry-run support.
  - Added optional `scripts/cloud/train_and_push.sh` for VM to auto-upload logs to GCS after training completes or on SIGINT.
  - Added `.cursor/quick-start.md` section documenting GCE instance details, SSH, and links to full docs.
  - Cross-linked training/monitoring and GCS sync docs from README, commands.md, and GCE docs.
- [2026-03-20] Created GCS bucket `gs://gg-swarm-training-logs` in GCP project `gg-swarm` for training artifact storage.
- [2026-03-20] Added helper scripts for training results workflow (videos excluded):
  - `scripts/cloud/push_results_to_gcs.sh` (VM): Push training logs to GCS after training, exclude videos
  - `scripts/cloud/pull_results_from_gcs.ps1` (Windows): Mirror GCS logs to local machine, exclude videos
  - `scripts/cloud/list_checkpoints.ps1` (Windows): List available checkpoints with metadata and print path for playback
  - Updated `docs/gce_results_sync.md` with quick-start guide and helper script examples
- [2026-03-21] Restored physics parameters to Isaac Lab reference values for Crazyflie:
  - `moment_scale`: 0.001 → 0.01 (10x increase). Previous reduction was 10x overcorrection; root causes (wide spawn yaw, missing upright reward) are now fixed. Restores physical attitude authority per Isaac-Quadcopter-Direct-v0 baseline.
  - `thrust_to_weight`: 2.0 → 1.9 for consistency with Isaac Lab reference (minor change, neutral action still hovers).
  - Rationale: Weak attitude correction in earlier playback indicated insufficient torque authority. Isaac Lab's validated Crazyflie parameters now provide accurate moment scaling.
- [2026-03-21] Adjusted curriculum timing to support 300k-step training budgets:
  - `curriculum_start_step`: 200000 → 50000 (formation rewards begin after basic hover is learned).
  - `curriculum_end_step`: 500000 → 200000 (formation reaches full strength by 200k, allowing 100k steps of full signal vs. 33% signal in previous run).
  - Rationale: Previous run to 300k steps only exposed agents to 33% formation reward. Early curriculum ensures agents learn coordination within typical training budgets.
- [2026-03-21] Documented playback and training guidance:
  - MLP policy trained without GNN cannot observe neighbors (12-dim obs: self state + goal; no neighbor info). GNN policy via `--gnn` flag uses adjacency matrix for neighbor-aware message passing.
  - For formation behavior, train with `python scripts/run.py phase2 train --headless --gnn` to enable GATv2 policy with graph structure.
  
## Phase 2: Evaluation & Training Optimization (Weeks 8-9)

- [2026-03-23] **Phase 2A PD6 prep** (TensorBoard contract + ground penalty knob — **before** GCE PD6 train):
  - **TensorBoard / SKRL:** `extras["log"]` values are **0-dim `torch.Tensor`** scalars (not Python `float`) so SKRL’s `environment_info: log` path passes `isinstance(v, torch.Tensor) and v.numel() == 1` and writes **`Info / rew_pos`**, **`Info / rew_vel`**, **`Info / rew_low_clearance`**, **`Info / rew_terminated`**, **`Info / mean_world_z`**, etc. CBF `cbf_intervention_rate` / obstacle rate return tensors from [`cbf_safety.py`](../../source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/cbf_safety.py).
  - **Reward (single knob):** `GGSwarmMarlHoverStabilityCfg.rew_scale_terminated` **`-5.0`**. `compute_stable_hover_rewards` adds **`rew_terminated = (z < min_height).float() * scale`** each step while below `min_height` (dense penalty; often one step before reset — not a one-shot lump-sum). Rationale: with `rew_scale_pos=18` and env **`step_dt=0.02`** (physics `dt` × decimation), max tanh position reward per step is **`18 × 0.02 = 0.36`**; **`-5.0`** per grounded step is ~**14×** that upper bound so the critic can separate crash-short episodes from 500-step survivors without swamping the dt-scaled pos/vel terms early on.
  - **Docs:** [`docs/design/architecture.md`](../design/architecture.md) telemetry note; Rule 22 table in [`docs/ops/pd5_rule22_checklist.md`](../ops/pd5_rule22_checklist.md) updated for PD6.
  - **Tests:** `TestComputeStableHoverRewards` includes `rew_terminated` in term-sum invariant + `test_ground_hit_applies_rew_scale_terminated`.
  - **Rule 22 smoke (local):** `python scripts/run.py debug smoke --task Template-GGSwarm-Marl-HoverStability-v0 --iterations 1 --gnn --headless` — PASS after PD6 doc/code pass (512 envs, `GGSwarmMarlHoverStabilityCfg` + `rew_scale_terminated=-5.0`).

- [2026-03-23] **`run.py` train/play:** accept `--gnn` (forwards to `train.py` / `play.py`); hover-stability
  already defaulted to GNN via `build_train_cmd`, but the flag was missing from argparse and was rejected.

- [2026-03-23] **Train CLI:** `--action_telemetry_steps` on `run.py … train` and `scripts/skrl/train.py`
  overrides `env_cfg.action_telemetry_max_env_steps` for short local TensorBoard diagnostics
  (omit on GCE). Documented in [`docs/ops/pd5_rule22_checklist.md`](ops/pd5_rule22_checklist.md).

- [2026-03-23] **Phase 2A PD5 prep** (first train under stable-hover rewards):
  - **Rule 22 smoke:** `python scripts/run.py debug smoke --task Template-GGSwarm-Marl-HoverStability-v0 --iterations 1 --gnn --headless` — PASS (env + GNN init, 512 envs).
  - **Checklist:** [`docs/ops/pd5_rule22_checklist.md`](ops/pd5_rule22_checklist.md) documents Rule 22 cfg fields + optional `action_telemetry_max_env_steps` workflow (PD6 extends table with `rew_scale_terminated`).
  - **Knob choice:** **zero** additional deltas vs PD4 bundle (`max_moment`, `entropy_loss_scale`, `initial_log_std` unchanged); PD5 = stable-hover reward migration + existing PD4 exploration caps.

## Phase 2A Run PD5 — 2026-03-23

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-22_23-01-57_mappo_torch`
- **Config class:** `GGSwarmMarlHoverStabilityCfg` — commit `2b87b83`
- **Train budget:** 92,000 iterations (GCE); stable-hover reward path (`use_stable_hover_rewards=True`).
- **Convergence:** entropy collapse not detected | peak reward **-0.48** @ step **21,000** | final **-0.77** @ step **92,000** | recommended budget **~106k** steps
- **Scorecard** (`post_train_assess.py`, seed **42**, `best_agent.pt`, 5 episodes):
  - survival_steps = **6.4**
  - airborne_ratio = **0.623**
  - ground_hit_rate = **0.542**
  - mean_roll_deg = **19.2°**
  - orientation_violation_rate = **0.168**
  - verdict = **FAIL**
- **vs Run PD4:** Attitude metrics improved (roll, orientation viol); altitude proxies and ground contact **worse** — stable-hover reshaping alone did not fix the Phase 2A gates.
- **Decision: FAIL** — do not advance to Phase 2B. Do **not** change reward or PD knobs until TensorBoard is reviewed (`Reward/Total`, `mean_world_z`, `rew_low_clearance`, per-term stable-hover logs).
- **Next action:** (1) TensorBoard on this run: confirm post-~21k reward drift vs policy entropy. (2) If adjusting, **one knob at a time** per [`pd_authority_tuning.md`](../ops/pd_authority_tuning.md) / [`phase2a_diagnostics.md`](../ops/phase2a_diagnostics.md) — e.g. bounded `rew_scale_low_clearance` / `rew_scale_vel` / `rew_scale_pos` nudge or PD authority only after TB narrative is clear; log any cfg change here before retrain.

- [2026-03-22] Phase 2A hover-stability uses Isaac-style stable hover rewards and diagnostics:
  - `GGSwarmMarlHoverStabilityCfg` sets `use_stable_hover_rewards=True` so `_get_rewards` calls `compute_stable_hover_rewards` (tanh position, squared body-frame velocity penalties, `step_dt`-scaled) with optional low-clearance shaping via `StableHoverRewardParams`.
  - Formation and default MARL configs keep `compute_marl_rewards` (Gaussian position, L2 velocity norms, curriculum).
  - Added `action_telemetry_max_env_steps` plus optional pre-clamp moment logging (`moment_pre_clamp_buf` on `compute_attitude_control`); documented env-side `[-1, 1]` action clamp in `docs/design/architecture.md`.
  - Ops guide: `docs/ops/pd_authority_tuning.md`; torch-only regression tests: `tests/unit/test_attitude_open_loop.py`.

- [2026-03-21] Implemented per-reward-term logging for TensorBoard visualization:
  - Modified `compute_marl_rewards()` to optionally return individual reward components (`rew_pos`, `rew_formation`, `rew_cohesion`, etc.)
  - Enhanced `drone_swarm_env.py` to log per-term rewards and curriculum_alpha in `self.extras["log"]` for SKRL to write to TensorBoard
  - Enables real-time diagnosis of which reward components dominate training and when curriculum transitions occur
  
- [2026-03-21] Extended `eval_phase2.py` with orientation and stability metrics:
  - Added `mean_roll_deg`, `mean_pitch_deg`: directly measure drone orientation (target: < 15° for level flight)
  - Added `orientation_violation_rate`: fraction of steps exceeding 45° roll/pitch (target: < 0.1)
  - Added `altitude_std_m`: standard deviation of altitude (detects oscillation)
  - Added `mean_episode_survival_steps`: tracks how long agents stay alive before termination
  - Rationale: Previous run showed severe flipping (45° roll/pitch observed); these metrics enable targeted diagnosis
  
- [2026-03-21] Created `scripts/analyze_checkpoints.py` for checkpoint progression analysis:
  - Evaluates multiple checkpoints at regular intervals (e.g., every 50k steps)
  - Produces CSV table showing metric evolution over training
  - Identifies degradation points (e.g., when formation curriculum kicks in at curriculum_start_step)
  - Usage: `python scripts/analyze_checkpoints.py --run_dir logs/skrl/ggswarm_marl/<run> --interval 50000`
  
- [2026-03-21] Optimized NVIDIA L4 GPU training for L4 (24GB VRAM, Ada Lovelace compute capability 8.9):
  - Scaled `num_envs` from 32 to 128 in `drone_swarm_env_cfg.py` (4x more parallel experience, better gradient variance)
  - Increased `rollouts` from 32 to 64 in `skrl_mappo_cfg.yaml` (larger data buffer per update)
  - Increased `mini_batches` from 4 to 8 (matches larger rollout buffer, keeps per-batch size reasonable)
  - Added explicit TF32 enable in `train.py`: `torch.backends.cuda.matmul.allow_tf32 = True` (Ada Lovelace supports ~2x FP32 speedup)
  - Effective samples per update: 32*3*32=3,072 → 128*3*64=24,576 (8x improvement in gradient diversity)
  - Rationale: Previous config left GPU severely underutilized; scaling increases both throughput and learning quality
  
- [2026-03-21] Applied reward rebalancing to address observed training failures:
  - `rew_scale_upright`: 1.0 → 2.0 (stronger penalty for flipping; was insufficient given weak ang_vel penalty)
  - `rew_scale_ang_vel`: -0.02 → -0.15 (7.5x stronger penalty for spinning; critical for preventing tumbles)
  - `rew_scale_vel`: -0.10 → -0.15 (slightly stronger to reduce wild oscillation)
  - `rew_scale_alive`: 0.5 → 1.0 (double the carrot for staying airborne)
  - `rew_scale_terminated`: -10.0 → -15.0 (50% stronger crash penalty to prioritize recovery)
  - Rationale: Previous 300k run showed mean_roll/pitch=45°, low airborne_ratio; balance was heavily skewed toward position reward at expense of stability
  
- [2026-03-21] Delayed curriculum transition to allow hover stabilization:
  - `curriculum_start_step`: 50,000 → 80,000 (give drones 80k steps to master hover before formation kicks in)
  - `curriculum_end_step`: 200,000 → 250,000 (slower ramp-in; formation reaches full strength by 250k)
  - `curriculum_pos_floor`: 0.3 → 0.4 (keep hover signal at 40% baseline instead of 30%; avoid formation overriding stability)
  - Rationale: Previous run jumped into formation training too early; drones weren't stable enough to learn multi-agent coordination
  
- [2026-03-21] Created comprehensive training workflow documentation (`docs/ops/training_workflow.md`):
  - End-to-end cycle: TRAIN → SYNC → INSPECT → EVAL → DIAGNOSE → ADJUST → LOG → RETRAIN
  - Detailed metric interpretation guide (what good/bad values mean for formation, stability, and survival)
  - Decision matrix for common failure modes and recommended adjustments
  - TensorBoard monitoring tips for per-term reward decomposition
  - L4 GPU optimization notes and VRAM monitoring guidance
  - Pre-flight checklist for long training runs (per project rule 7)
  - Troubleshooting section for common issues and solutions
  
- [2026-03-21] Updated project changelog with all changes and rationale for reproducibility (per project rule 5)

## Phase 2: Critical Stability Fixes (Week 9)

- [2026-03-21] Evaluated 300k-step training run baseline (best_agent.pt) with enhanced metrics:
  - **Catastrophic orientation failure**: mean_roll=78.5°, mean_pitch=80.1° (nearly inverted drones)
  - **Stability collapse**: 62% of steps violate 45° threshold, 65% ground hit rate, only 59% airborne
  - **Formation irrelevant**: 1.566m error (meaningless until hover works)
  - **Root cause confirmed**: Old `rew_scale_ang_vel=-0.02` was ~150x weaker than position reward (3.0), making spin penalty invisible to optimizer
  - **Policy behavior**: Drones learned to apply aggressive moments that flip them; when upside-down, thrust pushes into ground → crashes
  - Evaluated at 10 episodes (5000 steps total); metrics converged by step 2500
  - Full baseline: airborne_ratio=0.587, formation_error=1.566m, mean_speed=1.921m/s, altitude_std=0.571m

- [2026-03-21] Applied aggressive stability tuning based on eval baseline:
  - `rew_scale_upright`: 2.0 → **3.0** (match position reward importance; "stay level" = "reach goal")
  - `rew_scale_ang_vel`: -0.15 → **-0.25** (12.5x vs old -0.02; heavily penalize spinning)
  - Rationale: 78° average tilt indicates even 2.0 uprightness was still dominated by 3.0 position reward. Matching scales ensures level flight and position tracking are equally weighted in the loss.
  - New reward balance: `rew_scale_pos=3.0, rew_scale_upright=3.0, rew_scale_ang_vel=-0.25, rew_scale_alive=1.0, rew_scale_vel=-0.15, rew_scale_terminated=-15.0`
  - All other settings unchanged: curriculum (80k-250k, pos_floor=0.4), L4 optimization (num_envs=128, rollouts=64), architecture (GNN + MLP value function)
  - Next steps: Deploy to GCE L4 and train 300k steps; expect airborne_ratio > 0.9 and mean_roll/pitch < 15° if tuning successful

- [2026-03-21] **CONVERGENCE FINDINGS**: Analysis of 300k-step run shows catastrophic divergence from the fix:
  - Evaluated best_agent.pt from run (trained with fixed `rew_scale_upright=3.0`, `rew_scale_ang_vel=-0.25`)
  - **Eval results**: mean_roll=63.5°, mean_pitch=62.0°, 53.5% ground_hit_rate, 64.9% airborne_ratio, 1.126m formation_error
  - **Root cause analysis**: Policy learned to apply aggressive moments that tumble drones; when inverted, thrust pushes into ground
  - **Convergence analysis tool created**: `scripts/cloud/check_convergence.py` reads TFEvents and detects policy convergence via entropy collapse
  - **Key finding**: Training converged (entropy locked) at **105k steps**; last 195k steps (300k→105k) were wasted compute
  - Recommended training budget for next run: **120k steps** (105k + 15% buffer) instead of 300k; saves ~60% GPU hours per run
  - **New aggressive tuning**: `rew_scale_upright: 3.0 → 5.0` (exceed position reward), `rew_scale_ang_vel: -0.25 → -0.5` (2x stronger spin penalty), `rew_scale_terminated: -15.0 → -20.0`, `spawn_yaw_range: 0.3 → 0.1`
  - Rationale: Even 3.0 uprightness was insufficient; position reward (3.0) still dominated over spin penalty (-0.25). New scaling makes uprightness the top priority: 5.0 > 3.0 + 1.0 (alive) + 0.2 (cohesion).
  - **Faster eval**: Changed default `--num_episodes` from 10 to 5 and added `--headless` to eval by default (metrics converge by step 2500; halves eval time from ~15min to ~8min)
  - Next steps: Deploy tuned config to GCE and train for 120k steps (not 300k); expect full stability recovery.

## Phase 2: Run 4 Evaluation and Hover-Stability Pivot (2026-03-22)

- [2026-03-22] **Run 4 eval** (run: `2026-03-21_21-21-55_mappo_torch`, 120k iters, `best_agent.pt`, 5 episodes):
  - `survival_steps=1.1` | `airborne_ratio=0.582` | `ground_hit_rate=0.648`
  - `mean_roll=75.8°` | `mean_pitch=76.2°` | `orientation_violation_rate=0.582`
  - `mean_formation_error=1.551m`
  - **Decision: FAIL** — regression vs. Run 1; aggressive reward stacking (`upright=5.0`, `ang_vel=-0.5`, `terminated=-20.0`) destabilized policy. `survival_steps=1.1` indicates agents crash within one simulation step — policy is broken at spawn.
  - **Root cause**: `rew_scale_terminated=-20.0` combined with `rew_scale_upright=5.0` creates an enormous penalty gradient at episode start. Policy collapsed to a degenerate local minimum where any action results in immediate termination.
  - **Bug fix applied**: `train.py` `max_iterations * rollouts` multiplication removed — `--max_iterations 120000` now correctly runs 120k rollout collections (~2 hrs) instead of 7.68M timesteps (~60 hrs).
  - **Next action**: Pivot to hover-stability training mode (formation rewards disabled) with rebalanced rewards at Run 1 levels. Run 80k steps before re-introducing formation curriculum.

- [2026-03-22] Introduced hover-stability training mode (`Template-GGSwarm-Marl-HoverStability-v0`):
  - `GGSwarmMarlHoverStabilityCfg` subclass: `rew_scale_formation=0.0`, `rew_scale_cohesion=0.0`, `rew_scale_separation=0.0`
  - Curriculum locked off: `curriculum_start_step=999999`, `curriculum_pos_floor=1.0`
  - Reward rebalance back to Run 1 levels: `rew_scale_upright=3.0`, `rew_scale_ang_vel=-0.25`, `rew_scale_terminated=-10.0`
  - Wider spawn: `spawn_yaw_range=0.3` (up from 0.1)
  - Rationale: Isolate stability objective before reintroducing formation pressure; avoid Run 4's destabilization pattern.

- [2026-03-22] Added post-training assessment infrastructure:
  - `phase2 assess` / `hover-stability assess` subcommands in `scripts/run.py` — runs convergence check, checkpoint progression, best_agent eval, and prints PASS/WARN/FAIL scorecard
  - Rule 20 added: assessment gate mandatory before any reward change or retraining
  - `phase2 hover-stability train/eval` subcommands added for hover-stability workflow
  - Progress output cleaned up: tqdm suppressed in headless mode, heartbeat stopped after 100%

- [2026-03-22] Added cursor rules for operational consistency:
  - `gce-training-ops.mdc`: Added "GCE for Training Only" section — all eval/play/assess must run locally
  - `.cursor/rules/shell-syntax.mdc`: New rule documenting PowerShell syntax pitfalls (no `&&`, no `head`, `rsync` Windows bug, SSH `$` escaping)

## Phase 2A: PD Attitude Controller Refactor (2026-03-22)

- [2026-03-22] **Architecture overhaul: added Crazyflie-style PD attitude controller inner loop.**
  - Root cause of all 4 failed runs identified: the RL policy was outputting raw torques
    (`moment_scale * action`) and had to simultaneously learn flight dynamics and navigation.
    This is fundamentally misaligned with how real Crazyflie drones operate.
  - Added `attitude_controller.py`: pure-torch PD controller with no Isaac Lab imports.
    Action semantics changed from `[thrust, roll_moment, pitch_moment, yaw_moment]` to
    `[thrust, desired_roll, desired_pitch, desired_yaw_rate]`. The PD controller converts
    attitude errors to moments every physics step.
  - Reference: OmniDrones `AttitudeController` (deployed to real Crazyflie 2.1 hardware via
    crazyswarm2). Gains: `kp_att=0.03`, `kd_att=0.005`, `kp_yaw=0.01` (tunable per Rule 6).
  - Force application simplified: single `set_forces_and_torques` call on main body only
    (matches Isaac Lab Isaac-Quadcopter-Direct-v0 reference; removes parasitic prop-body torques
    and per-step `torch.zeros` allocation, fixing Rule 15 violation).
  - Reward function simplified to 3-term Isaac Lab structure for hover-stability training:
    `(1 - tanh(dist/0.8)) * scale * step_dt` (pos), `sum(sq(lin_vel)) * scale * step_dt` (vel),
    `sum(sq(ang_vel)) * scale * step_dt` (ang_vel). Scales: `pos=15.0, vel=-0.05, ang_vel=-0.01`.
  - Removed: `moment_scale`, `rew_scale_upright`, `rew_scale_alive`, `rew_scale_terminated`
    from hover-stability config (PD controller makes these redundant).
  - Added `GGSwarmMarlHoverStabilityCfg.scene.num_envs=512` for GCE L4 GPU (4x throughput).
  - Added episode length stagger on full reset in `_reset_idx` (Isaac Lab pattern).
  - Old checkpoints (Runs 1–A1) are incompatible with new action semantics; all 4 were FAIL
    so this is sunk cost. Phase 2A restarts from scratch.
  - Action/obs dims, GNN policy architecture, CBF/SwarmRaft/MINCO, eval/assess scripts: unchanged.

## Phase 2A: Run A1 Assessment (2026-03-22)

- [2026-03-22] **Run A1 eval** (run: `2026-03-22_00-32-56_mappo_torch`, 80k iters, hover-stability mode, `best_agent.pt`, 5 episodes):
  - `survival_steps=1.1` | `airborne_ratio=0.700` | `ground_hit_rate=0.423`
  - `mean_roll=59.8°` | `mean_pitch=63.8°` | `orientation_violation_rate=0.524`
  - `mean_formation_error=0.821m` (informational; not gated in Phase 2A)
  - **Verdict: FAIL** (3 FAIL, 3 WARN, 0 PASS)
  - **Convergence analysis**: Peak reward 6997 at step 14k; drifted to 5618 by step 80k (-19.7%). Episode length maxed at 496/500 from step 3k — agents reach survival gate quickly but are flying erratically (high roll/pitch). Policy std dev grew monotonically (0.35 → 5.8) — entropy never collapsed, policy still exploring at end of run.
  - **Root cause**: `survival_steps=1.1` is an artifact of the eval metric — the episode survival counter resets on ground hit, so a single crash shows as ~1 step. `airborne_ratio=0.700` (gate > 0.9) and `orientation_violation_rate=0.524` (gate < 0.1) indicate agents are staying up but tumbling severely.
  - **Decision**: Reward rebalance needed. Upright reward is insufficient vs. position reward — agents reach target altitude but don't stabilize orientation. Increase `rew_scale_upright` and `rew_scale_ang_vel` penalty. Consider reducing `rew_scale_pos` slightly to shift priority toward stability.
  - **Fixed assess pipeline bugs**: 3 bugs in `_cmd_assess` / `eval_phase2.py` prevented end-to-end execution: (1) `--log_dir` → `--run_dir` arg name; (2) missing `--output_json` in `eval_phase2.py`; (3) `mean_episode_survival_steps` key renamed to `survival_steps`. Also fixed Windows cp1252 `←` char in scorecard print.

## Phase 2A: Run PD1 Assessment (2026-03-22)

- [2026-03-22] **Run PD1 eval** (run: `2026-03-22_04-32-04_mappo_torch`, 80k iters, PD inner loop, hover-stability, `best_agent.pt`, 5 episodes):
  - `survival_steps=250.5` | `airborne_ratio=0.542` | `ground_hit_rate=0.815`
  - `mean_roll_deg=22.2°` | `orientation_violation_rate=0.219` | `mean_formation_error_m=12.0` (informational; formation reward is 0 in Phase 2A — do not gate on this)
  - **Verdict: FAIL** (3 FAIL, 3 WARN, 0 PASS)
  - **Progress vs raw-torque runs (Runs 1–A1)**: Mean roll dropped from ~60–75° to ~22° — the PD attitude loop materially stabilizes attitude; the policy is no longer systematically inverted.
  - **Remaining gap**: High `ground_hit_rate` and low `airborne_ratio` — the outer RL loop is not yet holding altitude / goal position; agents still dip to the ground often despite better roll.
  - **Decision: FAIL** — do not advance to Phase 2B until Phase 2A gates pass (`airborne_ratio` > 0.9, `ground_hit_rate` < 0.05, `mean_roll_deg` < 15°, etc.).
  - **Next action** (PD + 3-term hover reward — **do not** use `rew_scale_upright` / `rew_scale_alive`; they are 0 in hover-stability cfg):
    1. Tune low-level loop in `GGSwarmMarlEnvCfg`: try slightly higher `kp_att` and/or `max_moment`; confirm `thrust_to_weight` vs Isaac Lab Crazyflie baseline.
    2. Tune hover reward: `rew_scale_pos`, `rew_scale_vel`, `rew_scale_ang_vel` (Isaac Lab–style terms only); inspect TensorBoard for reward and policy std.
    3. Optional: longer train or curriculum tweak — not reward surgery until PD + reward sweep above is logged.
  - **Scorecard follow-up**: Updated `scripts/ggswarm_utils/scorecard.py` FAIL hints to match PD architecture (removed stale upright/alive/formation-only advice for Phase 2A).

- [2026-03-22] **Headless eval video**: Unified eval (`scripts/eval.py`) and `run_eval()` can record an offscreen `rgb_array` clip via `--video` (same `EncodingRecordVideo` path as `play.py`), writing to `<run_dir>/videos/eval/`. Refactored `EncodingRecordVideo` into `scripts/ggswarm_utils/encoding_record_video.py` for reuse. `run.py` eval/assess and `post_train_assess.py` forward `--video` / codec options; default `rendering_mode=quality` when `--video` is set without an override.
- [2026-03-22] **Eval checkpoint validation**: `validate_eval_checkpoint_path()` rejects missing `.pt` files and paths containing literal `<`/`>` (e.g. pasted `<run>` placeholders) before AppLauncher / `RecordVideo`, avoiding opaque Windows `WinError 123` from `os.makedirs`.

## Phase 2A: PD2 config (post–Run PD1 assess)

- [2026-03-22] **Hover-stability next run (PD2 bundle)** after PD1 FAIL on `airborne_ratio` / `ground_hit_rate` with improved roll (~22°):
  - **Base `GGSwarmMarlEnvCfg`:** `thrust_to_weight` **1.9 → 2.0** (neutral collective matches documented hover: `T/W × 0.5 = 1.0`); `kp_att` **0.03 → 0.045**; `max_moment` **0.02 → 0.03** (`kd_att` unchanged at 0.005 — raise if smoke shows oscillation).
  - **`GGSwarmMarlHoverStabilityCfg`:** `rew_scale_pos` **15.0 → 18.0** (stronger 3D goal signal, including spawn altitude).
  - **Rationale:** Prioritize vertical authority and faster attitude tracking under load; no `rew_scale_upright` / formation changes (Phase 2A scope). Docs: `training_workflow.md` Step 0 grep aligned to PD + hover; `post_train_analysis.md` Phase 2A matrix rows aligned to PD + 3-term reward; `phase2_brain_development.md` footnote on `mean_formation_error_m` in 2A.

## Phase 2A: Run PD2 Assessment (2026-03-22)

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-22_07-03-55_mappo_torch`
- **Config class:** `GGSwarmMarlHoverStabilityCfg` — commit `5bb33cf` (PD2 bundle: `thrust_to_weight=2.0`, `kp_att=0.045`, `max_moment=0.03`, `rew_scale_pos=18.0`)
- **Convergence:** entropy collapse not detected | peak reward **14925** @ step **53,000** | final **10721** @ step **80,000** (drawdown vs peak) | recommended budget **92k** steps
- **Scorecard** (`best_agent.pt`, 5 episodes, hover-stability assess):
  - `survival_steps=250.5` (**invalid metric** — pre-fix collector artifact; see `post_train_analysis.md` / `run_history.md` footnote) | `airborne_ratio=0.571` | `ground_hit_rate=0.723`
  - `mean_roll_deg=24.6°` | `orientation_violation_rate=0.296` | `mean_formation_error_m=5.34` (informational; not gated in Phase 2A)
  - **Verdict: FAIL** (scorecard overall FAIL; interpret gates on `airborne_ratio` / `ground_hit_rate` / `mean_roll_deg` — not legacy `survival_steps`)
- **Vs Run PD1:** `airborne_ratio` ↑ (0.542 → 0.571), `ground_hit_rate` ↓ (0.815 → 0.723) — altitude proxy improved; `mean_roll_deg` and `orientation_violation_rate` slightly worse (22.2° → 24.6°, 0.219 → 0.296).
- **Decision: FAIL** — do not advance to Phase 2B. PD2 moved vertical metrics but Phase 2A gates are still far from pass.
- **Next action:** (1) Inspect TensorBoard for the 53k→80k reward drawdown (noise vs regression). (2) If curves justify it, **90k–100k** hover-stability rerun before further reward surgery. (3) If another config pass is needed, stay in 3-term + PD space: e.g. small `rew_scale_vel` / `rew_scale_ang_vel` nudge or incremental PD (`kd_att` if oscillation, else bounded `max_moment` / `kp_att`); **do not** re-enable `rew_scale_upright` without an explicit design change and changelog entry.

## Phase 2A: PD3 prep — survival metric fix + low-clearance shaping (2026-03-22)

- [2026-03-22] **`survival_steps` assess fix:** `Phase2Collector` now records survival **once per eval episode** in `on_episode_end`: first step (1-based) where batch `ground_hit_rate > 0`, else full horizon. `EvalStats.update()` no longer ingests per-step survival; use `record_episode_survival()`. `scripts/analyze_checkpoints.py` uses the same episode boundary logic. Tests: `tests/unit/test_ggswarm_utils.py` (`TestPhase2CollectorSurvival`), `tests/unit/test_contract_logic.py`.
- [2026-03-22] **Low-clearance MDP alignment:** `MarlRewardParams` / `compute_marl_rewards` add `rew_low_clearance` from `rew_scale_low_clearance` × depth below `(min_height + low_clearance_margin_m)`. Base `GGSwarmMarlEnvCfg`: `rew_scale_low_clearance=0.0`, `low_clearance_margin_m=0.2`. `GGSwarmMarlHoverStabilityCfg`: `rew_scale_low_clearance=-8.0`. **Rationale:** penalise flight in the dead band between crash floor and eval “airborne” threshold.
- [2026-03-22] **TensorBoard telemetry:** `extras["log"]` adds `rew_low_clearance`, `mean_world_z`, `low_clearance_frac`.
- [2026-03-22] **Docs / rules:** `docs/ops/phase2a_diagnostics.md` (TB, baselines, ladder, train-length policy); `post_train_analysis.md` metric definitions; `run_history.md` footnote for PD1/PD2 survival; `training_workflow.md`, `commands.md`, `architecture.md`; `.cursor/rules/project-rules.mdc` (Rule 18 `attitude_controller.py`, Rule 22 hover table, inner-loop SHOULD); `scorecard.py` assess report note + FAIL hint.
- **Next:** Rule 22 smoke → GCE `hover-stability train` (≥80k) → pull → assess → append **`run_history.md`** with **post-fix** `survival_steps`.

## Phase 2A: Run PD3 Assessment (2026-03-22)

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-22_16-00-12_mappo_torch`
- **Train budget:** 92,000 iterations (GCE); **config:** `GGSwarmMarlHoverStabilityCfg` with PD3 prep (`rew_scale_low_clearance`, survival metric fix in codebase; VM pulled `main` including PD3 bundle).
- **Convergence:** peak reward **19071** @ step **90,000** | final **18615** @ step **92,000** | **entropy collapse @ step 81,000** (reward ~13748 at collapse) | recommended budget **~93k** steps
- **Scorecard** (`best_agent.pt`, 5 episodes, hover-stability assess, **post-fix** `survival_steps`, **eval seed 42** — train parity; superseded seed-1 pass):
  - `survival_steps=4.4` | `airborne_ratio=0.617` | `ground_hit_rate=0.494` (**scorecard WARN** vs threshold 0.5; was **FAIL** at 0.717 under seed 1)
  - `mean_roll_deg=24.6°` | `orientation_violation_rate=0.349` | `mean_formation_error_m=1.17` (informational; not gated in Phase 2A)
  - **Verdict: FAIL** (survival / airborne still far from pass; `ground_hit_rate` materially better with train-aligned seed)
- **Vs Run PD2** (re-assessed PD2 with fixed collector: `2026-03-22_07-03-55`, `survival_steps≈4.8`): **Training signal improved strongly** (peak/final reward, no mid-run collapse like PD2’s 53k→80k drawdown). **Eval scorecard:** small moves only — `airborne_ratio` ↑ 0.571→0.596, `ground_hit_rate` ↓ 0.723→0.717, `mean_roll_deg` ↓ 24.6°→23.4°; **`survival_steps` unchanged in practice** (~4.6 vs ~4.8) — still immediate batch ground-contact regime on eval.
- **Decision: FAIL** — do not advance to Phase 2B.
- **Next action:** (1) TensorBoard: confirm `rew_low_clearance` / `mean_world_z` / policy entropy post–81k collapse vs eval failures. (2) Consider eval vs train **sim / seed / num_envs** parity check if dips persist. (3) If staying in 3-term + PD space: bounded tweak to `rew_scale_vel` / `rew_scale_ang_vel` or PD limits **after** TB review; log any cfg change here before another GCE run.

- [2026-03-22] **Run PD3 next-step execution (local):**
  - **TensorBoard autopsy:** Summarized scalars for `2026-03-22_16-00-12_mappo_torch` — `Reward/Total reward (mean)` **140 → ~18.6k**; `Policy/Standard deviation` **~0.36 → ~7.39** (matches entropy-collapse warning ~81k). Added `scripts/summarize_tb_scalars.py` and §7 in `docs/ops/phase2a_diagnostics.md`.
  - **Train–eval parity:** Default eval/assess seed was **1** while `skrl_mappo_cfg.yaml` uses **42**. Defaults set to **42** in `post_train_assess.py`, `eval.py`, `eval_runner.py`, and `run.py` assess (`build_assess_cmd` forwards `--seed`).
  - **Run PD3 re-assess (seed 42):** `hover-stability assess` re-run on `2026-03-22_16-00-12_mappo_torch`; `assess_report.md` / `assess_metrics.json` updated. **`run_history.md` PD3 row** now matches seed **42**. vs seed **1**: `ground_hit_rate` **0.717→0.494**, `airborne_ratio` **0.596→0.617**; `orientation_violation_rate` **0.284→0.349**; overall **FAIL** unchanged.
  - **`analyze_checkpoints.py`:** Fixed missing `--num_envs` / `--seed` CLI (was crashing on `args_cli.num_envs`); removed non-reproducible `random.randint` seed — default **`--seed 42`**. Auto **`configure_gnn_policy`** when `--task` matches `PHASE_REGISTRY` with `gnn_default=True` (override with **`--no-gnn`**). Replaced Unicode checkmarks in prints (Windows **cp1252**). Doc’d hover-stability `--task` in `phase2a_diagnostics.md` §5.
  - **No reward / MAPPO YAML change** in this pass (policy exploration cap left as a documented optional follow-up in §7).

- [2026-03-22] **PD neutral (zero-action) baseline:** Added `scripts/pd_neutral_baseline.py` — local headless roll with constant neutral RL commands (no MAPPO checkpoint); prints time-averaged `mean_world_z`, per-step `ground_hit_rate` (any agent below `min_height` per env), and `airborne_ratio` (agent slots above `min_height + 0.2` m). Documented command and metrics in `docs/ops/phase2a_diagnostics.md` §3. Sample run (8 envs, 300 steps, seed 42): `mean_world_z ≈ 0.72 m`, `ground_hit_rate ≈ 0.23`, `airborne_ratio ≈ 0.80` — use as regression guard after PD/spawn edits.

## Phase 2A: PD4 prep — eval semantics, exploration cap, hover nudges (2026-03-22)

- **Eval uses mean actions:** Documented in `docs/ops/phase2a_diagnostics.md` §2.1 — `skrl` `GaussianMixin.act` samples for train, but `scripts/ggswarm_utils/sim_helpers.py` `extract_actions()` prefers `outputs["mean_actions"]` for eval/assess/checkpoint ladder. Wide TB policy std is **not** from stochastic eval sampling.
- **`skrl_mappo_cfg.yaml`:** `max_log_std` **2.0 → 1.0** (caps train-time σ); `initial_log_std` **-1.0 → -0.5**; `entropy_loss_scale` **0.01 → 0.008**. Comments note GNN-only keys are injected in `train.py` / `configure_gnn_policy()` so MLP instantiation stays clean.
- **`GGSwarmGNNPolicy`:** Constructor parameters `hidden_channels`, `num_heads`, `initial_log_std` (Rule 14). `train.py` logs active values when `--gnn`; `sim_helpers.configure_gnn_policy` sets GNN defaults for eval parity.
- **`GGSwarmMarlHoverStabilityCfg`:** `spawn_z_min` **0.5 → 0.65**, `spawn_z_max` **1.5 → 1.65** (goal Z still follows spawn Z); `rew_scale_vel` **-0.05 → -0.055**, `rew_scale_ang_vel` **-0.01 → -0.012**.
- **Ops:** `docs/status/run_history.md` **Run PD4** row `pending` until first post-train assess; Rule 22 smoke: `python scripts/run.py debug smoke --task Template-GGSwarm-Marl-HoverStability-v0 --iterations 1 --gnn`.
- **GCE (user-triggered):** After smoke, launch hover-stability train on VM per `docs/ops/training_workflow.md` / `.cursor/rules/gce-training-ops.mdc`; pull → assess → replace PD4 `TBD` row with scorecard metrics.

## Phase 2A: Run PD4 Assessment (2026-03-22)

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-22_18-38-39_mappo_torch`
- **Train budget:** 92,000 iterations (GCE); **config:** PD4 bundle (see Phase 2A PD4 prep entry).
- **Convergence:** peak reward **20338** @ step **87,000** | final **20163** @ step **92,000** | entropy collapse @ step **56,000** | recommended budget **~64k** steps (heuristic)
- **Scorecard** (`best_agent.pt`, 5 episodes, hover-stability assess, **seed 42**):
  - `survival_steps=5.0` | `airborne_ratio=0.687` | `ground_hit_rate=0.361` (**WARN** vs 0.5 threshold)
  - `mean_roll_deg=28.9°` | `orientation_violation_rate=0.373` | `mean_formation_error_m=0.85` (informational; not gated in Phase 2A)
  - **Verdict: FAIL** (same as PD3; `airborne_ratio` / `survival_steps` still far from pass)
- **Vs Run PD3:** Clear win on **`ground_hit_rate`** and **`airborne_ratio`**; small regression on roll/orientation violation.
- **Decision: FAIL** — do not advance to Phase 2B.
- **Next action:** TensorBoard review (`summarize_tb_scalars.py`); consider bounded **`rew_scale_pos`** or **`rew_scale_ang_vel`** nudge, or PD **`kp_att` / `max_moment`** only if TB shows attitude-dominated failure; log any cfg change before next GCE run.
