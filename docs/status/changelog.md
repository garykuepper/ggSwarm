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
