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

## Phase 2A Run PD6 — 2026-03-23

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-23_01-22-37_mappo_torch`
- **Config class:** `GGSwarmMarlHoverStabilityCfg` — commit `0a87fb4`
- **Train budget:** 80,000 iterations (GCE); PD6 = `rew_scale_terminated=-5.0` + `extras["log"]` tensor scalars for SKRL `Info / *` TensorBoard.
- **Convergence:** entropy collapse not detected | peak / final reward **0.36** @ step **80,000** (recommended budget **92k** steps per convergence script)
- **Scorecard** (`post_train_assess.py`, seed **42**, `best_agent.pt`, 5 episodes):
  - survival_steps = **4.4**
  - airborne_ratio = **0.612**
  - ground_hit_rate = **0.534**
  - mean_roll_deg = **21.1°**
  - orientation_violation_rate = **0.234**
  - verdict = **FAIL**
- **vs Run PD5:** Marginal `ground_hit_rate` improvement; `survival_steps` and attitude metrics **regressed**; scalar training reward curve **qualitatively different** (positive plateau vs PD5 negative drift) — review **`Info / rew_*`** and `Info / rew_terminated` on this run before further reward edits.
- **Decision: FAIL** — do not advance to Phase 2B.
- **Next action:** TensorBoard: `Info / mean_world_z`, `Info / rew_low_clearance`, `Info / rew_terminated`, `Policy/Standard deviation`; decide whether to tune **`rew_scale_terminated` magnitude**, **`rew_scale_pos` / vel**, or PD authority **one at a time** with changelog entry before retrain.

- [2026-03-23] **Phase 2A PD7 prep** (structural hover fix — **before** GCE PD7 train):
  - **Root cause (PD1–PD6):** `_reset_idx` assigned **formation-circle XY slots** while Phase 2A had **no formation reward**; stable-hover **`dist_to_goal`** and obs **`rel_pos_to_goal`** still pulled agents laterally → persistent **~21° / ~24°** eval tilt and train/eval narrative mismatch vs TensorBoard aggregates.
  - **Single logical change:** `GGSwarmMarlEnvCfg.hover_in_place: bool = False` (default); `GGSwarmMarlHoverStabilityCfg.hover_in_place = True`; in [`drone_swarm_env.py`](../../source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env.py) `_reset_idx`, after **Z = spawn Z**, **`desired_pos_w[:, :, :2] = root_pos_w[:, :, :2]`** when **`hover_in_place`** — goal = **full 3D spawn pose**. Phase 2B unchanged (inherits **`False`**).
  - **Reward scales (unchanged vs PD6):** `rew_scale_terminated=-5.0`, `rew_scale_pos=18.0`, `rew_scale_vel=-0.055`, `rew_scale_ang_vel=-0.012`, `rew_scale_low_clearance=-8.0`, spawn Z **0.65–1.65** m, `spawn_yaw_range=0.3`, **512** envs.
  - **Docs:** [`docs/design/architecture.md`](../design/architecture.md) Phase 2A **`hover_in_place`** semantics; Rule 22 note in [`docs/ops/pd5_rule22_checklist.md`](../ops/pd5_rule22_checklist.md).
  - **Rule 22 smoke (local):** `python scripts/run.py debug smoke --task Template-GGSwarm-Marl-HoverStability-v0 --iterations 1 --gnn --headless` — **PASS** (512 envs, `GGSwarmMarlHoverStabilityCfg`, GNN `hidden_channels=128`). **`hover_in_place`:** confirm with checklist one-liner under Isaac env or inspect `GGSwarmMarlHoverStabilityCfg` in [`drone_swarm_env_cfg.py`](../../source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env_cfg.py) (**`True`**).
  - **Tests:** removed duplicate root [`tests/test_contract_logic.py`](../../tests/test_contract_logic.py) (second copy without pytest fixtures; canonical suite is [`tests/unit/test_contract_logic.py`](../../tests/unit/test_contract_logic.py)). Full `pytest tests`: **403 passed**, 7 skipped.
  - **GCE launch:** command in [`docs/ops/training_workflow.md`](../ops/training_workflow.md) (PD7 block): `.\scripts\cloud\gce_train_launch.ps1 hover-stability train --headless --gnn --max_iterations 92000` after `git push`. **Pull/sync** after completion: `python scripts/cloud/pull_results_from_gcs.py --family marl --latest 1`.
  - **Train budget (GCE):** **92,000** iterations (`--max_iterations 92000`). **Post-train:** local `hover-stability assess`, then **Rule 23** row in [`run_history.md`](../status/run_history.md) + **Run PD7** section below.

- [2026-03-23] **Phase 2A PD8 prep** (ceiling-escape fix + VM verification — **before** GCE PD8 train):
  - **Root cause (PD7 trajectory diagnostics):** `hover_in_place` code was **never on the VM** during PD7 training — XY drift 4–8 m from spawn in every episode confirmed VM ran PD6 code (commit `0a87fb4`). Additionally, `rew_scale_terminated=-5.0` created ceiling escape: drone_0 climbed to 2.5–3.0 m every episode (altitude traces); stochastic policy exploited σ noise for ceiling hover, deterministic eval could not reproduce → train/eval gap.
  - **Single knob (Rule 22):** `GGSwarmMarlHoverStabilityCfg.rew_scale_terminated: -5.0 → 0.0`. Floor avoidance via `rew_scale_low_clearance=-8.0` (penalty per metre below 0.3 m) + position reward.
  - **VM verification:** SSH confirmed commit `7e9506c` with `hover_in_place: True` and `rew_scale_terminated: 0.0` before launch.
  - **New diagnostic:** `scripts/plot_trajectories.py` — standalone trajectory recording via `TrajectoryCollector(Phase2Collector)` subclass; generates altitude, XY, and attitude PNG plots per eval episode.
  - **Docs:** Rule 22 checklist PD8+ column; `training_workflow.md` PD8 launch command.
  - **Rule 22 smoke (local):** PASS (512 envs, GNN `hidden_channels=128`, `rew_scale_terminated=0.0`).

## Phase 2A Run PD8 — 2026-03-23

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-23_06-23-47_mappo_torch`
- **Config class:** `GGSwarmMarlHoverStabilityCfg` — commit `7e9506c` (VM verified via SSH before launch)
- **Train budget:** 92,000 iterations (GCE); `rew_scale_terminated=0.0`, `hover_in_place=True`
- **Convergence:** entropy collapse not detected | peak reward **243.06** @ step **85,000** | final **240.52** @ step **92,000** | recommended budget **105,799** steps
- **Scorecard** (`post_train_assess.py`, seed **42**, `best_agent.pt`, 5 episodes):
  - survival_steps = **7.0**
  - airborne_ratio = **0.732**
  - ground_hit_rate = **0.304**
  - mean_roll_deg = **32.2°** | mean_pitch_deg = **31.5°**
  - orientation_violation_rate = **0.525**
  - verdict = **FAIL**
- **TensorBoard (key scalars):**
  - `Policy / Standard deviation (drone_0)`: **0.609 → 2.704** (pinned at `max_log_std=1.0` ceiling = e^1 = 2.718)
  - `Policy / Standard deviation (drone_1)`: **0.609 → 2.718**; `(drone_2)`: **0.610 → 1.059**
  - `Info / mean_world_z`: **0.640 → 1.142 m** (learning to hover — major improvement)
  - `Info / low_clearance_frac`: **0.393 → 0.002** (floor penalty working, near zero by end)
  - `Info / rew_pos`: **0.111 → 0.308** (position reward improving)
  - `Info / rew_ang_vel`: **−0.112 → −0.116** (flat — attitude never improves)
- **Checkpoint ladder** (`analyze_checkpoints.py`, 10k interval, 2 eps): **10k is best checkpoint** (airborne **0.766**, formation error **0.675 m**); later checkpoints degrade as σ explodes. Policy gets **worse over training** — stochastic training reward rises while deterministic eval quality falls.
- **Trajectory plots:** shark-fin altitude pattern (~50-step crash-reset cycles, improved from PD7's 7-step saw-tooth); XY drift **1–3 m** (down from 4–8 m — `hover_in_place` confirmed working); attitude oscillating ±50–75° correlated with crash cycle.
- **vs Run PD7:** `airborne_ratio` **0.516→0.732** (+0.22 ✓), `ground_hit_rate` **0.712→0.304** (−0.41 ✓); `mean_roll_deg` **18.4°→32.2°** (worse ✗ — σ explosion), `orientation_violation_rate` **0.191→0.525** (worse ✗). Both PD8 fixes confirmed (ceiling escape gone, hover_in_place working), but σ explosion is new dominant failure.
- **Decision: FAIL** — do not advance to Phase 2B.
- **Next action:** `max_log_std: 1.0 → 0.0` in `skrl_mappo_cfg.yaml` (clamp σ ceiling from 2.72 to 1.0). Single Rule 22 knob. The 10k checkpoint outperforming 90k proves σ blowout degrades deterministic policy quality; tighter ceiling should make later checkpoints the best.

- [2026-03-23] **Phase 2A PD9 prep** (σ explosion fix + post-train tooling — **before** GCE PD9 train):
  - **Root cause (PD8 TB diagnostics):** `max_log_std=1.0` in `skrl_mappo_cfg.yaml` allowed σ ceiling of e^1.0=2.718; drone_0/drone_1 pinned at ceiling throughout training. Stochastic training reward rises (wider Gaussian = better exploration credit) while deterministic eval quality degrades. 10k checkpoint outperforms 90k.
  - **Single knob (Rule 22):** `skrl_mappo_cfg.yaml` `max_log_std: 1.0 → 0.0` (σ ceiling drops from 2.718 to 1.0; policy can still explore but can't blow out).
  - **Post-train tooling:** `post_train_assess.py` now embeds TB scalar diagnostics (Policy std dev, mean_world_z, reward components) directly in `assess_report.md` via `extract_scalar_summary()` + `format_tb_diagnostics()`. Also prints ready-to-paste `run_history.md` row at end of assessment (Rule 23 compliance).
  - **Docs:** Rule 22 checklist PD9+ column with `max_log_std (YAML) = 0.0`.

## Phase 2A Run PD9 — 2026-03-23

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-23_16-19-28_mappo_torch`
- **Config class:** `GGSwarmMarlHoverStabilityCfg` — `max_log_std=0.0` (σ ceiling = 1.0)
- **Train budget:** 92,000 iterations (GCE)
- **Convergence:** entropy collapse @ **58,000** | peak reward **235.97** @ step **84,000** | final **203.32** @ step **92,000** | recommended budget **66,700** steps
- **Scorecard** (`post_train_assess.py`, seed **42**, `best_agent.pt`, 5 episodes):
  - survival_steps = **6.2** (FAIL; gate > 500)
  - airborne_ratio = **0.629** (FAIL; gate > 0.9)
  - ground_hit_rate = **0.444** (WARN; gate < 0.5)
  - mean_roll_deg = **29.7°** (WARN; gate < 15°)
  - orientation_violation_rate = **0.469** (WARN; gate < 0.1)
  - mean_formation_error_m = **1.199** (WARN; gate < 1.5)
  - verdict = **FAIL**
- **TensorBoard (key scalars):**
  - `Policy / Standard deviation (drone_0)`: **0.609 → 1.000** (pinned at new `max_log_std=0.0` ceiling = e^0 = 1.0)
  - `Info / mean_world_z`: **0.640 → 1.161 m** (roughly same as PD8)
  - `Info / rew_pos`: **0.111 → 0.311** (improving — policy learns to be near goal, not to hover)
  - `Info / rew_ang_vel`: **−0.118 → −0.114** (flat — attitude NEVER improves; 9 runs, always flat)
  - `Info / rew_low_clearance`: **−0.654 → −0.002** (floor penalty working)
- **Trajectory plots:** shark-fin crash cycles; 2–4 m XY drift; ±50–100° attitude swings.
- **vs Run PD8:** σ ceiling fix worked (1.0 vs 2.7), but altitude/airborne **regressed** (0.629 vs 0.732).
  Entropy collapsed at 58k — narrower policy converged to crash-reset local optimum before discovering hover.
  Attitude marginally improved (29.7° vs 32.2°).
- **Decision: FAIL** — do not advance to Phase 2B.
- **Root cause analysis (PD1–PD9 deep dive):** The reward landscape has a deceptive local optimum.
  With `pos_tanh_sigma=0.8`, a drone 0.5 m off-target still gets 45% of max reward.
  Crash-reset cycles (~83 resets/episode) yield ~160 reward vs ~180 for hover — 1.12x ratio
  the critic cannot distinguish. Velocity penalty (327x weaker than position) is ignored
  (TB confirms `rew_ang_vel` flat across all 9 runs). `rew_scale_terminated=0.0` = free crash.
- **Next action:** PD10 — sharpen position reward discriminator (see PD10 prep below).

- [2026-03-23] **Phase 2A PD10 prep** (reward discriminator fix — **before** GCE PD10 train):
  - **Root cause (PD1–PD9 retrospective):** `pos_tanh_sigma=0.8` (hardcoded, Rule 6 violation) makes
    crash-reset nearly as rewarding as hover (1.12x ratio). Velocity/angular velocity penalties
    327x weaker than position reward — policy ignores attitude. `rew_scale_terminated=0.0` = free crash.
  - **Single conceptual change (Rule 22):** Sharpen position reward discriminator so crash-reset is
    clearly worse than hover. At `pos_tanh_sigma=0.25`: reward drops to 5% at 0.5 m drift (was 45%);
    crash-reset yields ~40 reward vs ~180 for hover (4.5x ratio without crash penalty).
  - **Parameters changed:**
    - `pos_tanh_sigma: 0.8 → 0.25` (new cfg param; Rule 6 fix for hardcoded 0.8 in `contract_logic.py`)
    - `rew_scale_vel: -0.055 → -0.3` (5.5x increase; makes velocity visible in TB, still 60x below pos peak)
    - `rew_scale_ang_vel: -0.012 → -0.06` (5x increase; makes attitude visible in TB)
    - `rew_scale_terminated: 0.0 → -2.0` (moderate crash cost; ceiling escape suppressed by tight sigma)
  - **No MAPPO hyperparameter changes:** `max_log_std=0.0`, `initial_log_std=-0.5`, `entropy_loss_scale=0.008` unchanged. Reward landscape fix should unblock exploration — standard PPO should suffice once crash-reset is no longer a competitive strategy.
  - **Files changed:** `contract_logic.py` (add `pos_tanh_sigma` to `StableHoverRewardParams`), `drone_swarm_env_cfg.py` (add field + override), `drone_swarm_env.py` (pass-through), `test_ggswarm_utils.py` (2 new sigma tests; 11/11 pass).
  - **Rule 22 smoke (local):** PASS (142 tests, 512 envs, GNN).

## Phase 2A Run PD10 — 2026-03-23

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-23_23-36-16_mappo_torch`
- **Config class:** `GGSwarmMarlHoverStabilityCfg` — `pos_tanh_sigma=0.25`, `rew_scale_vel=-0.3`,
  `rew_scale_ang_vel=-0.06`, `rew_scale_terminated=-2.0`, `max_log_std=0.0`
- **Train budget:** 92,000 iterations (GCE)
- **Convergence:** no entropy collapse | peak reward **419.97** @ step **56,000** | final **196.28** @ step
  **92,000** | recommended budget **105,799** steps
- **Scorecard** (`post_train_assess.py`, seed **42**, `best_agent.pt` = 80k checkpoint, 5 episodes):
  - survival_steps = **5.2** (FAIL; gate > 500)
  - airborne_ratio = **0.736** (FAIL; gate > 0.9)
  - ground_hit_rate = **0.373** (WARN; gate < 0.5)
  - mean_roll_deg = **23.9°** (WARN; gate < 60°)
  - orientation_violation_rate = **0.455** (WARN; gate < 0.5)
  - mean_formation_error_m = **0.676** (WARN; gate < 1.5)
  - verdict = **FAIL**
- **TensorBoard — TRAINING BREAKTHROUGH (first run with healthy dynamics):**
  - `Policy / Standard deviation (drone_0)`: **0.60 → 0.12** (healthy decay; no explosion/ceiling)
  - `Info / mean_dist_to_goal`: **0.91 → 0.015 m** (1.5 cm error — policy learned to hover!)
  - `Info / ground_hit_rate_step`: **0.22 → 0.0024** (99% crash reduction in training)
  - `Info / mean_lin_speed`: **1.59 → 0.035 m/s** (nearly stationary in training)
  - `Info / thrust_val_mean`: **0.41 → 0.499** (converged to hover thrust)
  - `Info / rew_ang_vel`: **-0.56 → -0.03** (attitude learning — first time ever across PD1-PD10)
  - `Info / moment_saturated_frac`: **0.98 → 0.33** (PD still saturating 33% of time)
  - `Episode / Total timesteps (mean)`: **9 → 217** (24x episode length improvement)
- **Checkpoint ladder** (10k intervals, 2 eps each): all checkpoints plateau at airborne **0.71-0.76**;
  90k is best (airborne **0.746**, ground_hit **0.337**, roll **22.0°**). mean_speed ≈ **1.7 m/s** in ALL
  checkpoints despite training converging to 0.035 m/s. Flat ceiling = architectural limit (PD saturation).
- **Train-eval gap: 155x** (ground_hit 0.0024 training vs 0.373 eval). Stochastic noise during training
  acts as dithering past PD saturation boundary; deterministic eval commits to saturating commands.
- **Decision: FAIL** — do not advance to Phase 2B.
- **Next action:** PD11 — increase PD moment authority (`max_moment: 0.03 → 0.05`) to eliminate saturation.

- [2026-03-23] **Phase 2A PD11 prep** (PD authority fix — **before** GCE PD11 train):
  - **Root cause (PD10 TB + checkpoint analysis):** `moment_saturated_frac = 33%` at training end.
    At `max_moment=0.03`, PD saturates at 6.4° error when ang_vel = 5 rad/s. Stochastic training
    survives via noise dithering; deterministic eval locks into saturation → 155x crash rate gap.
    All 9 checkpoints hit same eval ceiling (airborne 0.71-0.76) regardless of training step.
  - **Single knob (Rule 22):** `max_moment: 0.03 → 0.05`. At 0.05 Nm, PD can correct full 30° tilt
    envelope without saturating even at 5 rad/s (threshold = 32°). Deterministic policy no longer
    needs noise to escape saturation lock.
  - **All PD10 reward parameters preserved** — reward landscape is working correctly.
  - **No MAPPO changes** — `max_log_std=0.0`, `entropy_loss_scale=0.008`, `learning_rate=1e-4` unchanged.

## Phase 2A Run PD11 — 2026-03-24

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-24_02-19-01_mappo_torch`
- **Config class:** `GGSwarmMarlHoverStabilityCfg` — `max_moment=0.05` (PD authority increase),
  all PD10 reward parameters preserved
- **Train budget:** 92,000 iterations (GCE)
- **Convergence:** no entropy collapse | peak reward **-17.91** @ step **65,000** | final **-159.92** @ step
  **92,000** | recommended budget **105,799** steps
- **Scorecard** (`post_train_assess.py`, seed **42**, `best_agent.pt`, 5 episodes):
  - survival_steps = **4.2** (FAIL; gate > 500)
  - airborne_ratio = **0.757** (FAIL; gate > 0.9)
  - ground_hit_rate = **0.358** (WARN; gate < 0.5)
  - mean_roll_deg = **49.6°** (WARN; gate < 60°)
  - orientation_violation_rate = **0.582** (FAIL; gate < 0.1)
  - mean_formation_error_m = **0.548** (WARN; gate < 1.5)
  - verdict = **FAIL**
- **Training Curve Progression (new diagnostic — first run with `extract_training_progression`):**
  - `thrust_val_mean`: **0.43 → 0.07 → 0.12** — policy cut thrust by 10k and never recovered hover.
    PD10 converged to 0.499 (hover); PD11 converged to 0.12 (diving).
  - `mean_dist_to_goal`: **0.68 → 0.47** (flat; PD10 reached **0.015 m**)
  - `ground_hit_rate_step`: **0.19 → 0.28** (getting **worse** over training)
  - `rew_ang_vel`: **-1.57 → -0.36** (improving — but via "don't fly" exploit, not stable hover)
  - `moment_saturated_frac`: **0.97 → 0.45** (still high despite 67% larger `max_moment`)
- **Root cause:** `max_moment: 0.03 → 0.05` increased PD corrective moments, but stronger corrections
  produce larger angular velocities during recovery. `rew_scale_ang_vel=-0.06` punishes this heavily.
  The policy's cheapest escape: cut thrust → no flight → less angular velocity → less penalty. This
  "don't fly" exploit was not possible in PD10 because weaker moments (0.03) produced gentler corrections
  that didn't trigger the angular velocity penalty as hard.
- **vs PD10:** `mean_roll_deg` **23.9° → 49.6°** (2x worse); `thrust_val_mean` **0.499 → 0.125**
  (not hovering); `orientation_violation_rate` **0.455 → 0.582** (worse). Clear regression.
- **Decision: FAIL** — do not advance to Phase 2B.
- **Next action:** PD12 — reduce `rew_scale_ang_vel` to make the angular velocity penalty compatible
  with stronger PD moments. Keep `max_moment=0.05` (needed for train-eval gap closure).

- [2026-03-24] **Phase 2A PD12 prep** (ang_vel penalty rebalance — **before** GCE PD12 train):
  - **Root cause (PD11 regression):** `rew_scale_ang_vel=-0.06` + `max_moment=0.05` created a
    "don't fly" exploit. Stronger PD moments produce larger transient angular velocities during
    attitude correction. The ang_vel penalty punished corrections so heavily that the policy
    learned to cut thrust (0.12 vs 0.50 hover) rather than fly and risk angular velocity penalties.
  - **Fix (two knobs, one conceptual change):**
    - `rew_scale_ang_vel: -0.06 → -0.01` — weaker ang_vel penalty so the policy tolerates transient
      angular velocity from PD corrections. PD10 had -0.06 with weaker moments (0.03); with stronger
      moments (0.05), corrections are more vigorous and need a gentler penalty.
    - `rew_scale_vel: -0.3 → -0.1` — reduce velocity penalty proportionally. PD10 showed the policy
      converges to near-zero speed regardless; 3x weaker still provides signal without suppressing
      early exploration.
  - **Preserved from PD10/PD11:** `max_moment=0.05` (needed for train-eval gap), `pos_tanh_sigma=0.25`
    (sharp discriminator working), `rew_scale_pos=18.0`, `rew_scale_terminated=-2.0`,
    `rew_scale_low_clearance=-8.0`, `max_log_std=0.0`.
  - **No MAPPO changes.**

## Phase 2A Run PD12 — 2026-03-24

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-24_04-42-36_mappo_torch`
- **Config class:** `GGSwarmMarlHoverStabilityCfg` — `rew_scale_ang_vel=-0.01`, `rew_scale_vel=-0.1`,
  `max_moment=0.05`
- **Train budget:** early-stopped at **20,000** iterations (health check)
- **Early stop reason:** `ground_hit_rate_step rising: 0.170 -> 0.371; mean_dist_to_goal not improving: 0.503 -> 0.754`
- **Training progression (20k):**
  - `moment_saturated_frac`: **0.97 -> 0.98** (worse than PD11 — 98% saturation despite 0.05 max_moment)
  - `thrust_val_mean`: **0.44 -> 0.42** (not collapsing as badly as PD11)
  - `ground_hit_rate_step`: **0.19 -> 0.31** (rising — not learning)
- **Root cause:** Inertia extraction (`scripts/extract_crazyflie_inertia.py`) revealed Ixx=**1.66e-5** kg*m^2 —
  18x smaller than the estimated 0.0003. At this inertia, the current gains (kp=0.045, kd=0.005) give
  **zeta=2.9 (heavily overdamped)**, not 0.68 (underdamped) as assumed. The D-term (kd*omega) consumes
  **107% of the P-term budget** at 5 rad/s, leaving zero corrective authority. The PD was not oscillating —
  it was too sluggish to correct.
- **Decision: FAIL** — do not advance.
- **Next action:** PD13 — inertia-based critically-damped gains.

- [2026-03-24] **Phase 2A PD13 prep** (inertia-based PD gain tuning — **before** GCE PD13 train):
  - **Root cause (PD11/PD12 retrospective):** Gains were empirically chosen without knowing sim inertia.
    `scripts/extract_crazyflie_inertia.py` extracted Ixx=1.66e-5, Iyy=1.67e-5, Izz=2.93e-5 kg*m^2
    from PhysX. At I=1.66e-5, kd=0.005 gives zeta=2.9 (heavily overdamped); D-term at 5 rad/s
    produces 0.025 Nm, exceeding the P-term at 10 deg error (0.008 Nm). No corrective authority.
  - **Fix (inertia-derived gains):**
    - `kd_att: 0.005 -> 0.00173` — critical damping: kd = 2*sqrt(kp*I) = 2*sqrt(0.045*1.66e-5) = 0.00173.
      omega_n=52.1 rad/s, t_settle=0.077s (4 control steps). D-term at 5 rad/s now 0.0087 Nm (was 0.025).
    - `max_moment: 0.05 -> 0.08` — worst-case P+D moment at 30 deg tilt + peak omega is 0.070 Nm;
      0.08 provides 15% headroom. Previous 0.05 cap was 71% of needed budget.
    - `kp_att`: unchanged (0.045). `kp_yaw`: unchanged (0.01, fits in budget).
  - **Reward scales reverted to PD10:** `rew_scale_ang_vel: -0.01 -> -0.06`, `rew_scale_vel: -0.1 -> -0.3`.
    PD11/PD12 reduced these to work around overdamped PD; with critically-damped gains, corrections
    no longer produce excessive angular velocities.
  - **Verification:** `pd_neutral_baseline.py` (8 envs, 500 steps): airborne=1.000, ground_hit=0.000,
    mean_z=1.147m — perfect hover with zero RL actions. 434 unit tests pass.
  - **New tooling:** `scripts/extract_crazyflie_inertia.py`, inertia/zeta logging in env `__init__`,
    `test_no_saturation_at_max_tilt_and_moderate_ang_vel`, `test_damping_ratio_is_near_critical`.

## Phase 2A Run PD15b — 2026-03-24

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-24_15-13-09_mappo_torch`
- **Config class:** `GGSwarmMarlHoverStabilityCfg` — PD15 config: reverted to PD10 gains (`kd_att=0.005`,
  `max_moment=0.03`), `eval_noise_std=0.05` (PD saturation dithering added)
- **Train budget:** 92,000 iterations (GCE)
- **Convergence:** peak reward **419.97** @ step **56,000** | final **196.28** @ step **92,000** |
  entropy collapse: not detected | std **0.5951 → 0.1249** | recommended budget **~105,799** steps
- **Training diagnostics (TB scalars, 1k → 92k):**
  - `rew_ang_vel`: **-0.5631 → -0.0318** (strong attitude improvement)
  - `moment_saturated_frac`: **0.9789 → 0.3323** (saturation greatly reduced)
  - `ground_hit_rate_step`: **0.2209 → 0.0024** (near-zero by 60k — healthy in-training hover)
  - `mean_dist_to_goal`: **0.9119 → 0.0154** (nearly at goal)
  - `mean_world_z`: **0.6497 → 1.1441 m** (drones climbed to target altitude)
  - `thrust_val_mean`: **0.4125 → 0.4993** (converging to hover thrust)
- **Scorecard** (`best_agent.pt`, 5 episodes, seed 42, `eval_noise_std=0.05`):
  - `survival_steps` = **5.0** | `airborne_ratio` = **0.737** | `ground_hit_rate` = **0.353**
  - `mean_roll_deg` = **23.6°** | `mean_pitch_deg` = **27.7°** | `orientation_violation_rate` = **0.305**
  - `mean_formation_error_m` = **0.717** (informational)
  - **Verdict: FAIL**
- **Key observation:** Severe train–eval gap. In-training `ground_hit_rate_step` reached 0.002 by 60k
  (near-perfect hover), but eval scorecard shows `survival_steps=5` and `ground_hit_rate=0.353`.
  Reward peaked at 56k then decayed to 196 by 92k — policy likely overfit or destabilized in late
  training. `eval_noise_std=0.05` dithering may be amplifying the gap.
- **Decision: FAIL** — do not advance to Phase 2B.
- **Next action:** Diagnose train–eval gap before next run. Candidates: (1) reduce `eval_noise_std`
  or disable dithering during assess; (2) use 56k checkpoint (peak) instead of `best_agent.pt`;
  (3) early-stop or checkpoint at peak reward rather than running to 92k with reward decay.

## Phase 2A Run PD16 — 2026-03-24

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-24_22-37-14_mappo_torch`
- **Config class:** `GGSwarmMarlHoverStabilityCfg` — dropped PD controller, switched to Isaac Lab
  direct moment control (`moment_scale=0.01`). All PD10 reward params preserved.
- **Train budget:** 92,000 iterations (GCE, 512 envs)
- **Convergence:** peak reward **477.58** @ step **13,000** | final **433.75** @ step **92,000**
- **Training diagnostics (TB scalars, 1k → 92k):**
  - `ground_hit_rate_step`: **0.179 → 0.000** (zero crashes from 10k onward)
  - `mean_dist_to_goal`: **0.551 → 0.009 m** (9 mm precision)
  - `thrust_val_mean`: **0.452 → 0.503** (perfect hover by 10k)
  - `rew_ang_vel`: **-0.611 → -0.057**
- **Scorecard** (`best_agent.pt`, 5 episodes, seed 42):
  - `survival_steps` = **5.0** | `airborne_ratio` = **0.473** | `ground_hit_rate` = **0.706**
  - `mean_roll_deg` = **83.1°** | `mean_pitch_deg` = **85.2°**
  - **Verdict: FAIL** — train-eval gap even worse without PD damping
- **Root cause identified:** `self.robot.write_data_to_sim()` in `_apply_action()` (line 288).
  Isaac Lab's reference quadcopter does NOT call this — the `DirectRLEnv` base class handles it
  at the correct simulation boundary. The double-write corrupts physics integration; stochastic
  noise masks it during training, deterministic eval exposes it.
- **Next action:** PD17 — remove `write_data_to_sim()` (one-line fix), resume from PD16 checkpoint,
  4096 envs, 30k iterations.

## Phase 2A Run PD17 — 2026-03-25

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-25_00-35-49_mappo_torch`
- **Config:** Direct moments (no PD), removed `write_data_to_sim()` from `_apply_action()`.
  Resumed from PD16 checkpoint. 4096 envs, 30k iterations.
- **Convergence:** peak reward **497.47** @ step **7,000** | final **401.56** @ step **30,000**
- **Training:** Perfect — `ground_hit_rate=0.0008`, `mean_dist_to_goal=0.005m`, `thrust=0.500`
- **Scorecard:** `mean_roll=97.4°` | `ground_hit_rate=0.799` | `airborne_ratio=0.447`
- **Verdict: FAIL** — train-eval gap persists. `write_data_to_sim()` was NOT the root cause.
- **True root cause identified:** `entropy_loss_scale=0.008` in `skrl_mappo_cfg.yaml`.
  Isaac Lab's reference quadcopter uses `entropy_loss_scale=0.0`. The entropy bonus
  incentivizes the optimizer to keep policy noise high (std went UP 0.33→0.44 during PD17).
  The policy mean becomes biased — optimized for `E[reward(mean+noise)]` not `reward(mean)`.
  During deterministic eval, the mean alone produces invalid control commands.
- **Next action:** PD18 — match all Isaac Lab noise settings (`entropy_loss_scale=0.0`,
  `initial_log_std=0.0`, `max_log_std=2.0`), train from scratch (no checkpoint resume).

## Phase 2A Run PD18 — 2026-03-25

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-25_01-32-20_mappo_torch`
- **Config:** Direct moments, no PD, `entropy_loss_scale=0.0`, `initial_log_std=0.0`,
  `max_log_std=2.0` (all matched to Isaac Lab reference). GNN policy, 4096 envs, 30k iterations.
- **Convergence:** peak reward **508.24** @ step **12,000** | final **417.83** @ step **30,000**
- **Policy std:** **0.93 → 0.09** (correctly DECREASING — first run to show this)
- **Scorecard** (`best_agent.pt` @ 12k, 5 episodes):
  - `mean_roll=52.2°` | `ground_hit_rate=0.323` | `airborne_ratio=0.762`
  - **Verdict: FAIL** — gap reduced (was 97° in PD17) but not closed
- **Analysis:** Side-by-side config comparison found 8 more hyperparameter differences
  with Isaac Lab reference (learning_rate, rollouts, layers, rewards_shaper_scale, etc.).
  Also discovered GNN adjacency matrix never reaches policy (Phase 2B blocker, not gap cause).
- **Next action:** PD19 — match ALL Isaac Lab hyperparams + use MLP (--no_gnn) to isolate variables.

## Phase 2A Run PD19 — 2026-03-25

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-25_02-40-25_mappo_torch`
- **Config:** MLP (no GNN), matched Isaac Lab SKRL hyperparams (lr=5e-4, rollouts=24,
  rewards_shaper_scale=0.01, etc.). But env config NOT matched (different rewards, spawn, etc.).
- **Convergence:** peak **376.15** @ step **21,000** | final **294.32** @ step **30,000**
- **Scorecard:** `mean_roll=85.1°` | `ground_hit_rate=0.794` | `airborne_ratio=0.459`
- **Verdict: FAIL** — worse than PD18 (GNN, 52°). Matching SKRL hyperparams alone didn't help
  because env reward/spawn params were still different from Isaac Lab. Also, rewards_shaper_scale=0.01
  with our reward magnitudes (tuned for scale=1.0) made effective gradients 100x weaker.
- **Lesson:** Must match env config AND training config simultaneously. Changed too many
  variables at once (MLP + 8 hyperparams) — couldn't isolate cause.
- **Next action:** PD20 — match ALL 12 env parameters + keep Isaac Lab SKRL config + MLP.
  Only remaining difference: MAPPO vs PPO.

## Phase 2A Run PD20 — 2026-03-25

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-25_03-23-42_mappo_torch`
- **Config:** MLP, ALL Isaac Lab params matched (env + SKRL). Only difference: MAPPO vs PPO.
- **Training:** Converged by 10k, then collapsed after 20k (ground_hit_rate rose to 27%).
- **Scorecard:** `mean_roll=80.2°` | `ground_hit_rate=0.836` | `airborne_ratio=0.432`
- **Verdict: FAIL** — gap persists even with everything matched. Led to final investigation.

## ROOT CAUSE FOUND — Train-Eval Gap (2026-03-25)

**Root cause:** `load_policy_from_checkpoint()` in `checkpoint.py` only loaded neural network
weights — it did NOT restore the `RunningStandardScaler` preprocessor statistics
(`running_mean`, `running_variance`, `current_count`). During eval, the preprocessor used
fresh `mean=0, variance=1` instead of training-time statistics. This caused observations to
be wrongly scaled — e.g. angular velocity dimensions had training variance ~16.4 but were
normalized by variance=1 during eval, making the policy see values **16x too large**.

**Fix:** Replace `load_policy_from_checkpoint(agent, resume_path)` with
`agent.load(str(resume_path))` in `eval_runner.py` and `play.py`. SKRL's built-in `load()`
restores everything: policy weights, value network, preprocessor statistics, optimizer state.
This is what Isaac Lab's own `play.py` uses (line 208).

**Impact:** This bug affected every eval/assess run from PD1 through PD20. All the config
changes we tried (PD gains, entropy, noise settings, GNN vs MLP, reward tuning, Isaac Lab
matching) were irrelevant — the policy was receiving garbage inputs during eval regardless.

## Phase 2A COMPLETE — PD16 Re-Eval (2026-03-25)

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-24_22-37-14_mappo_torch` (PD16 checkpoint)
- **Eval fix applied:** `agent.load()` restoring preprocessor statistics
- **Scorecard** (`best_agent.pt`, 5 episodes, seed 42):
  - `survival_steps` = **240.8** (WARN, gate 500)
  - `airborne_ratio` = **0.9999** (PASS)
  - `ground_hit_rate` = **0.0001** (PASS)
  - `mean_roll_deg` = **0.08°** (PASS)
  - `orientation_violation_rate` = **0.0001** (PASS)
  - `mean_formation_error_m` = **0.47 m** (PASS)
  - **Overall: WARN** (5/6 PASS, 1 WARN)
- **Trajectory analysis:**
  - Altitude: rock-solid hold for 450+ steps
  - Attitude: flat at 0° roll/pitch (±1° noise)
  - XY: slow lateral drift ~0.4m over 500 steps due to uncontrolled yaw spin
  - Yaw control is the main remaining weakness — priority for Phase 2B
- **Decision: Phase 2A hover-stability COMPLETE.** Advance to Phase 2B (formation hover).

## Phase 2A Run PD7 — 2026-03-23

- **Run dir:** `logs/skrl/ggswarm_marl/2026-03-23_03-38-53_mappo_torch`
- **Config class:** `GGSwarmMarlHoverStabilityCfg` — `hover_in_place=True` (local source at assess); commit **`0a87fb4`** (verify VM revision matched for GCE train)
- **Train budget:** 92,000 iterations (GCE); stable-hover + PD6 reward scales unchanged
- **Convergence:** entropy collapse not detected | peak reward **0.50** @ step **91,000** | final **0.46** @ step **92,000** | recommended budget **~106k** steps
- **Scorecard** (`post_train_assess.py`, seed **42**, `best_agent.pt`, 5 episodes):
  - survival_steps = **7.0**
  - airborne_ratio = **0.516**
  - ground_hit_rate = **0.712**
  - mean_roll_deg = **18.4°** | mean_pitch_deg = **22.7°**
  - orientation_violation_rate = **0.191**
  - verdict = **FAIL**
- **TensorBoard (event scalars):** `Reward / Total reward (mean)` first/last **-42.2 → 0.46** @ 92k; `Info / rew_pos` **0.073 → 0.208** (first point is batch mean @ 1k logging step — same order as PD6, not a per-episode “all at spawn” probe); `Policy / Standard deviation (drone_0)` **0.61 → 0.22**
- **Checkpoint ladder** (`analyze_checkpoints.py`, 10k interval, 2 eps): printed roll/pitch **~22° / ~23°** @ 10k–30k; **worst airborne ratio 0.518** @ **90k**; CSV `survival_steps` **14.5** @ 10k then **1.0** for 20k–90k (cheap eval — compare to 5-ep assess)
- **vs Run PD6:** Attitude metrics slightly improved; **ground contact / airborne proxies regressed strongly** — **`hover_in_place` alone did not clear Phase 2A gates**
- **Decision: FAIL** — do not advance to Phase 2B
- **Next action:** (1) Confirm GCE trained repo revision with **`hover_in_place`** in `drone_swarm_env.py` / cfg. (2) TensorBoard + `logs/skrl/ggswarm_marl/2026-03-23_03-38-53_mappo_torch/checkpoint_progression.csv` for 70k–90k degradation. (3) **One** follow-up knob (clearance, vel damping, `spawn_dist`, or exploration) per ops docs — log in changelog before retrain.

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

## Phase 2B: Formation Training

- [2026-03-25] **Phase 2B infrastructure:**
  - Fixed GNN adj\_matrix pipeline: `patch_mappo_gnn_batched_act` batches all agents'
    obs into `[num_envs * num_agents, obs_dim]`, calls GNN once with full graph,
    splits per-agent. Syncs policy weights after `_update()`.
  - Added `--log_subdir` for phase-based log organisation (`phase2b/` subfolder).
  - Removed deprecated `gce_train_launch.ps1`; updated CLAUDE.md GCE rules.

- [2026-03-25] **Run p2b-1** (30k iters, 4096 envs, GNN, checkpoint PD16):
  - Used `compute_marl_rewards` (Gaussian, no dt-scaling) with Phase 2A reward scales.
  - **FAIL:** `survival_steps=4.2`, `mean_roll=27.3°`, drones tumbling.
  - Root cause: 185x reward magnitude mismatch between `compute_marl_rewards` and
    `compute_stable_hover_rewards`. Phase 2A scales are invalid for the Gaussian function.
  - Decision: FAIL — switch to hybrid reward approach.

- [2026-03-25] **Run p2b-2** (30k iters, 4096 envs, GNN, checkpoint PD16):
  - Hybrid rewards: `compute_stable_hover_rewards` base (Phase 2A scales) +
    `compute_formation_rewards` on top via curriculum alpha.
  - **Scorecard** (`best_agent.pt`, 5 episodes, seed 42):
    - `survival_steps=4.4` | `airborne_ratio=0.911` | `ground_hit_rate=0.002`
    - `mean_roll_deg=3.1°` | `orientation_violation_rate=0.004`
    - `mean_formation_error_m=0.471`
  - **Verdict: FAIL** — `survival_steps` gate not met (4.4 vs >500).
    But 5/6 gates pass. Stability massively improved vs p2b-1 (roll 27°→3.1°,
    ground\_hit 7.3%→0.2%). Late-episode crash at ~step 450 causes low survival.
  - **Next action:** investigate survival\_steps metric — drones hover stably for 450+
    steps then crash at episode end. May be truncation/boundary issue.

## Fresh Start: CTDE Rebuild (Week 12)

- [2026-03-26] **FRESH START.** Archived Phase 2 codebase to `archive/phase2-v1` branch.
  Rebuilt from Isaac Lab quadcopter reference. Key architectural change:
  `DirectMARLEnv` + MAPPO (per-agent weights) replaced by `DirectRLEnv` + PPO
  (single shared policy). All drones are homogeneous — separate weights were wasteful.
- [2026-03-26] New codebase: `GgswarmEnv(DirectRLEnv)`, 1 Crazyflie per env, shared PPO.
  Task ID: `ggswarm-v0`. CTDE: centralized training, decentralized execution.
- [2026-03-26] Added viz package: `ggswarm.viz.trajectory_plots` (2x2 summary),
  `ggswarm.viz.nvenc_recorder` (NVENC H.264 video).
- [2026-03-26] Added `--trajectories`, `--play_length`, `--video_prefix`, `--log_subdir`
  to play.py and train.py.

### Training runs

- [2026-03-26] **p2a-1** (sigma=0.8, vel=-0.05, quadcopter baseline):
  ep_len=497, std=0.077, reward=107. Converged by step 6k, slight overfit after.
- [2026-03-26] **p2a-2** (sigma=0.3, vel=-0.15, too tight):
  ep_len=78, reward=-0.09. Failed — reward landscape too steep from scratch.
- [2026-03-26] **p2a-3** (sigma=0.5, vel=-0.10, middle ground):
  ep_len=486, std=0.141, reward=97.5. Good convergence, healthier std.
- [2026-03-26] **p2a-4** (sigma=0.8, vel=-0.05, reset to quadcopter exact):
  Pending eval — testing whether drift is params or code.
