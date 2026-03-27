---
trigger: always_on
---

# Project Rules: ggSwarm

## 1. Architecture (MANDATORY)

- **Framework:** Isaac Lab DirectRLEnv + SKRL PPO (single-agent, shared policy)
- **CTDE:** Centralized Training, Decentralized Execution
- **Env:** `GgswarmEnv(DirectRLEnv)` — 1 Crazyflie per env, PPO trains shared policy
- **Config:** `GgswarmEnvCfg(DirectRLEnvCfg)` — all tunable params are cfg fields
- **Task ID:** `ggswarm-v0`
- **Viz:** `source/ggswarm/ggswarm/viz/` — trajectory plots, NVENC video recorder
- Source of truth: `docs/phases/` for phase documentation

## 2. Status Reporting (MANDATORY)

- **RULE:** Major phase transitions and technical milestones must be logged in `docs/status/changelog.md`.
- **RULE:** Weekly status in `docs/status/weekly_updates.md` (Tuesdays).

## 3. Python Coding Standards

- PEP 8, snake_case for filenames, type hints on public functions.
- Auto-resolve linting issues (line length 88-100, unused imports).
- **RULE:** No `print()` in per-step env code — use `logger.*`.
- **RULE:** `print()` allowed in `scripts/` for CLI output.

## 4. Documentation Standards

- **RULE:** Use professional, clear language in all `.md` files.
- **RULE:** Auto-resolve markdownlint warnings.
- GFM tables: header → separator (`| :--- |`) → body.

## 5. Technical Terminology (Spelling Exceptions)

- **Frameworks/Tools:** Isaac Lab, `isaaclab`, `isaacsim`, `conda`, `PyPI`.
- **Project Specific:** `ggSwarm`, `ggswarm`, `Crazyflie`, CTDE, PPO.
- **Technical Shorthand:** `cfg`, `envs`, `quat`, `lin_vel`, `ang_vel`,
  `pos_w`, `rel_pos`, `multirotor`.
- **User Context:** `gkuep`.

## 6. Tensor Shape Contracts (MANDATORY)

- **RULE:** Every function that manipulates drone tensors MUST include
  a shape comment on first access: `# shape: [num_envs, dim]`
- **REQUIRED FOR:** pos_w, quat_w, lin_vel_b, ang_vel_b, actions, rewards.
- **RATIONALE:** Silent shape mismatches are the #1 source of bugs.

## 7. Per-Step Allocation Ban (MANDATORY)

- **RULE:** No `torch.zeros`, `torch.ones`, `torch.empty`, `torch.full`,
  or `tensor.clone()` inside hot-path methods: `_pre_physics_step`,
  `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`.
- **RULE:** All scratch buffers pre-allocated in `__init__` and reused.

## 8. Reward Function Hygiene (MANDATORY)

- **RULE:** Every reward component MUST have a corresponding scale
  parameter in `GgswarmEnvCfg` (e.g. `lin_vel_reward_scale`).
- **RULE:** Reward components that are planned but not yet active MUST
  be set to `0.0` in the config, NOT commented out.
- **RULE:** Any change to the reward function MUST be logged in
  `docs/status/changelog.md` with the rationale.
- **RULE:** No magic numbers — all thresholds, sigmas, radii must be cfg fields.

## 9. GCE Training Operations

- Instance: `isaacsim`, zone: `us-central1-a`, project: `gg-swarm`
- GCE is for training only — play/video/TB must run locally.
- Always run a local smoke test (5 iterations) before GCE launch.
- After GCE training, push logs to `gs://gg-swarm-training-logs`.
- **Log naming:** Serialized run labels — `train_ggswarm_p2a-1.log`, `p2a-2.log`, etc. Must match `--log_subdir` phase.

## 10. Video Recording

- Always pass `--video_prefix <run_label>` for traceability.
- Videos use NVENC H.264 via `ggswarm.viz.nvenc_recorder`.
- Default resolution: 1920×1080.

## 11. Hard Deadline

- **Apr 24, 2026** — Capstone Festival. Immovable.
- Gary makes all scope-cut calls. Never propose extending deadlines.

## 12. Post-Training Review (MANDATORY)

- After every training run: check TB, run play with `--trajectories`.
- Log results in `docs/status/changelog.md` before any config change.
