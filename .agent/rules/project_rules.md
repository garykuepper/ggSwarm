---
trigger: always_on
---

# Project Rules: ggSwarm

## 1. Architecture Maintenance (MANDATORY)

The file `docs/architecture.md` is the source of truth for the system design.

- **RULE:** Any change to the environment structure, message passing logic, or
  coordination layers MUST be accompanied by an update to
  `docs/architecture.md`.
- **RATIONALE:** Ensures scalability and maintainability of the decentralized control logic.

## 2. Status Reporting (MANDATORY)

- **RULE:** A weekly status summary must be added to `docs/status/weekly_updates.md` every Monday.
- **RULE:** Major phase transitions and technical milestones must be logged in `docs/status/changelog.md`.
- **RATIONALE:** Provides transparency for project stakeholders and website updates.

## 3. Python Coding Standards

- Follow the global Python standards defined in the user settings (PEP 8, Snake_case for filenames, Type Hinting).
- Explicitly documented public interfaces in all module-level components.
- **RULE:** Always auto-resolve common linting issues like "line too long" (wrap
  at 88-100 chars) and "unused imports" (remove them) as they occur.

## 4. Documentation Standards

- **RULE:** Use professional, clear language in all `.md` files.
- **RULE:** Always auto-resolve any problems indicated by `markdownlint` to maintain formatting consistency.
- **RATIONALE:** Ensures high-quality, professional-grade documentation for the project.

## 5. Technical Terminology (Spelling Exceptions)

- **RULE:** The following terms are correct technical jargon for this project and
  should NOT be flagged as spelling errors:
  - **Frameworks/Tools:** Isaac Lab, `isaaclab`, `isaacsim`, `conda`, `PyPI`, `Py`.
  - **Project Specific:** `ggSwarm`, `ggswarm`, `Crazyflie`.
  - **Technical Shorthand:** `cfg`, `envs`, `quat`, `lin_vel`, `ang_vel`,
    `pos_w`, `rel_pos`, `multirotor`, `multirotors`.
  - **User Context:** `gkuep`.
- **RATIONALE:** Prevents false positives in spelling checks for common robotics and simulation terminology.

## 6. Tensor Shape Contracts (MANDATORY)
- **RULE:** Every function that manipulates swarm tensors MUST include 
  a shape comment on the first access of each major tensor.
- **FORMAT:** `# shape: [num_envs, num_agents, dim]`
- **REQUIRED FOR:** pos_w, quat_w, lin_vel_b, ang_vel_b, adj_matrix, 
  actions, rewards, and any intermediate computation tensors.
- **RATIONALE:** Silent shape mismatches (e.g. [num_envs*num_agents, dim] 
  vs [num_envs, num_agents, dim]) are the #1 source of bugs in this codebase.

## 7. Adjacency Matrix Integrity (MANDATORY)
- **RULE:** The adjacency matrix MUST always have zeros on the diagonal 
  (no self-connections) before being stored in `self.extras["adj_matrix"]`.
- **RULE:** Any function consuming `extras["adj_matrix"]` must assert or 
  document the expected shape: [num_envs, num_agents, num_agents].
- **RULE:** Do NOT modify the adjacency threshold (2.0m) without updating 
  `docs/architecture.md` and the Phase 2 GATv2 edge construction logic.
- **RATIONALE:** The adjacency matrix is the contract between L1 sensing 
  and L2 GATv2. Silent changes corrupt the message-passing graph.

## 8. Logging Standards
- **RULE:** NEVER use bare `print()` statements in environment or training 
  code. Always use `logger.info()`, `logger.debug()`, or `logger.warning()`.
- **RULE:** `sys.stdout.flush()` calls are forbidden outside of 
  standalone demo scripts.
- **EXCEPTION:** `scripts/run.py` and scripts under `scripts/reference/` may
  use print for human-readable CLI output (standalone examples).
- **RATIONALE:** Bare prints spam the console during 100k+ timestep 
  training runs and cannot be filtered by log level.

## 9. Reward Function Versioning (MANDATORY)
- **RULE:** Every reward component MUST have a corresponding scale 
  parameter in `GgswarmMarlEnvCfg` (e.g. `rew_scale_formation`).
- **RULE:** Reward components that are planned but not yet active MUST 
  be set to `0.0` in the config, NOT commented out.
- **RULE:** Any change to the reward function MUST be logged in 
  `docs/status/changelog.md` with the rationale.
- **RATIONALE:** Reward shaping is the primary control lever across 
  phases. Silent reward changes make training curves uninterpretable.

## 10. Training Config Checklist (MANDATORY)
- **RULE:** Before any training run exceeding 10,000 timesteps, verify:
  1. `experiment.directory` is set to `"ggswarm_marl"` (not a template name)
  2. `trainer.timesteps` reflects the intended run length
  3. `entropy_loss_scale` is non-zero during early-phase training
  4. `network.layers` matches the current phase target capacity
- **RULE:** The agent MUST NOT modify `skrl_mappo_cfg.yaml` autonomously 
  without flagging the change for human review.
- **RATIONALE:** Misconfigured training runs waste GPU hours and produce 
  uninterpretable results.

## 11. Phase Boundary Protection (MANDATORY)
- **RULE:** Code for a future phase MUST NOT be added to current-phase 
  files. Use `# TODO (Phase N):` comments as placeholders only.
- **RULE:** The agent MUST NOT implement SwarmRaft, CBF, or MINCO logic 
  until the corresponding phase is explicitly started.
- **RULE:** `self.extras["adj_matrix"]` is the ONLY data channel between 
  the environment and the GATv2 policy. Do not add extra keys to 
  `self.extras` without updating `docs/architecture.md`.
- **RATIONALE:** Premature implementation of future phases introduces 
  untested complexity and breaks the incremental validation strategy.

## 12. No Magic Numbers in Environment Code
- **RULE:** All numerical constants in `drone_swarm_env.py` MUST be 
  defined as parameters in `GgswarmMarlEnvCfg`, not hardcoded inline.
- **KNOWN VIOLATIONS TO FIX:** 
  - Adjacency threshold `2.0` → `cfg.graph_connectivity_radius`
  - Yaw range `±3.14` → `cfg.spawn_yaw_range`
  - Reward sigma `0.5` in `exp(-dist/0.5)` → `cfg.rew_pos_sigma`
- **RATIONALE:** Hardcoded constants make hyperparameter tuning 
  impossible without modifying environment logic.