# CLAUDE.md — ggSwarm Project Rules

This file is automatically loaded by Claude Code at session start.
All rules below are MANDATORY unless marked SHOULD.

---

## Shell / Environment

- **Shell is PowerShell on Windows.** Use Unix shell syntax in Bash tool calls (forward slashes, `/dev/null`), but observe the PowerShell rules below for any commands the user runs locally.
- No `&&` chaining in PowerShell — use `;` or separate tool calls.
- No `head` in PowerShell — use `Select-Object -First N`.
- No `rg` as a shell command — use the Grep tool.
- `gcloud storage rsync` corrupts paths on Windows — use `gcloud storage cp`.
- Escape `$` as `\$` in `gcloud compute ssh --command="..."`, or wrap in single quotes.

---

## GCE Training Operations

- Instance: `isaacsim`, zone: `us-central1-a`, project: `gg-swarm`
- Repo path on VM: `~/ggSwarm`
- GCS bucket: `gs://gg-swarm-training-logs` (only valid bucket)
- **Launch training via `gcloud compute ssh`** using `scripts/run.py` syntax. `train_and_push.sh` wraps `run.py` and auto-pushes logs to GCS on completion.
- Pattern:

  ```text
  gcloud compute ssh isaacsim --zone=us-central1-a --project=gg-swarm \
    --command='cd ~/ggSwarm && git pull origin main && \
    GGSWARM_GCS_URI=gs://gg-swarm-training-logs \
    nohup bash scripts/cloud/train_and_push.sh \
    phase2b train --headless --max_iterations 30000 \
    --checkpoint <path/to/best_agent.pt> \
    > ~/train_ggswarm_<label>.log 2>&1 &'
  ```

- The args after `train_and_push.sh` are passed directly to `python scripts/run.py`.

- Always inline `GGSWARM_GCS_URI=gs://gg-swarm-training-logs` before `nohup` — `source ~/.bashrc` does NOT propagate to backgrounded processes.
- **GCE is for training only.** eval / play / assess / tensorboard / git must run locally.
- Do NOT suggest running eval, play, assess, or analysis scripts via `gcloud compute ssh --command`.

## Video Recording (MANDATORY)

- Always pass `--video_prefix <run_label>` when recording video (e.g. `--video_prefix p2b-3`).
- The run label must match the label used in `docs/status/run_history.md` for traceability.
- Example: `python scripts/run.py phase2b play --video --video_prefix p2b-3 --checkpoint <path>`
- Output filename: `p2b-3__best_agent-episode-0.mp4` (1080p, H.264 NVENC).

---

## Architecture + Interfaces (MANDATORY)

- Source of truth: `docs/design/architecture.md`.
- Update when changing: environment structure, observation/action contracts, `extras["adj_matrix"]`, or coordination layer boundaries.

## Phase Boundary Protection (MANDATORY)

- Do not implement features tagged for a future phase until that phase starts.
- Future work: `# TODO (Phase N): ...` placeholders only.

## Tensor Shape Contracts (MANDATORY)

- Any function manipulating swarm tensors must add a shape comment on first access: `# shape: [num_envs, num_agents, dim]`
- Required for: `pos_w`, `quat_w`, `lin_vel_b`, `ang_vel_b`, `actions`, `rewards`, `adj_matrix`, and intermediate tensors.

## Graph / Adjacency Contract (MANDATORY)

- `extras["adj_matrix"]`: shape `[num_envs, num_agents, num_agents]`, diagonal zeros, radius controlled by config (no hardcoded threshold).
- Any consumer must assert or document the expected shape.

## Reward + Hyperparameter Hygiene (MANDATORY)

- Every reward term must have a scale in `GGSwarmMarlEnvCfg` and (if applicable) a sigma/threshold parameter.
- Planned-but-disabled reward terms must be `0.0` in cfg (not commented out).
- Any reward change must be logged in `docs/status/changelog.md` with rationale.
- Also check Rule 6 (no magic numbers) when touching reward code.

## No Tunable Magic Numbers in Env Core (MANDATORY)

- In `drone_swarm_env.py`, any tunable constant (thresholds, sigmas, radii, limits) must be a `GGSwarmMarlEnvCfg` parameter.

## Training Safety Checklist (MANDATORY for runs > 10k steps)

Before long runs, verify:
- Logging directory is correct (`ggswarm_marl`)
- Timesteps match intended run length
- `entropy_loss_scale` non-zero early-phase (unless deliberately disabled)
- Network capacity matches phase goal
- If `skrl_mappo_cfg.yaml` changed, call out diff + rationale in commit message

## Logging Policy (MANDATORY)

- No `print()` inside per-step environment code — use `logger.*`.
- One-time init → `logger.info`; per-episode diagnostics → `logger.debug`; unexpected recoverable → `logger.warning`.
- `print()` allowed in `scripts/` and demos for CLI output.

## Documentation + Status (SHOULD)

| Document | Update trigger |
| :--- | :--- |
| `docs/status/changelog.md` | Every reward change, phase transition, or major bug fix |
| `docs/status/weekly_updates.md` | Weekly snapshot (preferred Mondays) |
| `docs/design/architecture.md` | Any change to env I/O contracts or layer boundaries |

## Markdown Hygiene (MANDATORY)

- GFM tables: header → separator (`| :--- |`) → body. Every row starts with exactly one `|`.
- Every fenced code block must be surrounded by blank lines on both sides.
- Ordered lists interrupted by a code fence must restart numbering at `1.` after the fence.

## Spelling / Terminology (SHOULD)

- Maintain spelling exceptions in `cspell.json`.

## Hard Deadline: Apr 24, 2026 (MANDATORY)

Capstone Festival presentation and all submissions due **Apr 24, 2026**. No phase may be extended past this date.

### Remaining schedule (read-only — do not revise these dates)

| Phase | Window | Gate |
| :--- | :--- | :--- |
| 3. Muscle Refinement | Mar 25 – Apr 7 | M2 complete by Apr 7 |
| 4. Stress Testing | Apr 8 – Apr 14 | M3 complete by Apr 14 |
| 5. Showcase Prep | Apr 14 – Apr 21 | M4 complete by Apr 21 |
| 6. Delivery | Apr 22 – Apr 24 | **Submission due Apr 24** |

### Scope-cut rules (MANDATORY when a phase risks overrun)

- **Phase 3 overrun →** Drop MINCO polynomial upgrade; EMA smoother (`alpha=0.3`) is the shipped implementation. SwarmRaft nearest-alive-slot fallback replaces full Raft state machine if needed.
- **Phase 4 overrun →** Reduce 100-episode suite to 25 episodes (5 per scenario). Drop urban canyon scenario; cluttered forest only. Skip scale benchmark beyond 20 agents.
- **Phase 5 overrun →** Drop shape-transition recording sequence. Ship single 60-second demo clip. Compile Testing Report with available data; mark unrun scenarios as "planned."
- **Phase 6 overrun →** Impossible — Apr 24 is immovable.

### Decision authority

Gary makes all scope-cut calls. The assistant **must not propose extending any deadline** — only propose what to cut. If a suggestion would push work past Apr 24, flag it explicitly and offer the scope-cut alternative.

### Timeline references

- Authoritative source: `docs/project/proposal.md` § 7 (Week 9 baseline).
- Current status: `docs/status/weekly_updates.md`.

## Phase Gate Checklist (MANDATORY before advancing)

Before declaring any phase complete:

1. All phase objectives (P{N}.x) in `docs/design/phase{N}_*.md` have a recorded pass/fail result in `docs/status/changelog.md`.
2. Any failed objective has either a scope-cut decision logged or a targeted fix scoped within remaining calendar time.
3. `docs/status/weekly_updates.md` current-week snapshot reflects the new phase start date.
4. No open `# TODO (Phase N)` placeholders remain in files that belong to completed phases.

## GNN Policy Config Hygiene (MANDATORY)

- `hidden_channels`, `num_heads`, and `initial_log_std` in `skrl_gnn_policy.py` must be constructor parameters with defaults, not hardcoded literals.
- When `--gnn` mode is active, a `logger.info` line must state the policy class name and active `hidden_channels` value before training starts.
- `skrl_mappo_cfg.yaml` must contain a comment on every block silently ignored in GNN mode (e.g. `models.policy.network`).

## Per-Step Allocation Ban (MANDATORY)

- No calls to `torch.zeros`, `torch.ones`, `torch.empty`, `torch.full`, or any other tensor constructor inside: `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`.
- Also banned per-step in hot-path functions: `contract_logic.py` (`compute_adjacency_matrix`, `compute_marl_rewards`), `cbf_safety.py` (`apply_cbf_safety`).
- `tensor.clone()` is also banned inside any of the above — use in-place ops or pre-allocated destination buffers.
- All scratch buffers must be pre-allocated in `__init__` and reused. Declarations must carry: `# pre-allocated; reused every step`.

## Eval Script Parity (MANDATORY)

- Every phase's `eval` and `play` subcommands in `scripts/run.py` must use the correct task ID and eval script for that phase.
- Silent cross-phase delegation only permitted with a `# intentional: <reason>` comment at the call site AND a rationale note in `docs/ops/commands.md`.
- `--gnn` / `--no_gnn` must be explicit CLI arguments on all `eval` and `play` subcommands — never hardcoded.
- Adding a new phase requires its own `_cmd_eval_phaseN` function in `run.py`.

## RNG Seeding Policy (MANDATORY)

- Global RNG state must only be set at top-level entry points (`scripts/skrl/train.py`, `scripts/run.py`), never inside env class methods.
- Env methods requiring randomness must use a local `torch.Generator` or `np.random.default_rng(seed)` stored on `self`, initialized in `__init__`.
- Seed value must be a `cfg` parameter, not a hardcoded literal.

## Type Annotation Completeness (SHOULD)

- All new public functions and dataclass fields must carry full type annotations.
- All new public classes and functions in pure-torch modules must have at minimum a one-line docstring.
- When editing an existing file, type any bare-typed fields you touch.

## Post-Training Assessment Gate (MANDATORY)

After **every** training run, before any reward change or relaunch:

1. Run assess locally (NOT on GCE):

   ```bash
   python scripts/run.py phase2 assess --run_dir logs/skrl/ggswarm_marl/<run>
   # or:
   python scripts/run.py hover-stability assess --run_dir logs/skrl/ggswarm_marl/<run>
   # with trajectory diagnostic plots:
   python scripts/run.py hover-stability assess --run_dir logs/skrl/ggswarm_marl/<run> --trajectories
   ```

1. Log scorecard to `docs/status/changelog.md`:

   ```markdown
   - [YYYY-MM-DD] Run N eval (checkpoint: <run>/best_agent.pt, N episodes):
     - survival_steps=X | airborne_ratio=X | ground_hit_rate=X
     - mean_roll=X° | orientation_violation_rate=X
     - mean_formation_error=Xm
     - Decision: FAIL/WARN/PASS — <one line rationale>
     - Next action: <specific config change or "advance to Phase 3">
   ```

The assistant **MUST NOT** suggest reward changes or a new training launch until the `assess` output has been shown (or explicitly waived) and a `Decision:` line has been added to `changelog.md`.

## Curriculum Resume Contract (MANDATORY)

When resuming from a checkpoint, `self.common_step_counter` resets to 0 — curriculum always restarts from the beginning.

Before launching a Phase B (formation) run resuming a Phase A (hover-stability) checkpoint:

1. Set `curriculum_start_step: 0` (max `5000` for a short ramp).
2. Document in `docs/status/changelog.md`: source checkpoint, target Phase B config, rationale.
3. Never assume curriculum continues from prior phase's final step.

## Config Snapshot Before Training (MANDATORY for runs > 10k steps)

Run a 1-iteration smoke test before any long training run:

```bash
python scripts/run.py debug smoke --task <TASK_ID> --iterations 1 --gnn
```

Confirm these 5 fields match the intended config:

**Phase 2B / formation (`GGSwarmMarlFormationCfg`):**

| Field | Expected value |
| :--- | :--- |
| `rew_scale_upright` | per planned config |
| `rew_scale_ang_vel` | per planned config |
| `rew_scale_terminated` | per planned config |
| `curriculum_start_step` | 0 for Phase B resumes |
| `spawn_yaw_range` | per planned config |

**Phase 2A hover-stability (`GGSwarmMarlHoverStabilityCfg`):**

| Field | Expected value |
| :--- | :--- |
| `rew_scale_pos` | per planned config |
| `rew_scale_vel` | per planned config |
| `rew_scale_terminated` | typically 0.0 |
| `curriculum_start_step` | 999999 (hover-only lock) |
| `spawn_yaw_range` | per planned config |

The assistant **must not** initiate a training launch via `train_and_push.sh` until these 5 values have been confirmed in the current session (or explicitly waived by the user).

## Post-Train Record (MANDATORY)

After every training run, append a row to `docs/status/run_history.md` from the `assess` scorecard before making any config change or relaunching.

Required fields: Run label, Timestamp, Phase, `survival_steps`, `airborne_ratio`, `ground_hit_rate`, `mean_roll_deg`, `orientation_violation_rate`, Verdict (`PASS` / `WARN` / `FAIL` / `aborted`).

Also log in `docs/status/changelog.md` using the template in `docs/ops/post_train_analysis.md`.

Full assessment workflow: `docs/ops/post_train_analysis.md`.

## Cyclomatic Complexity Cap (SHOULD)

- New functions must stay at or below cyclomatic complexity **15**.
- Grandfathered exceptions (must not grow further without refactor):
  - `_build_parser` in `scripts/run.py`
  - `_get_dones` in `drone_swarm_env.py`
  - Both carry `# noqa: C901  # grandfathered — do not add branches`
