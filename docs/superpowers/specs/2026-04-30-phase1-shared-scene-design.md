# Phase 1: Shared-Scene Multi-Drone Training (sim only) — Design

**Date:** 2026-04-30
**Status:** Approved (awaiting user spec review)
**Phase reference:** [`docs/ggswarm_live/phases/phase1_sim.md`](../../ggswarm_live/phases/phase1_sim.md)
**Capstone anchor:** `v1.0.0-capstone` on `capstone` branch
**Compute envelope:** Local 3070 only (no GCE — credits exhausted)

---

## 1. Summary

Phase 1 introduces real inter-drone aerodynamics into the training distribution. The capstone trained 1 drone per physics scene with no aero coupling; Phase 1 puts all 8 drones in one shared physics scene per env, adds downwash modeling, and upgrades the GATv2 policy to consume relative-velocity edge features.

Phase 1 is delivered as **three sequenced sub-phases (1a → 1b → 1c)**, each with its own gate and TensorBoard scalar baseline. The final milestone is a single 2-panel comparison video (Phase 0 vs 1c final) plus per-sub-phase scalar deltas in the changelog.

| Sub-phase | Scope | Anchor | Gate |
| :--- | :--- | :--- | :--- |
| 1a | Shared-scene refactor + B3 stacked-spawn fix + group-plumbing cleanup | `v1.0.0-capstone` | Replay-only: capstone checkpoint rollouts in shared-scene env (downwash off) match capstone-env rollouts statistically (within 2σ on four metrics) |
| 1b | Analytic downwash (Panerati) + learned residual (Neural-Swarm2) trained on a published Crazyflie aero dataset | 1a checkpoint | Three retrains (`disabled` / `analytic` / `residual`) all converge; deltas documented |
| 1c | GATv2 `edge_dim=6` (rel_pos + rel_vel as edge features), obs vector unchanged | 1b winning anchor | 1c retrain converges; TB scalar deltas vs 1b documented |

## 2. Decisions locked in (brainstorm record)

| # | Decision | Rationale |
| :--- | :--- | :--- |
| Q1 | Sub-phased: 1a → 1b → 1c with per-sub-phase checkpoints | Isolates aero from architectural noise; each sub-phase is independently bisectable |
| Q2 | 1a scope = refactor + B3 + cleanup | Group plumbing collapses naturally when group == env; clean it up while in the area |
| Q3 | Compute envelope = empirical via 1a throughput sweep | Shared-scene PhysX cost is unknown; pick the knee from measurement |
| Q4 | 1a gate = replay-only against `v1.0.0-capstone` | Tests env layer in isolation, no PPO noise; requires obs vector byte-identity |
| Q5 | 1b ships both analytic and learned residual as ablation | Phase doc explicitly mentions "both as an ablation" |
| Q6 | Residual data source = higher-fidelity reference (per Q7) | Bootstrap-from-analytic is circular; real data is Phase 2 |
| Q7 | Reference = published Crazyflie aero dataset (Neural-Swarm2 release if available) | No second sim to maintain; fixed traces are sufficient for residual training |
| Q8 | Edge attr = 6D (`Δpos + Δvel`); obs vector unchanged | Downwash is velocity-coupled; obs stability preserves 1b → 1c warm-start |
| Q9 | Milestone = single 2-panel video + per-sub-phase TB scalar deltas in changelog | Video for storytelling, TB scalars for engineering attribution |
| Q10 | Local-only training, anchor = `v1.0.0-capstone`, no new reward terms, no explicit aero curriculum, A2 deferred to 1c-followup | All confirmed by user |

## 3. Architecture

### 3.1 Code-organization shape

In-place refactor of `GgswarmEnv` and `GgswarmEnvCfg`. After 1a, the `num_agents == 1` branch and the `_num_groups = N // A` plumbing are gone — groups *are* envs. The capstone env remains accessible only via the `capstone` branch and the `v1.0.0-capstone` tag. No backwards-compat shim, no parallel env class, no flag whose lifetime is "1a development only".

Rationale: `CLAUDE.md` rules are unambiguous on no backwards-compat hacks, no half-finished implementations, no feature flags when the code can just change. The "incremental smoke-testability during 1a" risk that motivated alternative approaches is mitigated at the *implementation plan* layer (commit ordering: scene topology → reshape math → obs audit → replay gate), not the design layer.

### 3.2 Invariants across all of Phase 1

- Reward terms and scales (no new terms, no scale changes — phase doc says aero is the variable)
- Action space (4D: thrust + 3 moments)
- Obs vector layout (12D local + K×3 neighbor rel_pos = 18D for K=2) — locked by the 1a replay gate
- MINCO trajectory filter, CBF shield, agent-dropout/SwarmRaft logic
- PPO hyperparams (`agents/skrl_ppo_cfg.yaml`)

### 3.3 Per-sub-phase deltas

- **1a:** scene topology, reshape math, spawn pattern, dead group plumbing removed
- **1b:** pairwise downwash forces in `_pre_physics_step`, optional residual network module
- **1c:** `GATv2Conv(edge_dim=6)` and a published `edge_attr` tensor alongside `edge_index`

## 4. Components

### 4.1 1a components

| Component | Touches | Notes |
| :--- | :--- | :--- |
| `GgswarmEnv._setup_scene` (modified) | `ggswarm_env.py:_setup_scene`, `ggswarm_env_cfg.py:robot` | Spawns A=8 drone articulations per env at `/World/envs/env_.*/Drone_0..7` |
| Group-plumbing collapse | `_get_observations`, `_compute_formation_reward`, `_compute_cloud_reward`, `_get_dones`, `_reset_idx`, `_pre_physics_step` (forest deflection block) | All `[N, ...] → [G, A, ...]` reshapes become `[N_envs, A, ...]` directly. `num_agents == 1` branch removed. |
| Spawn-pattern fix (B3) | `_reset_idx` near line 988 | Replace stacked-vertical spawn with `cfg.spawn_radius` circle (compute already exists at `ggswarm_env_cfg.py:93`) |
| Replay-gate harness | `scripts/skrl/play.py` or new `scripts/skrl/replay_gate.py` | Loads `v1.0.0-capstone`, runs N rollouts in both envs, statistical comparison, single pass/fail report |

### 4.2 1b components

| Component | Touches | Notes |
| :--- | :--- | :--- |
| `ggswarm/aero/downwash_analytic.py` (new) | New module under `source/ggswarm/ggswarm/aero/` | Panerati pairwise downwash. Inputs: `[N_envs, A, 3]` pos and vel. Output: `[N_envs, A, 3]` external force. Pre-allocated scratch. |
| `ggswarm/aero/downwash_residual.py` (new) | New module + one-off `scripts/aero/train_residual.py` | Neural-Swarm2-style MLP/graph-net over rel-pose graph. Trained offline against published dataset. **Hard prerequisite: dataset availability (R1).** |
| `_pre_physics_step` aero hook | `_pre_physics_step` near `self._thrust[:, 0, 2] = ...` | Adds downwash force to wrench based on `cfg.downwash_mode ∈ {"off", "analytic", "residual"}`. "off" preserves 1a behavior exactly. |
| `GgswarmEnvCfg` aero config | `ggswarm_env_cfg.py` | `downwash_mode` plus tunables. Planned-but-disabled flags default to "off" per reward-hygiene rule. |

### 4.3 1c components

| Component | Touches | Notes |
| :--- | :--- | :--- |
| `_get_observations` edge-attr publication | `_expand_obs_with_neighbors`, `gnn_policy.py:set_knn_edges` API | Builds `edge_attr` `[num_edges, 6]` = `[Δx, Δy, Δz, Δvx, Δvy, Δvz]` per edge. Published via renamed `set_knn_edges_and_attrs(edge_index, edge_attr, batch_size)`. Pre-allocated `_knn_edge_attr` buffer. |
| `GgswarmGNNPolicy` | `gnn_policy.py` throughout | `GATv2Conv(edge_dim=6)`. `_prepare_graph` returns `(node_features, edge_index, edge_attr)`. Edge cache stores tuples. `_gnn_forward` passes `edge_attr` to each layer. |
| 1c warm-start | SKRL load logic in `train.py` | Inherits 1b weights for GNN/policy/value heads; new edge-feature MLP weights init fresh. May need partial-load shim. |

### 4.4 Cross-cutting

- **Run labels:** `train_ggswarm_p1a-N.log`, `train_ggswarm_p1b-{disabled,analytic,residual}-N.log`, `train_ggswarm_p1c-N.log` (local instead of GCE).
- **Changelog:** one entry per sub-phase landing in `docs/ggswarm_live/status/changelog.md` with TB scalar deltas and rationale.
- **Phase doc:** `docs/ggswarm_live/phases/phase1_sim.md` status updated as 1a/1b/1c land.
- **References:** Panerati 2021 and Shi 2022 entries already cited in `docs/ggswarm_live/references.md`.

## 5. Data Flow

### 5.1 Per-step flow (post-1c end state)

```
1. Policy outputs actions          [N_envs, A, 4]
                                       │
                                       ▼
2. _pre_physics_step:
   ├── action.clamp(-1, 1)
   ├── (forest deflection — preserved, reshapes adjusted)
   ├── dropout / agent_alive mask
   ├── MINCO smoothing
   ├── CBF shield
   ├── compute thrust + moment from clamped actions
   └── compute downwash force                        [aero-hook, NEW in 1b]
       ├── if downwash_mode == "off"   → zero
       ├── if "analytic"                → Panerati pairwise model
       └── if "residual"                → MLP over rel-pose graph
                                       │
                                       ▼
3. _apply_action:
   set_forces_and_torques(thrust + downwash_force, moment)
                                       │
                                       ▼
4. PhysX step  (decimation=2 → policy 50 Hz, physics 100 Hz)
                                       │
                                       ▼
5. _get_observations:
   ├── pos, vel, quat from articulation     [N_envs, A, *]
   ├── desired_pos in body frame
   ├── KNN over within-env drone positions
   ├── build edge_index                      [2, num_edges]
   ├── build edge_attr  [num_edges, 6]       [NEW in 1c]
   └── publish (edge_index, edge_attr) to GgswarmGNNPolicy
                                       │
                                       ▼
6. _get_rewards:                              (unchanged)
                                       │
                                       ▼
7. _get_dones:                                (unchanged — collision check, collective resets)
                                       │
                                       ▼
8. _reset_idx (for done envs):
   ├── sample centroid per env
   ├── spawn drones on `spawn_radius` circle + jitter   [B3 fix]
   └── nearest-slot goal assignment within env    (unchanged logic)
```

**Critical contract:** Steps 2/5/8 reshape via `[N_envs, A, *]` directly. No `_num_groups = N // A` indirection — that plumbing is gone after 1a.

**Aero force composition:** Downwash force is *additive on top of* the policy's thrust command, not replacing it. PhysX integrates the combined wrench. Policy must learn to compensate.

### 5.2 Training-loop flow (post-1c)

```
                  ┌────────── PPO collection phase ──────────┐
                  │                                          │
   Env step ───►  obs                       edges + attrs    │
                  │                            │             │
                  ▼                            ▼             │
            policy.compute()  ◄── set_knn_edges_and_attrs    │
                  │                                          │
                  │  GgswarmGNNPolicy._edge_cache ◄── append  │
                  │  (deque of (edge_index, edge_attr))      │
                  │                                          │
                  ▼                                          │
              actions → env  ──────────────────────────────► │
                                                             │
                  ┌──────── PPO update phase (mini-batches) ─┘
                  │
                  ▼
       _prepare_graph(states, B):
         ├── if B == N_drones_total       → use latest cached
         ├── if B % N_drones_total == 0   → reconstruct by replaying
         │                                  cached (edge_index, edge_attr)
         │                                  per block, offsetting indices
         └── else                          → empty edges (fallback)
                  │
                  ▼
       _gnn_forward(node_feat, edge_index, edge_attr)
       GATv2Conv layers consume both edge_index AND edge_attr
                  │
                  ▼
            policy / value heads → loss → backprop
```

**Critical contract:** Edge cache stores `(edge_index, edge_attr)` tuples in a single deque. Replay reconstructs both for mini-batch updates. SKRL `sample_all` returns sequential contiguous slices (verified at `gnn_policy.py:208`); block-aligned replay still works.

### 5.3 Residual-training flow (offline, before 1b retrain)

```
Published Crazyflie aero dataset    (Q7-C — must be sourced first)
                  │
                  ▼
      scripts/aero/train_residual.py:
        ├── parse dataset → (rel-pose graph, ground-truth force)
        ├── train MLP/graph-net to minimize MSE(pred_force, gt_force)
        ├── hold-out validation split for early stop
        └── export checkpoint to logs/aero/residual_<date>.pt
                  │
                  ▼
       cfg.downwash_residual_checkpoint = "logs/aero/residual_<date>.pt"
                  │
                  ▼
       _pre_physics_step aero-hook loads it once at env init
       and runs inference per tick when downwash_mode == "residual"
```

**Hard prerequisite:** Dataset must exist and be usable. Cleared at gate G1b-0; if unmet, reconvene per stop condition #2.

## 6. Error Handling & Risks

### 6.1 Runtime error handling posture

Same posture as the capstone env: fail loud, no silent fallbacks. New code paths:

- **Shared-scene scene setup (1a):** Don't catch Isaac Lab exceptions. Shape-contract assertions added in 1a fail loudly the first step.
- **Analytic downwash (1b):** Clamp pairwise distance to small epsilon before normalization (single-line guard against NaN at zero-distance, not a fallback).
- **Residual downwash (1b):** Clamp MLP output to `cfg.downwash_residual_max_force` (default ≈ 0.5 × drone weight). If clamp triggers more than `cfg.downwash_residual_clamp_warn_frac` (default 0.05 = 5%) of steps in a rollout, log a `logger.warning` — signals the residual is extrapolating.
- **Edge attribute publication (1c):** Single deque of tuples eliminates cache desync by construction. Assertion `edge_attr.shape[1] == 6` at policy-side consumption.
- **Replay-gate harness (1a):** Fails the gate if metric distributions differ beyond a tolerance set in the script (2σ on mean formation error, etc.). Does not try to recover.
- **Logging:** No `print()` in env code per `CLAUDE.md`. Aero modules use `logger.{info,debug,warning}`.

### 6.2 Phase-level risks

| # | Risk | P × Blast | Mitigation |
| :--- | :--- | :--- | :--- |
| R1 | No usable published Crazyflie aero dataset | M × M | G1b-0 literature check **before** any 1b code. If empty, 1b ships analytic-only; residual becomes documented backlog item. |
| R2 | Local 3070 can't run shared-scene at any meaningful env count | M × H | 1a throughput sweep is the discovery. Fallbacks: reduce A, drop policy frequency, accept smaller learning signal + longer wall clock. |
| R3 | 1a replay gate fails (env divergence even with downwash off) | M-L × M | Diagnostic playbook: per-step state tensor diff, shape-contract bisect, obs vector byte-identity verify. Gate failing is gate working. |
| R4 | Downwash destabilizes 1a checkpoint policy without curriculum | M-H × L | No explicit curriculum in default scope (Q10). Contingency: ramp downwash via `cfg.downwash_curriculum_iters` over first ~1000 iters. |
| R5 | Edge-attr cache replay misalignment under SKRL mini-batch shuffling | L × M | Single-deque-of-tuples eliminates desync. Plus first-PPO-update assertion `edge_attr.shape[0] == edge_index.shape[1]`. |
| R6 | Group-plumbing collapse breaks forest mode | L-M × L | Forest reshapes audited as part of 1a cleanup. Forest play-mode smoke test in 1a verification (G1a-3). |
| R7 | B3 spawn fix invalidates replay gate | L × L | Gate runs B3-off first (must pass), then B3-on (drift recorded, not gating). |
| R8 | A2 (cloud retrain) needs structural changes 1c didn't anticipate | L × L | A2 is 1c-followup; not a Phase 1 gate. |

### 6.3 Stop conditions

1. R2 fully realized — pause, talk through structural-only 1a tag and defer 1b/1c until cloud credits return.
2. R1 + dataset unavailable — reconvene; revise Q5 to "analytic-only" or accept "residual-deferred" caveat in 1b.
3. 1a replay gate fails three times after diagnostic passes — sub-design pass before more flailing.

## 7. Testing & Gates

No formal unit test suite exists or is being added. Discipline = gates per sub-phase.

### 7.1 1a gates (in order)

- **G1a-1: Smoke test.** `train.py --num_envs 16 --max_iterations 5`. Pass = no errors, training progresses, A=8 drones spawn per env.
- **G1a-2: Throughput sweep.** 16 / 32 / 64 / 128 / (higher if 3070 holds) at 5 iters. No preset throughput bar — the gate is *deciding the knee and recording it*. Pass = a chosen `num_envs` value is selected with steps/sec recorded in the changelog as the 1a/1b/1c retrain envelope.
- **G1a-3: Forest-mode play smoke.** Headless play with `forest_enabled=True` for ~200 steps, no crash. Catches R6.
- **G1a-4: Replay gate (the real 1a milestone).** N rollouts of `v1.0.0-capstone` in capstone env (reference, captured once). Same N rollouts in shared-scene env on `main` with `downwash_mode="off"`, B3 off. Pass = mean formation error / collision pairs / episode reward / final_distance_to_goal all within 2σ of capstone. Re-run with B3 on; drift recorded but not gating.
- **G1a-5: Tag.** `phase1a-shared-scene` tag + changelog with TB scalar baseline.

### 7.2 1b gates (in order)

- **G1b-0: Dataset prerequisite (R1 clearance).** Literature check confirming usable published Crazyflie aero dataset exists (URL identified, license verified). **Required before any 1b code is written.** Failure → stop condition #2.
- **G1b-1: Analytic downwash sanity.** Stand-alone repro: two drones stacked at 0.3 m, equal hover thrust, lower drone loses altitude under analytic downwash. Script under `scripts/aero/sanity_check.py`.
- **G1b-2: Residual training run.** MLP converges on dataset training split, validation loss within expected magnitude. Output checkpoint loadable.
- **G1b-3: Three retrains** warm-started from 1a checkpoint:
  - `p1b-disabled-1` (sanity — 1b code paths default-disabled don't regress 1a)
  - `p1b-analytic-1`
  - `p1b-residual-1`
  - Iteration budget set after G1a-2 throughput sweep (3070-constrained).
- **G1b-4: TB scalar comparison.** Pass = `disabled` within 2σ of 1a; `analytic` and `residual` both converge (no policy collapse). Comparison between analytic/residual is information, not gate. Pick better-converging as 1c warm-start anchor.
- **G1b-5: Tag.** `phase1b-downwash` on chosen anchor commit.

### 7.3 1c gates

- **G1c-1: GATv2 edge-dim shape audit.** After wiring `edge_dim=6` and edge_attr publication, single PPO update step. Assert `edge_index.shape[1] == edge_attr.shape[0]`; cache replay produces matching shapes for mini-batches.
- **G1c-2: 1c retrain.** Warm-start from 1b anchor. New edge-MLP weights init fresh; rest inherit 1b weights.
- **G1c-3: Cold-start MLP doesn't collapse policy.** First ~50 iterations: episode reward must not crater. If not recovered by iter 200, contingency = freeze 1b weights for warmup window (documented option, not default scope).
- **G1c-4: TB scalar comparison vs 1b.** Pass = 1c converged + deltas documented. No hard magnitude bar (we don't know in advance how much edge features help).
- **G1c-5: Tag.** `phase1c-edge-features`.

### 7.4 Phase 1 milestone gates

- **GM-1: Comparison video.** 2-panel render: `v1.0.0-capstone` vs Phase 1 final (1c). Same scenario seed, same play length, NVENC encode per `CLAUDE.md`. Pass = video renders, both panels show stable formation, write-up draft exists.
- **GM-2: Phase 1 changelog summary.** Rolled-up entry aggregating 1a/1b/1c TB scalar deltas. Phase doc status → "Complete".
- **GM-3: Tag and write-up.** `phase1-complete` tag + social-media write-up the phase doc calls for.

### 7.5 Explicit non-goals

- No formal unit tests for new code (matches existing codebase posture).
- No CI gating on Phase 1 (no CI exists for env code).
- No sim-to-real validation (Phase 2).
- No A2 cloud-mode retrain inside Phase 1 gates (1c-followup).
- No type-check / lint gating beyond what `CLAUDE.md` already specifies as SHOULD.

## 8. Out of scope (explicit deferrals)

- A2 cloud-mode retrain — 1c-followup, separate small-scope item.
- Sim-to-real (Phase 2).
- Real Crazyflie flight data collection (Phase 2).
- Higher-fidelity in-Isaac BEMT model — considered and rejected at Q7 in favor of dataset replay.
- Skybrush integration — Phase 4b and beyond.
- Reward function changes — explicitly excluded per Q10 and phase doc framing.
- Aero curriculum — contingency only (R4); not in default scope.

## 9. Implementation handoff

The next step is `superpowers:writing-plans` to produce a step-by-step implementation plan covering 1a → 1b → 1c with commit-level granularity, including:

- 1a commit ordering (scene topology → reshape math → obs audit → spawn fix → replay gate harness)
- 1b precondition gate (dataset availability) **before** any 1b code
- 1c partial-load shim for SKRL warm-start
- Per-sub-phase tag points and changelog entries
