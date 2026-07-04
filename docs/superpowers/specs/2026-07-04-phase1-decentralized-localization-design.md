# Phase 1 Goal A: Decentralized Localization, Stages 0–4 (shadow mode) — Design

**Date:** 2026-07-04
**Status:** Approved (awaiting user spec review)
**Plan reference:** [`docs/ggswarm_live/decentralization_plan.md`](../../ggswarm_live/decentralization_plan.md)
**Phase reference:** [`docs/ggswarm_live/phases/phase1_sim.md`](../../ggswarm_live/phases/phase1_sim.md)
**Anchor checkpoint:** `logs/ref/v1.0.0-capstone/best_agent.pt` (canonical, already consumed by `scripts/skrl/replay_gate.py`)
**Compute envelope:** Local 3070 only (no GCE)

---

## 1. Summary

This spec covers Stages 0–4 of the decentralization plan: build the simulated
UWB ranging stack and per-drone localization estimator, integrate them into
`GgswarmMarlEnv` in **shadow mode** (estimator runs and logs, observations stay
ground truth), and calibrate the residual fault test against the trained
policy's maneuver envelope. Stage 5 (observation swap + retrain) is explicitly
out of scope and gets its own spec once shadow-mode numbers exist.

Delivered as **two sequenced sub-phases**, mirroring the 1a/1b/1c precedent:

| Sub-phase | Scope | Harness | Gate |
| :--- | :--- | :--- | :--- |
| Loc-a | Stages 0–2: DropoutGuard rename + `ranging.py` + `localization.py`, pure torch | `pytest tests/` (no Isaac) | Unit + synthetic-trajectory tests green; steady-state RMSE ≤ 0.10 m |
| Loc-b | Stages 3–4: shadow-mode env integration + residual-test calibration + eval scripts | `replay_gate.py` + new eval scripts | Three-part gate (§6) |

## 2. Decisions locked in (brainstorm record)

| # | Decision | Rationale |
| :--- | :--- | :--- |
| Q1 | Spec scope = Stages 0–4 only; Stage 5 specced separately later | No training cost; curriculum choices for Stage 5 are better made with real shadow-mode RMSE data |
| Q2 | Compute envelope = local 3070 only | Replay and Monte Carlo calibration are inference-only; keeps the spec free of GCE dependencies |
| Q3 | Structure = two sub-phases Loc-a / Loc-b | One test harness per sub-phase; independently bisectable, matches 1a/1b/1c precedent |
| Q4 | Anchor = `logs/ref/v1.0.0-capstone/best_agent.pt` | The promoted canonical checkpoint; `replay_gate.py` already loads it via `checkpoint_utils` |
| Q5 | Tests live in new repo-root `tests/`, runnable without Isaac Sim | `ranging.py`/`localization.py` are pure torch by design; fast local iteration on any machine |
| Q6 | Gauge = odometry-anchored (takeoff-layout seed + dead-reckon + distributed GN correction) | Inherited from the approved plan doc; resolves anchor-free translation/rotation/reflection ambiguity |
| Q7 | No leader, no Raft; per-drone local verdicts | Inherited; see `docs/ggswarm_live/archive/consensus_mechanisms.md` |
| Q8 | Formation goal in obs will be centroid-relative when swapped (Stage 5; recorded here so Loc-b's diagnostics log gauge drift as a first-class metric) | Common-mode gauge drift cancels out of the formation-keeping signal |
| Q9 | Rewards/dones keep ground truth | Legal privileged information under CTDE |
| Q10 | "SwarmRaft" (code) renamed DropoutGuard in the active MARL path only | Paper-verified misnomer; legacy `ggswarm_env.py` left untouched |

## 3. Architecture

### 3.1 Module shape

Two new pure-torch modules under `source/ggswarm/ggswarm/`, no Isaac imports,
mirroring the isolation of `minco.py`/`cbf.py`:

- `ranging.py` — `UwbRangingSim`. Owns the simulated measurement channel:
  pairwise ranges with bias + Gaussian noise, per-link Bernoulli dropout,
  latency ring buffer, fault injection.
- `localization.py` — `DecentralizedLocalizer`. Owns per-drone state
  estimation: dead-reckoning propagation, distributed damped Gauss-Newton
  correction against latency-delayed peer broadcast estimates, residual fault
  test, IRLS multilateration recovery, RMSE/diagnostics.

The env (`ggswarm_marl_env.py`) composes them behind a single
`_update_localization()` call at the top of `_get_observations`. Contract: the
localizer never reads `self._robot` directly — the env hands it tensors
(`pos_g [N_envs, A, 3]`, `lin_vel_w`, alive mask), keeping both modules
unit-testable and hardware-portable (the correct/test/recover math is the
companion-computer algorithm).

### 3.2 Invariants across Loc-a and Loc-b

- Observation vector layout and sources (shadow mode: obs stay ground truth)
- Reward terms and scales; termination conditions
- Action space, PPO/MAPPO hyperparams, MINCO filter, CBF shield
- Formation offsets + greedy reset-time slot assignment
- `replay_gate.py` behavior with `loc_enabled=False` (bit-identical)

### 3.3 Per-step data flow (shadow mode, `loc_enabled=True`)

```text
_get_observations()
  └─ _update_localization()
       1. localizer.propagate(lin_vel_w, dt, noise_scale)   # dead-reckon p̂
       2. ranging.measure(pos_true_g) -> ranges, valid_mask # noisy UWB matrix
       3. localizer.correct(ranges, valid_mask, alive)      # damped GN steps
       4. localizer.update_residuals()                      # per-drone residual
       5. localizer.run_fault_test(mu, sigma, k)            # flags (if enabled)
       6. localizer.recover(ranges, valid_mask)             # IRLS re-multilateration
       7. extras log: loc_rmse_m, loc_gauge_drift_m, flag counts
  └─ obs built from ground truth exactly as today (unchanged in this spec)
```

Dead (DropoutGuard) drones are masked out of `valid_mask` and excluded from
correction/recovery neighbor sets. `_reset_idx` re-seeds estimates from true
spawn poses and clears per-env latency ring-buffer slices.

## 4. Components

### 4.1 Loc-a components

| Component | Touches | Notes |
| :--- | :--- | :--- |
| DropoutGuard rename | `ggswarm_marl_env.py`, `ggswarm_marl_env_cfg.py`, `scripts/skrl/play.py`, `scripts/eval_metrics.py` | Strings/comments only; no behavior change. Legacy `ggswarm_env.py` untouched. |
| `UwbRangingSim` (new) | `source/ggswarm/ggswarm/ranging.py` | Preallocated `[N, A, A]` buffers; in-place `.normal_()`/`.bernoulli_()` (allocation ban — `randn_like` is banned in step paths). Symmetric per-link noise; ring-buffer latency holds last valid reading. |
| `DecentralizedLocalizer` (new) | `source/ggswarm/ggswarm/localization.py` | `propagate` / `correct` / `update_residuals` / `run_fault_test` / `recover` / `rmse_vs` / `reset_idx`. GN scratch buffers preallocated in `__init__`. Shape comments per CLAUDE.md. |
| Test infra (new) | `tests/test_ranging.py`, `tests/test_localization.py` | First `tests/` dir in repo; pure torch, no Isaac. Monte Carlo stats, latency, symmetry, allocation checks; synthetic 8-drone scripted trajectory for the localizer. |

### 4.2 Loc-b components

| Component | Touches | Notes |
| :--- | :--- | :--- |
| Cfg params | `ggswarm_marl_env_cfg.py` | All `loc_*`, `uwb_*`, `odom_*`, `residual_*`, `recovery_*`, `fault_*`, anneal params per the plan doc §5. Defaults: `loc_enabled=False`, `loc_obs_source="ground_truth"`. Planned-but-off features default off per reward-hygiene rule. |
| `_update_localization()` hook | `ggswarm_marl_env.py` (`__init__`, `_get_observations` top, `_reset_idx`) | Single integration point; instantiates both modules only when `loc_enabled`. TB extras: `Metrics/loc_rmse_m`, `Metrics/loc_gauge_drift_m`, flag/recovery counters. |
| Fault injection | `ranging.py` + per-env schedule in `_reset_idx` | Mirrors the existing dropout-step scheduling pattern (`fault_step_min/max`, `fault_count`, `fault_bias_m`). |
| `scripts/calibrate_residual_threshold.py` (new) | Standalone headless script | Replays the anchor checkpoint under honest noise, dumps residual distribution, prints `residual_mu`/`residual_sigma` to paste into cfg. Calibrates against the maneuver envelope, not hover. |
| `scripts/eval_localization.py` (new) | Standalone headless script | Honest runs → RMSE + FP rate; fault-injected runs → FN rate + recovery-time CDF + formation-collapse count. Reuses `eval_metrics.py` helpers where possible. |

## 5. Error handling

- **Ill-conditioned recovery** (< 4 non-flagged, alive, in-range peers): skip
  multilateration, hold dead-reckoned estimate for that drone this step. No
  exception paths in per-step code.
- **All links dropped for a drone** (Bernoulli dropout streak): correction is a
  no-op for that drone (weights zero), dead-reckoning carries it; latency
  buffer returns last valid ranges.
- **NaN guard:** GN damping bounds the step; tests assert no NaNs over long
  synthetic runs at 10× default noise.
- **Misconfiguration:** `loc_obs_source="estimate"` with `loc_enabled=False`
  raises at env construction (fail fast, init-time only).

## 6. Gates

**Loc-a gate (pytest, no Isaac):**

1. Measured-minus-true range stats match cfg (mean ≈ bias, std ≈ σ) over Monte Carlo.
2. Link dropout rate and latency (t−L retrieval) match cfg; range matrix symmetric.
3. No tensor allocations after warmup in `measure`/`propagate`/`correct` (buffer-identity asserts).
4. Synthetic 8-drone trajectory, 500 steps, default noise: steady-state RMSE ≤ 0.10 m; gauge drift within the odometry random-walk envelope; zero reflection flips; <4-peer fallback exercised.

**Loc-b gate (three parts, anchor checkpoint):**

1. **Off means off:** `replay_gate.py` results bit-identical with `loc_enabled=False`.
2. **Shadow is free:** with `loc_enabled=True`, formation metrics within 2σ of the loc-off replay; `Metrics/loc_rmse_m` ≤ 0.10 m steady state; step time stable (allocation ban holds).
3. **Scorecard:** after calibration — residual-test FP ≤ 0.01 (honest runs), FN ≤ 0.05 (injected faults), recovery ≤ 1.0 s. Same targets as the plan doc §6.

## 7. Testing strategy

- Loc-a: pure-torch pytest, runnable on any machine (`pytest tests/`).
- Loc-b: replay-driven — no training anywhere in this spec. All Isaac runs are
  local (play/replay per CLAUDE.md; GCE not used).
- 5-iteration smoke train after the rename and after env integration
  (`--num_envs 64 --max_iterations 5`) to confirm nothing in the training path
  regressed, per the CLAUDE.md smoke-test rule.

## 8. Risks

| Risk | Mitigation |
| :--- | :--- |
| RNG allocations violate the per-step allocation ban | Preallocated buffers + in-place ops only; buffer-identity asserts in Loc-a tests |
| Latency ring buffer leaks stale ranges across staggered resets | `reset_idx` clears per-env slices; covered by a dedicated unit test |
| Residual threshold calibrated on hover underestimates maneuver transients | Calibration script replays the trained policy, not a hover setpoint |
| GN correction diverges under high noise | Damping factor `loc_gn_damping`; NaN-free long-run test at 10× noise |
| Shadow-mode overhead slows training throughput later | Step-time measurement is part of the Loc-b gate; `loc_enabled=False` short-circuits entirely |

## 9. Out of scope

Observation swap + noise-anneal retrain (Stage 5), milestone video (Stage 6),
distributed slot assignment, gossip/CRDT propagation, downwash (Goal B),
Byzantine resilience, hardware bring-up.

## See Also

- [Decentralization detail plan](../../ggswarm_live/decentralization_plan.md)
- [Phase 1: Sim](../../ggswarm_live/phases/phase1_sim.md)
- [Archived Phase 2a localization design](../../ggswarm_live/archive/phases/phase2a_localization.md)
- [Consensus mechanisms rationale](../../ggswarm_live/archive/consensus_mechanisms.md)
- [Phase 1a shared-scene spec](2026-04-30-phase1-shared-scene-design.md) (format precedent)
