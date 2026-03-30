# Run History: Phase 2 Training Scorecard

Cross-run scorecard for all Phase 2 training runs. One row per run.

**After every run:** append a row from the `assess` scorecard output before changing any config
or relaunching (Rule 23). Full assessment workflow: [`docs/ops/post_train_analysis.md`](../ops/post_train_analysis.md).

---

## Scorecard Table

> **Architecture reset (2026-03-22):** Runs 1–A1 used raw-torque action semantics and are
> incompatible with the new PD attitude controller. All were FAIL. Logs deleted; history
> preserved below for reference only. New runs use `[thrust, desired_roll, desired_pitch,
> desired_yaw_rate]` action semantics with PD inner loop.

### Pre-Reset Runs (raw-torque action semantics — retired)

| Run | Timestamp | Phase | survival\_steps | airborne\_ratio | ground\_hit\_rate | mean\_roll\_deg | orientation\_viol | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Run 1 | 2026-03-21_14-44-56 | Phase 2A | 487 | 0.835 | 0.165 | 63.5° | 0.535 | FAIL |
| Run 2 | 2026-03-21_20-46-16 | Phase 2A | — | — | — | — | — | aborted |
| Run 3 | 2026-03-21_21-21-55 | Phase 2A | 1.1 | 0.582 | 0.648 | 75.8° | 0.582 | FAIL |
| Run A1 | 2026-03-22_00-32-56 | Phase 2A | 1.1 | 0.700 | 0.423 | 59.8° | 0.524 | FAIL |

### Post-Reset Runs (PD1–PD15b: PD attitude controller; PD16+: direct moments)

| Run | Timestamp | Phase | survival\_steps | airborne\_ratio | ground\_hit\_rate | mean\_roll\_deg | orientation\_viol | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Run PD1 | 2026-03-22_04-32-04 | Phase 2A | 250.5 | 0.542 | 0.815 | 22.2° | 0.219 | FAIL |
| Run PD2 | 2026-03-22_07-03-55 | Phase 2A | 250.5 | 0.571 | 0.723 | 24.6° | 0.296 | FAIL |
| Run PD3 | 2026-03-22_16-00-12 | Phase 2A | 4.4 | 0.617 | 0.494 | 24.6° | 0.349 | FAIL |
| Run PD4 | 2026-03-22_18-38-39 | Phase 2A | 5.0 | 0.687 | 0.361 | 28.9° | 0.373 | FAIL |
| Run PD5 | 2026-03-22_23-01-57 | Phase 2A | 6.4 | 0.623 | 0.542 | 19.2° | 0.168 | FAIL |
| Run PD6 | 2026-03-23_01-22-37 | Phase 2A | 4.4 | 0.612 | 0.534 | 21.1° | 0.234 | FAIL |
| Run PD7 | 2026-03-23_03-38-53 | Phase 2A | 7.0 | 0.516 | 0.712 | 18.4° | 0.191 | FAIL |
| Run PD8 | 2026-03-23_06-23-47 | Phase 2A | 7.0 | 0.732 | 0.304 | 32.2° | 0.525 | FAIL |
| Run PD9 | 2026-03-23_16-19-28 | Phase 2A | 6.2 | 0.629 | 0.444 | 29.7° | 0.469 | FAIL |
| Run PD10 | 2026-03-23_23-36-16 | Phase 2A | 5.2 | 0.736 | 0.373 | 23.9° | 0.455 | FAIL |
| Run PD11 | 2026-03-24_02-19-01 | Phase 2A | 4.2 | 0.757 | 0.358 | 49.6° | 0.582 | FAIL |
| Run PD12 | 2026-03-24_04-42-36 | Phase 2A | — | — | — | — | — | aborted |
| Run PD15b | 2026-03-24_15-13-09 | Phase 2A | 5.0 | 0.737 | 0.353 | 23.6° | 0.305 | FAIL |
| Run PD16 | 2026-03-24_22-37-14 | Phase 2A | 5.0 | 0.473 | 0.706 | 83.1° | 0.585 | FAIL |
| Run PD17 | 2026-03-25_00-35-49 | Phase 2A | 5.2 | 0.447 | 0.799 | 97.4° | 0.661 | FAIL |
| Run PD18 | 2026-03-25_01-32-20 | Phase 2A | 5.2 | 0.762 | 0.323 | 52.2° | 0.617 | FAIL |
| Run PD19 | 2026-03-25_02-40-25 | Phase 2A | 5.0 | 0.459 | 0.794 | 85.1° | 0.540 | FAIL |
| Run PD20 | 2026-03-25_03-23-42 | Phase 2A | 4.4 | 0.432 | 0.836 | 80.2° | 0.518 | FAIL |
| **PD16 re-eval** | 2026-03-24_22-37-14 | Phase 2A | **240.8** | **1.000** | **0.000** | **0.08°** | **0.000** | **WARN** |

> **PD16 re-eval (2026-03-25):** Root cause of the train-eval gap found: `load_policy_from_checkpoint()` did not restore the `RunningStandardScaler` preprocessor statistics. Fix: `agent.load()`. PD16 re-evaluated with fix — **Phase 2A hover-stability effectively solved.** Only WARN is `survival_steps=240.8` (gate 500) due to one late episode wobble.

### Phase 2B Runs (Formation — hybrid stable-hover + formation rewards)

| Run | Timestamp | Phase | survival\_steps | airborne\_ratio | ground\_hit\_rate | mean\_roll\_deg | orientation\_viol | mean\_formation\_error\_m | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| p2b-1 | 2026-03-25_06-30-30 | Phase 2B | 4.2 | 0.921 | 0.073 | 27.3° | 0.250 | 0.431 | FAIL |
| p2b-2 | 2026-03-25_07-23-42 | Phase 2B | 4.4 | 0.911 | **0.002** | **3.1°** | **0.004** | 0.471 | FAIL |

> **p2b-1 (2026-03-25):** First Phase 2B run. Used `compute_marl_rewards` (Gaussian, no dt-scaling) — 185x magnitude mismatch vs Phase 2A's `compute_stable_hover_rewards`. Drones tumbled (roll ±150°) and crashed repeatedly. Formation error "passed" only because crashed drones cluster on the floor.

> **p2b-2 (2026-03-25):** Hybrid rewards — `compute_stable_hover_rewards` base (Phase 2A scales) + `compute_formation_rewards` on top via curriculum. Stability restored: mean\_roll 27°→3.1°, ground\_hit 7.3%→0.2%, orientation\_viol 25%→0.4%. `survival_steps=4.4` FAIL is a late-episode crash (~step 450), not early instability. Formation error 0.47m (PASS). 5/6 gates pass.

> **Run PD8 scorecard:** GCE train **92,000** iters; PD8 = `rew_scale_terminated=0.0` (ceiling-escape fix) + `hover_in_place=True` confirmed on VM. **`post_train_assess.py`** (seed **42**, `best_agent.pt`, 5 episodes). **TensorBoard:** `Reward/Total` peak **243** @ **85k**, final **241** @ **92k** (massive vs PD7 ~0.5 — reward structure changed with terminated=0); **`Policy / Standard deviation (drone_0)`** **0.61 → 2.70** (pinned at `max_log_std=1.0` ceiling = e^1 = 2.718); `Info / mean_world_z` **0.64 → 1.14 m** (learning to hover!); `Info / rew_ang_vel` flat **−0.112 → −0.116** (attitude never improves). **Checkpoint ladder** (10k interval, 2 eps): **10k is best** (airborne 0.766, formation 0.675 m); performance **degrades** at later checkpoints as σ explodes — confirms `best_agent.pt` (peak stochastic reward) ≠ best deterministic eval. **Trajectory plots:** shark-fin altitude pattern (~50-step crash-reset cycles, NOT 7-step PD7 saw-tooth); XY drift **1–3 m** (down from 4–8 m in PD7 — `hover_in_place` working); attitude ±50–75° oscillating with crash cycle. **vs PD7:** `airborne_ratio` **0.516→0.732** (+0.22), `ground_hit_rate` **0.712→0.304** (−0.41); `mean_roll_deg` **18.4°→32.2°** (worse — σ explosion), `orientation_violation_rate` **0.191→0.525** (worse). **FAIL** — ceiling escape fixed, hover_in_place working, but σ explosion is new dominant failure. **Next:** `max_log_std: 1.0 → 0.0` (clamp σ ceiling to 1.0).

> **Run PD7 scorecard:** GCE train **92,000** iters; **`hover_in_place=True`** (spawn-hold XY+Z goals). **`post_train_assess.py`** (seed **42**, `best_agent.pt`, 5 episodes). **TensorBoard:** `Reward/Total` peak **0.50** @ **91k**, final **0.46** @ **92k**; `Info/rew_pos` tail ~**0.21** (batch mean, not per-reset). **Checkpoint ladder** (`analyze_checkpoints.py`, 2 eps): roll/pitch still **~20–24°** @ 10k–30k (not collapsed to ~0°); **worst airborne** **0.518** @ **90k**. **vs PD6:** `mean_roll_deg` **21.1°→18.4°**, `orientation_violation_rate` **0.234→0.191** (slightly better); `ground_hit_rate` **0.534→0.712**, `airborne_ratio` **0.612→0.516** (worse); `survival_steps` **4.4→7.0** (still far below gate). **FAIL** — gates not met; lateral/altitude failure mode not fully explained by formation-slot XY alone (GNN coupling, `spawn_dist`, or train/eval stress). **Next:** confirm VM trained **current** `drone_swarm_env.py`; review TB + `checkpoint_progression.csv`; tune **one** knob (e.g. clearance / vel / spawn) with changelog before retrain — see [`changelog.md`](changelog.md) Run PD7.

> **Run PD6 scorecard:** GCE train **80,000** iters; PD6 bundle (`rew_scale_terminated=-5.0`, TB `Info / *` tensor logging, commit `0a87fb4`). **`post_train_assess.py`** (seed **42**, `best_agent.pt`, 5 episodes). **vs Run PD5:** `ground_hit_rate` **0.542→0.534** (small improvement); `airborne_ratio` **0.623→0.612** (flat); `survival_steps` **6.4→4.4** (worse); `mean_roll_deg` **19.2°→21.1°**, `orientation_violation_rate` **0.168→0.234** (attitude worse). TensorBoard mean reward **~0.36** end-state (vs PD5 negative drift) — compare `Info / rew_terminated` + `Reward/Total` for credit assignment. **FAIL** — gates unchanged; do not advance to Phase 2B.

> **Run PD5 scorecard:** GCE train **92,000** iters; first full run under **`use_stable_hover_rewards`** / `compute_stable_hover_rewards` (stable-hover MDP). **`post_train_assess.py`** (seed **42**, `best_agent.pt`, 5 episodes). **vs Run PD4:** `mean_roll_deg` **28.9°→19.2°**, `orientation_violation_rate` **0.373→0.168** (attitude improved); `airborne_ratio` **0.687→0.623**, `ground_hit_rate` **0.361→0.542** (altitude proxy regressed); `survival_steps` **5.0→6.4** (still far below gate). **FAIL** — do not advance to Phase 2B. TensorBoard: reward fell from peak **~-0.48 @ 21k** to **~-0.77 @ 92k**; review stable-hover term balance + low-clearance vs pos/vel before next cfg change.

> **Run PD4 scorecard:** GCE train **92k** iters; config = PD4 bundle (`max_log_std=1.0`, spawn/reward nudges). **`post_train_assess.py`** (seed **42**, `best_agent.pt`, 5 episodes). vs **Run PD3:** `airborne_ratio` **0.617→0.687**, `ground_hit_rate` **0.494→0.361** (WARN); `mean_roll_deg` **24.6°→28.9°**, `orientation_violation_rate` **0.349→0.373** — altitude proxy improved, attitude slightly worse; overall **FAIL** (gates unchanged).

> **Run PD3 scorecard:** row uses **eval seed 42** (matches `skrl_mappo_cfg.yaml` / training). An earlier assess pass used **seed 1** (`survival_steps` 4.6, `ground_hit_rate` 0.717, `airborne_ratio` 0.596) — superseded for cross-run comparison.
>
> **`survival_steps` for Run PD1 / Run PD2:** these values are **artifacts of a broken metric**
> (pre-2026-03-22 `Phase2Collector`). They are **not** comparable to `survival_steps` from assess runs
> after the fix. See [`post_train_analysis.md`](../ops/post_train_analysis.md) § Metric definitions.

### Phase 3 Runs (Cloud formation — centroid-to-goal + CBF + GNN)

| Run | Timestamp | Reward (mean) | Ep Len (mean) | Std | KNN Separation | Key Change | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| p3-1 | 2026-03-27_15-04-41 | 122.4 | — | — | — | Initial cloud formation | — |
| p3-2 | 2026-03-27_15-24-55 | 120.7 | — | — | — | — | — |
| p3-3 | 2026-03-27_16-11-50 | 104.9 | — | — | — | — | — |
| p3-4 | 2026-03-27_16-50-25 | 129.9 | — | — | — | — | — |
| p3-5 | 2026-03-27_17-25-05 | 89.7 | 383 | 0.17 | Collapsed to ~0m | Per-drone goal, drones stack | FAIL |
| p3-6 | 2026-03-27_19-51-55 | 118.3 | 499 | 0.35 | 0.2–0.5m, dips to 0 | Centroid-to-goal fix | WARN |
| p3-7 | 2026-03-27_20-20-20 | 118.7 | 499 | 0.25 | 0.25–0.3m, dips to 0 | min_spacing 0.50, cbf_d_safe 0.30 | WARN |
| p3-8 | 2026-03-27_21-14-49 | -107.4 | 16 | 0.76 | — | Inverse-distance² repulsion (scale 30) | FAIL |
| p3-9 | 2026-03-27_21-41-25 | -16.3 | 18 | 0.26 | — | Linear inverse-distance (scale 15) | FAIL |
| p3-10 | 2026-03-27_22-03-29 | -7.1 | 19 | 0.33 | Spreads but flips | Clamped repulsion (scale 5) | FAIL |
| p3-11 | 2026-03-27_22-36-52 | 4.3 | 60 | 0.17 | 0.5–1.5m but crashes | CBF lateral moments (scale 0.5) | FAIL |
| p3-12 | 2026-03-28_09-28-26 | 97.2 | 479 | 0.23 | Collapses to ~0m | CBF lateral scale 0.15 | WARN |
| p3-13 | 2026-03-28 (p4 subdir) | — | — | — | — | L2 fix: proper GNN edges | pending |
| p3-14 | 2026-03-28_10-28-26 | — | — | — | — | FC edges during collection | pending |
| p3-15 | 2026-03-28_16-25-29 | 61.9 | 431 | — | 0.10-0.50m | K-hop sparse + edge cache | PASS |
| p3-16 | 2026-03-28_17-15-47 | -14.5 | 18 | 0.70 | — | CBF-QP (unclamped) | FAIL |
| p3-17 | 2026-03-28_17-58-29 | 65.3 | 463 | 0.29 | 0.10-0.50m | CBF-QP fix (clamped 0.15) | PASS |
| p3-18 | 2026-03-28_20-22-22 | 2.6 | 51 | 0.24 | — | MINCO T=0.10s | FAIL |
| p3-19 | 2026-03-29_01-24-49 | 46.3 | 472 | 0.24 | 0.15-0.60m | MINCO T=0.04s | PASS |
| p3-20 | 2026-03-29_02-00-50 | 17.5 | 99 | 0.23 | 0.15-0.40m | Collision termination (500 iter) | Learning |
| p3-21 | 2026-03-29_08-03-19 | 22.8 | 131 | 0.13 | 0.20-0.50m | Collision termination (1000 iter) | PASS |
| p3-22 | 2026-03-29_11-10-15 | 17.0 | 111 | 0.16 | 0.30-0.80m | KNN cohesion (replaces centroid) | PASS |
| p3-23 | 2026-03-29_12-20-33 | 19.0 | 109 | 0.15 | 0.25-0.60m | MINCO-CBF sync + spawn 0.5m | PASS |
| p3-24 | 2026-03-29_13-29-35 | 36.4 | 242 | 0.16 | 0.30-0.60m | Separation penalty 20 + random Z | **BEST** |
| p3-25 | 2026-03-29_14-30-20 | 12.3 | 132 | 0.32 | — | SwarmRaft dropout (death bug) | FAIL |
| p3-26 | 2026-03-29_17-13-15 | 19.6 | 332 | 0.26 | 0.30-0.60m | SwarmRaft fixed (dead drone excl) | PASS |

> **p3-5 (2026-03-27):** Root cause identified — all drones in cloud mode shared same
> `_desired_pos_w`, per-drone goal reward (15.0) pulled them all to one point.

> **p3-6 (2026-03-27):** Centroid-to-goal fix. Reward now tracks group centroid to goal.
> Drones hover stably (499 ep_len) but still converge — CBF was only clamping thrust.

> **p3-8 through p3-10:** Reward-based separation iteration. Every approach failed:
> too strong killed hover, too weak had no effect, smooth repulsion made drones flip.

> **p3-11 (2026-03-27):** CBF rewrite with lateral moment injection. Drones separate
> (0.5–1.5m) but lateral_scale=0.5 causes ±150° flips and crashes.

> **p3-12 (2026-03-28):** CBF lateral_scale reduced to 0.15. Stable hover restored
> (reward 97, ep_len 479) but separation still collapses. Conclusion: CBF tuning is
> insufficient — need proper L2 GNN message passing for spatial awareness.

> **p3-13 (2026-03-28):** L2 fix — proper within-group fully-connected edges in GATv2.
> First run with actual GNN message passing. Results pending.

> **p3-15 (2026-03-28):** K-hop sparse edges with edge cache for PPO replay. Baseline
> for all subsequent Phase 3 work. Reward 61.9, ep_len 431, stable hover + cloud formation.

> **p3-16 (2026-03-28):** CBF-QP rewrite — unclamped gradient projection caused massive
> corrections that flipped drones. ep_len 18, reward -14.5. Root cause: correction_scale
> unbounded when drones are close.

> **p3-17 (2026-03-28):** CBF-QP fix — clamped corrections to MAX=0.15, normalized escape
> direction. Restored to baseline quality. Best non-collision run.

> **p3-18 (2026-03-28):** MINCO min-jerk filter at T=0.10s (5 env steps). Too sluggish —
> drones couldn't respond to hover corrections fast enough. ep_len 51.

> **p3-19 (2026-03-29):** MINCO T=0.04s (2 env steps). Responsive enough for hover,
> visibly smoother attitude (+/-15 deg vs +/-30 deg). Lower reward than EMA but better
> quality behavior.

> **p3-20/21 (2026-03-29):** Virtual collision detection (r=0.10m). Group resets on
> collision create strong separation signal. 500 iter not enough; 1000 iter shows KNN
> floor rising to 0.20m. Policy actively learning to avoid close approaches.

> **p3-22 (2026-03-29):** KNN cohesion replacing centroid cohesion. KNN range widened
> to 0.3-0.8m (centroid version was 0.2-0.5m). Drones spread more naturally.

> **p3-23 (2026-03-29):** MINCO-CBF state sync — syncing _minco_pos to post-CBF output
> makes safety corrections persistent. All 8 drones survive full 500 steps. KNN floor 0.25m.

> **p3-24 (2026-03-29):** Best overall. Separation penalty 20 (2x) + random spawn Z.
> Reward 36.4, ep_len 242. KNN 0.3-0.6m centered on 0.5m target. Attitude +/-2 deg.

> **p3-25 (2026-03-29):** SwarmRaft dropout — dead drone fell (thrust=0), hit z<0.05,
> triggered collective group reset. Death bug: dead drones cascade-crash alive group.

> **p3-26 (2026-03-29):** SwarmRaft fixed — `died & _agent_alive` excludes dead drones
> from altitude check. 7/8 survive after dropout. KNN topology self-heals.

---

## Column Definitions

| Column | Description | Units / Values |
| :--- | :--- | :--- |
| `Run` | Sequential run label within the phase | e.g. Run A1, Run B1 |
| `Timestamp` | Run directory timestamp prefix | `YYYY-MM-DD_HH-MM-SS` |
| `Phase` | Phase 2 sub-phase this run belongs to | `Phase 2A` / `Phase 2B` / `Phase 2C` |
| `survival_steps` | Mean over eval episodes: steps until first batch ground hit (`z < min_height`) or full horizon | steps (higher = better; gate: > 500) |
| `airborne_ratio` | Fraction of agent-steps where altitude > `min_height + 0.2 m` | 0–1 (higher = better; gate: > 0.9) |
| `ground_hit_rate` | Fraction of env-steps where any agent is below `min_height` | 0–1 (lower = better; gate: < 0.05) |
| `mean_roll_deg` | Mean absolute roll across all agents and steps | degrees (lower = better; gate: < 15°) |
| `orientation_viol` | Fraction of agent-steps where roll or pitch exceeds 45° | 0–1 (lower = better; gate: < 0.1) |
| `Verdict` | Overall assess result | `PASS` / `WARN` / `FAIL` / `aborted` |

> `mean_formation_error_m` is not in this table because it is not gated in Phase 2A.
> Add it as an extra column when Phase 2B runs appear.

---

## How to Fill In a Row

1. After training finishes and results are synced locally, run:

```powershell
python scripts/run.py hover-stability assess --run_dir logs/skrl/ggswarm_marl/<timestamp>_mappo_torch --num_episodes 5
```

1. The scorecard block at the end of the output prints each metric value and the overall verdict.

1. Copy the values into a new row in the table above, using the run timestamp as the `Timestamp` column.

1. Log the same results in [`docs/status/changelog.md`](changelog.md) using the changelog template
   in `docs/ops/post_train_analysis.md`.
