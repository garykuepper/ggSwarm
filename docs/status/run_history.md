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

### Post-Reset Runs (PD attitude controller — current)

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
