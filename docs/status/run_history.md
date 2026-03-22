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
