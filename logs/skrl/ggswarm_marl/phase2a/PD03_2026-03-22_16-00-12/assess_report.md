# Assessment Report: 2026-03-22_16-00-12_mappo_torch

Generated: 2026-03-22 17:57 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | 19070.52 @ step 90,000 |
| Final reward | 18614.88 @ step 92,000 |
| Entropy collapse step | 81,000 |
| Recommended budget | 93,150 steps |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 4.4000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.6166 | FAIL | > 0.9 |
| `ground_hit_rate` | 0.4940 | WARN | < 0.5 |
| `mean_roll_deg` | 24.5799 | WARN | < 60.0 |
| `orientation_violation_rate` | 0.3489 | WARN | < 0.5 |
| `mean_formation_error_m` | 1.1728 | WARN | < 1.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.616588 |
| `altitude_std_m` | 0.449409 |
| `ground_hit_rate` | 0.493964 |
| `mean_altitude_error_m` | 0.458742 |
| `mean_formation_error_m` | 1.172831 |
| `mean_pitch_deg` | 30.729706 |
| `mean_roll_deg` | 24.579881 |
| `mean_speed_mps` | 1.739940 |
| `orientation_violation_rate` | 0.348862 |
| `separation_event_rate` | 0.004337 |
| `survival_steps` | 4.400000 |
