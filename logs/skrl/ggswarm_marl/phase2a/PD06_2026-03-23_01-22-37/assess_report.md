# Assessment Report: 2026-03-23_01-22-37_mappo_torch

Generated: 2026-03-23 02:57 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence


| Metric                | Value              |
| --------------------- | ------------------ |
| Peak reward           | 0.36 @ step 80,000 |
| Final reward          | 0.36 @ step 80,000 |
| Entropy collapse step | Not detected       |
| Recommended budget    | 92,000 steps       |


## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)


| Metric                       | Value   | Verdict | Threshold |
| ---------------------------- | ------- | ------- | --------- |
| `survival_steps`             | 4.4000  | FAIL    | > 500.0   |
| `airborne_ratio`             | 0.6124  | FAIL    | > 0.9     |
| `ground_hit_rate`            | 0.5344  | FAIL    | < 0.05    |
| `mean_roll_deg`              | 21.1224 | WARN    | < 60.0    |
| `orientation_violation_rate` | 0.2338  | WARN    | < 0.5     |
| `mean_formation_error_m`     | 2.5757  | FAIL    | < 0.5     |


**Overall: FAIL**

## All Metrics


| Metric                       | Value     |
| ---------------------------- | --------- |
| `airborne_ratio`             | 0.612388  |
| `altitude_std_m`             | 0.719912  |
| `ground_hit_rate`            | 0.534404  |
| `mean_altitude_error_m`      | 0.663378  |
| `mean_formation_error_m`     | 2.575699  |
| `mean_pitch_deg`             | 24.397689 |
| `mean_roll_deg`              | 21.122422 |
| `mean_speed_mps`             | 2.452344  |
| `orientation_violation_rate` | 0.233831  |
| `separation_event_rate`      | 0.003191  |
| `survival_steps`             | 4.400000  |


