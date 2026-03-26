# Assessment Report: 2026-03-22_23-01-57_mappo_torch

Generated: 2026-03-23 00:48 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence


| Metric                | Value               |
| --------------------- | ------------------- |
| Peak reward           | -0.48 @ step 21,000 |
| Final reward          | -0.77 @ step 92,000 |
| Entropy collapse step | Not detected        |
| Recommended budget    | 105,799 steps       |


## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)


| Metric                       | Value   | Verdict | Threshold |
| ---------------------------- | ------- | ------- | --------- |
| `survival_steps`             | 6.4000  | FAIL    | > 500.0   |
| `airborne_ratio`             | 0.6233  | FAIL    | > 0.9     |
| `ground_hit_rate`            | 0.5416  | FAIL    | < 0.05    |
| `mean_roll_deg`              | 19.1829 | WARN    | < 60.0    |
| `orientation_violation_rate` | 0.1677  | WARN    | < 0.5     |
| `mean_formation_error_m`     | 2.6968  | FAIL    | < 0.5     |


**Overall: FAIL**

## All Metrics


| Metric                       | Value     |
| ---------------------------- | --------- |
| `airborne_ratio`             | 0.623292  |
| `altitude_std_m`             | 0.832378  |
| `ground_hit_rate`            | 0.541648  |
| `mean_altitude_error_m`      | 0.701428  |
| `mean_formation_error_m`     | 2.696831  |
| `mean_pitch_deg`             | 20.928530 |
| `mean_roll_deg`              | 19.182862 |
| `mean_speed_mps`             | 2.131727  |
| `orientation_violation_rate` | 0.167688  |
| `separation_event_rate`      | 0.003213  |
| `survival_steps`             | 6.400000  |


