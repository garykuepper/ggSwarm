# Assessment Report: 2026-03-22_07-03-55_mappo_torch

Generated: 2026-03-22 15:51 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence


| Metric                | Value                  |
| --------------------- | ---------------------- |
| Peak reward           | 14925.35 @ step 53,000 |
| Final reward          | 10720.96 @ step 80,000 |
| Entropy collapse step | Not detected           |
| Recommended budget    | 92,000 steps           |


## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)


| Metric                       | Value   | Verdict | Threshold |
| ---------------------------- | ------- | ------- | --------- |
| `survival_steps`             | 4.8000  | FAIL    | > 500.0   |
| `airborne_ratio`             | 0.5709  | FAIL    | > 0.9     |
| `ground_hit_rate`            | 0.7229  | FAIL    | < 0.05    |
| `mean_roll_deg`              | 24.5663 | WARN    | < 60.0    |
| `orientation_violation_rate` | 0.2960  | WARN    | < 0.5     |
| `mean_formation_error_m`     | 5.3385  | FAIL    | < 0.5     |


**Overall: FAIL**

## All Metrics


| Metric                       | Value     |
| ---------------------------- | --------- |
| `airborne_ratio`             | 0.570878  |
| `altitude_std_m`             | 0.974184  |
| `ground_hit_rate`            | 0.722898  |
| `mean_altitude_error_m`      | 0.737267  |
| `mean_formation_error_m`     | 5.338468  |
| `mean_pitch_deg`             | 28.168980 |
| `mean_roll_deg`              | 24.566347 |
| `mean_speed_mps`             | 3.247706  |
| `orientation_violation_rate` | 0.295957  |
| `separation_event_rate`      | 0.003737  |
| `survival_steps`             | 4.800000  |


