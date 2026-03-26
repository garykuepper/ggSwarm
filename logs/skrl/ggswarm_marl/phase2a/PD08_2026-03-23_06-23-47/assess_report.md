# Assessment Report: 2026-03-23_06-23-47_mappo_torch

Generated: 2026-03-23 15:58 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | 243.06 @ step 85,000 |
| Final reward | 240.52 @ step 92,000 |
| Entropy collapse step | Not detected |
| Recommended budget | 105,799 steps |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 7.0000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.7324 | FAIL | > 0.9 |
| `ground_hit_rate` | 0.3044 | WARN | < 0.5 |
| `mean_roll_deg` | 32.2015 | WARN | < 60.0 |
| `orientation_violation_rate` | 0.5251 | FAIL | < 0.1 |
| `mean_formation_error_m` | 1.2421 | WARN | < 1.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.732374 |
| `altitude_std_m` | 0.471700 |
| `ground_hit_rate` | 0.304380 |
| `mean_altitude_error_m` | 0.447453 |
| `mean_formation_error_m` | 1.242117 |
| `mean_pitch_deg` | 31.500567 |
| `mean_roll_deg` | 32.201530 |
| `mean_speed_mps` | 2.028548 |
| `orientation_violation_rate` | 0.525055 |
| `separation_event_rate` | 0.004746 |
| `survival_steps` | 7.000000 |
