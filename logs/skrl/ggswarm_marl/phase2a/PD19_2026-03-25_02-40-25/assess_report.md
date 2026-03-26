# Assessment Report: 2026-03-25_02-40-25_mappo_torch

Generated: 2026-03-25 03:09 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | 376.15 @ step 21,000 |
| Final reward | 294.32 @ step 30,000 |
| Entropy collapse step | Not detected |
| Recommended budget | 34,500 steps |

## Training Diagnostics

| Scalar | First | Last | Steps |
| :--- | :--- | :--- | :--- |
| `Policy / Standard deviation (drone_0)` | 0.7635 | 0.0105 | 1k -> 30k |
| `Info / mean_world_z` | 1.3348 | 1.1198 | 1k -> 30k |
| `Info / rew_pos` | 0.0750 | 0.2933 | 1k -> 30k |
| `Info / rew_ang_vel` | -0.3074 | -0.0044 | 1k -> 30k |
| `Info / rew_low_clearance` | -0.4102 | -0.0015 | 1k -> 30k |
| `Info / rew_terminated` | -0.3456 | -0.0013 | 1k -> 30k |

## Training Curve Progression

| Scalar | 1k | 10k | 20k | 30k |
| :--- | :--- | :--- | :--- | :--- |
| `rew_ang_vel` | -0.3074 | -0.0154 | -0.0028 | -0.0044 |
| `ground_hit_rate_step` | 0.1728 | 0.0093 | 0.0006 | 0.0007 |
| `mean_dist_to_goal` | 1.3927 | 0.2009 | 0.0371 | 0.0625 |
| `mean_lin_speed` | 1.6647 | 0.1133 | 0.0320 | 0.0201 |
| `thrust_val_mean` | 0.4741 | 0.4950 | 0.4999 | 0.5001 |
| `mean_world_z` | 1.3348 | 1.1633 | 1.1380 | 1.1198 |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 5.0000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.4592 | FAIL | > 0.9 |
| `ground_hit_rate` | 0.7941 | FAIL | < 0.05 |
| `mean_roll_deg` | 85.0677 | FAIL | < 15.0 |
| `orientation_violation_rate` | 0.5396 | FAIL | < 0.1 |
| `mean_formation_error_m` | 2.1563 | FAIL | < 0.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.459207 |
| `altitude_std_m` | 0.499861 |
| `ground_hit_rate` | 0.794132 |
| `mean_altitude_error_m` | 0.724469 |
| `mean_formation_error_m` | 2.156301 |
| `mean_pitch_deg` | 85.322321 |
| `mean_roll_deg` | 85.067666 |
| `mean_speed_mps` | 1.407107 |
| `orientation_violation_rate` | 0.539566 |
| `separation_event_rate` | 0.002534 |
| `survival_steps` | 5.000000 |
