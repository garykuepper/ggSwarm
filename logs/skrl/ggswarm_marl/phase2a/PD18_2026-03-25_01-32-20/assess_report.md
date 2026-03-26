# Assessment Report: 2026-03-25_01-32-20_mappo_torch

Generated: 2026-03-25 04:08 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | 508.24 @ step 12,000 |
| Final reward | 417.83 @ step 30,000 |
| Entropy collapse step | Not detected |
| Recommended budget | 34,500 steps |

## Training Diagnostics

| Scalar | First | Last | Steps |
| :--- | :--- | :--- | :--- |
| `Policy / Standard deviation (drone_0)` | 0.9268 | 0.0910 | 1k -> 30k |
| `Info / mean_world_z` | 0.7320 | 1.1492 | 1k -> 30k |
| `Info / rew_pos` | 0.1176 | 0.3553 | 1k -> 30k |
| `Info / rew_ang_vel` | -0.7480 | -0.0047 | 1k -> 30k |
| `Info / rew_low_clearance` | -0.4334 | -0.0023 | 1k -> 30k |
| `Info / rew_terminated` | -0.3565 | -0.0020 | 1k -> 30k |

## Training Curve Progression

| Scalar | 1k | 10k | 20k | 30k |
| :--- | :--- | :--- | :--- | :--- |
| `rew_ang_vel` | -0.7480 | -0.0107 | -0.0054 | -0.0047 |
| `ground_hit_rate_step` | 0.1782 | 0.0000 | 0.0008 | 0.0010 |
| `mean_dist_to_goal` | 0.5422 | 0.0101 | 0.0051 | 0.0045 |
| `mean_lin_speed` | 2.0533 | 0.0721 | 0.0533 | 0.0456 |
| `thrust_val_mean` | 0.4480 | 0.5004 | 0.4998 | 0.4997 |
| `mean_world_z` | 0.7320 | 1.1504 | 1.1495 | 1.1492 |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 5.2000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.3809 | FAIL | > 0.9 |
| `ground_hit_rate` | 0.9303 | FAIL | < 0.05 |
| `mean_roll_deg` | 109.5811 | FAIL | < 15.0 |
| `orientation_violation_rate` | 0.6259 | FAIL | < 0.1 |
| `mean_formation_error_m` | 2.4098 | FAIL | < 0.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.380924 |
| `altitude_std_m` | 0.500922 |
| `ground_hit_rate` | 0.930297 |
| `mean_altitude_error_m` | 0.614995 |
| `mean_formation_error_m` | 2.409763 |
| `mean_pitch_deg` | 109.733367 |
| `mean_roll_deg` | 109.581080 |
| `mean_speed_mps` | 0.142594 |
| `orientation_violation_rate` | 0.625934 |
| `separation_event_rate` | 0.000821 |
| `survival_steps` | 5.200000 |
