# Assessment Report: 2026-03-25_03-23-42_mappo_torch

Generated: 2026-03-25 04:19 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | 418.21 @ step 11,000 |
| Final reward | 179.38 @ step 30,000 |
| Entropy collapse step | Not detected |
| Recommended budget | 34,500 steps |

## Training Diagnostics

| Scalar | First | Last | Steps |
| :--- | :--- | :--- | :--- |
| `Policy / Standard deviation (drone_0)` | 0.8122 | 0.0338 | 1k -> 30k |
| `Info / mean_world_z` | 0.7255 | 0.7215 | 1k -> 30k |
| `Info / rew_pos` | 0.1199 | 0.1963 | 1k -> 30k |
| `Info / rew_ang_vel` | -0.0567 | -0.0166 | 1k -> 30k |
| `Info / rew_low_clearance` | 0.0000 | 0.0000 | 1k -> 30k |
| `Info / rew_terminated` | 0.0000 | 0.0000 | 1k -> 30k |

## Training Curve Progression

| Scalar | 1k | 10k | 20k | 30k |
| :--- | :--- | :--- | :--- | :--- |
| `rew_ang_vel` | -0.0567 | -0.0013 | -0.0070 | -0.0166 |
| `ground_hit_rate_step` | 0.2685 | 0.0001 | 0.0008 | 0.2697 |
| `mean_dist_to_goal` | 1.6639 | 0.0530 | 0.2116 | 1.3673 |
| `mean_lin_speed` | 1.4412 | 0.0593 | 0.1710 | 0.4425 |
| `thrust_val_mean` | 0.4896 | 0.5265 | 0.5281 | 0.5234 |
| `mean_world_z` | 0.7255 | 0.9795 | 1.1055 | 0.7215 |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 409.2000 | WARN | > 500.0 |
| `airborne_ratio` | 1.0000 | PASS | > 0.9 |
| `ground_hit_rate` | 0.0000 | PASS | < 0.05 |
| `mean_roll_deg` | 0.6601 | PASS | < 15.0 |
| `orientation_violation_rate` | 0.0000 | PASS | < 0.1 |
| `mean_formation_error_m` | 1.9466 | FAIL | < 0.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.999975 |
| `altitude_std_m` | 0.289795 |
| `ground_hit_rate` | 0.000036 |
| `mean_altitude_error_m` | 0.019135 |
| `mean_formation_error_m` | 1.946598 |
| `mean_pitch_deg` | 0.767263 |
| `mean_roll_deg` | 0.660138 |
| `mean_speed_mps` | 0.082568 |
| `orientation_violation_rate` | 0.000013 |
| `separation_event_rate` | 0.000698 |
| `survival_steps` | 409.200000 |
