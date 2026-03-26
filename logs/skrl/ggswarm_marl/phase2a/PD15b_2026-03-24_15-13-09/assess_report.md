# Assessment Report: 2026-03-24_15-13-09_mappo_torch

Generated: 2026-03-24 21:58 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | 419.97 @ step 56,000 |
| Final reward | 196.28 @ step 92,000 |
| Entropy collapse step | Not detected |
| Recommended budget | 105,799 steps |

## Training Diagnostics

| Scalar | First | Last | Steps |
| :--- | :--- | :--- | :--- |
| `Policy / Standard deviation (drone_0)` | 0.5951 | 0.1249 | 1k -> 92k |
| `Info / mean_world_z` | 0.6497 | 1.1441 | 1k -> 92k |
| `Info / rew_pos` | 0.0848 | 0.3435 | 1k -> 92k |
| `Info / rew_ang_vel` | -0.5631 | -0.0318 | 1k -> 92k |
| `Info / rew_low_clearance` | -0.5496 | -0.0053 | 1k -> 92k |
| `Info / rew_terminated` | -0.4418 | -0.0049 | 1k -> 92k |

## Training Curve Progression

| Scalar | 1k | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 92k |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rew_ang_vel` | -0.5631 | -0.3563 | -0.1031 | -0.1095 | -0.0993 | -0.0553 | -0.0321 | -0.0321 | -0.0316 | -0.0324 | -0.0318 |
| `moment_saturated_frac` | 0.9789 | 0.8584 | 0.5876 | 0.6980 | 0.5472 | 0.3840 | 0.3316 | 0.3319 | 0.3322 | 0.3330 | 0.3323 |
| `ground_hit_rate_step` | 0.2209 | 0.1536 | 0.5935 | 0.4025 | 0.3033 | 0.0933 | 0.0019 | 0.0022 | 0.0022 | 0.0038 | 0.0024 |
| `mean_dist_to_goal` | 0.9119 | 0.4725 | 0.8383 | 0.6566 | 0.5336 | 0.2201 | 0.0242 | 0.0160 | 0.0159 | 0.0191 | 0.0154 |
| `mean_lin_speed` | 1.5948 | 2.1215 | 0.2058 | 0.2044 | 0.1636 | 0.1206 | 0.0456 | 0.0390 | 0.0365 | 0.0357 | 0.0345 |
| `thrust_val_mean` | 0.4125 | 0.0827 | 0.2614 | 0.3637 | 0.4103 | 0.4668 | 0.4995 | 0.4994 | 0.4992 | 0.4989 | 0.4993 |
| `mean_world_z` | 0.6497 | 0.6904 | 0.4747 | 0.6957 | 0.8058 | 1.0621 | 1.1524 | 1.1504 | 1.1499 | 1.1534 | 1.1441 |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 5.0000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.7371 | FAIL | > 0.9 |
| `ground_hit_rate` | 0.3528 | WARN | < 0.5 |
| `mean_roll_deg` | 23.6060 | WARN | < 60.0 |
| `orientation_violation_rate` | 0.3045 | WARN | < 0.5 |
| `mean_formation_error_m` | 0.7169 | WARN | < 1.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.737084 |
| `altitude_std_m` | 0.469701 |
| `ground_hit_rate` | 0.352750 |
| `mean_altitude_error_m` | 0.449923 |
| `mean_formation_error_m` | 0.716926 |
| `mean_pitch_deg` | 27.709090 |
| `mean_roll_deg` | 23.605988 |
| `mean_speed_mps` | 1.695863 |
| `orientation_violation_rate` | 0.304496 |
| `separation_event_rate` | 0.007389 |
| `survival_steps` | 5.000000 |

## Trajectory Plots

- `trajectories/altitude_trace_ep0.png`
- `trajectories/altitude_trace_ep1.png`
- `trajectories/altitude_trace_ep2.png`
- `trajectories/altitude_trace_ep3.png`
- `trajectories/altitude_trace_ep4.png`
- `trajectories/attitude_trace_ep0.png`
- `trajectories/attitude_trace_ep1.png`
- `trajectories/attitude_trace_ep2.png`
- `trajectories/attitude_trace_ep3.png`
- `trajectories/attitude_trace_ep4.png`
- `trajectories/xy_trace_ep0.png`
- `trajectories/xy_trace_ep1.png`
- `trajectories/xy_trace_ep2.png`
- `trajectories/xy_trace_ep3.png`
- `trajectories/xy_trace_ep4.png`
