# Assessment Report: 2026-03-24_22-37-14_mappo_torch

Generated: 2026-03-25 04:43 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | 477.58 @ step 13,000 |
| Final reward | 433.75 @ step 92,000 |
| Entropy collapse step | Not detected |
| Recommended budget | 105,799 steps |

## Training Diagnostics

| Scalar | First | Last | Steps |
| :--- | :--- | :--- | :--- |
| `Policy / Standard deviation (drone_0)` | 0.5743 | 0.6880 | 1k -> 92k |
| `Info / mean_world_z` | 0.7383 | 1.1516 | 1k -> 92k |
| `Info / rew_pos` | 0.1190 | 0.3472 | 1k -> 92k |
| `Info / rew_ang_vel` | -0.6114 | -0.0568 | 1k -> 92k |
| `Info / rew_low_clearance` | -0.4339 | 0.0000 | 1k -> 92k |
| `Info / rew_terminated` | -0.3579 | 0.0000 | 1k -> 92k |

## Training Curve Progression

| Scalar | 1k | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 92k |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rew_ang_vel` | -0.6114 | -0.0187 | -0.0277 | -0.0416 | -0.0489 | -0.0534 | -0.0546 | -0.0572 | -0.0578 | -0.0564 | -0.0568 |
| `ground_hit_rate_step` | 0.1789 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0002 | 0.0003 | 0.0000 | 0.0000 |
| `mean_dist_to_goal` | 0.5514 | 0.0154 | 0.0122 | 0.0112 | 0.0112 | 0.0108 | 0.0098 | 0.0098 | 0.0099 | 0.0088 | 0.0089 |
| `mean_lin_speed` | 1.9591 | 0.1056 | 0.1229 | 0.1352 | 0.1370 | 0.1411 | 0.1373 | 0.1391 | 0.1408 | 0.1357 | 0.1370 |
| `thrust_val_mean` | 0.4515 | 0.5012 | 0.5016 | 0.5024 | 0.5029 | 0.5033 | 0.5029 | 0.5029 | 0.5030 | 0.5028 | 0.5029 |
| `mean_world_z` | 0.7383 | 1.1508 | 1.1586 | 1.1567 | 1.1472 | 1.1529 | 1.1607 | 1.1541 | 1.1479 | 1.1456 | 1.1516 |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 240.8000 | WARN | > 500.0 |
| `airborne_ratio` | 0.9999 | PASS | > 0.9 |
| `ground_hit_rate` | 0.0001 | PASS | < 0.05 |
| `mean_roll_deg` | 0.0805 | PASS | < 15.0 |
| `orientation_violation_rate` | 0.0001 | PASS | < 0.1 |
| `mean_formation_error_m` | 0.4665 | PASS | < 0.5 |

**Overall: WARN**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.999944 |
| `altitude_std_m` | 0.289476 |
| `ground_hit_rate` | 0.000118 |
| `mean_altitude_error_m` | 0.012762 |
| `mean_formation_error_m` | 0.466506 |
| `mean_pitch_deg` | 0.081124 |
| `mean_roll_deg` | 0.080458 |
| `mean_speed_mps` | 0.005791 |
| `orientation_violation_rate` | 0.000062 |
| `separation_event_rate` | 0.010338 |
| `survival_steps` | 240.800000 |

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
