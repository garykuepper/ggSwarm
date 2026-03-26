# Assessment Report: 2026-03-25_00-35-49_mappo_torch

Generated: 2026-03-25 01:15 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | 497.47 @ step 7,000 |
| Final reward | 401.56 @ step 30,000 |
| Entropy collapse step | Not detected |
| Recommended budget | 34,500 steps |

## Training Diagnostics

| Scalar | First | Last | Steps |
| :--- | :--- | :--- | :--- |
| `Policy / Standard deviation (drone_0)` | 0.3285 | 0.4434 | 1k -> 30k |
| `Info / mean_world_z` | 1.1516 | 1.1497 | 1k -> 30k |
| `Info / rew_pos` | 0.3410 | 0.3545 | 1k -> 30k |
| `Info / rew_ang_vel` | -0.0189 | -0.0175 | 1k -> 30k |
| `Info / rew_low_clearance` | -0.0002 | -0.0017 | 1k -> 30k |
| `Info / rew_terminated` | -0.0002 | -0.0015 | 1k -> 30k |

## Training Curve Progression

| Scalar | 1k | 10k | 20k | 30k |
| :--- | :--- | :--- | :--- | :--- |
| `rew_ang_vel` | -0.0189 | -0.0198 | -0.0197 | -0.0175 |
| `ground_hit_rate_step` | 0.0001 | 0.0000 | 0.0005 | 0.0008 |
| `mean_dist_to_goal` | 0.0134 | 0.0071 | 0.0054 | 0.0048 |
| `mean_lin_speed` | 0.1024 | 0.1014 | 0.1031 | 0.1001 |
| `thrust_val_mean` | 0.5011 | 0.5009 | 0.5004 | 0.5002 |
| `mean_world_z` | 1.1516 | 1.1498 | 1.1494 | 1.1497 |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 5.2000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.4474 | FAIL | > 0.9 |
| `ground_hit_rate` | 0.7988 | FAIL | < 0.05 |
| `mean_roll_deg` | 97.3874 | FAIL | < 15.0 |
| `orientation_violation_rate` | 0.6607 | FAIL | < 0.1 |
| `mean_formation_error_m` | 1.5964 | FAIL | < 0.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.447447 |
| `altitude_std_m` | 0.448104 |
| `ground_hit_rate` | 0.798826 |
| `mean_altitude_error_m` | 0.794115 |
| `mean_formation_error_m` | 1.596371 |
| `mean_pitch_deg` | 95.854390 |
| `mean_roll_deg` | 97.387423 |
| `mean_speed_mps` | 1.085441 |
| `orientation_violation_rate` | 0.660681 |
| `separation_event_rate` | 0.002538 |
| `survival_steps` | 5.200000 |

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
