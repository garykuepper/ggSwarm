# Assessment Report: 2026-03-23_23-36-16_mappo_torch

Generated: 2026-03-24 01:43 UTC  
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

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 5.2000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.7360 | FAIL | > 0.9 |
| `ground_hit_rate` | 0.3729 | WARN | < 0.5 |
| `mean_roll_deg` | 23.8919 | WARN | < 60.0 |
| `orientation_violation_rate` | 0.4549 | WARN | < 0.5 |
| `mean_formation_error_m` | 0.6763 | WARN | < 1.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.736015 |
| `altitude_std_m` | 0.470446 |
| `ground_hit_rate` | 0.372915 |
| `mean_altitude_error_m` | 0.444309 |
| `mean_formation_error_m` | 0.676252 |
| `mean_pitch_deg` | 35.007971 |
| `mean_roll_deg` | 23.891884 |
| `mean_speed_mps` | 1.789902 |
| `orientation_violation_rate` | 0.454858 |
| `separation_event_rate` | 0.007680 |
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
