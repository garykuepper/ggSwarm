# Assessment Report: 2026-03-23_16-19-28_mappo_torch

Generated: 2026-03-23 22:54 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | 235.97 @ step 84,000 |
| Final reward | 203.32 @ step 92,000 |
| Entropy collapse step | 58,000 |
| Recommended budget | 66,700 steps |

## Training Diagnostics

| Scalar | First | Last | Steps |
| :--- | :--- | :--- | :--- |
| `Policy / Standard deviation (drone_0)` | 0.6091 | 1.0000 | 1k -> 92k |
| `Info / mean_world_z` | 0.6402 | 1.1613 | 1k -> 92k |
| `Info / rew_pos` | 0.1109 | 0.3113 | 1k -> 92k |
| `Info / rew_ang_vel` | -0.1118 | -0.1136 | 1k -> 92k |
| `Info / rew_low_clearance` | -0.6535 | -0.0016 | 1k -> 92k |
| `Info / rew_terminated` | 0.0000 | 0.0000 | 1k -> 92k |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 6.2000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.6290 | FAIL | > 0.9 |
| `ground_hit_rate` | 0.4439 | WARN | < 0.5 |
| `mean_roll_deg` | 29.6662 | WARN | < 60.0 |
| `orientation_violation_rate` | 0.4691 | WARN | < 0.5 |
| `mean_formation_error_m` | 1.1986 | WARN | < 1.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.628975 |
| `altitude_std_m` | 0.483235 |
| `ground_hit_rate` | 0.443898 |
| `mean_altitude_error_m` | 0.555421 |
| `mean_formation_error_m` | 1.198593 |
| `mean_pitch_deg` | 33.397635 |
| `mean_roll_deg` | 29.666161 |
| `mean_speed_mps` | 2.158709 |
| `orientation_violation_rate` | 0.469078 |
| `separation_event_rate` | 0.005134 |
| `survival_steps` | 6.200000 |

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
