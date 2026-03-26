# Assessment Report: 2026-03-25_06-30-30_mappo_torch

Generated: 2026-03-25 07:16 UTC  
Task: `Template-GGSwarm-Marl-Formation-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | 52.63 @ step 1,000 |
| Final reward | -2.75 @ step 30,000 |
| Entropy collapse step | Not detected |
| Recommended budget | 34,500 steps |

## Training Diagnostics

| Scalar | First | Last | Steps |
| :--- | :--- | :--- | :--- |
| `Policy / Standard deviation (drone_0)` | 0.3336 | 0.3348 | 1k -> 30k |
| `Info / mean_world_z` | 0.6660 | 0.5994 | 1k -> 30k |
| `Info / rew_pos` | 4.9012 | 1.5012 | 1k -> 30k |
| `Info / rew_ang_vel` | -1.3194 | -1.5441 | 1k -> 30k |
| `Info / rew_low_clearance` | 0.0000 | 0.0000 | 1k -> 30k |
| `Info / rew_terminated` | -0.3521 | -0.3495 | 1k -> 30k |

## Training Curve Progression

| Scalar | 1k | 10k | 20k | 30k |
| :--- | :--- | :--- | :--- | :--- |
| `rew_ang_vel` | -1.3194 | -1.3658 | -1.4685 | -1.5441 |
| `ground_hit_rate_step` | 0.1761 | 0.1793 | 0.1765 | 0.1747 |
| `mean_dist_to_goal` | 0.6751 | 0.6550 | 0.6506 | 0.6489 |
| `mean_lin_speed` | 1.8232 | 1.7236 | 1.7015 | 1.7003 |
| `mean_world_z` | 0.6660 | 0.5999 | 0.5969 | 0.5994 |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 4.2000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.9207 | PASS | > 0.9 |
| `ground_hit_rate` | 0.0726 | WARN | < 0.5 |
| `mean_roll_deg` | 27.3058 | WARN | < 60.0 |
| `orientation_violation_rate` | 0.2495 | WARN | < 0.5 |
| `mean_formation_error_m` | 0.4315 | PASS | < 0.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.920721 |
| `altitude_std_m` | 0.331960 |
| `ground_hit_rate` | 0.072565 |
| `mean_altitude_error_m` | 0.252085 |
| `mean_formation_error_m` | 0.431509 |
| `mean_pitch_deg` | 31.874701 |
| `mean_roll_deg` | 27.305809 |
| `mean_speed_mps` | 0.755342 |
| `orientation_violation_rate` | 0.249510 |
| `separation_event_rate` | 0.004224 |
| `survival_steps` | 4.200000 |

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
