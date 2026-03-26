# Assessment Report: 2026-03-25_07-23-42_mappo_torch

Generated: 2026-03-25 14:51 UTC  
Task: `Template-GGSwarm-Marl-Formation-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | -18.56 @ step 30,000 |
| Final reward | -18.56 @ step 30,000 |
| Entropy collapse step | Not detected |
| Recommended budget | 34,500 steps |

## Training Diagnostics

| Scalar | First | Last | Steps |
| :--- | :--- | :--- | :--- |
| `Policy / Standard deviation (drone_0)` | 0.3332 | 0.3341 | 1k -> 30k |
| `Info / mean_world_z` | 0.6704 | 0.5466 | 1k -> 30k |
| `Info / rew_pos` | 0.0320 | 0.0202 | 1k -> 30k |
| `Info / rew_ang_vel` | -0.9411 | -0.6235 | 1k -> 30k |
| `Info / rew_low_clearance` | -0.4412 | -0.5218 | 1k -> 30k |
| `Info / rew_terminated` | -0.3522 | -0.3979 | 1k -> 30k |

## Training Curve Progression

| Scalar | 1k | 10k | 20k | 30k |
| :--- | :--- | :--- | :--- | :--- |
| `rew_ang_vel` | -0.9411 | -0.8108 | -0.7015 | -0.6235 |
| `ground_hit_rate_step` | 0.1761 | 0.1857 | 0.1946 | 0.1990 |
| `mean_dist_to_goal` | 0.6770 | 0.6685 | 0.6811 | 0.6847 |
| `mean_lin_speed` | 1.8369 | 1.6462 | 1.6009 | 1.6087 |
| `mean_world_z` | 0.6704 | 0.5843 | 0.5567 | 0.5466 |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 4.4000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.9113 | PASS | > 0.9 |
| `ground_hit_rate` | 0.0021 | PASS | < 0.05 |
| `mean_roll_deg` | 3.1364 | PASS | < 15.0 |
| `orientation_violation_rate` | 0.0040 | PASS | < 0.1 |
| `mean_formation_error_m` | 0.4714 | PASS | < 0.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.911301 |
| `altitude_std_m` | 0.292343 |
| `ground_hit_rate` | 0.002119 |
| `mean_altitude_error_m` | 0.285927 |
| `mean_formation_error_m` | 0.471428 |
| `mean_pitch_deg` | 10.394810 |
| `mean_roll_deg` | 3.136402 |
| `mean_speed_mps` | 0.286884 |
| `orientation_violation_rate` | 0.004024 |
| `separation_event_rate` | 0.003640 |
| `survival_steps` | 4.400000 |

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
