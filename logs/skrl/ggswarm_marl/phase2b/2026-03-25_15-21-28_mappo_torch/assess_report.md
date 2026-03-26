# Assessment Report: p2c-eval

Generated: 2026-03-26 01:02 UTC  
Run: `p2c-eval`  
Task: `Template-GGSwarm-Marl-Perturbation-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | -15.02 @ step 29,000 |
| Final reward | -15.05 @ step 30,000 |
| Entropy collapse step | Not detected |
| Recommended budget | 34,500 steps |

## Training Diagnostics

| Scalar | First | Last | Steps |
| :--- | :--- | :--- | :--- |
| `Policy / Standard deviation (drone_0)` | 0.3333 | 0.3344 | 1k -> 30k |
| `Info / mean_world_z` | 0.6750 | 0.5483 | 1k -> 30k |
| `Info / rew_pos` | 0.0320 | 0.0206 | 1k -> 30k |
| `Info / rew_ang_vel` | -0.9654 | -0.6192 | 1k -> 30k |
| `Info / rew_low_clearance` | -0.4400 | -0.5230 | 1k -> 30k |
| `Info / rew_terminated` | -0.3517 | -0.3987 | 1k -> 30k |

## Training Curve Progression

| Scalar | 1k | 10k | 20k | 30k |
| :--- | :--- | :--- | :--- | :--- |
| `rew_ang_vel` | -0.9654 | -0.8115 | -0.6985 | -0.6192 |
| `ground_hit_rate_step` | 0.1758 | 0.1833 | 0.1933 | 0.1994 |
| `mean_dist_to_goal` | 0.6820 | 0.6625 | 0.6787 | 0.6860 |
| `mean_lin_speed` | 1.8625 | 1.6718 | 1.6171 | 1.6172 |
| `mean_world_z` | 0.6750 | 0.5946 | 0.5624 | 0.5483 |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 64.2000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.8056 | FAIL | > 0.9 |
| `ground_hit_rate` | 0.1853 | WARN | < 0.5 |
| `mean_roll_deg` | 27.1398 | WARN | < 60.0 |
| `orientation_violation_rate` | 0.2142 | WARN | < 0.5 |
| `mean_formation_error_m` | 0.8165 | WARN | < 1.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.805559 |
| `altitude_std_m` | 0.354863 |
| `ground_hit_rate` | 0.185290 |
| `mean_altitude_error_m` | 0.361266 |
| `mean_formation_error_m` | 0.816480 |
| `mean_pitch_deg` | 31.257153 |
| `mean_roll_deg` | 27.139811 |
| `mean_speed_mps` | 1.088089 |
| `orientation_violation_rate` | 0.214247 |
| `separation_event_rate` | 0.004227 |
| `survival_steps` | 64.200000 |

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
- `trajectories/distance_trace_ep0.png`
- `trajectories/distance_trace_ep1.png`
- `trajectories/distance_trace_ep2.png`
- `trajectories/distance_trace_ep3.png`
- `trajectories/distance_trace_ep4.png`
- `trajectories/xy_trace_ep0.png`
- `trajectories/xy_trace_ep1.png`
- `trajectories/xy_trace_ep2.png`
- `trajectories/xy_trace_ep3.png`
- `trajectories/xy_trace_ep4.png`
