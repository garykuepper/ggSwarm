# Assessment Report: 2026-03-24_02-19-01_mappo_torch

Generated: 2026-03-24 04:14 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | -17.91 @ step 65,000 |
| Final reward | -159.92 @ step 92,000 |
| Entropy collapse step | Not detected |
| Recommended budget | 105,799 steps |

## Training Diagnostics

| Scalar | First | Last | Steps |
| :--- | :--- | :--- | :--- |
| `Policy / Standard deviation (drone_0)` | 0.5875 | 0.0826 | 1k -> 92k |
| `Info / mean_world_z` | 0.6731 | 0.6258 | 1k -> 92k |
| `Info / rew_pos` | 0.0993 | 0.1086 | 1k -> 92k |
| `Info / rew_ang_vel` | -1.5705 | -0.3619 | 1k -> 92k |
| `Info / rew_low_clearance` | -0.4721 | -0.6585 | 1k -> 92k |
| `Info / rew_terminated` | -0.3724 | -0.5670 | 1k -> 92k |

## Training Curve Progression

| Scalar | 1k | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 92k |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `rew_ang_vel` | -1.5705 | -1.0030 | -0.6808 | -0.5678 | -0.4968 | -0.4463 | -0.4093 | -0.3806 | -0.3571 | -0.3617 | -0.3619 |
| `moment_saturated_frac` | 0.9716 | 0.9076 | 0.8543 | 0.7543 | 0.6758 | 0.6059 | 0.5550 | 0.5104 | 0.4572 | 0.4484 | 0.4473 |
| `ground_hit_rate_step` | 0.1862 | 0.1534 | 0.1545 | 0.1543 | 0.1530 | 0.1538 | 0.1539 | 0.1532 | 0.1659 | 0.2716 | 0.2835 |
| `mean_dist_to_goal` | 0.6837 | 0.4719 | 0.4719 | 0.4730 | 0.4733 | 0.4723 | 0.4729 | 0.4723 | 0.4707 | 0.5283 | 0.5372 |
| `mean_lin_speed` | 1.9291 | 2.1448 | 2.1062 | 2.1045 | 2.0994 | 2.0918 | 2.0874 | 2.0631 | 1.9098 | 1.5075 | 1.4683 |
| `thrust_val_mean` | 0.4328 | 0.1248 | 0.0881 | 0.0758 | 0.0725 | 0.0691 | 0.0656 | 0.0713 | 0.0943 | 0.1224 | 0.1246 |
| `mean_world_z` | 0.6731 | 0.6907 | 0.6877 | 0.6885 | 0.6893 | 0.6876 | 0.6881 | 0.6883 | 0.6890 | 0.6327 | 0.6258 |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 4.2000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.7569 | FAIL | > 0.9 |
| `ground_hit_rate` | 0.3579 | WARN | < 0.5 |
| `mean_roll_deg` | 49.6422 | WARN | < 60.0 |
| `orientation_violation_rate` | 0.5819 | FAIL | < 0.1 |
| `mean_formation_error_m` | 0.5478 | WARN | < 1.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.756934 |
| `altitude_std_m` | 0.467227 |
| `ground_hit_rate` | 0.357909 |
| `mean_altitude_error_m` | 0.421107 |
| `mean_formation_error_m` | 0.547771 |
| `mean_pitch_deg` | 45.144249 |
| `mean_roll_deg` | 49.642191 |
| `mean_speed_mps` | 1.800529 |
| `orientation_violation_rate` | 0.581930 |
| `separation_event_rate` | 0.008129 |
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
