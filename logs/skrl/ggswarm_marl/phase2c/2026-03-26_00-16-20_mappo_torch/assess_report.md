# Assessment Report: p2c-1

Generated: 2026-03-26 00:54 UTC  
Run: `p2c-1`  
Task: `Template-GGSwarm-Marl-Perturbation-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence


| Metric                | Value                |
| --------------------- | -------------------- |
| Peak reward           | -17.58 @ step 29,000 |
| Final reward          | -17.79 @ step 30,000 |
| Entropy collapse step | Not detected         |
| Recommended budget    | 34,500 steps         |


## Training Diagnostics


| Scalar                                  | First   | Last    | Steps     |
| --------------------------------------- | ------- | ------- | --------- |
| `Policy / Standard deviation (drone_0)` | 0.3344  | 0.3363  | 1k -> 30k |
| `Info / mean_world_z`                   | 0.5964  | 0.5217  | 1k -> 30k |
| `Info / rew_pos`                        | 0.0146  | 0.0134  | 1k -> 30k |
| `Info / rew_ang_vel`                    | -1.1454 | -0.4359 | 1k -> 30k |
| `Info / rew_low_clearance`              | -0.7215 | -0.7360 | 1k -> 30k |
| `Info / rew_terminated`                 | -0.5741 | -0.5881 | 1k -> 30k |


## Training Curve Progression


| Scalar                 | 1k      | 10k     | 20k     | 30k     |
| ---------------------- | ------- | ------- | ------- | ------- |
| `rew_ang_vel`          | -1.1454 | -0.5233 | -0.4694 | -0.4359 |
| `ground_hit_rate_step` | 0.2871  | 0.2921  | 0.2921  | 0.2940  |
| `mean_dist_to_goal`    | 8.1857  | 2.7304  | 2.4065  | 2.3998  |
| `mean_lin_speed`       | 6.9405  | 5.1362  | 4.8802  | 4.8486  |
| `mean_world_z`         | 0.5964  | 0.5212  | 0.5199  | 0.5217  |


## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)


| Metric                       | Value   | Verdict | Threshold |
| ---------------------------- | ------- | ------- | --------- |
| `survival_steps`             | 7.6000  | FAIL    | > 500.0   |
| `airborne_ratio`             | 0.6112  | FAIL    | > 0.9     |
| `ground_hit_rate`            | 0.4688  | WARN    | < 0.5     |
| `mean_roll_deg`              | 41.9074 | WARN    | < 60.0    |
| `orientation_violation_rate` | 0.4306  | WARN    | < 0.5     |
| `mean_formation_error_m`     | 3.2274  | FAIL    | < 0.5     |


**Overall: FAIL**

## All Metrics


| Metric                       | Value     |
| ---------------------------- | --------- |
| `airborne_ratio`             | 0.611216  |
| `altitude_std_m`             | 0.488075  |
| `ground_hit_rate`            | 0.468765  |
| `mean_altitude_error_m`      | 0.519555  |
| `mean_formation_error_m`     | 3.227438  |
| `mean_pitch_deg`             | 43.258929 |
| `mean_roll_deg`              | 41.907421 |
| `mean_speed_mps`             | 4.525476  |
| `orientation_violation_rate` | 0.430597  |
| `separation_event_rate`      | 0.005019  |
| `survival_steps`             | 7.600000  |


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

