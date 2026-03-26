# Assessment Report: 2026-03-22_18-38-39_mappo_torch

Generated: 2026-03-22 21:41 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | 20338.03 @ step 87,000 |
| Final reward | 20162.80 @ step 92,000 |
| Entropy collapse step | 56,000 |
| Recommended budget | 64,399 steps |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 5.0000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.6871 | FAIL | > 0.9 |
| `ground_hit_rate` | 0.3608 | WARN | < 0.5 |
| `mean_roll_deg` | 28.9024 | WARN | < 60.0 |
| `orientation_violation_rate` | 0.3727 | WARN | < 0.5 |
| `mean_formation_error_m` | 0.8540 | WARN | < 1.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.687091 |
| `altitude_std_m` | 0.476353 |
| `ground_hit_rate` | 0.360814 |
| `mean_altitude_error_m` | 0.492009 |
| `mean_formation_error_m` | 0.854037 |
| `mean_pitch_deg` | 29.871065 |
| `mean_roll_deg` | 28.902414 |
| `mean_speed_mps` | 1.779888 |
| `orientation_violation_rate` | 0.372661 |
| `separation_event_rate` | 0.005730 |
| `survival_steps` | 5.000000 |
