# Assessment Report: 2026-03-23_03-38-53_mappo_torch

Generated: 2026-03-23 05:27 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | 0.50 @ step 91,000 |
| Final reward | 0.46 @ step 92,000 |
| Entropy collapse step | Not detected |
| Recommended budget | 105,799 steps |

## Scorecard

`survival_steps` = mean over eval episodes of steps until the first batch ground hit (`z < min_height`) or the full horizon if none. (PD1/PD2 `run_history` rows used a broken collector — see `docs/ops/post_train_analysis.md`.)

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 7.0000 | FAIL | > 500.0 |
| `airborne_ratio` | 0.5161 | FAIL | > 0.9 |
| `ground_hit_rate` | 0.7121 | FAIL | < 0.05 |
| `mean_roll_deg` | 18.4066 | WARN | < 60.0 |
| `orientation_violation_rate` | 0.1909 | WARN | < 0.5 |
| `mean_formation_error_m` | 3.9851 | FAIL | < 0.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.516118 |
| `altitude_std_m` | 0.882055 |
| `ground_hit_rate` | 0.712103 |
| `mean_altitude_error_m` | 0.831396 |
| `mean_formation_error_m` | 3.985081 |
| `mean_pitch_deg` | 22.729995 |
| `mean_roll_deg` | 18.406570 |
| `mean_speed_mps` | 2.347530 |
| `orientation_violation_rate` | 0.190901 |
| `separation_event_rate` | 0.001395 |
| `survival_steps` | 7.000000 |
