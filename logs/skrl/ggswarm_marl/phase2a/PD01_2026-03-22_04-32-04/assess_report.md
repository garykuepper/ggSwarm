# Assessment Report: 2026-03-22_04-32-04_mappo_torch

Generated: 2026-03-22 06:26 UTC  
Task: `Template-GGSwarm-Marl-HoverStability-v0`  
Checkpoint: `best_agent.pt`  
Episodes: 5

## Convergence

| Metric | Value |
| :--- | :--- |
| Peak reward | 2758.64 @ step 77,000 |
| Final reward | 1808.31 @ step 80,000 |
| Entropy collapse step | Not detected |
| Recommended budget | 92,000 steps |

## Scorecard

| Metric | Value | Verdict | Threshold |
| :--- | :--- | :--- | :--- |
| `survival_steps` | 250.5000 | WARN | > 500.0 |
| `airborne_ratio` | 0.5421 | FAIL | > 0.9 |
| `ground_hit_rate` | 0.8146 | FAIL | < 0.05 |
| `mean_roll_deg` | 22.1970 | WARN | < 60.0 |
| `orientation_violation_rate` | 0.2194 | WARN | < 0.5 |
| `mean_formation_error_m` | 11.9660 | FAIL | < 0.5 |

**Overall: FAIL**

## All Metrics

| Metric | Value |
| :--- | :--- |
| `airborne_ratio` | 0.542089 |
| `altitude_std_m` | 1.086835 |
| `ground_hit_rate` | 0.814577 |
| `mean_altitude_error_m` | 0.753440 |
| `mean_formation_error_m` | 11.965997 |
| `mean_pitch_deg` | 23.049328 |
| `mean_roll_deg` | 22.196969 |
| `mean_speed_mps` | 4.978697 |
| `orientation_violation_rate` | 0.219437 |
| `separation_event_rate` | 0.003481 |
| `survival_steps` | 250.500000 |
