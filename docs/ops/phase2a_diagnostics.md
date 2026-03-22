# Phase 2A diagnostics and baselines

Operational checklist for hover-stability tuning (PD inner loop + MAPPO/GNN).
Complements [`post_train_analysis.md`](post_train_analysis.md) and
[`training_workflow.md`](training_workflow.md).

## 1. TensorBoard autopsy after each pull

Open the run in TensorBoard (`tensorboard --logdir logs/skrl/ggswarm_marl`) and compare:

| Curve | What to look for |
| :--- | :--- |
| `Reward/Total reward` | Peak step vs end step — sustained drawdown after ~50k may mean overshooting train length or misaligned objective. |
| `Reward/rew_pos`, `rew_vel`, `rew_ang_vel`, `rew_low_clearance` | Low-clearance term should correlate with episodes spending time below the eval airborne band. |
| `Reward/mean_world_z`, `Reward/low_clearance_frac` | Batch mean altitude and fraction of agents below `min_height + low_clearance_margin_m`. |
| Policy std (SKRL) | Flat from step 1 vs gradual decay — exploration health. |

## 2. Zero-action (PD-only) baseline

**Purpose:** Separate “physics + PD + spawn can hover” from “RL is mis-shaping commands.”

1. Use `play` or a short eval with **constant neutral actions** (thrust command 0 in `[-1,1]` semantics → nominal hover collective, attitude commands 0).
2. Observe whether altitude stays in the **clearance band** without RL.
3. If the open-loop PD stack drifts or dips without RL, tune `thrust_to_weight`, `kp_att`, `max_moment`, or spawn heights **before** large reward sweeps.

Exact CLI depends on your playback entry point; keep `num_envs` small for a readable trace.

## 3. Complexity ladder (single drone vs swarm)

- **`Template-GGSwarm-Marl-HoverStability-v0`** uses **three agents** and `compute_marl_rewards` requires **`num_agents >= 2`**. There is no supported `num_agents=1` path in that task without a contract change.
- For **true single-quad** validation, use the hover baseline task documented in [`commands.md`](commands.md) (`GGS-Hover-v0` / `hover` family) to verify SKRL + Isaac Lab before chasing 3-agent GNN issues.

## 4. Checkpoint ladder without full assess

```powershell
.\env_isaaclab\Scripts\python.exe scripts\analyze_checkpoints.py `
    --run_dir logs\skrl\ggswarm_marl\<timestamp>_mappo_torch `
    --num_episodes 2
```

Review `survival_steps`, `airborne_ratio`, and `ground_hit_rate` in the printed table / CSV as a cheap progression check.

## 5. Train length policy (post–PD2)

- **80k iterations** remains a reasonable **minimum** exploration budget.
- If TensorBoard shows a **clear peak** followed by **reward drawdown**, prefer **92k–100k** after MDP/reward alignment, or add **eval-based best checkpoint** selection in a future iteration.
- The convergence script “recommended budget” is a **heuristic**, not a PASS guarantee.
