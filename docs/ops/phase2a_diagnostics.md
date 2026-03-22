# Phase 2A diagnostics and baselines

Operational checklist for hover-stability tuning (PD inner loop + MAPPO/GNN).
Complements [`post_train_analysis.md`](post_train_analysis.md) and
[`training_workflow.md`](training_workflow.md).
See also [`pd_authority_tuning.md`](pd_authority_tuning.md) for `max_moment` / gain sweeps
and `action_telemetry_max_env_steps` (TensorBoard diagnostics).

PD5 Rule 22 smoke + field table: [`pd5_rule22_checklist.md`](pd5_rule22_checklist.md).

## 1. TensorBoard autopsy after each pull

Open the run in TensorBoard (`tensorboard --logdir logs/skrl/ggswarm_marl`) and compare:

| Curve | What to look for |
| :--- | :--- |
| `Reward/Total reward` | Peak step vs end step — sustained drawdown after ~50k may mean overshooting train length or misaligned objective. |
| `Reward/rew_pos`, `rew_vel`, `rew_ang_vel`, `rew_low_clearance` | Low-clearance term should correlate with episodes spending time below the eval airborne band. |
| `Reward/mean_world_z`, `Reward/low_clearance_frac` | Batch mean altitude and fraction of agents below `min_height + low_clearance_margin_m`. |
| Policy std (SKRL) | Flat from step 1 vs gradual decay — exploration health. |

Quick scalar dump (no TensorBoard UI):

```powershell
.\env_isaaclab\Scripts\python.exe scripts\summarize_tb_scalars.py `
  --run_dir logs\skrl\ggswarm_marl\<timestamp>_mappo_torch
```

## 2. Train vs eval parity (Phase 2A)

| Item | Training (`hover-stability train`) | Eval / assess |
| :--- | :--- | :--- |
| `scene.num_envs` | From task cfg (e.g. **512** in `GGSwarmMarlHoverStabilityCfg`) | `run_eval(..., num_envs=None)` keeps cfg → **same** |
| Seed | `skrl_mappo_cfg.yaml` **`seed: 42`** (unless `--seed` overrides train) | **`--seed` default 42** in `post_train_assess.py`, `eval.py`, and `run.py … assess` (was **1** before 2026-03-22 — fixed for parity) |

### 2.1 Eval / assess: mean actions (not Gaussian samples)

Training samples from the policy Gaussian (`skrl` `GaussianMixin.act` uses `rsample()`). **Eval, assess, `analyze_checkpoints`, and `run_eval`** call [`extract_actions()`](../../scripts/ggswarm_utils/sim_helpers.py) after `agent.set_running_mode("eval")`, which takes the **third** return bundle from `agent.act(...)` and prefers **`mean_actions`** when present:

- Multi-agent: `outputs[-1][agent_id].get("mean_actions", sampled_actions)`.
- Single-agent: `outputs[-1].get("mean_actions", sampled_actions)`.

`GaussianMixin` always populates `outputs["mean_actions"]` with the pre-sample mean. So **scorecards are not inflated by eval-time action noise**; wide train-time `Policy / Standard deviation` is still a training/exploration issue, not a “we accidentally sampled σ≈7 at assess” issue.

## 3. Zero-action (PD-only) baseline

**Purpose:** Separate “physics + PD + spawn can hover” from “RL is mis-shaping commands.”

Use [`scripts/pd_neutral_baseline.py`](../../scripts/pd_neutral_baseline.py): it builds the hover-stability task via Hydra (same entry point as training), steps with **constant neutral actions** (zeros in `[-1,1]` → nominal hover collective + zero roll/pitch/yaw-rate setpoints), and prints aggregates **without** SKRL or a checkpoint.

**Metrics printed:**

| Field | Meaning |
| :--- | :--- |
| `mean_world_z` | Time average of batch mean altitude (world z). |
| `ground_hit_rate` | Per decimated step: fraction of envs where **any** agent has `z < min_height`. |
| `airborne_ratio` | Per decimated step: fraction of **agent** slots with `z > min_height + 0.2` m (scorecard-style band). |

**Pass / fail intuition:** If `mean_world_z` drifts far below the spawn/goal band or `ground_hit_rate` saturates near **1.0** without RL, fix `thrust_to_weight`, `kp_att` / `kd_att`, `max_moment`, or spawn heights **before** another long MAPPO run. If the neutral run looks sane but trained policy assess still FAIL, prioritize RL / exploration / reward shaping.

**Command (PowerShell, local Isaac Lab venv):**

```powershell
.\env_isaaclab\Scripts\python.exe scripts\pd_neutral_baseline.py `
  --headless `
  --task Template-GGSwarm-Marl-HoverStability-v0 `
  --num_envs 8 `
  --num_steps 500 `
  --seed 42
```

**Reference snapshot (2026-03-22, repo cfg at that date):** `--num_envs 8`, `--num_steps 300`, `--seed 42` → `mean_world_z ≈ 0.72 m`, `ground_hit_rate ≈ 0.23`, `airborne_ratio ≈ 0.80` — enough headroom that PD+physics are not the sole explanation for a FAILing trained policy; re-run after any inner-loop or spawn change.

## 4. Complexity ladder (single drone vs swarm)

- **`Template-GGSwarm-Marl-HoverStability-v0`** uses **three agents** and `compute_marl_rewards` requires **`num_agents >= 2`**. There is no supported `num_agents=1` path in that task without a contract change.
- For **true single-quad** validation, use the hover baseline task documented in [`commands.md`](commands.md) (`GGS-Hover-v0` / `hover` family) to verify SKRL + Isaac Lab before chasing 3-agent GNN issues.

## 5. Checkpoint ladder without full assess

**Phase 2A hover-stability runs** must pass the hover task ID (default script target is formation `Template-GGSwarm-Marl-Direct-v0`).

```powershell
.\env_isaaclab\Scripts\python.exe scripts\analyze_checkpoints.py `
    --run_dir logs\skrl\ggswarm_marl\<timestamp>_mappo_torch `
    --task Template-GGSwarm-Marl-HoverStability-v0 `
    --interval 30000 `
    --num_episodes 2 `
    --seed 42
```

GNN is enabled automatically for registered tasks with `gnn_default=True` (e.g. hover-stability). For MLP-only checkpoints, add **`--no-gnn`**.

Review `survival_steps`, `airborne_ratio`, and `ground_hit_rate` in the printed table / CSV as a cheap progression check.

## 6. Train length policy (post–PD2)

- **80k iterations** remains a reasonable **minimum** exploration budget.
- If TensorBoard shows a **clear peak** followed by **reward drawdown**, prefer **92k–100k** after MDP/reward alignment, or add **eval-based best checkpoint** selection in a future iteration.
- The convergence script “recommended budget” is a **heuristic**, not a PASS guarantee.

## 7. PD3 run autopsy (`2026-03-22_16-00-12_mappo_torch`)

Executed after Run PD3 assess (2026-03-22):

- **TensorBoard (SKRL tags only):** `Reward / Total reward (mean)` rises **140 → ~18.6k** by step 92k; `Policy / Standard deviation` rises **~0.36 → ~7.39** per drone — very wide Gaussians by end of run (consistent with `check_convergence` **entropy collapse @ 81k**).
- **Per-step env extras** (`rew_low_clearance`, `mean_world_z`) are **not** in the default SKRL TB export for this run — use `extras["log"]` plumbing / WandB if you need them on the same dashboard.
- **PD4 bundle (applied in repo):** tighter `max_log_std`, lower `entropy_loss_scale`, GNN `initial_log_std` + logged `hidden_channels` / `num_heads`, hover-stability spawn Z band + vel/ang_vel nudges — see `changelog.md` Phase 2A PD4 prep.
