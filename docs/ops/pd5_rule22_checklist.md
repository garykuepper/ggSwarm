# Phase 2A PD5 — Rule 22 smoke checklist

Before any long hover-stability GCE run (PD5+), run a **1-iteration** local smoke and
confirm the five fields below match the **intended** `GGSwarmMarlHoverStabilityCfg`.

## Command

```powershell
cd C:\Users\gkuep\Code\isaaclab\ggSwarm
python scripts/run.py debug smoke --task Template-GGSwarm-Marl-HoverStability-v0 --iterations 1 --gnn --headless
```

Expect: clean exit, env parses `GGSwarmMarlHoverStabilityCfg`, **512** envs (L4 cfg),
GNN policy line with `hidden_channels=128`.

## Rule 22 fields (hover-stability)

Cross-check against [`drone_swarm_env_cfg.py`](../../source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env_cfg.py) (`GGSwarmMarlHoverStabilityCfg`); smoke stdout does **not** dump these — read the cfg file or Hydra log if in doubt.

| Field | Expected (PD5 baseline) |
| :--- | :--- |
| `rew_scale_upright` | `0.0` |
| `rew_scale_ang_vel` | `-0.012` |
| `rew_scale_terminated` | `0.0` |
| `curriculum_start_step` | `999999` (hover-only lock) |
| `spawn_yaw_range` | `0.3` |

Also confirm **`use_stable_hover_rewards: True`** (stable-hover reward path).

## Short local run (behavior + optional telemetry)

Yes — use your Windows GPU for a **short** hover-stability train (minutes, not hours).
Reduce parallel envs so an **RTX 3070-class** card fits VRAM; GCE can keep **512** envs.

**1. Short headless train** (random policy will move drones; TB shows rewards / telemetry):

```powershell
cd C:\Users\gkuep\Code\isaaclab\ggSwarm
python scripts/run.py hover-stability train --headless --gnn --max_iterations 50 --num_envs 64 `
  --action_telemetry_steps 200
```

- Omit `--action_telemetry_steps` on GCE (defaults to cfg **`0`** = no extra telemetry buffers).
- Tune `--num_envs` (e.g. `32`–`128`) if you hit OOM.
- Logs: `logs\skrl\ggswarm_marl\<timestamp>_mappo_torch\`

**2. TensorBoard** (separate terminal):

```powershell
tensorboard --logdir logs\skrl\ggswarm_marl
```

Watch **`Reward/act_clamp_hit_frac`**, **`Reward/moment_saturated_frac`**, **`Reward/act_raw_thrust_mean`**
(only written for the first **200** env steps when `--action_telemetry_steps 200` is set).

**3. Visual playback** (needs a checkpoint — use `best_agent.pt` from the short run, or any saved ckpt):

```powershell
python scripts/run.py hover-stability play --checkpoint "logs\skrl\ggswarm_marl\<run>\checkpoints\best_agent.pt" --gnn
```

Add `--video` if you want a clip (see `play --help`).

**Alternative (no code edit):** temporarily set `action_telemetry_max_env_steps = 200` in
[`drone_swarm_env_cfg.py`](../../source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env_cfg.py)
on `GGSwarmMarlHoverStabilityCfg` — **do not commit**; prefer the CLI flag above.

See also [`pd_authority_tuning.md`](pd_authority_tuning.md).

## PD5 knob policy

**Default:** no extra deltas beyond the stable-hover migration — PD5 is the first full
train under `compute_stable_hover_rewards`. At most **one** of `max_moment` or
exploration (`entropy_loss_scale` / `initial_log_std`) if smoke or TB forces it;
document any change in [`changelog.md`](../status/changelog.md).
