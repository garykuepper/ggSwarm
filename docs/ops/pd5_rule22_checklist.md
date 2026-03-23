# Phase 2A PD5+ — Rule 22 smoke checklist

Before any long hover-stability GCE run (PD5 onward), run a **1-iteration** local smoke and
confirm the checklist fields below match the **intended** `GGSwarmMarlHoverStabilityCfg`.

## Command

```powershell
cd C:\Users\gkuep\Code\isaaclab\ggSwarm
python scripts/run.py debug smoke --task Template-GGSwarm-Marl-HoverStability-v0 --iterations 1 --gnn --headless
```

Expect: clean exit, env parses `GGSwarmMarlHoverStabilityCfg`, **512** envs (L4 cfg),
GNN policy line with `hidden_channels=128`.

## Rule 22 fields (hover-stability)

Cross-check against [`drone_swarm_env_cfg.py`](../../source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env_cfg.py) (`GGSwarmMarlHoverStabilityCfg`); smoke stdout does **not** dump these — read the cfg file or Hydra log if in doubt.

| Field | Expected (PD5 baseline) | Expected (PD6+) | Expected (PD7+) | Expected (PD8+) | Expected (PD9+) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rew_scale_upright` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` |
| `rew_scale_ang_vel` | `-0.012` | `-0.012` | `-0.012` | `-0.012` | `-0.012` |
| `rew_scale_terminated` | `0.0` | **`-5.0`** (dense ground penalty; see changelog PD6) | `-5.0` | **`0.0`** (ceiling-escape fix; see changelog PD8) | `0.0` |
| `curriculum_start_step` | `999999` (hover-only lock) | `999999` | `999999` | `999999` | `999999` |
| `spawn_yaw_range` | `0.3` | `0.3` | `0.3` | `0.3` | `0.3` |
| `max_log_std` (YAML) | — | — | — | `1.0` | **`0.0`** (σ explosion fix; see changelog PD9) |

Also confirm **`use_stable_hover_rewards: True`** (stable-hover reward path).

**PD7+:** confirm **`hover_in_place: True`** on `GGSwarmMarlHoverStabilityCfg` (spawn-hold goals; Phase 2B keeps default **`False`**). Quick check:

```powershell
python -c "from ggSwarm.tasks.direct.ggswarm_marl.drone_swarm_env_cfg import GGSwarmMarlHoverStabilityCfg; print('hover_in_place:', GGSwarmMarlHoverStabilityCfg().hover_in_place)"
```

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

**PD6+:** With `extras["log"]` tensor scalars, also watch **`Info / rew_pos`**, **`Info / rew_vel`**,
**`Info / rew_low_clearance`**, **`Info / rew_terminated`**, **`Info / mean_world_z`**.

**3. Visual playback** (needs a checkpoint — use `best_agent.pt` from the short run, or any saved ckpt):

```powershell
python scripts/run.py hover-stability play --checkpoint "logs\skrl\ggswarm_marl\<run>\checkpoints\best_agent.pt" --gnn
```

Add `--video` if you want a clip (see `play --help`).

**Alternative (no code edit):** temporarily set `action_telemetry_max_env_steps = 200` in
[`drone_swarm_env_cfg.py`](../../source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env_cfg.py)
on `GGSwarmMarlHoverStabilityCfg` — **do not commit**; prefer the CLI flag above.

See also [`pd_authority_tuning.md`](pd_authority_tuning.md).

## PD5 / PD6 knob policy

**PD5 default:** no extra deltas beyond the stable-hover migration — first full train under
`compute_stable_hover_rewards`. **PD6:** single reward knob `rew_scale_terminated=-5.0` on hover-stability
(documented in [`changelog.md`](../status/changelog.md)).

At most **one** further knob per smoke (`max_moment` or exploration) if TB forces it;
document any change in [`changelog.md`](../status/changelog.md).
