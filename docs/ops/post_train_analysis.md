# Post-Training Analysis: Phase 2A and Phase 2B

Reference guide for assessing every Phase 2 training run consistently.
Run `post_train_assess.py` once after each GCE training job finishes —
it handles GCS sync, convergence check, Isaac Lab eval, scorecard, and
the markdown report in a single command.

Cross-run scorecard history: **[`docs/status/run_history.md`](../status/run_history.md)** — fill in a row
there before changing any config (Rule 23).

**Diagnostics playbook (TensorBoard, baselines, train-length policy):**
[`phase2a_diagnostics.md`](phase2a_diagnostics.md).

---

## Metric definitions (Phase 2 eval / assess)

### `survival_steps`

**Current definition (assess / `Phase2Collector`, fixed 2026-03-22):** mean over **eval episodes** of the
step index (1-based within the episode) at which the **batch** first shows any ground contact
(`z < cfg.min_height` for any agent in any parallel env), **or** the full episode horizon if none.

**Historical note:** PD-era rows **Run PD1** and **Run PD2** in [`run_history.md`](../status/run_history.md) list
`survival_steps ≈ 250.5` because the collector incorrectly averaged a monotonic step counter every
simulation step (always near `(max_episode_length + 1) / 2` for 500-step episodes). **Do not compare**
those numbers to post-fix assess runs.

### Training telemetry (TensorBoard)

Logged from `GGSwarmMarlEnv` (`extras["log"]`) as **0-dim tensors** (required for SKRL's `Info /` path — see `architecture.md`):

| TensorBoard key | Description |
| :--- | :--- |
| `Info / rew_pos` | Mean tanh position reward per step |
| `Info / rew_vel` | Mean linear velocity squared penalty |
| `Info / rew_ang_vel` | Mean angular velocity squared penalty |
| `Info / rew_low_clearance` | Mean low-clearance depth penalty (non-zero when `rew_scale_low_clearance ≠ 0`) |
| `Info / rew_terminated` | Mean ground-hit penalty (non-zero when `rew_scale_terminated ≠ 0`; PD6+: −5.0) |
| `Info / mean_world_z` | Mean agent altitude (m) — tracks whether policy is hovering or sinking |
| `Info / low_clearance_frac` | Fraction of agents below `min_height + low_clearance_margin_m` |
| `Info / curriculum_alpha` | Curriculum blend weight (0 = hover only, 1 = full formation) |
| `Info / rew_formation` etc. | Formation/cohesion/separation terms (zero in Phase 2A hover-stability) |

Action telemetry keys (`Info / act_raw_thrust_mean`, `Info / thrust_val_mean`, `Info / moment_saturated_frac`, etc.)
are written only during the first `action_telemetry_max_env_steps` env steps when that cfg field is > 0.

---

## Phase 2A (hover-stability) — Post-Training Steps

```powershell
# Via run.py (default: pulls checkpoints/events from GCS if the run folder is missing locally)
python scripts/run.py hover-stability assess `
    --run_dir logs/skrl/ggswarm_marl/<timestamp>_mappo_torch `
    --num_episodes 5

# Offline / already pulled: skip GCS
python scripts/run.py hover-stability assess `
    --run_dir logs/skrl/ggswarm_marl/<timestamp>_mappo_torch `
    --num_episodes 5 `
    --no_sync

# Optional: record a short eval clip (headless rgb_array → run_dir/videos/eval/)
python scripts/run.py hover-stability assess `
    --run_dir logs/skrl/ggswarm_marl/<timestamp>_mappo_torch `
    --video --video_length 200
```

Direct script (same pipeline):

```powershell
# Direct script: same pipeline
# Full assess: sync from GCS + convergence + eval + scorecard + report
# (GCS sync is automatic if run_dir doesn't exist locally)
.\env_isaaclab\Scripts\python.exe scripts/post_train_assess.py `
    --run_dir logs/skrl/ggswarm_marl/<timestamp>_mappo_torch `
    --task Template-GGSwarm-Marl-HoverStability-v0 `
    --num_episodes 5 `
    --headless

# If data is already local (skip GCS sync):
.\env_isaaclab\Scripts\python.exe scripts/post_train_assess.py `
    --run_dir logs/skrl/ggswarm_marl/<timestamp>_mappo_torch `
    --task Template-GGSwarm-Marl-HoverStability-v0 `
    --num_episodes 5 `
    --headless --no_sync

# With eval clip (videos land under run_dir/videos/eval/):
.\env_isaaclab\Scripts\python.exe scripts/post_train_assess.py `
    --run_dir logs/skrl/ggswarm_marl/<timestamp>_mappo_torch `
    --task Template-GGSwarm-Marl-HoverStability-v0 `
    --num_episodes 5 `
    --headless --video --video_length 200
```

After the script completes:

1. Review `logs/skrl/ggswarm_marl/<timestamp>/assess_report.md` (written automatically)
2. Check TensorBoard (see checklist below) for visual confirmation
3. Fill in a row in `docs/status/run_history.md` (Rule 23)
4. Log the result in `docs/status/changelog.md` using the template below

```powershell
# Optional: TensorBoard visual inspection
tensorboard --logdir logs/skrl/ggswarm_marl
```

---

## Phase 2B (formation) — Post-Training Steps

```powershell
# Full assess
.\env_isaaclab\Scripts\python.exe scripts/post_train_assess.py `
    --run_dir logs/skrl/ggswarm_marl/<timestamp>_mappo_torch `
    --task Template-GGSwarm-Marl-Formation-v0 `
    --num_episodes 5 `
    --headless

# Skip GCS sync if already local:
.\env_isaaclab\Scripts\python.exe scripts/post_train_assess.py `
    --run_dir logs/skrl/ggswarm_marl/<timestamp>_mappo_torch `
    --task Template-GGSwarm-Marl-Formation-v0 `
    --num_episodes 5 `
    --headless --no_sync
```

After the script completes:

1. Review `assess_report.md` and check formation reward curves in TensorBoard
2. Fill in a row in `docs/status/run_history.md`
3. Log in `docs/status/changelog.md`

```powershell
tensorboard --logdir logs/skrl/ggswarm_marl
```

---

## Pass Gates

These thresholds are what `run.py assess` uses to compute PASS / WARN / FAIL.
Listed here so you can interpret partial scorecard output without opening code.

| Metric | Phase 2A gate | Phase 2B gate | WARN range | FAIL |
| :--- | :--- | :--- | :--- | :--- |
| `survival_steps` | > 500 | > 500 | 10–500 | < 10 |
| `airborne_ratio` | > 0.9 | > 0.9 | 0.5–0.9 | < 0.5 |
| `ground_hit_rate` | < 0.05 | < 0.05 | 0.05–0.5 | ≥ 0.5 |
| `mean_roll_deg` | < 15° | < 15° | 15°–60° | ≥ 60° |
| `orientation_violation_rate` | < 0.1 | < 0.1 | 0.1–0.5 | ≥ 0.5 |
| `mean_formation_error_m` | not gated | < 0.5 m | 0.5–1.5 m | ≥ 1.5 m |

---

## Decision Matrix

| Verdict | Stability ok? | Formation ok? | Action |
| :--- | :--- | :--- | :--- |
| Phase 2A PASS | yes | n/a | Launch Phase 2B: `phase2b train --checkpoint <2A_run>/checkpoints/best_agent.pt --max_iterations 120000` |
| Phase 2A WARN | partial | n/a | Extend Phase 2A by 20k iters OR reduce `rew_scale_pos` to 1.5 and retrain |
| Phase 2A FAIL (survival < 10) | no | n/a | Check `Info/rew_terminated`: negative = floor hits; ~0 = penalty off (verify `rew_scale_terminated ≠ 0`). Also `Info/thrust_val_mean` < 0.5 = under-hover; check entropy + PD authority. |
| Phase 2A FAIL (roll ≥ 60°) | no | n/a | Hover-stability (PD era): raise `kp_att` / `max_moment` first; then `rew_scale_ang_vel` in the 3-term reward — **not** `rew_scale_upright` (disabled in `GGSwarmMarlHoverStabilityCfg` unless you explicitly re-enable). |
| Phase 2B PASS | yes | yes | Advance to Phase 3 |
| Phase 2B WARN | yes | partial | Reduce `curriculum_end_step` by 20k so full formation pressure applies for longer |
| Phase 2B FAIL (stability) | no | — | Roll back to Phase 2A checkpoint; retry with `rew_scale_upright=3.5` and `curriculum_pos_floor=0.4` |
| Phase 2B FAIL (formation) | yes | no | Verify `curriculum_start_step=0` landed on VM (`grep curriculum_start_step` on the env cfg) |

---

## TensorBoard Inspection Checklist

Check these four curves **before** running `assess`. If any show a bad pattern, read
the convergence check output first — a very early entropy collapse means the scorecard
metrics will be uninformative.

| Curve | Good (Phase 2A) | Bad |
| :--- | :--- | :--- |
| `Reward/Total reward (mean)` | rises to ~5–8 and plateaus | flat from step 1 or collapses after peak |
| `Reward/rew_upright` | increases steadily and stays > 2.0 | stays near 0 or drops after early spike |
| `Reward/rew_ang_vel` | small negative value, magnitude decreasing | large negative throughout (still spinning) |
| `Policy/Standard deviation (drone_0)` | decreases from ~1.0 to ~0.3 over training | stays flat from step 1 (never explored) |

**Phase 2A stable-hover (PD6+):** SKRL logs env extras under the `Info /` prefix when values are 0-dim tensors. Prefer these over the legacy `Reward/rew_*` names above (upright is disabled in hover-stability).

| Curve | Good (Phase 2A stable-hover) | Bad |
| :--- | :--- | :--- |
| `Info / rew_pos` | positive, stable band mid-training | collapses to ~0 from step 1 |
| `Info / rew_vel` | small negative, not diverging | large negative throughout (high speed) |
| `Info / rew_low_clearance` | not pegged at extreme negative | dominates total return (mis-tuned scale) |
| `Info / rew_terminated` | sparse / near zero once policy lifts | frequently negative (repeated ground contact) |
| `Info / mean_world_z` | tracks spawn band | drifts to min_height |

For Phase 2B, also check:

| Curve | Good (Phase 2B) | Bad |
| :--- | :--- | :--- |
| `Reward/rew_formation` | increases after step ~5k (curriculum active) | stays at 0 (curriculum not activating) |
| `Reward/curriculum_alpha` | ramps from 0 to 1 over 80k steps | stuck at 0 (curriculum_start_step misconfigured) |

---

## Changelog Template

Copy this into [`docs/status/changelog.md`](../status/changelog.md) immediately after
the assess scorecard, before changing any config.

```markdown
## Phase 2A Run N — YYYY-MM-DD

- **Run dir:** `logs/skrl/ggswarm_marl/<timestamp>_mappo_torch`
- **Config class:** `GGSwarmMarlHoverStabilityCfg` — commit `<git hash>`
- **Convergence:** entropy collapse @ step N | recommended budget: Nk steps
- **Scorecard:**
  - survival_steps = N
  - airborne_ratio = N
  - ground_hit_rate = N
  - mean_roll_deg = N°
  - orientation_violation_rate = N
  - verdict = PASS / WARN / FAIL
- **Decision:** [next action and rationale]
```

For Phase 2B runs, also add `mean_formation_error_m` to the scorecard block and change
the config class to `GGSwarmMarlFormationCfg`.

---

## References

- Cross-run scorecard: [`docs/status/run_history.md`](../status/run_history.md)
- Training workflow (launch + monitor): [`docs/ops/training_workflow.md`](training_workflow.md)
- Changelog: [`docs/status/changelog.md`](../status/changelog.md)
- Assessment script (single entry point): [`scripts/post_train_assess.py`](../../scripts/post_train_assess.py)
- `run.py assess` delegates to the above script; use it for quick CLI access
- Manual eval tool (standalone): [`scripts/eval_phase2.py`](../../scripts/eval_phase2.py)
- Convergence check (standalone): [`scripts/cloud/check_convergence.py`](../../scripts/cloud/check_convergence.py)
