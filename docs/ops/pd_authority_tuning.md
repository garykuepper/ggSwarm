# PD attitude authority and early-run diagnostics

This note supports tuning the inner-loop PD (`attitude_controller.py`) and
interpreting early training crashes when drones lose altitude immediately.

## 1. What limits “self-balancing”

- **`max_moment`:** Roll/pitch/yaw torques are clamped to ±`max_moment` (Nm) after the
  PD law. Large attitude errors can **saturate** the output; the vehicle then cannot
  correct fast enough.
- **`kp_att` / `kd_att`:** Higher `kp_att` demands more torque for the same error;
  saturation becomes more likely. `kd_att` adds damping on body rates.

Use the unit tests in `tests/unit/test_ggswarm_utils.py` (`TestAttitudeController`)
and `tests/unit/test_attitude_open_loop.py` as regression checks when changing gains.

## 2. Suggested sweep order (small, bounded)

After any change, run Rule 22 smoke (see `docs/ops/commands.md`).

1. **`max_moment`:** If TensorBoard shows `moment_saturated_frac` pegged near 1.0 for
   early steps, raise slightly (e.g. 0.03 → 0.04) and re-smoke.
2. **`kp_att`:** If saturation is rare but attitude oscillates, reduce `kp_att` or
   raise `kd_att` slightly.
3. **`thrust_to_weight`:** Only adjust after thrust telemetry shows the collective
   mapping is wrong relative to weight (see env cfg comments for neutral-thrust semantics).

Do **not** combine large changes; one knob per smoke avoids ambiguous results.

## 3. Action telemetry (first N env steps)

In `GGSwarmMarlEnvCfg` / `GGSwarmMarlHoverStabilityCfg`, set:

| Field | Purpose |
| :--- | :--- |
| `action_telemetry_max_env_steps` | If `> 0`, writes thrust/moment diagnostics into `extras["log"]` for TensorBoard for the first N **env** steps |

**CLI (local only):** `python scripts/run.py … train --action_telemetry_steps 200` forwards to
`train.py` and overrides the cfg for that process — no file edit; omit on GCE (default `0`).
See [`pd5_rule22_checklist.md`](pd5_rule22_checklist.md) § Short local run.

Logged keys include:

- `act_raw_thrust_*` — before env clamp (detects Gaussian samples outside `[-1, 1]`).
- `act_clamp_hit_frac` — fraction of elements where raw ≠ clamped.
- `thrust_val_mean` — mean mapped collective `(action0+1)/2` after the full action pipeline.
- `moment_pre_abs_max_mean` / `moment_saturated_frac` — PD authority (requires
  `action_telemetry_max_env_steps > 0` so `_moment_pre_clamp` is allocated).

Set back to `0` for long production runs to avoid extra work and log volume.

## 4. Full sim check (optional)

For altitude drift under **zero policy output**, use Isaac headless smoke / play with a
deterministic zero-action policy when available, or extend `tests/test_env_smoke.py`
under `@pytest.mark.isaacsim_ci`. Torch-only open-loop tests are in
`tests/unit/test_attitude_open_loop.py`.
