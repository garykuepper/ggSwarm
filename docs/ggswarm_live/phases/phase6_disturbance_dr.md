# Phase 6: Outdoor Disturbance Domain Randomization (sim only)

**Status:** Planned. Sim phase. Pulled forward from the old "Phase 5
outdoor" framing under the *exhaust sim before hardware* principle. Can
proceed in parallel with Phases 3, 4, 5, 7, 8, 9.

**New capability:** the policy is trained against simulated outdoor
disturbances (wind, gust, thermal, sensor multipath under outdoor
conditions) before any real outdoor flight is attempted.

## Scope

1. **Wind domain randomization.** Steady wind 0–5 m/s + gust models
   (Dryden-style turbulence; Karman; or a learned residual on real-flight
   logs once Phase 11+ produces them).
2. **Thermal-effect modeling.** Vertical air-column updrafts / downdrafts
   per the operating-altitude envelope.
3. **Outdoor-multipath sensor noise.** Inflated UWB noise tail for outdoor
   conditions (different multipath profile vs. indoor); GPS *intentionally
   off* to preserve the GPS-denied invariant.
4. **Multi-simultaneous fault drills under disturbance.** Phase 2c fault
   catalog re-run with wind + thermal active. Headline metric: does the
   recovery-time CDF degrade gracefully or cliff?
5. **Curriculum schedule.** Anneal disturbance intensity over training;
   freeze the Phase 1c / 2 checkpoint as warm-start.

## Inputs from prior phase

- Phase 1b downwash modeling (interacts with wind near the ground)
- Phase 2c fault catalog (re-run under disturbance here)
- Phase 5 animated formations (morph-during-gust scenarios)

## Sim methodology

- Wind models implemented as a force perturbation in the existing physics
  step.
- Disturbance intensity sampled per episode from a curriculum schedule.
- Evaluation harness reuses the Phase 2c Monte Carlo machinery, just with
  the disturbance dimension added.

## Milestone artifact

Sim demo: 8-drone formation hold under 3 m/s steady + 5 m/s gust, with a
multi-dropout drill mid-disturbance. Wind-vs-formation-error scorecard
plot. Video recorded with `--video_prefix p6-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Formation error vs. wind speed (steady, 0–5 m/s) | Documented; flag if RMSE > 2× still-air baseline at 5 m/s |
| Formation collapse rate under gust + multi-dropout | ≤ 0.05 per 1k episodes |
| Phase 2c recovery-time CDF degradation under disturbance | ≤ 1.5× still-air CDF |
| Wind-handling generalization to held-out wind direction | Within 1.30× training distribution |

## FAA evidence produced

Indirect — establishes the disturbance envelope the safety case can claim
the swarm trained against. Real disturbance numbers come from Phase 15;
Phase 6 sets the *bound* the safety case relies on.

## Risks

- Wind model under-represents real outdoor turbulence → calibrate against
  Phase 11+ logs once available; treat Phase 6 numbers as upper-bound for
  pre-hardware claims.
- Disturbance + multi-dropout interaction destabilizes policy →
  curriculum schedule with both dimensions slowly increased; ablation.
- Wind interacts non-trivially with downwash (Phase 1b) at low altitude →
  joint scenario in Phase 6's evaluation harness.

## Decline list

- **Real wind data collection in Phase 6** — declined; that requires
  outdoor flight (Phase 14b solo content / Phase 15). Phase 6 closes only
  what sim alone can close.
- **Outdoor-specific airframe DR** — declined; that is Phase 9
  (multi-platform DR).

## See Also

- [Phase 1 Shared-Scene Sim](phase1_shared_scene_sim.md)
- [Phase 2c Fault Tolerance](phase2c_fault_tolerance.md)
- [Phase 15 Outdoor Hardware](phase15_outdoor_hw.md) — calibration target
- [Vision § Phase 6](../vision.md)
