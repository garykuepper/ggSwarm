# Phase 9: Multi-Platform Domain Randomization (sim only)

**Status:** Planned. Sim phase. Pulled forward from the old "Phase 7
hardware-agnostic" stretch framing under the *exhaust sim before hardware*
principle. Can proceed in parallel with Phases 3–8.

**New capability:** the policy is trained against an extended airframe
domain (mass, inertia, thrust-curve, size ranges spanning Crazyflie 2.1
through Holybro X500-class) so that *one checkpoint* can be deployed to
multiple real platforms in Phase 18.

## Scope

1. **Extended airframe-family DR.** Mass 0.030 kg → 1.5 kg; inertia
   tensors per family; thrust-curve families; rotor-arm length families.
2. **Per-airframe simulator presets.** Isaac Lab scene configurations for
   Crazyflie 2.1, Crazyflie Bolt, Holybro X500 v2, ModalAI Seeker (each
   with documented mass / inertia / motor model from vendor data).
3. **Scale-aware policy.** Policy conditioned on airframe-parameter
   observations (mass, inertia diagonal, max thrust) so that the same
   weights generalize.
4. **Generalization measurement.** Held-out airframe parameters (within
   the DR envelope) scored without retraining.
5. **Mission-conditioning preliminary.** Optional task-conditioning hook
   (formation / area-coverage / escort / navigation) with formation-only
   trained at this phase; other missions deferred to Phase 18.

## Inputs from prior phase

- Phase 1c shared-scene MAPPO + GATv2 policy
- Phase 6 disturbance DR (interacts with airframe choice)
- Vendor specifications for the candidate airframe family

## Sim methodology

- Airframe sampled per episode from the DR envelope.
- Curriculum: start in the Crazyflie-tight envelope, slowly anneal toward
  the Holybro-wide envelope.
- Evaluation harness reuses Phase 1a replay-gate machinery; per-airframe
  rollouts scored against per-airframe baselines.

## Milestone artifact

Sim demo: same policy checkpoint flying formations on simulated Crazyflie
2.1 and simulated Holybro X500 v2, no retraining. Per-airframe
generalization plot. Video recorded with `--video_prefix p9-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Formation error on training-distribution airframes (median) | Within 1.20× single-airframe baseline |
| Formation error on held-out-airframe parameters | Within 1.50× training-distribution median |
| Number of airframe families in training distribution | ≥ 4 (Crazyflie 2.1, Crazyflie Bolt, Holybro X500, one synthetic) |
| Conditioning observation dimension | ≤ 16 |
| Performance on the Phase 1c canonical Crazyflie scenario | Within 1.10× pre-Phase-9 checkpoint |

## FAA evidence produced

None directly — this phase's evidence is consumed by Phase 18 (hardware
validation of the cross-platform claim). Phase 9 establishes that the
policy *exists* before any hardware deployment.

## Risks

- DR envelope too wide → policy regresses on Crazyflie. Mitigation:
  curriculum schedule from narrow to wide; floor on Crazyflie performance.
- Per-airframe physics fidelity uneven → document each preset's source
  (vendor data vs. measured); flag uncertainty in scorecard.
- Mission-conditioning hook destabilizes formation training → keep
  mission-conditioning behind a feature flag; Phase 18 scope.

## Decline list

- **Real cross-platform deployment** — declined; Phase 18 (post-show vision).
- **Heterogeneous-swarm training** (mixed airframes in one swarm) —
  declined; backlog item B4. Phase 9 trains per-airframe; B4 trains across
  mixed populations.
- **Non-quadrotor platforms** — declined; out of program scope (vision §5
  out-of-scope list).

## See Also

- [Phase 6 Outdoor Disturbance DR](phase6_disturbance_dr.md)
- [Phase 18 Multi-Platform Hardware](phase18_multiplatform_hw.md) — post-show
- [Vision § Phase 9](../vision.md)
