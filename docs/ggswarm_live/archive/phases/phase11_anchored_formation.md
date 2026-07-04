# Phase 11: Anchored Multi-Drone Formation (LPS scaffolding)

**Status:** Planned. Depends on Phase 10.

**New capability:** the Phase 1c shared-scene policy flies a formation on
3–5 real Crazyflies with LPS anchors providing absolute positioning as a
safety scaffold.

## Scope

1. **3–5 Crazyflies in shared indoor volume** under LPS anchor positioning.
2. **Phase 1c MAPPO + GATv2 policy** loaded onboard the companion-computer
   for each drone (one companion per drone, or one shared host with per-drone
   processes — decision recorded in Phase 10's stack decisions).
3. **Centralized greedy-nearest slot assignment retained** at this stage —
   the decentralized auction (Phase 2b) does not enter until Phase 12. This
   is the *one variable at a time* rule: we are testing the policy on real
   hardware, not the assignment stack.
4. **LPS anchors active** — peer ranging is recorded in parallel for the
   Phase 12 calibration loop, but is not yet driving control.
5. **Centralized command injection** from a laptop GCS; no gossip
   dissemination yet (Phase 2d enters in Phase 12).

## Inputs from prior phase

- Phase 10 stack decisions, single-drone failsafe verification, battery /
  motor envelope measurements
- Phase 1c shared-scene MAPPO + GATv2 checkpoint

## Hardware methodology

- Same indoor volume as Phase 10; LPS anchor calibration verified before
  each session.
- Domain randomization in the *training distribution* (Phases 1, 6, 9) is
  what closes the sim-to-real gap; the gap is *measured* here, not closed.
- Each session records: ground-truth trajectories (LPS), peer ranges (raw
  UWB packets), commanded vs. achieved formation, battery telemetry.

## Milestone artifact

Multi-drone formation-hold video with 3–5 Crazyflies under LPS, Phase 1c
policy in control. Sim-to-real gap report (formation error vs. sim
baseline). Video recorded with `--video_prefix p11-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Formation error (steady state, 3-drone) | Documented; targets vs. sim baseline TBD after first session |
| Formation error (steady state, 5-drone) | Documented; targets TBD |
| Sim-to-real gap (real RMSE / sim RMSE) | ≤ 3.0× initial; tighten after retrain |
| Formation collapse rate over 30 min | 0 |
| Peer-range data captured per session | Hours of ranging logs sufficient to calibrate Phase 2a noise model |

## FAA evidence produced

Feeds **architecture.md § 6.1** evidence pipeline as the "real-hardware
calibrated formation control" baseline. Sim-to-real gap measurement
documents what the safety case can claim about Phase 2 sim numbers.

## Risks

- Sim-to-real gap larger than expected → retrain Phase 1c with broader
  domain randomization (mass, motor response, sensor latency); retain LPS
  scaffolding through Phase 12 retrain cycles.
- LPS anchor calibration drift session-to-session → document calibration
  procedure; recalibrate per session.
- Multi-drone CRTP / radio interference at small N → document the limit;
  factor into Phase 12 entry.

## Decline list

- **Anchor removal in Phase 11** — declined; that is the explicit Phase 12
  scope.
- **Decentralized assignment in Phase 11** — declined; isolate the policy
  variable.
- **Skybrush integration in Phase 11** — declined; that is Phase 13.

## See Also

- [Phase 10 Single-Drone Bring-Up](phase10_singledrone.md)
- [Phase 12 Anchor Removal + Decentralized Stack](phase12_decentralized_hw.md)
- [Vision § Phase 11](../vision.md)
