# Phase 12: Anchor Removal + Decentralized Stack on Hardware

**Status:** Planned. Depends on Phase 11. The most consequential gate in
the hardware block.

**New capability:** the full Phase 2a/b/c/d decentralized + fault-tolerant
stack runs on real Crazyflies with no fixed positioning infrastructure.
Real UWB and real radio-loss measurements feed back to recalibrate the
Phase 2a noise model (and, indirectly, Phases 6 and 7 disturbance / obstacle
sim models).

## Scope

1. **LPS anchors removed.** Peer UWB ranging only.
2. **Phase 2a residual-test + multilateration recovery** active onboard.
3. **Phase 2b auction** running over the peer mesh; centralized greedy-nearest
   assignment retired.
4. **Phase 2c multi-dropout drill** executed on real hardware (controlled
   power-down of 1–2 drones mid-flight, verified recovery).
5. **Phase 2d versioned gossip + average consensus** carrying command state
   and centroid.
6. **Calibration loop back to sim phases.** Real UWB ranging data and real
   mesh-loss measurements feed back into Phase 2a's sim noise model
   (primary), and into Phase 6 disturbance DR + Phase 7 obstacle sim
   parameters where applicable. Policy retrained on the calibrated sim
   before the next Phase 12 session.

## Inputs from prior phase

- Phase 11 anchored sessions of peer-range logs and sim-to-real gap
  measurements
- Phase 2 (all four sub-phases) sim-validated with scorecards passing
- Phase 11 retrained Phase 1c checkpoint with Phase 11-measured DR

## Hardware methodology

- Same indoor volume; LPS anchors physically removed (or powered down) for
  scorecard captures.
- Optional LPS scaffolding may remain for *bring-up only* — a documented
  bring-up flight with anchors on, then anchors off for the milestone
  capture.
- Multi-dropout drill: scheduled controlled power-down of N drones at a
  pre-announced episode step; recovery measured against Phase 2c sim CDFs.
- Calibration loop is iterative: each session produces ranging / loss logs;
  Phase 2a noise model parameters updated; 2a (and downstream) re-trained;
  next Phase 12 session repeats with the updated policy.

## Milestone artifact

Multi-drone formation-hold video with no anchors visible, with a recorded
intentional dropout event mid-flight and demonstrated recovery. Calibration
report comparing Phase 2a sim noise model parameters before and after.
Video recorded with `--video_prefix p12-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Formation error (steady state, 3–5 drones, anchors off) | ≤ 1.5× Phase 11 anchored baseline |
| Multi-dropout (k=1) recovery time on hardware | Within 2× Phase 2c sim CDF |
| Multi-dropout (k=2) recovery time on hardware | Within 3× Phase 2c sim CDF |
| Auction convergence under real radio (median) | ≤ 100 sim-step-equivalent updates |
| Sim-noise-model recalibration delta (per Phase 2a parameter) | Documented; flag if any parameter shifts > 50% |
| Formation collapse rate over 30 min, anchors off | ≤ 0.05 |

## FAA evidence produced

Feeds the **most weighted rows** of the architecture.md § 6.1 evidence
pipeline: every safety-case claim about decentralized operation is backed
by Phase 12 data, not just sim. Recovery-time CDFs for multi-dropout
become the hardware-measured numbers that the waiver application cites.

## Risks

- Real peer-range outliers degrade auction convergence faster than the
  calibrated Phase 2a model predicted → outlier rejection layer in
  multilateration; revisit Phase 2a noise tail.
- Sim-noise model recalibration drives a large policy retrain → time cost
  is real; mitigation: keep the Phase 1c checkpoint as a warm-start,
  retrain only the head if possible.
- Multi-dropout drill destroys hardware → controlled power-down (telemetry
  command, not crash); spare-drone stock; one drill at a time.
- Auction does not converge under real loss → fall back to Phase 2b's E1
  stepping-stone (slot-pref logits) for show ops while Phase 2b auction is
  hardened.

## Decline list

- **Outdoor flying in Phase 12** — declined; Phase 14b solo outdoor content
  / Phase 15 outdoor hardware.
- **Skybrush integration in Phase 12** — declined; that is Phase 13.
- **N > 5 drones in Phase 12** — declined; scale is Phase 3 (sim) and
  Phase 14c rehearsals (hardware).

## See Also

- [Phase 11 Anchored Multi-Drone Formation](phase11_anchored_formation.md)
- [Phase 13 Skybrush End-to-End](phase13_skybrush_e2e.md)
- [Phase 2a Localization](phase2a_localization.md) — recalibration target
- [Phase 2b Slot Assignment](phase2b_assignment.md)
- [Phase 2c Fault Tolerance](phase2c_fault_tolerance.md)
- [Phase 2d Consensus + Dissemination](phase2d_consensus_dissemination.md)
- [Vision § Phase 12](../vision.md)
