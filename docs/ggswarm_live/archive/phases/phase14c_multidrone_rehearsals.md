# Phase 14c: Multi-Drone Rehearsals on Private Land

**Status:** Planned. Depends on Phase 14b. Cannot start until § 107.35
waiver is granted.

**New capability:** multi-drone outdoor choreography flown reliably under
the § 107.35 waiver, on owned or rented private land, before any paid
booking.

## Scope

1. **Multi-drone outdoor flight under § 107.35 waiver.** 3–5 drones to
   start; scale toward 10 only as reliability holds.
2. **Full-show rehearsals.** Music sync + LED / color choreography +
   3–5 minute show duration with the Phase 5 animated formation library
   and the Phase 13 Skybrush + RL overlay stack.
3. **Graceful failure-mode rehearsal.** Intentional drone-drop scenarios
   on rehearsal flights to confirm "one drone drops, show continues"
   under outdoor conditions (Phase 12 sim CDFs + Phase 6 sim DR
   recalibrated against rehearsal data).
4. **Reliability bar definition.** A documented internal bar (success
   rate over N rehearsals, formation-error envelope, crash rate) that
   gates Phase 14d entry.
5. **Choreography authoring tooling.** Skybrush Studio workflow
   ergonomics, LED scripting integration, music-sync alignment;
   tooling improvements as discovered.

## Inputs from prior phase

- Phase 14b § 107.35 waiver granted; insurance policy in force
- Phase 5 choreography primitive library
- Phase 13 Skybrush + RL overlay integration
- Phase 12 multi-dropout recovery CDFs
- Phase 6 disturbance DR (calibrated against any Phase 14b outdoor logs)

## Methodology

- Owned or rented private land within driving range; LAANC for any
  controlled airspace overlap.
- Pre-flight checklist + crash-recovery SOP + RPIC + visible-observer
  roles per the architecture.md § 6 safety case.
- Rehearsal cadence: weekly to start; success rate tracked across
  rehearsals.
- Each rehearsal records: full telemetry, mesh logs, telemetry-commanded
  dropout drills (where scheduled), audio + video for show-quality
  review.

## Milestone artifact

Recorded rehearsal show, multi-drone, full music-sync + LEDs + 3–5 minute
duration, graceful failure on at least one rehearsal scenario. Reliability
bar met across N rehearsals (N TBD per the scorecard). Video recorded with
`--video_prefix p14c-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Rehearsal show completion rate (no formation collapse, no crash) | ≥ 0.90 over last 10 rehearsals |
| Show duration (multi-drone, full choreography) | ≥ 3 minutes |
| Multi-dropout recovery on outdoor rehearsal (k=1) | Within 1.5× Phase 12 CDF |
| Audience-perceived show quality (subjective panel) | Clearly "show-shaped" |
| Crash rate per rehearsal | ≤ 0.05 |

## FAA evidence produced

Real-flight evidence that closes the loop on the § 107.35 waiver's safety
claims. Documented rehearsal logs become reference data for future waiver
amendments and for any rehearsals that scale beyond the current envelope.

## Risks

- Reliability bar repeatedly missed → expand the rehearsal window;
  diagnose the failure modes; revisit Phases 6 / 12 calibration before
  scheduling Phase 14d.
- Outdoor sim-to-real gap re-opens at multi-drone scale (was small at
  single-drone in 14b) → multi-drone-specific DR retrain; possibly extend
  Phase 6 DR envelope.
- Choreography authoring takes much longer than engineering predicts →
  budget time honestly; don't promise paid-show timeline based on
  rehearsal-tooling assumptions.
- Hardware crashes destroy multiple airframes → conservative scaling
  (start at 3 drones, prove reliability before going to 5; same again
  before 10); spare stock; budget for replacements.

## Decline list

- **Paid bookings in Phase 14c** — declined; that is Phase 14d. Rehearsals
  here are non-revenue.
- **Show duration > 5 minutes** — declined for the milestone capture;
  longer shows are post-Phase-14d once reliability proves out.
- **Outdoor obstacle scenarios** — declined; Phase 17 (post-show vision).

## See Also

- [Phase 14b Outdoor Solo Content](phase14b_outdoor_solo_content.md)
- [Phase 14d First Paid Booking](phase14d_first_paid_booking.md)
- [Phase 12 Decentralized Hardware](phase12_decentralized_hw.md)
- [Phase 13 Skybrush End-to-End](phase13_skybrush_e2e.md)
- [Phase 5 Animated Formations](phase5_animated.md)
- [Architecture § 6 Part 107.35 alignment](../architecture.md#6-faa-part-10735-waiver-alignment)
