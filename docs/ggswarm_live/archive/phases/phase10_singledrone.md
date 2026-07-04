# Phase 10: Single-Drone Hardware Bring-Up

**Status:** Planned. First hardware phase. Resolves architectural decisions
before any multi-drone hardware work.

**New capability:** one Crazyflie 2.1 flies under offboard control with
Skybrush waypoint ingestion and all four failsafe layers
([architecture.md § 4](../architecture.md#4-failsafe-architecture)) verified
individually.

## Scope

1. **Stack decisions resolved.** Pegasus Simulator vs. custom PX4 integration;
   Crazyswarm2 (CRTP) vs. PX4 offboard mode. Decisions recorded here before
   any flying.
2. **Companion-computer setpoint stream.** Setpoint publisher running at ≥ 50 Hz.
   Trajectory loader reading Skybrush CSV, interpolating reference waypoint,
   3–5 waypoint lookahead buffer.
3. **All four failsafe layers verified individually.** Layer 1 (RL fallback)
   stubbed off but the bypass path works; Layer 2 (offboard timeout) triggers
   on stopped setpoint stream; Layer 3 (data link loss) triggers on GCS
   heartbeat loss; Layer 4 (kill switch) triggers on RC activation.
4. **Battery-sag and motor-saturation envelope measured.** Real-data
   collection that feeds Phase 6 sim disturbance-DR calibration and the
   Phase 12 recalibration loop back to Phase 2a.
5. **No RL policy yet.** Pure waypoint following at this phase. The RL
   overlay enters in Phase 13.

## Inputs from prior phase

- Phases 0–9 (sim block) complete and milestones met.
- Architectural-decisions list resolved here as a Phase 10 prerequisite.

## Hardware methodology

- Single Crazyflie 2.1 on a known-good battery, indoor controlled volume.
- LPS anchor scaffolding active (LPS is not removed until Phase 12).
- Companion computer: laptop initially; on-airframe option deferred to
  Phase 16.
- Pre-flight checklist documented; battery rotation procedure documented;
  crash-recovery SOP documented.

## Milestone artifact

Single-drone hover video with each failsafe layer triggered on camera in
sequence. Stack-decisions document committed. Pre-flight checklist + SOPs
committed. Video recorded with `--video_prefix p10-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Setpoint publication rate (steady) | ≥ 50 Hz |
| Layer 2 (offboard timeout) trigger latency | ≤ 1.5 s after stream stop |
| Layer 3 (data link loss) trigger latency | ≤ 12 s after GCS heartbeat loss |
| Layer 4 (kill switch) disarm latency | ≤ 250 ms after RC activation |
| Hover position drift (wind-still indoor) | ≤ 0.10 m over 60 s |
| Battery-sag voltage range observed | Documented from full → land trigger |

## FAA evidence produced

Foundational evidence for the [architecture.md § 6 safety
case](../architecture.md#6-faa-part-10735-waiver-alignment): each failsafe
layer has measured trigger latency on real hardware, not just sim.

## Risks

- Pegasus / Crazyswarm2 / PX4 stack-decision misjudgement → revisit in
  Phase 10; cost is phase rework, not a hardware-block restart.
- Battery sag breaks waypoint tracking before RL policy enters → voltage as
  explicit observation in Phase 12 retrain, or randomized thrust scaling.
- Companion-computer setpoint scheduling jitter → measure and document; if
  > 5% jitter at 50 Hz, escalate to a real-time scheduler before Phase 11.

## Decline list

- **Onboard companion computer in Phase 10** — declined; laptop companion
  is sufficient. Onboard compute is Phase 16.
- **Outdoor flying in Phase 10** — declined; indoor only until Phase 14b
  outdoor solo content.
- **Multi-drone in Phase 10** — declined; that is Phase 11.

## See Also

- [Phase 11 Anchored Multi-Drone Formation](phase11_anchored_formation.md)
- [Architecture § 3 Selected architecture](../architecture.md#3-selected-architecture-option-b-on-px4)
- [Architecture § 4 Failsafe Cascade](../architecture.md#4-failsafe-architecture)
- [Vision § Phase 10](../vision.md)
