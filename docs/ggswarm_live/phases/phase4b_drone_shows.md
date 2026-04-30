# Phase 4b: Drone Show Capability (income stream)

**Status:** Planned. Gates Phase 5+ hardware spend in practice.

**New capability:** small-venue drone show performance as a repeatable
revenue stream. Builds on Phase 4's formation library and Phase 2's
sim-to-real pipeline. Funds the hardware upgrades needed for Phases 5–7.

## Scope

- Music-synced choreography on top of Phase 4's formation library
- Staged LED / color changes synchronized with shape morphs
- Outdoor ops as the primary venue (indoor space and Crazyflie noise make
  indoor shows impractical beyond a few drones)
- Minimum viable show duration (3–5 minutes) with graceful failure modes
  (one drone drops, show continues)
- Safety stack: pre-flight checklist, geofencing, emergency land triggers,
  visible observer and pilot-in-command roles defined

## Skybrush integration

Phase 4b is the natural home for landing the
[Option B architecture](../architecture.md) end-to-end: Skybrush Studio
choreography → CSV export → companion-computer trajectory loader → RL
overlay → PX4 offboard mode. The RL overlay's value proposition is most
visible here (wind compensation, formation coherence) and also most
constrained (zero tolerance for show-killing failures).

## Regulatory infrastructure (first-class, not deferred)

- **Part 107 Remote Pilot Certificate** for the operator. Prerequisite for
  any paid outdoor flying. Cost ~$175 plus study time.
- **Waiver under 14 CFR § 107.35** ("operation of multiple small unmanned
  aircraft"). Required for one pilot to operate more than one drone
  simultaneously outdoors. Granted via FAA safety case; multiple small
  companies hold these (Sky Elements, Verge Aero, Pixis Drones). Months
  of paperwork and iteration expected.
- **Airspace authorizations** (LAANC for controlled airspace near airports;
  not needed in Class G rural areas).
- **Insurance**: commercial UAS liability policy, typically $1M to $2M coverage.

Note: private land ownership does not waive any of the above. FAA
jurisdiction runs from the ground up. Indoor ops are the only Part 107
exemption, and indoor space plus Crazyflie noise make that path
impractical for shows beyond a few drones.

## Purpose in the roadmap

**Revenue stream.** Small-scale shows fund the hardware upgrades (Holybro
outdoor airframes, Jetson Orin compute) that Phases 5 and 6 need, where
per-drone cost jumps 10x to 100x over Crazyflie.

## Milestone artifacts

- Part 107 certificate (individual milestone, ~1 month)
- § 107.35 waiver granted (paperwork milestone)
- First paid booking completed
- Recorded show video + social-media posts

## Staged entry plan

1. Get Part 107 certificate.
2. Start single-drone outdoor cinematic content for social media while
   the § 107.35 waiver application is in flight.
3. Multi-drone rehearsals on own or rented private land after waiver granted.
4. First small paid booking when reliability clears an internal bar.

## Risk hot-spots

- § 107.35 waiver timeline and grant likelihood
- Outdoor sim-to-real gap for the Crazyflie formation work (wind, GPS vs LPS, battery)
- Choreography authoring tooling ergonomics
- Reliability under live performance constraints — one crashed drone during a paid show costs the booking

## See Also

- [Architecture](../architecture.md)
- [Vision § Phase 4b](../vision.md)
