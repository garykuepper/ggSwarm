# Phase 14: First Drone Show — The Major Milestone

**Status:** Planned. Sub-phased per the regulatory + revenue gates that
sequence the work.

**New capability:** small-venue paid drone show performance. This is the
program's headline milestone — everything in Phases 0–13 builds toward it,
and post-show phases (15–18) build on top of it.

## Sub-phase index

| Sub-phase | Title | Gate |
| :--- | :--- | :--- |
| 14a | [Part 107 Remote Pilot Certificate](phase14a_part107.md) | Certificate granted |
| 14b | [Single-Drone Outdoor Cinematic Content](phase14b_outdoor_solo_content.md) | Published outdoor content while § 107.35 waiver application is in flight |
| 14c | [Multi-Drone Rehearsals on Private Land](phase14c_multidrone_rehearsals.md) | Reliable multi-drone outdoor rehearsals post-waiver |
| 14d | [First Paid Booking](phase14d_first_paid_booking.md) | A paid show flown to completion |

## Sub-phase dependencies

```text
Phase 13 (Skybrush end-to-end on hardware)
       │
       ▼
      14a  Part 107 certificate
       │
       ▼
      14b  Single-drone outdoor content (waiver application in flight, parallel)
       │
       ▼
      14c  Multi-drone rehearsals (post-waiver)
       │
       ▼
      14d  First paid booking
       │
       ▼
   Post-show phases 15+ (high-level vision)
```

Strict sequence: each gate is a real-world prerequisite for the next.
Outdoor content (14b) can be authored and published while waiver paperwork
is in flight; multi-drone rehearsals (14c) cannot start until the waiver
is granted.

## Why a top-level numbered phase

Promoted from a separable revenue track. The drone show is *the program's
major milestone*; treating it as a numbered phase makes the sequencing
visible alongside engineering work, ensures regulatory gates get the same
status-tracking discipline as engineering gates, and aligns the docs with
the actual program shape (everything before is "make the show possible";
everything after is "expand beyond the show").

## Whole-Phase-14 milestone artifact

- Part 107 certificate held
- § 107.35 waiver granted
- Recorded paid-show video + social-media posts
- Repo release tagged `v3.0.0-first-show`
- Booking ledger entry: first paid show completed

## See Also

- [Phase 13 Skybrush End-to-End](phase13_skybrush_e2e.md) — the integration this phase consumes
- [Phase 4 Expressive Shape Library](phase4_shapes.md)
- [Phase 5 Animated Formations](phase5_animated.md) — choreography primitive library
- [Architecture](../architecture.md)
- [Vision § Phase 14](../vision.md)
