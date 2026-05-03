# Phase 14b: Single-Drone Outdoor Cinematic Content

**Status:** Planned. Depends on Phase 14a. § 107.35 waiver application is
in flight in parallel with this sub-phase.

**New capability:** social-media presence with outdoor drone footage,
single-drone choreography pieces flown legally under Part 107.

## Scope

1. **Single-drone outdoor flight under Part 107.** Crazyflie 2.1 or a
   step-up airframe (Holybro X500-class) that handles outdoor wind
   reasonably. Cinematic content as a deliverable, not just engineering
   validation.
2. **§ 107.35 waiver application.** Filed and iterated on with the FAA
   throughout this sub-phase. This is the regulatory long-pole; budget
   months.
3. **Social-media content cadence.** Single-drone outdoor pieces published
   on a regular cadence (the social-media distribution thesis is part of
   the program — see `MEMORY.md` post-capstone-mode entry). Each piece is
   both a content artifact and a flight-ops rehearsal.
4. **Insurance acquisition.** Commercial UAS liability policy (typically
   $1M–$2M coverage) in force before any paid work.
5. **LAANC / airspace authorization workflow.** Documented for whatever
   airspace classes the regular flight sites fall under.

## Inputs from prior phase

- Phase 14a Part 107 certificate
- Phase 13 single-drone Skybrush integration (provides the trajectory
  authoring stack used for the cinematic pieces)
- Phase 6 disturbance DR (informs which wind conditions are flyable)

## Methodology

- Outdoor flight sessions on private land or in unregulated Class G
  airspace where possible; LAANC for any controlled airspace.
- Pre-flight checklist + crash-recovery SOP from Phase 10 inherited.
- Skybrush Studio + Phase 13 RL overlay used to author each piece;
  single-drone is a constraint of the regulatory phase, not the engineering
  envelope.

## Milestone artifact

A published portfolio of single-drone outdoor pieces (≥ 5 pieces over the
sub-phase window). § 107.35 waiver application filed (acknowledged by FAA).
Insurance policy in force. Public social presence active.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Published outdoor pieces over sub-phase window | ≥ 5 |
| § 107.35 waiver application filed | Yes (acknowledgement received) |
| Insurance policy in force before first paid touchpoint | Yes |
| LAANC authorization workflow used | At least once (documented) |
| Crash rate per outdoor session | ≤ 0.05 (1 per 20 sessions) |

## FAA evidence produced

The § 107.35 waiver application itself draws on architecture.md § 6 and
the Phase 2/3/12/13 evidence pipeline. This sub-phase converts engineering
evidence into the legal artifact that unblocks Phase 14c.

## Risks

- § 107.35 waiver timeline drags or grant likelihood is lower than expected
  → continue 14b indefinitely; pursue parallel paths (study other waiver
  holders' approaches; engage a UAS attorney if needed).
- Outdoor wind exceeds Phase 6 DR envelope → fly only in calm conditions
  initially; extend DR envelope before pushing weather limits.
- Crash on private land destroys the only outdoor airframe → spare-airframe
  stock; rotate batteries; conservative flight envelope until a second
  airframe is in inventory.
- Social cadence becomes a distraction from engineering work → cap content
  production at one piece per N engineering days; do not let content drive
  scope.

## Decline list

- **Multi-drone outdoor flight in Phase 14b** — declined; § 107.35 waiver
  is the legal gate. Multi-drone is Phase 14c.
- **Paid flying in Phase 14b** — declined; that is Phase 14d. Solo content
  here is portfolio / brand building, not revenue.
- **Indoor content in Phase 14b** — declined; indoor work continues in the
  hardware-block phases for engineering validation, but the *content* this
  sub-phase produces is outdoor (visually distinct and legally
  significant).

## See Also

- [Phase 14a Part 107 Certificate](phase14a_part107.md)
- [Phase 14c Multi-Drone Rehearsals](phase14c_multidrone_rehearsals.md)
- [Phase 13 Skybrush End-to-End](phase13_skybrush_e2e.md)
- [Phase 6 Outdoor Disturbance DR](phase6_disturbance_dr.md)
- [Architecture § 6 Part 107.35 alignment](../architecture.md#6-faa-part-10735-waiver-alignment)
