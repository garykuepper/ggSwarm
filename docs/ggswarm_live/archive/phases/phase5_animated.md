# Phase 5: Animated / Time-Varying Formations (sim only)

**Status:** Planned. Sim phase. Soft-depends on Phase 4 (interesting morphs
need interesting shapes); otherwise can proceed in parallel with Phases 3
and 4.

**New capability:** the policy executes time-varying formations — morphing
between shapes, rotating formations, simple flocking — without losing
formation coherence during the transition.

## Scope

1. **Morph between two shapes mid-flight.** Per-drone slot mapping from
   shape A → shape B; trajectory blending; reward shaping for smooth
   transition.
2. **Rotating formations.** Continuous rotation of the formation around
   the swarm centroid; constant slot assignment, time-varying slot pose.
3. **Temporal reward structure.** Reward is a function of trajectory over
   a window, not just instantaneous position; addresses FR-4 from
   `vision.md` § 2.
4. **Choreography primitive library.** Morph / rotate / hold / translate as
   composable building blocks for the Phase 14 drone-show choreography.

## Inputs from prior phase

- Phase 4 shape library (provides the source / target shapes for morphs)
- Phase 2 sim-validated policy

## Sim methodology

- Morph slot-mapping: solved as an optimal-transport / Hungarian step
  *offline* (during choreography authoring), encoded into the per-drone
  trajectory; the swarm itself does not solve assignment mid-morph.
- Reward shaping: penalty for formation collapse during transition;
  reward for smooth jerk-bounded slot trajectory.
- Curriculum: start with hold → translate → simple morph (N-gon to same
  N-gon scaled), then morph between different shapes, then rotation, then
  composed primitives.

## Milestone artifact

Video of a smooth morph between two shapes (e.g., circle → 5-letter word)
mid-flight without formation collapse, plus a rotating formation hold.
Choreography primitive library committed. Video recorded with
`--video_prefix p5-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Maximum formation error during morph (peak) | ≤ 2× steady-state baseline |
| Settling time at end of morph (back to steady state) | ≤ 4 s |
| Per-drone jerk during morph (95th percentile) | Bounded by hardware envelope (TBD from Phase 10 battery / motor envelope) |
| Rotation rate range stable | 0.05 → 0.50 rad/s without collapse |
| Choreography primitive library size at milestone | ≥ 4 primitives (morph, rotate, hold, translate) |

## FAA evidence produced

Indirect. Establishes the *temporal* operating envelope (max morph rate,
max rotation rate) that the safety case can claim. Important input to
Phase 14 drone-show choreography limits.

## Risks

- Morph triggers formation collapse → tighter jerk bound; longer transition
  time; restrict morph rate per shape.
- Temporal reward structure destabilizes the existing policy → start with a
  small temporal window; ablation against instantaneous-reward baseline.
- Optimal-transport slot mapping produces crossing trajectories →
  collision avoidance during the transition (existing CBF inherited from
  capstone, reactivate if needed).

## Decline list

- **Online morph slot reassignment** — declined; offline solve is
  sufficient and avoids the auction-during-morph stability question.
- **Truly emergent flocking (no choreography)** — declined; out of scope
  for the drone-show product. Backlog item A2 (cloud / boid retraining)
  is preserved for a separate one-off demo.
- **Music sync in Phase 5** — declined; that is Phase 14.

## See Also

- [Phase 3 Scale](phase3_scale.md)
- [Phase 4 Expressive Shape Library](phase4_shapes.md)
- [Phase 14 First Drone Show](phase14_drone_show.md) — consumes the primitive library
- [Vision § Phase 5](../vision.md)
