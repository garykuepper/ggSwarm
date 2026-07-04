# Phase 3: Scale (5 → 10 → 20+ drones, sim only)

**Status:** Planned. Sim phase. Independent of Phases 4 and 5; can proceed
in parallel.

**New capability:** the policy + decentralized stack scales to 20+ drones in
simulation without retraining beyond curriculum extension.

## Scope

1. **Curriculum extension over swarm size N.** Train with N drawn from
   {5, 8, 10, 16, 20, 24} per the existing dynamic-spawn-radius mechanism.
2. **Auction convergence retest at N ≥ 20.** Phase 2b's risk register
   flagged that auction does not necessarily scale; this is the phase
   that measures it. If convergence collapses, fall back to hierarchical
   clustering or consensus-based bundle algorithm before scaling further.
3. **Crazyflie density / downwash limits documented.** At what spacing does
   downwash dominate? At what swarm size does formation density force
   spacing that becomes show-impractical? Documented per shape from the
   existing shape library.
4. **Onboard inference budget retest.** GATv2 KNN observation cost is
   bounded by K, but per-drone gossip / consensus bandwidth grows with N.
   Document the bandwidth ceiling.

## Inputs from prior phase

- Phase 2 sim-validated decentralized + fault-tolerant stack
- Phase 2b auction convergence baseline at N ≤ 8

## Sim methodology

- Curriculum schedule: N sampled per episode from a slowly-widening
  distribution, anneal toward 24.
- Aerial Gym ([Kulkarni 2023](../references.md#kulkarni2023)) confirms
  simulator throughput at thousands of multirotors; the bottleneck is
  expected to be auction convergence and policy generalization, not sim.
- Downwash modeling from Phase 1b stays in the loop — high-N close-spaced
  formations are exactly where it matters.

## Milestone artifact

20-drone formation-hold video in sim. Auction convergence-vs-N plot.
Density / spacing limit table per shape. Video recorded with
`--video_prefix p3-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Formation error (steady state, N=20) | ≤ 1.50× N=8 baseline |
| Auction convergence time (N=20, healthy comms) | ≤ 5× N=8 baseline |
| Auction convergence time (N=20, 30% loss) | ≤ 10× N=8 baseline |
| Formation collapse rate, N=20 | ≤ 0.05 per 1k episodes |
| Policy inference latency per drone (sim) | Unchanged from N=8 baseline |
| Per-drone gossip bandwidth (steady state) | Documented; flag if > 10 KB/s |

## FAA evidence produced

Indirect — establishes the swarm-size envelope downstream hardware ops
(Phases 11–15) will operate within. Not a primary evidence source for the
safety case behavior tables, but the bandwidth ceiling and density limits
are operating-envelope inputs.

## Risks

- Auction does not scale → hierarchical clustering / consensus-based
  bundle algorithm; documented in Phase 2b decline list as a Phase 3
  follow-up.
- Downwash dominates at tight spacing → enforce minimum spacing constraint
  in the parametric shape generator (Phase 4 reads this limit).
- Curriculum destabilizes existing checkpoint → freeze policy core, train a
  curriculum-aware adapter, or retain Phase 2 checkpoint as the production
  drop.

## Decline list

- **Heterogeneous agents in Phase 3** — declined; backlog item B4 is partial
  fold-in only if the curriculum motivates it. Default: pure homogeneous
  Crazyflie.
- **Outdoor scale in Phase 3** — declined; Phase 6 covers outdoor disturbance
  DR in sim, Phase 15 covers real outdoor.

## See Also

- [Phase 4 Expressive Shape Library](phase4_shapes.md)
- [Phase 5 Animated Formations](phase5_animated.md)
- [Phase 2b Slot Assignment](phase2b_assignment.md) — auction-scale follow-up
- [Phase 11 Anchored Multi-Drone Formation](phase11_anchored_formation.md) — first hardware contact
- [Vision § Phase 3](../vision.md)
