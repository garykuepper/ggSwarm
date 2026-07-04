# Phase 7: Obstacle-Aware Formation Control (sim only)

**Status:** Planned. Sim phase. Pulled forward from the old "Phase 6
onboard + obstacles" framing under the *exhaust sim before hardware*
principle. Can proceed in parallel with Phases 3, 4, 5, 6, 8, 9.

**New capability:** the policy composes formation control with reactive
obstacle avoidance in unknown simulated environments, before any onboard
perception hardware enters the picture (Phase 17).

## Scope

1. **CBF reactive avoidance, reactivated.** Capstone CBF module
   (`cbf.py`) is reactivated and tuned for joint formation + obstacle
   reward. Backlog C1.
2. **Learned obstacle avoidance revisit.** Backlog C2 — richer obstacle
   encodings (occupancy grid, ray-cast, relative-obstacle graph edges) vs.
   the capstone-era column experiment.
3. **Unknown-environment curriculum.** Procedurally generated forest /
   urban-canyon / cluttered scenes; held-out test set never seen during
   training.
4. **Composition with formation control.** CBF treats obstacles as virtual
   agents (capstone Phase 4 thread); reward shaping balances goal
   deflection with formation coherence.
5. **Urban canyon scenario.** Backlog C3 — tall rectangular walls, narrow
   passages, vertical maneuver pressure; complementary to forest.

## Inputs from prior phase

- Phase 1c shared-scene MAPPO + GATv2 policy
- Phase 2 decentralized + fault-tolerant stack (obstacle scenarios stress
  it)
- Capstone `cbf.py` (retained but disabled in capstone)

## Sim methodology

- Obstacle scenes as a registry of procedural generators (forest, canyon,
  cluttered, mixed).
- CBF QP solve in the sim training loop (no onboard latency budget yet —
  that is Phase 8).
- Curriculum: easy → hard environment density; held-out scenes scored.

## Milestone artifact

Sim demo: 8-drone swarm flying through a held-out forest + a held-out urban
canyon, holding formation. Obstacle-encoding ablation plot (column /
occupancy / ray-cast / graph-edges). Video recorded with
`--video_prefix p7-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Collision rate, training-distribution scenes | 0 per 1k episodes |
| Collision rate, held-out scenes | ≤ 0.01 per 1k episodes |
| Formation error during obstacle traversal (vs. obstacle-free baseline) | ≤ 1.50× |
| Goal-arrival success rate, forest curriculum | ≥ 0.95 |
| Goal-arrival success rate, urban-canyon curriculum | ≥ 0.85 |
| CBF QP solve time (sim, single drone) | Documented; flag if > 5 ms (informs Phase 8 budget) |

## FAA evidence produced

Indirect — establishes the obstacle envelope and the composability claim
(formation + obstacle avoidance simultaneously). Phase 17 provides the
real-perception evidence.

## Risks

- CBF QP solve time too high to compose with policy inference → reduce
  obstacle count per drone; switch to differentiable barrier formulation;
  defer the QP to Phase 8 distillation.
- Learned obstacle encoding does not improve over capstone column
  experiment → keep CBF as the load-bearing avoidance, treat learned
  encoding as an optional enhancement.
- Curriculum collapses — policy ignores formation under obstacle pressure
  → reward-shaping rebalance; explicit formation-coherence floor.

## Decline list

- **Real perception (stereo / lidar / VIO) in Phase 7** — declined;
  Phase 17 (post-show vision).
- **Onboard-latency-budgeted inference** — declined; Phase 8 covers
  distillation + latency profiling.
- **Outdoor obstacle scenes** — declined; Phase 6 covers outdoor
  disturbance separately. Joint sim-only outdoor + obstacle is a Phase 17
  pre-step at most.

## See Also

- [Phase 2 Decentralized + Fault-Tolerant Stack](phase2_decentralized.md)
- [Phase 8 Onboard Inference Profiling](phase8_onboard_distill.md) — consumes the QP solve-time bound
- [Phase 17 Obstacle-Aware Hardware](phase17_obstacle_hw.md) — post-show
- [Vision § Phase 7](../vision.md)
