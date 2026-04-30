# Phase 1: Shared-Scene Multi-Drone Training (sim only)

**Status:** Planned. No timeline.

**New capability:** real inter-drone aerodynamics (downwash, wake
turbulence) enters the training distribution.

## Scope

- 8 drones in one shared Isaac physics scene per env (replaces 8 isolated envs)
- Still perfect state, still centrally precomputed slots — isolate the aero variable
- Downwash modeling: analytic force model ported from gym-pybullet-drones
  ([Panerati 2021](../references.md#panerati2021)), or learned residual on
  the relative-pose graph following Neural-Swarm2
  ([Shi 2022](../references.md#shi2022)), or both as an ablation
- Retrain shared GATv2 policy; compare against Phase 0 checkpoint

## Keeps simple

Perfect GPS, no peer ranging, no assignment, no outdoor, no obstacles.

## Backlog items folded into this phase

- **B1** Wind + downwash modeling ([backlog](../backlog.md#b1))
- **B3** Stacked-spawn downwash artifact ([backlog](../backlog.md#b3))
- **A1** GATv2 edge features (`edge_dim=3`) — natural to land while retraining ([backlog](../backlog.md#a1))
- **A2** Cloud / boid retraining ([backlog](../backlog.md#a2))

## Milestone artifact

Side-by-side comparison video (shared-scene vs isolated-scene), updated
checkpoint in the repo, short social-media write-up on what changed when
aero entered the loop.

## Risk hot-spots

- Shared-scene training too slow on local 3070 → debug locally, sweep on cloud
- Downwash model destabilizes existing policy without curriculum

## See Also

- [Vision § Phase 1](../vision.md)
- [Capstone Phase 0](phase0_capstone_baseline.md)
