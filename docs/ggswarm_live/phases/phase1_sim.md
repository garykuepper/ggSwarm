# Phase 1: Sim — Decentralization + Downwash Physics

**Status:** MAPPO shared-scene groundwork complete (2026-05-01). Two goals
remain, no timeline, no fixed order.

This phase replaces the old multi-sub-phase decentralization stack
(2a–2d) and the separate downwash sub-phase (1b/1c) with two flat goals
pursued in whatever order makes sense as work proceeds. Detailed prior
design work on both (auction-based assignment, gossip/consensus,
residual-test fault detection, analytic vs. learned downwash) is
preserved in [`../archive/`](../archive/) and can be pulled back in if
useful — it isn't being thrown away, just no longer a committed
sequence of numbered phases.

## Goal A: Proper decentralization

No central slot assignment, no anchors, no single point of failure.
Each drone reasons from peer information only:

- Peer-to-peer localization (no ground-truth state, no fixed anchors)
  (in progress — sim-side stack implemented, hardware-analog validation
  pending; see decentralization_plan.md)
- Distributed slot/formation assignment negotiated between drones
- Fault tolerance to single and multi-drone dropout without a central coordinator
- Command and state propagate peer-to-peer (gossip-style), not from a hub

Prior detailed design for this goal (Bertsekas auction, versioned
gossip/CRDT, residual-test localization fault detection, average
consensus) lives in [`../archive/phases/phase2_decentralized.md`](../archive/phases/phase2_decentralized.md)
and its 2a–2d sub-docs, plus [`../archive/consensus_mechanisms.md`](../archive/consensus_mechanisms.md)
for the "why not blockchain / not Raft" rationale.

The peer-to-peer localization bullet now has a committed detailed
plan — see [`../decentralization_plan.md`](../decentralization_plan.md)
for the staged implementation, scorecard, and the takeoff-frame
odometry-anchored gauge decision.

## Goal B: Downwash / aero physics fidelity

Real inter-drone aerodynamics (downwash, wake turbulence) in the
training distribution, so the policy isn't learning against
unrealistically clean physics:

- Analytic downwash force model (ported from gym-pybullet-drones,
  [Panerati 2021](../references.md#panerati2021)) and/or a learned
  residual on the relative-pose graph
  ([Shi 2022](../references.md#shi2022))
- Retrain shared GATv2 policy against the richer physics; compare to
  the Phase 0 capstone checkpoint

## What's already done (2026-05-01)

The shared-scene groundwork for both goals — 8 drones in one PhysX
scene per env instead of 8 isolated envs — is built and validated:

- Pivoted from a planned in-place facade refactor to `DirectMARLEnv` +
  SKRL `MAPPO` (shared GNN actor + shared centralized critic) after
  discovering Isaac Lab's scene cloner doesn't support multiple sibling
  Articulations per env, and `DirectRLEnv.num_envs` is hardcoded as the
  gym vec dim.
- Full env logic ported (formation/cloud reward, MINCO smoothing, CBF
  shield, dropout handling, KNN obs, forest deflection, collision
  detection). Throughput swept to `num_envs=512` on local 3070
  (near-linear scaling, no bottleneck found).
- Replay gate (`scripts/skrl/replay_gate.py`) validated against the
  capstone checkpoint: mean formation error preserved, std broadens —
  expected, since shared-scene wake coupling is exactly the new
  variable this phase introduces.
- First from-scratch MAPPO smoke run (500 iters): architecture trains
  end-to-end (reward −907.7 → −43.2), but formation not yet converged
  (drones cluster near centroid rather than spreading to slots — likely
  undertrained, not broken). Longer runs / warm-start from capstone
  weights / higher formation reward scale are the next things to try.

Tag: `phase1a-shared-scene-mappo`. Branch: `phase1a-shared-scene`. Full
narrative in [changelog 2026-05-01](../status/changelog.md).

## Keeps simple

No outdoor, no obstacles, no shape library beyond the capstone
baseline, no onboard-inference constraints yet — see
[`../backlog.md`](../backlog.md) for capabilities deliberately deferred
out of this phase.

## Risk hot-spots (carried from the pre-restructure phase doc)

- Shared-scene training too slow on local 3070 → debug locally, sweep on cloud.
- Downwash model destabilizes the existing policy without a curriculum.

## Milestone artifact

Recorded demo of the decentralized, downwash-aware policy running in
the shared-scene sim with anchors/central assignment fully removed,
plus an updated checkpoint in the repo.

## See Also

- [Vision](../vision.md)
- [Capstone Phase 0](phase0_capstone_baseline.md)
- [Phase 2: Hardware](phase2_hardware.md)
