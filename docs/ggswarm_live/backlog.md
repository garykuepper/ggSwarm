# ggSwarm Live Backlog

Loose, unordered ideas that aren't part of the active plan
([Phase 1: Sim](phases/phase1_sim.md), [Phase 2: Hardware](phases/phase2_hardware.md)).
No phase mapping, no effort/impact scoring — picked up only if one
turns out to matter. Detailed prior design for several of these
(shape generation, morphing, obstacle encodings, scale sweeps) is
preserved in [`archive/backlog_detailed.md`](archive/backlog_detailed.md).

- **Expressive shape library** — parametric shapes beyond the capstone
  baseline (N-gons, alphanumeric glyphs, arbitrary uploaded point clouds)
- **Animated / time-varying formations** — morphing between shapes,
  rotating formations
- **Obstacle-aware formation control** — CBF reactivation, learned
  obstacle encodings, urban-canyon-style scenarios
- **Scale beyond 20 drones** — retest auction/assignment convergence,
  edge-sparsity tuning at higher N
- **Onboard inference profiling + distillation** — fit the policy to
  real onboard compute (Jetson-class), ONNX export
- **Multi-platform domain randomization** — one checkpoint deployable
  across more than one real airframe family
- **Outdoor disturbance domain randomization** — wind/gust/thermal
  models, before any real outdoor flight
- **Heterogeneous agents** — mixed mass/inertia/thrust within one swarm
- **Full SwarmRaft implementation (incl. Raft consensus transport)** —
  parked, not dead. The estimation/residual-test/recovery layers ARE
  being adopted (see [decentralization_plan.md](decentralization_plan.md));
  what's parked is the paper's Raft verdict-replication transport.
  Revisit triggers: hardware trials show per-drone local verdicts are
  insufficient (inconsistent fault verdicts across the swarm), or a
  slow mission-level agreed log turns out to be genuinely needed
  beyond what versioned gossip/CRDTs provide. See
  [archive/consensus_mechanisms.md](archive/consensus_mechanisms.md)
  and the SwarmRaft paper (Dev et al., arXiv:2508.00622).
