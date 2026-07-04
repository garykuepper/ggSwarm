# Phase 2d: Distributed Dissemination + Consensus (sim only)

**Status:** Planned. Can run in parallel with Phases 2a and 2b.

**New capability:** any-drone-hears-it command propagation, and a swarm-wide
shape anchor / centroid that does not depend on a central planner or a leader.

## Scope

1. **Versioned gossip + CRDT command channel.** A command (e.g., "form a
   triangle, scale 1.5 m") is a versioned record. Drones flood received
   commands on every broadcast. CRDT semantics resolve concurrent writes
   without coordination.
2. **Average-consensus shape anchor.** Each drone holds an estimate of the
   swarm centroid (used as the formation anchor in relative-frame ops).
   Each round, a drone averages its own estimate with received neighbor
   estimates. In a connected graph this converges to the global mean in
   O(diameter) rounds.
3. **Convergence-under-loss benchmarks.** Simulated lossy mesh (loss rate,
   latency, partition events) drives the scorecard.

See [consensus_mechanisms.md](../consensus_mechanisms.md) for why these
primitives are the right fit (cooperative, lossy, low-latency-required) and
why blockchain / Raft alternatives are declined.

## Inputs from prior phase

- Phase 1c MAPPO + GATv2 checkpoint (the consensus centroid replaces any
  remaining ground-truth-derived shape anchor in observations)
- Phase 2a localization (peer state estimates feed into both the gossip
  channel and the average-consensus iteration)

## Sim methodology

### Versioned gossip

- Each command carries `(version: monotonic_int, payload, origin_drone_id)`.
- On every broadcast cycle, each drone re-broadcasts its current command set.
- A receiver replaces its current command with any received command of
  higher version; ties broken by `origin_drone_id` for determinism.
- Command set is bounded (e.g., last K versions) to cap memory.

### CRDT command state

- For state with multi-writer semantics (e.g., the alive-set from Phase 2c),
  use last-write-wins keyed by `(drone_id, monotonic_timestamp)`.
- Convergence: any two drones whose membership graph eventually shares an
  edge will converge to the same alive-set without explicit coordination.

### Average consensus for centroid

- Each drone `i` maintains `c_i,k` (its estimate of the swarm centroid at
  step k).
- Update rule: `c_i,k+1 = (1 / |N_i ∪ {i}|) · (c_i,k + Σ_{j ∈ N_i} c_j,k)`
  where `N_i` is the set of currently-trusted neighbors (post-residual-test
  from Phase 2a).
- In a connected graph this converges to the global mean of all `c_i,0`
  in O(graph diameter) rounds; degrades gracefully under partition.

### Comms model

- Per-link drop probability sampled per broadcast.
- Latency modeled as a uniform delay window.
- Partitions modeled as a scheduled loss-rate spike that drops the mesh
  into two components for N steps, then heals.

## Milestone artifact

Sim demo: command propagation under 30% link loss + a 2-drone partition,
showing the swarm receives the new command within the convergence target
and reconverges centroid after partition heal. Video recorded with
`--video_prefix p2d-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Command propagation time, healthy comms (8 drones) | ≤ 5 sim steps |
| Command propagation time, 30% link loss | ≤ 50 sim steps |
| Command propagation time after partition heal (8 drones, 2-drone partition) | ≤ 100 sim steps |
| Centroid drift over 60s relative-frame flight (steady state) | ≤ 0.05 m |
| Average-consensus convergence time (8 drones, healthy) | ≤ 10 sim steps |
| Concurrent-write resolution (CRDT) — divergent state observed across any 2 drones at any step | 0 |

## FAA evidence produced

Feeds **architecture.md §4 Layer 3 (data link loss)** evidence column.
Command propagation under loss is the headline number; centroid-drift bound
feeds Layer 4 (formation collapse). The behavior under partition + heal
demonstrates the "graceful degradation" property the safety case rests on.

## Risks

- Average consensus drifts under sustained partition → the shape-anchor
  divergence between partitions exceeds tolerance; mitigation: partition-
  detect heuristic (e.g., neighbor count drop) freezes anchor updates until
  reconvergence.
- Gossip bandwidth grows with N × command-history-depth → cap history depth,
  prune by monotonic version, broadcast deltas not full state once both
  sides confirm a baseline.
- Stale command set during partition leads to two halves executing different
  commands → latest-version-wins resolves at heal, but the divergence
  during partition is real and must be in the safety case.

## Decline list

- **Raft replication for command state** — declined; quorum cliff under
  partition violates graceful-degradation requirement. See
  `consensus_mechanisms.md`.
- **Total-order broadcast** — declined; not needed for command semantics
  (last-version-wins is sufficient) and the cost is not justified.
- **Blockchain-anchored command log** — declined; energy cost, latency, and
  threat-model mismatch (no adversary, no need for tamper-evidence beyond
  what command versioning + signed broadcasts already give).

## See Also

- [Phase 2 parent index](phase2_decentralized.md)
- [Phase 2b Slot Assignment](phase2b_assignment.md) — auction bids ride on this gossip channel
- [Phase 2c Fault Tolerance](phase2c_fault_tolerance.md) — alive-set is one of the CRDTs
- [Architecture § 4 Failsafe Cascade](../architecture.md#4-failsafe-architecture)
- [Consensus mechanisms reference](../consensus_mechanisms.md)
- [Vision § Phase 2d](../vision.md)
