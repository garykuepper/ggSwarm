# Consensus Mechanisms — What ggSwarm Live Uses and Why

This is a single-source-of-truth reference for the question "how do drones in
the swarm agree on things, and isn't blockchain one way of doing that?"

The Phase 2 sub-phase docs and `vision.md` link here rather than re-deriving
the comparison each time.

---

## TL;DR

| Mechanism | Threat model | Synchrony | Failure mode | ggSwarm Live use |
| :--- | :--- | :--- | :--- | :--- |
| Blockchain (PoW / PoS / BFT-SMR) | Byzantine adversaries | Partial / async with quorum | Energy + latency cliff | **Declined** |
| Raft / SwarmRaft Raft layer | Crash-fault, honest peers | Synchronous, reliable links | Cliff at quorum loss | **Declined** |
| Eventually-consistent primitives (gossip, version vectors, CRDTs, average consensus) | Crash-fault, cooperative | Asynchronous, lossy | Graceful degradation | **Adopted (Phase 2d)** |
| Bertsekas auction | Crash-fault, cooperative | Asynchronous, stale-aware | Eventual convergence | **Adopted (Phase 2b)** |
| SwarmRaft (paper) residual-test + multilateration recovery | Crash-fault + sensor faults | Per-update, no global agreement | Falls back to INS / RL fallback | **Adopted (Phase 2a / 2c)** |

---

## The threat model and the comms model decide the answer

Two facts about ggSwarm Live make most "famous" consensus mechanisms wrong
for it:

1. **The swarm is cooperative.** Every drone is yours, on your mesh, in your
   own airspace, running firmware you signed. There is no adversary trying to
   make a drone lie about its position to gain economic advantage. Byzantine
   fault tolerance — the property blockchain pays for in energy and latency —
   is not earned here.
2. **The radio is lossy and asynchronous.** WiFi/UWB mesh between Crazyflies
   (or Holybro / ModalAI airframes later) drops packets, varies in latency,
   and partitions under range. Anything that requires synchronous reliable
   delivery of every message inside a bounded round (Raft, PBFT) cliffs
   under real conditions.

The combination — cooperative + lossy + low-latency-required — points
directly at **eventually-consistent** primitives. They tolerate loss, have no
leader to lose, and converge to agreement in bounded time when the network
heals.

## Why blockchain is the wrong tool

Blockchain consensus protocols (Proof-of-Work, Proof-of-Stake, BFT-SMR
families like PBFT and Tendermint) exist to solve **trustless agreement
between mutually suspicious parties**. They guarantee that no coalition of
malicious nodes below some threshold can rewrite history.

For a swarm of cooperating drones flying a paid drone show:

- **Energy cost** — PoW is a non-starter on a 250 mAh battery.
- **Latency** — every block-finalization round is hundreds of milliseconds at
  best; control loops run at ≥ 50 Hz.
- **Threat-model mismatch** — there is no economic incentive for a Crazyflie
  to lie about its position; spending compute to defend against
  not-going-to-happen attacks is wasted budget.
- **Quorum requirements** — most BFT protocols halt without 2f+1 honest
  participants. A swarm of 5 that loses 2 drones to crashes is now stuck.

Blockchain solves a problem ggSwarm Live does not have, and pays a price the
hardware cannot afford.

## Why Raft (and SwarmRaft's Raft layer) is also a poor fit

Raft is much closer to the right shape — it tolerates **crash faults** rather
than Byzantine faults, which matches the actual threat model. A leader is
elected, log entries are replicated, and any quorum-acknowledged entry is
durable.

The SwarmRaft paper (Dev et al., arXiv:2508.00622v2) wraps a residual-test +
multilateration recovery mechanism inside a Raft transport. The fault-detection
mechanism is independent of the transport and is genuinely useful (see next
section); the Raft transport itself fails on three counts:

1. **Synchronous-comms assumption.** The paper assumes every node receives
   every message from step k before step k+1 begins. Real WiFi / UWB mesh
   does not provide that. The paper validates in a Python simulator, not on
   a real radio.
2. **Quorum cliff.** Raft halts when a majority of nodes is unreachable. A
   5-drone show that loses 3 freezes — the opposite of the graceful
   degradation the safety case (architecture.md §4) is built around.
3. **Reintroduced single point of failure.** Phase 2's whole point is to
   remove fixed infrastructure. Electing a leader at the protocol layer puts
   centralization back in, even if the leader is a different drone each
   election.

Raft is the right answer to "how do we agree on a write-once log entry?"
ggSwarm Live's coordination problems are not write-once-log-entry problems.

## Why eventually-consistent primitives are the right fit

For the things the swarm actually needs to agree on — current command,
current shape anchor, who is alive, where each peer is — the requirements are:

- Convergence in bounded time when the network is healthy.
- Graceful degradation when the network is partitioned.
- No single point of failure.
- Per-update cost that fits the radio's bandwidth budget (~5–10 Hz
  broadcasts of ~40 byte state packets).

Several primitives meet these requirements and compose well:

- **Versioned gossip** for command dissemination. Every command carries a
  monotonic version. Drones flood received commands on every broadcast.
  Any drone that hears the command gets it; drones that hear it later
  receive the same thing because the version is the tiebreaker.
- **CRDTs** (conflict-free replicated data types) for state that multiple
  drones write to — for example, the alive-set of drones. Last-write-wins
  keyed by drone-ID + heartbeat-timestamp converges without coordination.
- **Average consensus** for shape-anchor / centroid agreement. Each drone
  averages its own estimate with received neighbor estimates each round; in
  a connected graph this converges to the global mean in O(diameter) rounds.

All three degrade smoothly: under heavy loss they converge slower; under
partition each component converges within itself; when the network heals,
they re-converge across the merged graph.

## Why Bertsekas auction is the right fit for slot assignment specifically

Slot assignment is a special case: assign N drones to N slots in a way that
no two drones pick the same slot. Auction algorithms (Bertsekas) solve this
in a distributed, eventually-consistent way:

- Each drone bids on its preferred slot, weighted by cost (e.g., distance).
- Bids propagate via the same gossip mesh.
- A drone outbid for its current slot picks the next-best slot.
- The system converges to a valid assignment in O(N) rounds under healthy
  comms; longer under loss.

Auction tolerates stale bids: a drone acting on a one-round-old view of the
bid state will at worst trigger a re-bid, not a collision. This matches the
loss profile of a real radio. Phase 2b adopts this as the production path,
preceded by a learning-friendly stepping stone (backlog item E1, slot-
preference logits with env-side tie-breaking) that lets the policy develop
spatial reasoning before the auction is layered in.

## What we adopt from the SwarmRaft paper

The paper (arXiv:2508.00622v2) bundles two ideas:

1. A **fault-detection-and-recovery layer**: each drone reports its position
   and inter-peer ranges; a verifier (in the paper, the Raft leader)
   computes a residual `e_{i,k} = min‖x − z_{i,k}‖` against the feasible
   region defined by neighbor ranges, flags drones whose residual exceeds
   `μ + 3σ`, and recovers their position via least-squares multilateration
   over non-faulty peers.
2. A **Raft-consensus transport** that replicates the verifier's decision
   across all drones.

We adopt (1) and decline (2). The fault-detection layer is implemented in
Phase 2a (as part of the localization stack) and extended in Phase 2c (as
part of the multi-dropout fault catalog). The verifier role does not have to
be a leader-elected Raft role; in our implementation each drone runs the
residual test on its own neighbors using gossiped peer state. This is
strictly less powerful than the paper's design (we cannot force agreement on
a single global verdict per round) and strictly more robust (no quorum cliff,
no synchronous-comms requirement). For a cooperative drone show this is the
correct trade.

The threat model is also different: the paper defends against GNSS spoofing
(adversarial); ggSwarm Live operates GPS-denied with cooperative peers, so
the residual-test and multilateration recovery are repurposed for **honest
sensor faults** (UWB ranging glitches, IMU drift, filter divergence) rather
than adversarial position reports.

## Naming note: "SwarmRaft" in this codebase

The capstone code (frozen at `v1.0.0-capstone`) uses the term `SwarmRaft` to
refer to a single-dropout alive-mask mechanism — unrelated to the paper.
Post-capstone code renames that mechanism to `AliveMask` / `DropoutGuard`
(Phase 2c), and the term `SwarmRaft` is reserved in docs and new code for
the paper-derived residual-test + multilateration recovery layer described
above. See [phase2c_fault_tolerance.md](phases/phase2c_fault_tolerance.md)
for the rename details.

## See Also

- [Vision § 6 Phase Breakdown](vision.md#6-phase-breakdown)
- [Phase 2 parent index](phases/phase2_decentralized.md)
- [Phase 2a Localization](phases/phase2a_localization.md)
- [Phase 2b Slot Assignment](phases/phase2b_assignment.md)
- [Phase 2c Fault Tolerance](phases/phase2c_fault_tolerance.md)
- [Phase 2d Consensus + Dissemination](phases/phase2d_consensus_dissemination.md)
- [Architecture § 4 Failsafe Cascade](architecture.md#4-failsafe-architecture)
- [SwarmRaft paper, Dev et al. 2025](references.md)
