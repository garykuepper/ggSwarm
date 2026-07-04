# Phase 2: Decentralized + Fault-Tolerant Stack (sim only)

**Status:** Planned. Sub-phased to isolate variables and produce a clean
go/no-go gate per capability before sim-to-real.

**New capability:** the entire decentralized + fault-tolerant stack — peer
ranging, distributed slot assignment, multi-dropout fault tolerance, and
gossip-based command + centroid consensus — validated in simulation before
any real hardware.

**Why before sim-to-real:** these subsystems are largely algorithmic
(multilateration, auction convergence, gossip versioning, average consensus).
Debugging them in Isaac Lab with full state observability is dramatically
cheaper than debugging them on real Crazyflies. The hardware block
(Phases 10–13) then takes a fully decentralized, fault-tolerant policy to
hardware in disciplined per-variable steps.

**Why sub-phased:** the original Phase 2 bundled five orthogonal subsystems
behind one milestone. That made regressions un-attributable during MAPPO
training and prevented a clean go/no-go gate. The sub-phase split below
isolates one variable per sub-phase, each with its own milestone artifact
and scorecard. The scorecards double as Part 107.35 evidence rows feeding
the failsafe cascade in [architecture.md § 4](../architecture.md#4-failsafe-architecture).

## Sub-phase index

| Sub-phase | Title | Status |
| :--- | :--- | :--- |
| 2a | [Decentralized Localization](phase2a_localization.md) | Planned |
| 2b | [Decentralized Slot Assignment](phase2b_assignment.md) | Planned |
| 2c | [Multi-Dropout + Fault Catalog](phase2c_fault_tolerance.md) | Planned |
| 2d | [Distributed Dissemination + Consensus](phase2d_consensus_dissemination.md) | Planned |

## Sub-phase dependencies

```text
Phase 1c (planned)
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
      2a ──────────► 2c
      │
      2b ◀── auction bids ride on Phase 2d gossip channel
      │
      2d
       │              │              │
       └──────────────┼──────────────┘
                      ▼
                   Sim Phases 3–9 (parallel-capable) → Hardware Phases 10–13
```

- **2a** is the prerequisite for **2c** — you need ranging before you can
  detect ranging faults.
- **2b** and **2d** can proceed in parallel with 2a once the localization
  interface is defined; **2b**'s auction bids ride on **2d**'s gossip
  channel, so 2d is a soft prerequisite for 2b's step-2 (full auction)
  but not for 2b's step-1 (E1 stepping stone).
- All four converge before the rest of the sim block (Phases 3–9) and
  before any hardware work (Phases 10+). Calibration loop from Phase 12
  flows back to Phase 2a.

## Consensus-mechanism rationale

The user asked: *isn't blockchain one way of dealing with consensus?* The
short answer is no, for this swarm — see
[consensus_mechanisms.md](../consensus_mechanisms.md) for the full
comparison. Summary:

- **Blockchain:** wrong threat model (cooperative swarm, no adversary),
  energy cost, latency cliff. **Declined.**
- **Raft / SwarmRaft Raft layer:** synchronous-comms assumption violated by
  real radio; quorum cliff violates graceful degradation. **Declined.**
- **Eventually-consistent primitives** (gossip, CRDTs, average consensus):
  graceful degradation, no leader to lose, matches cooperative low-latency
  loss-tolerant regime. **Adopted in 2d.**
- **Bertsekas auction:** right specialization for slot assignment. **Adopted
  in 2b.**
- **SwarmRaft (paper) residual-test + multilateration recovery:** the
  fault-detection portion is independent of the Raft transport. **Adopted
  in 2a / 2c.**

## SwarmRaft naming resolution

The capstone code used `SwarmRaft` for an alive-mask dropout mechanism. The
Dev et al. paper uses `SwarmRaft` for residual-test + multilateration +
Raft. Resolution:

- **Capstone (`v1.0.0-capstone`, frozen):** untouched.
- **Post-capstone code:** rename `SwarmRaft` → `AliveMask` / `DropoutGuard`
  in Phase 2c.
- **Docs and new code:** the term `SwarmRaft` is reserved exclusively for
  the paper-derived residual-test + multilateration recovery layer adopted
  in 2a / 2c.

## Milestone artifact (whole-Phase-2 rollup)

When all four sub-phases pass their scorecards, the rollup is a sim demo +
write-up of the Phase 1c shared-scene policy now running with peer-ranging-
only localization, decentralized auction-based slot assignment, multi-dropout
fault tolerance, and gossip-based command + centroid consensus — with all
simulated anchors removed. Repo checkpoint of the full decentralized
fault-tolerant policy. Social: "Decentralized swarm in sim — no anchors,
no central controller, fault-tolerant under multi-dropout, command
propagation under packet loss."

## See Also

- [Vision § Phase 2](../vision.md)
- [Architecture § 4 Failsafe Cascade](../architecture.md#4-failsafe-architecture)
- [Architecture § 6 Part 107.35 alignment](../architecture.md#6-faa-part-10735-waiver-alignment)
- [Consensus mechanisms reference](../consensus_mechanisms.md)
