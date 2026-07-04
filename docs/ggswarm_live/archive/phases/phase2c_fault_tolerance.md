# Phase 2c: Multi-Dropout + Fault Catalog (sim only)

**Status:** Planned. Depends on Phase 2a.

**New capability:** the swarm tolerates **multiple simultaneous failures**
across a documented catalog of fault classes, with a recovery-time scorecard
per class that becomes Part 107.35 evidence.

## Scope

1. Extend the existing single-dropout alive-mask (`_agent_alive` in
   `ggswarm_env.py:343-365`) to N-simultaneous dropouts.
2. Codify a failure-mode catalog with a labelled scenario per class, runnable
   as a Monte Carlo evaluation harness.
3. Rename the in-code `SwarmRaft` symbol → `AliveMask` / `DropoutGuard`.
   Reserve the `SwarmRaft` term for the paper-derived residual-test +
   multilateration recovery layer (also adopted here, extending Phase 2a's
   sensor-fault detection to peer-loss detection).
4. Wire the per-class recovery-time CDFs into the FAA safety-case table in
   `architecture.md` §6.

## Inputs from prior phase

- Phase 2a residual-test + multilateration recovery (drone health flag derived
  from peer-ranging consistency, not just heartbeat presence)
- Existing alive-mask gating sites: `ggswarm_env.py:432-433` (action zeroing),
  `ggswarm_env.py:500, 509` (collision exclusion), `ggswarm_env.py:717`
  (centroid exclusion), KNN edge construction (alive-mask applied to
  `edge_index`)

## Sim methodology

### Fault catalog

Each class is a labelled scenario with an injection schedule and an evaluator:

| Class | Injection | Evaluator |
| :--- | :--- | :--- |
| `single_dropout` | One drone alive→dead at random step | Recovery time, formation error after recovery |
| `multi_dropout_simultaneous` | k≥2 drones alive→dead at the same step | Recovery time, formation error, collapse rate |
| `multi_dropout_staggered` | k≥2 drones alive→dead within a 1s window | Recovery time, transient overshoot |
| `sensor_degradation` | Per-drone UWB ranging σ inflated 5× | Residual-test detection time, recovery time |
| `comms_partition` | Mesh split into two components for N steps | Time to reconverge after heal, divergence during partition |
| `mesh_thinning` | Random links dropped at increasing rate | Convergence time as a function of loss rate |
| `stale_state` | One drone's gossiped state delayed by N steps | Auction stability, assignment thrashing |

Each class runs 1000 Monte Carlo episodes per swarm size N ∈ {3, 5, 8}.

### Multi-dropout extension

The dropout schedule (`ggswarm_env.py:915-917`) currently picks a single
random step and a single random drone per group. Extend to:

- `dropout_count_max` parameter (currently effectively 1).
- Per-step dropout decisions sampled from the schedule, not pre-committed
  at reset.
- Alive-mask gating sites are unchanged (already handles per-drone
  liveness independently).

### Rename

Search-and-replace `SwarmRaft` → `AliveMask` / `DropoutGuard` across
post-capstone code only. Capstone (`v1.0.0-capstone` tag, `docs/capstone/**`)
is frozen and untouched. Recorded in `status/changelog.md`. Recommendation
from the plan: ships as a standalone refactor PR ahead of the substantive
2c work, so review is not entangled.

## Milestone artifact

Sim demo + Monte Carlo report covering the fault catalog above. Per-class
recovery-time CDF plots. Threshold table feeding architecture.md §4
failsafes. Video recorded with `--video_prefix p2c-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| `single_dropout` recovery time (median) | ≤ 1.0 s (matches capstone target) |
| `multi_dropout_simultaneous` (k=2) recovery time (median) | ≤ 2.0 s |
| `multi_dropout_simultaneous` (k=3) recovery time (median) | ≤ 3.0 s |
| Formation collapse rate, all classes, all swarm sizes | ≤ 0.01 (per 1k episodes) |
| Residual-test detection time, sensor degradation | ≤ 0.5 s |
| Reconvergence time after partition heal | ≤ 5.0 s |

Targets are placeholders pending the first calibration runs; replace with
measured 95th-percentile thresholds before the FAA safety case is filed.

## FAA evidence produced

Feeds **architecture.md §4 Layer 2 (offboard timeout / peer fault)** and
**Layer 4 (kill switch / formation collapse)** evidence columns. The
collapse-rate threshold is the headline number for "swarm never goes from
working to catastrophic in one step." The per-class recovery-time CDFs are
the behavior tables that justify the failsafe timeouts.

## Risks

- Multi-dropout exposes a policy that overfit to single-drone gracefully
  recover scenarios → curriculum training over the fault catalog rather
  than only the easy classes.
- Renaming `SwarmRaft` → `AliveMask` touches many sites; incidental bugs
  introduced during rename → run the standalone refactor PR through the
  full smoke suite + Phase 1a replay-gate harness before merging.
- Mesh-partition scenarios are not realistic for indoor Crazyflie ops but
  matter for outdoor (Phase 15) — calibrate loss rates against real
  Phase 11+ comms logs once available.

## Decline list

- **Byzantine-fault-tolerant peer death detection** — declined; cooperative
  swarm, no adversary forging "drone X is dead" messages. CRDT alive-set
  with timestamp tiebreaker is sufficient.
- **Heartbeat-only liveness (no residual-test)** — declined; heartbeat
  presence does not catch a drone whose pose estimate has diverged but is
  still publishing. Residual test from Phase 2a covers this.
- **Hot-spare drones held in reserve** — out of scope for 2c, may revisit
  in Phase 5 (extended fault tolerance for outdoor / paid show ops).

## See Also

- [Phase 2 parent index](phase2_decentralized.md)
- [Phase 2a Localization](phase2a_localization.md) — residual-test source
- [Architecture § 4 Failsafe Cascade](../architecture.md#4-failsafe-architecture)
- [Architecture § 6 Part 107.35 alignment](../architecture.md#6-faa-part-10735-waiver-alignment)
- [Consensus mechanisms reference](../consensus_mechanisms.md)
- [Vision § Phase 2c](../vision.md)
