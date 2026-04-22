# Phase 7: Post-Capstone Plan

**Timeline:** After Apr 24, 2026  |  **Gate:** None — this is the
ongoing-work backlog beyond the capstone deliverable.

**Status: Planning.** Consolidates future-work items that were
intentionally deferred from Phases 1–6 so the capstone could ship on
time. Nothing here is on the critical path for the Apr 24, 2026
deadline — see [Phase 6: Delivery](phase6_delivery.md) for that.

> **See also:** [ggSwarm v2 Research Program](../project/ggswarm_v2_plan.md)
> for the broader sim-to-real and hardware-track vision beyond this
> deferred-items backlog.

Sources that previously held these items individually:

- [docs/design/architecture.md](../design/architecture.md) §
  Post-Capstone Enhancements
- [docs/design/assumptions.md](../design/assumptions.md) §§ 4–7
- [docs/project/testing_report.md](../project/testing_report.md) § 4.2
- [docs/phases/phase5_showcase_prep.md](phase5_showcase_prep.md) §
  Deferred, § Next iterations
- [docs/status/weekly_updates.md](../status/weekly_updates.md) (stacked-spawn
  note, 2026-04-06)

Those sections remain the detailed source for each item; this doc is the
index.

---

## 1. Prioritization

Rough effort / impact ranking to guide what to pick up first after the
capstone freeze.

| ID | Item | Area | Effort | Impact |
| :--- | :--- | :--- | :--- | :--- |
| A1 | GATv2 edge features (`edge_dim=3`) | Policy | S | M–H |
| A2 | Cloud / boid retraining (GCE) | Policy | M | M |
| B1 | Wind + downwash modeling | Env | L | H |
| B2 | Non-flat terrain / 3D environments | Env | L | M |
| B3 | Stacked-spawn downwash artifact | Env | S | L |
| B4 | Heterogeneous agents | Env | M | M |
| C1 | Action-space CBF reactive avoidance | Obstacles | M | M |
| C2 | Learned obstacle avoidance revisit | Obstacles | M | M |
| C3 | Urban canyon scenario | Obstacles | M | M |
| D1 | Beyond 20 agents | Scale | S | M |
| E1 | Semi-decentralized slot allocation | Consensus | L | H |
| F1–F6 | Cinematic polish bundle | Presentation | S | L |

Effort: S ≤ 1 week, M ≤ 1 month, L > 1 month.

---

## 2. Policy / Model (A)

### A1. GATv2 edge features

`GATv2Conv` is currently called without `edge_dim`, so attention
conditions only on node features. The obs already carries per-neighbor
relative positions (`rel_pos_n0`, `rel_pos_n1` at `obs[12:18]`) that are
discarded as node features today. Wiring these in as `edge_attr` with
`edge_dim=3` would let attention condition on *how far and in what
direction* each neighbor sits — typically a meaningful gain for spatial
GNN tasks.

**Touches:** [gnn_policy.py](../../source/ggswarm/ggswarm/gnn_policy.py),
env `_get_observations` to publish `edge_attr` alongside `edge_index`.

### A2. Cloud / boid retraining

The cloud/boid formation mode is implemented but was never retrained
fresh with `formation_mode = "cloud"` for ~500 iterations — Phase 5 ran
out of GCE credits. A dedicated training run would produce a better
demo clip of emergent swarm cohesion vs fixed-slot formations.

**Touches:** GCE training budget, training launch only.

---

## 3. Environment / Physics Fidelity (B)

### B1. Wind and downwash

The current sim has no wind and no downwash interaction between drones.
Real tight-formation flight has significant downwash effects from upper
to lower drones. Two tiers: (1) add simple random-field wind as a force
perturbation, (2) add pairwise downwash forces based on relative XY
distance + vertical stacking. See
[assumptions.md § 6](../design/assumptions.md).

### B2. Non-flat terrain / 3D environments

Current world is a flat plane with only cylindrical obstacles. Uneven
ground, elevation changes, indoor rooms with walls and ceilings would
push the policy harder. See
[assumptions.md § 7](../design/assumptions.md).

### B3. Stacked-spawn downwash artifact

Drones currently spawn hovering stacked vertically above the same XY
point. Real Crazyflies interact via downwash — the upper drone
destabilizes the lower. Recent runs don't bunch up at spawn so this
isn't blocking anything. Logged
[2026-04-06 weekly update](../status/weekly_updates.md#L223) as a
hardware-fidelity improvement.

### B4. Heterogeneous agents

Shared policy assumes identical Crazyflie 2.x mass, inertia, and thrust
capacity. Real swarms have battery differences, payload variance, and
potentially mixed platforms. Would need agent-specific observation
conditioning or a policy that takes agent parameters as input. See
[assumptions.md § 5](../design/assumptions.md).

---

## 4. Obstacle Avoidance (C)

### C1. Action-space CBF for reactive avoidance

The CBF obstacle module is retained in `cbf.py` but disabled — goal
deflection works better with the current policy. A future policy trained
to respect CBF corrections could enable harder obstacle environments.
See [testing_report.md § 4.2](../project/testing_report.md).

### C2. Learned obstacle avoidance revisit

Six-commit experiment preserved on the
`experimental/learned-obstacle-avoidance` branch. Adding obstacle
observation columns showed no measurable effect across six retrains.
Worth revisiting with richer obstacle encodings (occupancy grid,
ray-cast, or relative-obstacle graph edges).

### C3. Urban canyon scenario

The proposal mentioned both *forest* and *urban canyon*; only the forest
is in the final deliverable. A canyon scenario (tall rectangular walls,
narrower passages, vertical maneuver pressure) is the natural next
obstacle generalization test.

---

## 5. Scale (D)

### D1. Beyond 20 agents

Scale tests stopped at 20 because that meets O4. KNN obs structure and
dynamic spawn radius are designed to extend further but were not
measured. Next sweep: 32 / 64 agents with edge-sparsity tuning.

---

## 6. Decentralization (E)

### E1. Semi-decentralized slot allocation

Current slot assignment is greedy-nearest run by the env — spatially
aware but centralized. A semi-decentralized path would:

1. Add all slot positions to observations
2. Extend action space with slot-preference logits
3. Env resolves conflicts using preferences as tie-breakers
4. GNN learns emergent spatial reasoning ("I'm on the right, prefer
   rightmost slot")

Full decentralization via auction / Raft consensus breaks the single-pass
CTDE inference model. See
[assumptions.md § 4](../design/assumptions.md) for the full rationale.

---

## 7. Cinematic / Presentation Polish (F)

All deferred from [phase5_showcase_prep.md](phase5_showcase_prep.md) —
non-critical once the trailer was exported.

| ID | Item |
| :--- | :--- |
| F1 | Drone color refinement (currently yellow-leaning; push amber or magenta) |
| F2 | Cinematic 3-point lighting in cyan tones (currently flat under grid emission) |
| F3 | Volumetric fog reinstate (caused white-outs earlier — add carefully) |
| F4 | Lift Tron constants (`TRON_AMBER`, `TRON_TEAL`, `TRON_LINE_WIDTH`) into `GgswarmEnvCfg` |
| F5 | Cold-open slower orbit speed (halve current orbit for true establishing-shot feel) |
| F6 | `scripts/showcase.py` auto-edited cinematic pipeline (already scaffolded, not wired end-to-end) |

---

## 8. Out of Scope

Items explicitly *not* in Phase 7:

- Hardware sim-to-real transfer to real Crazyflies — separate project.
- Publishing a paper — capstone deliverable stands on its own.
- Rewriting to MAPPO — CTDE + shared PPO met objectives; no motivation
  to redo.

---

## See Also

- [Phase 6: Delivery](phase6_delivery.md)
- [Architecture](../design/architecture.md)
- [Assumptions](../design/assumptions.md)
- [Testing Report](../project/testing_report.md)
