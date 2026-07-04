# ggSwarm Live Backlog

Two sources flow in here:

1. **Capstone deferrals.** Items the capstone explicitly did not do.
   These were originally tracked in the deleted `phase7_post_capstone.md`.
2. **New ggSwarm Live work** that doesn't yet have a phase home.

Each item below is tagged with the phase that will absorb it.
"Unmapped" means it hasn't been assigned to a phase yet.

---

## Prioritization

| ID | Item | Area | Effort | Impact | Phase |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A1 | GATv2 edge features (`edge_dim=3`) | Policy | S | M–H | Phase 1 |
| A2 | Cloud / boid retraining (GCE) | Policy | M | M | Phase 1 |
| B1 | Wind + downwash modeling | Env | L | H | Phase 1 |
| B2 | Non-flat terrain / 3D environments | Env | L | M | Phase 15 |
| B3 | Stacked-spawn downwash artifact | Env | S | L | Phase 1 |
| B4 | Heterogeneous agents | Env | M | M | Phase 3 (partial) / 9 / 15 |
| C1 | Action-space CBF reactive avoidance | Obstacles | M | M | Phase 7 |
| C2 | Learned obstacle avoidance revisit | Obstacles | M | M | Phase 7 |
| C3 | Urban canyon scenario | Obstacles | M | M | Phase 7 |
| D1 | Beyond 20 agents | Scale | S | M | Phase 3 |
| E1 | Semi-decentralized slot allocation (slot-pref logits) | Consensus | M | H | Phase 2b (stepping stone) |
| E2 | Bertsekas auction over peer mesh | Consensus | L | H | Phase 2b (production path) |
| E3 | SwarmRaft (paper) residual-test + multilateration recovery | Localization | M | H | Phase 2a |
| E4 | Multi-dropout + fault catalog Monte Carlo harness | Fault tolerance | M | H | Phase 2c |
| E5 | `SwarmRaft` → `AliveMask` / `DropoutGuard` rename (post-capstone code only) | Refactor | S | L | Phase 2c (standalone PR ahead) |
| E6 | Versioned gossip + CRDT command channel | Consensus | M | M | Phase 2d |
| E7 | Average-consensus shape-anchor / centroid | Consensus | S | M | Phase 2d |
| H1 | Pegasus Simulator vs. custom PX4 stack decision | Hardware | S | H | Phase 10 (pre-kickoff) |
| H2 | Crazyswarm2 (CRTP) vs. PX4 offboard mode decision | Hardware | S | H | Phase 10 (pre-kickoff) |
| H3 | Single-drone failsafe-layer trigger-latency measurements | Hardware | S | H | Phase 10 |
| H4 | LPS anchored multi-drone sim-to-real gap measurement | Hardware | M | H | Phase 11 |
| H5 | Real UWB / mesh logs → Phase 2a noise-model recalibration loop | Calibration | M | H | Phase 12 |
| H6 | Multi-dropout drill on real hardware | Hardware | M | H | Phase 12 |
| H7 | Skybrush CSV importer + unit tests | Tooling | M | M | Phase 13 |
| H8 | RL overlay vs. raw-waypoint side-by-side comparison | Hardware | S | M | Phase 13 |
| H9 | Outdoor disturbance DR (wind / gust / thermal) sim models | Sim | M | H | Phase 6 |
| H10 | CBF + reactive obstacle avoidance composition (sim) | Sim | M | H | Phase 7 |
| H11 | GATv2 → MLP distillation + ONNX export pipeline | Sim | M | H | Phase 8 |
| H12 | Multi-platform DR (Crazyflie → Holybro family) | Sim | L | M | Phase 9 |
| R1 | Part 107 Remote Pilot Certificate | Regulatory | S | H | Phase 14a |
| R2 | § 107.35 waiver application | Regulatory | L | H | Phase 14b |
| R3 | Insurance policy in force | Regulatory | S | H | Phase 14b |
| R4 | First paid booking acquisition | Revenue | M | H | Phase 14d |
| S1 | Glyph-to-points pipeline (font → uniform-density slot set) | Tooling | M | M | Phase 4 |
| S2 | Held-out-shape generalization scorecard | Eval | S | M | Phase 4 |
| S3 | Choreography primitive library (morph / rotate / hold / translate) | Tooling | M | H | Phase 5 |
| S4 | Optimal-transport / Hungarian morph slot-mapping (offline) | Tooling | S | M | Phase 5 |
| F1–F6 | Cinematic polish bundle | Capstone trailer | S | L | Capstone (frozen, not pursuing) |

Effort: S ≤ 1 week, M ≤ 1 month, L > 1 month.

---

## Capstone deferrals (detail)

### Policy / Model

#### A1. GATv2 edge features

`GATv2Conv` is currently called without `edge_dim`, so attention conditions
only on node features. The obs already carries per-neighbor relative
positions (`rel_pos_n0`, `rel_pos_n1` at `obs[12:18]`) that are discarded
as node features today. Wiring these in as `edge_attr` with `edge_dim=3`
would let attention condition on *how far and in what direction* each
neighbor sits.

**Touches:** `source/ggswarm/ggswarm/gnn_policy.py`, env
`_get_observations` to publish `edge_attr` alongside `edge_index`.

#### A2. Cloud / boid retraining

The cloud/boid formation mode is implemented but was never retrained fresh
with `formation_mode = "cloud"` for ~500 iterations — capstone Phase 5 ran
out of GCE credits. A dedicated training run would produce a better demo
clip of emergent swarm cohesion vs fixed-slot formations.

### Environment / Physics fidelity

#### B1. Wind and downwash

The current sim has no wind and no downwash interaction between drones.
Real tight-formation flight has significant downwash effects from upper to
lower drones. Two tiers: (1) add simple random-field wind as a force
perturbation, (2) add pairwise downwash forces based on relative XY
distance + vertical stacking.

#### B2. Non-flat terrain / 3D environments

Current world is a flat plane with only cylindrical obstacles. Uneven
ground, elevation changes, indoor rooms with walls and ceilings would push
the policy harder.

#### B3. Stacked-spawn downwash artifact

Drones currently spawn hovering stacked vertically above the same XY point.
Real Crazyflies interact via downwash — the upper drone destabilizes the
lower. Recent runs don't bunch up at spawn so this isn't blocking anything.

#### B4. Heterogeneous agents

Shared policy assumes identical Crazyflie 2.x mass, inertia, and thrust
capacity. Real swarms have battery differences, payload variance, and
potentially mixed platforms. Would need agent-specific observation
conditioning or a policy that takes agent parameters as input.

### Obstacle avoidance

#### C1. Action-space CBF for reactive avoidance

The CBF obstacle module is retained in `cbf.py` but disabled — goal
deflection works better with the current policy. A future policy trained
to respect CBF corrections could enable harder obstacle environments.

#### C2. Learned obstacle avoidance revisit

Six-commit experiment preserved on the
`experimental/learned-obstacle-avoidance` branch. Adding obstacle
observation columns showed no measurable effect across six retrains. Worth
revisiting with richer obstacle encodings (occupancy grid, ray-cast, or
relative-obstacle graph edges).

#### C3. Urban canyon scenario

The capstone proposal mentioned both *forest* and *urban canyon*; only the
forest is in the final deliverable. A canyon scenario (tall rectangular
walls, narrower passages, vertical maneuver pressure) is the natural next
obstacle generalization test.

### Scale

#### D1. Beyond 20 agents

Capstone scale tests stopped at 20 because that meets O4. KNN obs structure
and dynamic spawn radius are designed to extend further but were not
measured. Next sweep: 32 / 64 agents with edge-sparsity tuning.

### Decentralization

See [consensus_mechanisms.md](consensus_mechanisms.md) for the full
adopt / decline rationale across blockchain / Raft / eventually-consistent /
auction.

#### E1. Semi-decentralized slot allocation (slot-pref logits)

Current slot assignment is greedy-nearest run by the env — spatially aware
but centralized. The semi-decentralized stepping stone (Phase 2b step 1):

1. Add all slot positions to observations.
2. Extend action space with slot-preference logits.
3. Env resolves conflicts using preferences as tie-breakers.
4. GNN learns emergent spatial reasoning ("I'm on the right, prefer
   rightmost slot").

E1 ships as a curriculum aid, not as a permanent layer. Phase 2b step 2
replaces it with a full Bertsekas auction (E2).

#### E2. Bertsekas auction over peer mesh

Production path for Phase 2b. Each drone bids on its preferred slot
weighted by cost (e.g., distance); bids propagate via the Phase 2d
versioned gossip channel; outbid drones pick the next-best slot.
Eventually-consistent under stale bids. See
[phase2b_assignment.md](phases/phase2b_assignment.md).

#### E3. SwarmRaft (paper) residual-test + multilateration recovery

Phase 2a fault-detection and recovery layer. Per-drone residual
`e_{i,k} = min‖x − z_{i,k}‖` against feasible region from neighbor ranges,
threshold `μ + 3σ`; multilateration recovery via least-squares over
non-flagged peers. Adopts the fault-detection layer of the SwarmRaft paper
(Dev et al. 2025, arXiv:2508.00622v2); declines the Raft transport layer.
See [phase2a_localization.md](phases/phase2a_localization.md).

#### E4. Multi-dropout + fault catalog

Phase 2c. Extend single-dropout alive-mask to N-simultaneous; codify
labelled scenarios per fault class (single dropout, multi-simultaneous,
staggered, sensor degradation, comms partition, mesh thinning, stale
state); Monte Carlo evaluation harness produces per-class recovery-time
CDFs that feed the FAA Part 107.35 safety case. See
[phase2c_fault_tolerance.md](phases/phase2c_fault_tolerance.md).

#### E5. `SwarmRaft` → `AliveMask` / `DropoutGuard` rename

Post-capstone code only — the capstone tag (`v1.0.0-capstone`,
`docs/capstone/**`) is frozen and untouched. Resolves the naming clash
between the capstone's alive-mask mechanism and the Dev et al. paper's
SwarmRaft. The term `SwarmRaft` is reserved going forward for the
paper-derived residual-test + multilateration recovery layer (E3).
Standalone refactor PR ahead of the substantive Phase 2c work.

#### E6. Versioned gossip + CRDT command channel

Phase 2d. Any-drone-hears-it command propagation. Versioned record
per command; CRDT semantics (last-write-wins keyed by version + origin)
resolve concurrent writes. See
[phase2d_consensus_dissemination.md](phases/phase2d_consensus_dissemination.md).

#### E7. Average-consensus shape-anchor / centroid

Phase 2d. Each drone averages its own centroid estimate with received
neighbor estimates per round; converges to global mean in O(diameter)
rounds; degrades gracefully under partition. See
[phase2d_consensus_dissemination.md](phases/phase2d_consensus_dissemination.md).

---

## New ggSwarm Live work (unmapped)

Placeholder. As Phase 1+ work begins, items that don't cleanly fit a
phase doc land here first and get promoted out.

- *(none yet)*
