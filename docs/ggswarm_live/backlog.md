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
| B2 | Non-flat terrain / 3D environments | Env | L | M | Phase 5 |
| B3 | Stacked-spawn downwash artifact | Env | S | L | Phase 1 |
| B4 | Heterogeneous agents | Env | M | M | Phase 4 / 5 |
| C1 | Action-space CBF reactive avoidance | Obstacles | M | M | Phase 6 |
| C2 | Learned obstacle avoidance revisit | Obstacles | M | M | Phase 6 |
| C3 | Urban canyon scenario | Obstacles | M | M | Phase 6 |
| D1 | Beyond 20 agents | Scale | S | M | Phase 4 |
| E1 | Semi-decentralized slot allocation | Consensus | L | H | Phase 3 |
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

#### E1. Semi-decentralized slot allocation

Current slot assignment is greedy-nearest run by the env — spatially aware
but centralized. A semi-decentralized path would:

1. Add all slot positions to observations.
2. Extend action space with slot-preference logits.
3. Env resolves conflicts using preferences as tie-breakers.
4. GNN learns emergent spatial reasoning ("I'm on the right, prefer
   rightmost slot").

Full decentralization via auction / Raft consensus breaks the single-pass
CTDE inference model — that path is the proper Phase 3 work.

---

## New ggSwarm Live work (unmapped)

Placeholder. As Phase 1+ work begins, items that don't cleanly fit a
phase doc land here first and get promoted out.

- *(none yet)*
