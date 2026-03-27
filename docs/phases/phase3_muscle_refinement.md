# Phase 3: Muscle Refinement

**Timeline:** Mar 27 -- Apr 7 (Weeks 12--13)  |  **Gate:** M2 -- Logic integration complete by Apr 7

---

## 1. Goals

Phase 3 adds the GATv2 GNN policy backbone (core proposal deliverable) and
post-policy filters for safety, smoothing, and fault recovery. All post-policy
components are config-gated and do not require retraining.

| ID | Objective | Success Criteria | Status |
| :--- | :--- | :--- | :--- |
| P3.1 | GATv2 GNN policy replaces MLP | Same or better formation performance as MLP | In progress (p3-1, p3-2) |
| P3.2 | EMA action smoothing | >= 20% reduction in `std(lin_vel)` | Implemented (enabled by default) |
| P3.3 | CBF collision avoidance | Zero collisions across 10 episodes | Planned |
| P3.4 | Agent loss recovery | Formation re-syncs within 2.0 s | Planned |
| P3.5 | Circular orbit formation (optional) | Drones orbit center while maintaining spacing | Stretch goal |

---

## 2. Architecture

### GATv2 GNN Policy (P3.1)

The MLP policy `[64, 64]` is replaced with a GATv2 Graph Neural Network.
This is Layer 2 of the GNSC 5-Layer Model from the proposal.

```text
Per-drone obs (12D)     K-nearest edges
        |                       |
        v                       v
   Node encoder          Edge construction
   Linear(12, 64)        from neighbor positions
        |                       |
        +--------> GATv2Conv <--+
                   (64->64, heads=2)
                       |
                   GATv2Conv
                   (64->64, heads=2)
                       |
                  Action head        Value head
                  Linear(64,4)       Linear(64,64,1)
```

- Node features: 12D local obs (lin_vel, ang_vel, proj_grav, desired_pos)
- Edges: K-nearest neighbors (K=2), same as Phase 2C obs expansion
- Still PPO training, DirectRLEnv with 1-drone-per-env
- Custom SKRL policy class (`GgswarmGNNPolicy`), bypasses model instantiator
- Train with 3 agents, deploy with N (K-nearest scales)

**Key files:**

- `source/ggswarm/ggswarm/gnn_policy.py` — GATv2 policy class
- `scripts/skrl/train.py` — `--policy gnn` flag (default)

### EMA Action Smoother (P3.2)

Exponential Moving Average on raw policy actions before thrust/moment mapping.
Reduces jittery commands without retraining.

```text
smoothed_action = alpha * raw_action + (1 - alpha) * prev_smoothed
```

- Config: `smoothing_enabled = True`, `smoothing_alpha = 0.3`
- Applied in `_pre_physics_step` before thrust computation
- Smoothed actions reset on episode reset

### CBF Safety Shield (P3.3)

Control Barrier Function for pairwise collision avoidance. Projects
unsafe actions onto the safe half-space defined by minimum separation.

```text
h_ij = ||p_i - p_j||^2 - d_safe^2
if h_dot + gamma * h < 0: project action to safe set
```

- Config: `cbf_enabled`, `cbf_d_safe = 0.12m`, `cbf_gamma = 1.0`
- Operates across swarm group (reads grouped env positions)
- New file: `ggswarm/cbf.py`

### Agent Loss Recovery (P3.4)

Nearest-slot fallback (simplified SwarmRaft):

- Simulated drone kill via config flag
- Remaining drones redistribute to nearest unoccupied formation slots
- Formation offsets recomputed for N-1 agents

### Circular Orbit (P3.5 — stretch goal)

Moving centroid goal — the group centroid orbits a center point.
Drones maintain formation while the centroid moves in a circle.

---

## 3. Training Runs

| Run | Policy | Iterations | ep_len | Reward | Formation | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| p3-1 | GNN | 300 | 492 | 122 | N/A | Learning but rough |
| p3-2 | GNN | 1000 | TBD | TBD | TBD | Running |

### p3-1 Assessment

GNN learns to hover (ep_len 492) but trajectory shows:

- Altitude oscillation (jittery, not smooth)
- Wild XY paths (not converging to formation)
- Roll/pitch +/-100 deg (tumbling)
- Inter-drone distance 0-2m (not converging to 0.5m target)

Conclusion: GNN needs more training time. MLP converged in ~250 iterations;
GNN has more parameters and needs 1000+.

---

## 4. Scope-Cut Rules

- **P3.1 (GNN) slips:** Ship MLP policy. Mention GNN as "in progress."
- **P3.2 (EMA) slips:** Already implemented and on by default.
- **P3.3 (CBF) slips:** Ship without collision avoidance.
- **P3.4 (Agent loss) slips:** Ship with static formation only.
- **P3.5 (Orbit) slips:** Ship with hover formation.

---

## 5. Implementation Schedule

```text
Day 1-2 (Mar 27-28): EMA smoother — DONE
Day 3-6 (Mar 28-31): GATv2 GNN — implemented, training p3-2
Day 7-8 (Apr 1-2):   CBF safety shield
Day 9-10 (Apr 3-4):  Agent loss recovery
Day 11 (Apr 5):      Integration testing
Apr 7:               M2 gate
```

---

## See Also

- [Phase 2: Brain Development](phase2_brain_development.md)
- [Architecture](../design/architecture.md)
- [Changelog](../status/changelog.md)
