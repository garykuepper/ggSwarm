# Phase 2: Brain Development (Weeks 7–8)

Phase 2 trains the GATv2 coordination policy using MAPPO so agents learn basic formation keeping in open space. This phase covers GNSC Layer 2 (GNN Message Passing) and Layer 3 groundwork.

---

## Objectives

| ID | Objective | Success Criteria |
| :--- | :--- | :--- |
| P2.1 | Train a shared MAPPO policy that keeps agents in a stable formation | **Mean formation error < 0.5m** (defined below) after **≤ 50k environment steps** |
| P2.2 | Integrate GATv2 as the policy backbone | Policy handles batched graphs from `extras["adj_matrix"]` |
| P2.3 | Implement Curriculum Reward Shaping | Smooth transition from hover-in-place to formation-aware |
| P2.4 | Validate training pipeline end-to-end | TensorBoard logs show converging reward curve without collapsing |

Aligns with proposal **Milestone M1 (Week 8):** "GNN policy training."

> **Note on proposal targets:** The proposal's steady-state objective of **< 0.1m**
> formation error is a *project-level* target and is expected to be reached after
> later phases (e.g., safety constraints, trajectory smoothing, stress testing).
> Phase 2's goal is to establish a working GNN coordination policy and
> demonstrate basic formation keeping.

---

## Phase 2 Definition of "Mean Formation Error"

For an environment with \(N\) agents and desired spacing `target_formation_dist`, define the per-step spacing error as the mean absolute pairwise spacing error over unique pairs:

\[
e_t = \frac{2}{N(N-1)} \sum_{i < j} \left| d_{ij}(t) - d^* \right|
\]

Where \(d_{ij}(t)\) is the Euclidean distance between agents \(i\) and \(j\) at step \(t\), and \(d^* =\) `target_formation_dist`.

The **mean formation error** for an evaluation run is the average of \(e_t\) over all evaluation steps across all evaluation episodes.

### Evaluation Procedure (Pass/Fail)

- **Checkpoint**: evaluate `best_agent.pt` (or the newest `agent_*.pt`) under
  `logs/skrl/ggswarm_marl/**/checkpoints/`.
- **Episodes**: 10 evaluation episodes.
- **Episode length**: use the environment's configured episode length.
- **Pass condition**: mean formation error < 0.5m.
- **Secondary metrics (sanity)**:
  - separation event rate (fraction of steps where any pair violates minimum separation)
  - mean linear speed \(\|\mathrm{lin\_vel}\|\) (jitter proxy)

Implementation lives in `scripts/ggswarm_utils/eval_runner.py` with phase-specific
collectors in `scripts/ggswarm_utils/phases/`.

---

## Architecture Changes

### Custom GATv2 Policy Network

Replace the current MLP policy (`[32, 32]` layers) with a GATv2-based network that bridges PyTorch Geometric and SKRL.

```text
Input: agent_obs (12-dim) + extras["adj_matrix"]
  │
  ▼ (Batched 3D Adjacency Flattened to 2D Sparse Edge Index)
  │
GATv2Conv (2 attention heads, max 2 hops)
  │
  ▼
MLP Head → Actions (4-dim)
```

**Curriculum-Based Rewards**
Rewards dynamically scale based on training progress to prevent early-stage training collapse.

| Component | Scale | Formula |
| :--- | :--- | :--- |
| **Separation Penalty** | `-5.0` | Applied if `dist < min_separation_dist` (0.10m) |
| **Formation error** | `+1.0 * α` | `exp(-mean_spacing_error / 0.3)` |
| **Cohesion** | `+0.2 * α` | `exp(-max_neighbor_dist / connectivity_radius)` |
| Position (Hover) | `+15.0 * (1-α+floor)` | `exp(-dist_to_goal / 0.5)` (floor=0.3) |
| Velocity penalty | `-0.05` | L2 body-frame linear velocity |
| Angular vel penalty | `-0.06` | L2 body-frame angular velocity |
| Alive bonus | `0.0` | Disabled (direct moment control) |
| Terminated | `-2.0` | Height-bounds violation |

*(Where `α` scales from 0.0 to 1.0 between `curriculum_start_step=0` and `curriculum_end_step=10000` in `GGSwarmMarlFormationCfg`. At 4096 envs × 3 agents = 12,288 experiences/step, this is ~123M total experiences for the curriculum ramp.)*

---

## SKRL Configuration (`skrl_mappo_cfg.yaml`)

Current production values (tuned during Phase 2A):

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `network.layers` | `[128, 64]` | Larger capacity for multi-agent coordination |
| `trainer.timesteps` | `300000` | Default cap; `--max_iterations` controls actual run length |
| `agent.rollouts` | `64` | Large buffer per update (L4 GPU throughput) |
| `agent.mini_batches` | `8` | Balanced memory/gradient quality |
| `agent.learning_rate` | `1.0e-04` | Stable learning with KLAdaptiveLR scheduler |
| `agent.entropy_loss_scale` | `0.0` | No entropy incentive — PD4-PD17 showed non-zero values cause train-eval gap |
| `state_preprocessor` | `RunningStandardScaler` | Must be restored via `agent.load()` during eval (PD1-PD20 lesson) |

---

## Phase 2 Sub-phases

Phase 2 is split into three sequential sub-phases. Each sub-phase has its own
CLI family, cfg class, and gym task. Advance to the next sub-phase only after
the current assess gate passes (Rule 20).

| Sub-phase | CLI command | Task ID | Config class | num_envs | Gate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A: Hover-Stability | `hover-stability train` | `Template-GGSwarm-Marl-HoverStability-v0` | `GGSwarmMarlHoverStabilityCfg` | 4096 | airborne_ratio > 0.9, mean_roll < 15° (PASSED) |
| B: Formation | `phase2b train --checkpoint <Phase_A/best_agent.pt>` | `Template-GGSwarm-Marl-Formation-v0` | `GGSwarmMarlFormationCfg` | 4096 | formation_error < 0.5m, stability maintained |
| C: Circular Orbit | `phase2c train --checkpoint <Phase_B/best_agent.pt>` | TBD | TBD | 4096 | formation_error < 0.5m while orbiting, stability maintained |

**Phase A note:** `mean_formation_error_m` on the assess scorecard is **not** a Phase 2A pass/fail gate — formation rewards are off; the metric is a rough position/spread proxy only.

**Phase A → B handoff:** pass `best_agent.pt` from the Phase A run via `--checkpoint`.
Per Rule 21, `curriculum_start_step` is set to `0` in `GGSwarmMarlFormationCfg`
because `common_step_counter` resets to 0 on env re-init regardless of checkpoint.

**Phase B → C handoff:** pass `best_agent.pt` from the Phase B run via `--checkpoint`.
Do not implement Phase C until Phase B assess gate passes (Rule 2).

### Phase 2C: Circular Orbit Formation

Phase 2C extends static formation (2B) to dynamic coordination. Drones maintain
a **horizontal circle formation** (equal angular spacing, same altitude) while the
entire formation follows a circular orbit path at constant angular velocity.

**Goal positions:** each agent's target is a slot on a circle of radius
`orbit_radius`, rotating at `orbit_angular_vel` rad/s. Slot `i` is offset by
`2π·i/N` radians from a shared reference angle that advances with time.
All slots share the same Z coordinate (level flight).

**What changes from 2B:**

- Position reward target becomes time-varying (moving slot on orbit)
- Formation reward still uses inter-agent pairwise distance (carried from 2B)
- New metric: **orbit tracking error** — mean distance from each agent to its
  moving slot
- Velocity penalty may need relaxation since agents must sustain forward motion

**Gate criteria:**

- `formation_error < 0.5m` while orbiting (same threshold as 2B)
- `orbit_tracking_error < 0.5m` (agents follow their slots)
- Stability metrics maintained (airborne_ratio, orientation)

---

## Code Implementation

### 1. GNN Policy (`skrl_gnn_policy.py`)

See `source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/agents/skrl_gnn_policy.py`.

Key design: `GGSwarmGNNPolicy` (GATv2 + `GaussianMixin`) receives
`extras["adj_matrix"]` via a monkey-patch on MAPPO's `act()` method
(`patch_mappo_gnn_adj_matrix` in `scripts/ggswarm_utils/sim_helpers.py`).
The 3D adjacency matrix `[num_envs, num_agents, num_agents]` is flattened
to a 2D sparse `edge_index` via `adjacency_to_edge_index` in
`contract_logic.py`, treating each environment as a disconnected subgraph.

Constructor params (`hidden_channels`, `num_heads`, `initial_log_std`) are cfg-driven per Rule 14.

### 2. Curriculum Reward Logic

See `source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/contract_logic.py` (`compute_marl_rewards`, `compute_curriculum_alpha`).

Phase 2A hover-stability uses `compute_stable_hover_rewards` (tanh position,
squared velocity, dt-scaled — Isaac Lab style). Phase 2B formation uses
`compute_marl_rewards` with curriculum `α` that fades formation rewards in
while maintaining a position reward floor.

---

## Training Workflow

```powershell
# Phase A: hover-stability (run on GCE via train_and_push.sh)
python scripts/run.py hover-stability train --headless --gnn

# After GCS pull — assess locally (Rule 20):
python scripts/run.py hover-stability assess \
  --run_dir logs/skrl/ggswarm_marl/<run>

# Phase B: formation resume (run on GCE, pass Phase A checkpoint)
python scripts/run.py phase2b train --headless --gnn \
  --checkpoint logs/skrl/ggswarm_marl/<phase_a_run>/checkpoints/best_agent.pt

# After GCS pull — assess locally (Rule 20):
python scripts/run.py phase2b assess \
  --run_dir logs/skrl/ggswarm_marl/<run>

# Eval and play are always local (GCE is training-only):
python scripts/run.py phase2b eval --gnn \
  --checkpoint logs/skrl/ggswarm_marl/<run>/checkpoints/best_agent.pt
python scripts/run.py phase2b play --gnn \
  --checkpoint logs/skrl/ggswarm_marl/<run>/checkpoints/best_agent.pt

# Phase C: circular orbit (run on GCE, pass Phase B checkpoint)
python scripts/run.py phase2c train --headless --gnn \
  --checkpoint logs/skrl/ggswarm_marl/<phase_b_run>/checkpoints/best_agent.pt

# After GCS pull — assess locally (Rule 20):
python scripts/run.py phase2c assess \
  --run_dir logs/skrl/ggswarm_marl/<run>

tensorboard --logdir logs/skrl/ggswarm_marl
```

---

## Dependencies

- `torch_geometric` — for `GATv2Conv` implementation
- `skrl >= 1.4.3` — MAPPO multi-agent trainer
- `tensorboard` — training visualization

---

## Risks

| Risk | Mitigation |
| :--- | :--- |
| GATv2 over-smoothing with deep layers | Limit to 2–3 attention heads, max 3-hop neighborhood |
| VRAM saturation with 20+ agents | Use headless training; reduce `num_envs` if needed |
| Reward hacking (agents collapse to same point) | Add minimum separation penalty to reward |
| Training instability with formation rewards | Curriculum: start with hover rewards, gradually increase formation weight |
