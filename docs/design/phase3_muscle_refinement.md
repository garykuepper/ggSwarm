# Phase 3: Muscle Refinement (Weeks 9–11, Mar 25 – Apr 7)

Phase 3 adds the three remaining GNSC layers on top of the Phase 2 GATv2 policy:
**L4 CBF Safety Shield**, **L3 SwarmRaft Consensus**, and **L5 MINCO Trajectory
Smoother**. All components are post-policy filters or goal managers — they do **not**
require retraining the Phase 2 checkpoint and are individually config-gated so the
Phase 2 `phase2 train/play` path is unaffected.

---

## Objectives

| ID | Objective | Success Criteria |
| :--- | :--- | :--- |
| P3.1 | Hard inter-agent collision avoidance via CBF | **Zero** collision events across 10 evaluation episodes |
| P3.2 | Autonomous gap-filling after agent loss via SwarmRaft | Formation re-syncs within **2.0 s** of a simulated failure |
| P3.3 | Velocity jitter reduction via MINCO smoother | **≥ 20%** reduction in `std(‖lin_vel‖)` vs. Phase 2 baseline |
| P3.4 | Steady-state formation error maintained or improved | Mean formation error **< 0.5 m** (Phase 2 criterion maintained) |

Aligns with proposal **Milestone M2 (Week 13, Apr 7):** "Logic integration."

> **Proposal project-level targets** (< 0.1 m formation error, 0 collisions, 2 s recovery)
> are the *cumulative* outcome expected after Phase 3 + Phase 4 stress testing.
> Phase 3 establishes the mechanisms; Phase 4 validates them at scale.

---

## Architecture Changes

### Data Flow (Phase 3)

```text
┌─────────────┐   ┌──────────────┐   ┌──────────────┐
│  L1: Sensing │──▶│ L2: GATv2    │──▶│ raw actions  │
│  12-dim obs  │   │  Policy      │   │  [N×4]       │
└─────────────┘   └──────────────┘   └──────┬───────┘
        ▲                                    │
        │                             ┌──────▼───────┐
        │                             │ L4: CBF      │
        │                             │  Safety      │
        │                             └──────┬───────┘
        │                                    │ safe actions
        │                             ┌──────▼───────┐
        │                             │ L5: MINCO    │
        │                             │  Smoother    │
        │                             └──────┬───────┘
        │                                    │ smooth actions
        │                             ┌──────▼───────┐
        │                             │ L5: Force    │
        │                             │  Control     │
        │                             └──────────────┘
        │
┌───────┴──────┐
│ L3: SwarmRaft│──▶ updates _desired_pos_w on agent loss
│  Consensus   │
└──────────────┘
```

### New Files

| File | Layer | Purpose |
| :--- | :--- | :--- |
| `ggswarm_marl/cbf_safety.py` | L4 | Pairwise CBF barriers + analytical projection |
| `ggswarm_marl/swarm_raft.py` | L3 | Heartbeat detection, leader election, formation redistribution |
| `ggswarm_marl/minco_trajectory.py` | L5 | EMA action smoother (MVP); polynomial upgrade path |
| `scripts/eval_phase3.py` | — | Phase 3 evaluation script |
| `docs/design/phase3_muscle_refinement.md` | — | This document |

All three new modules are **pure-torch** with no Isaac Lab imports (same pattern as
`contract_logic.py`) so they are unit-testable in isolation.

---

## 3A. CBF Safety Shield (L4)

### Design

The Control Barrier Function for pairwise agent safety:

```
h_ij(x) = ‖p_i − p_j‖² − d_safe²
```

When `h_ij` approaches zero the raw action for agent `i` is projected onto the
safe half-space:

```
ȧ_safe = ȧ_raw − min(0, (∇h · ȧ_raw + γ·h) / ‖∇h‖²) · ∇h
```

where `γ` is the barrier decay rate (config param `cbf_gamma`). Only the thrust
component (action dim 0) is corrected since that dominates horizontal velocity;
moment corrections are applied proportionally if the barrier is critically violated.
No QP solver is required for pairwise spherical constraints.

### Config Parameters (`GGSwarmMarlEnvCfg`)

```python
cbf_enabled: bool = False          # False keeps Phase 2 behavior unchanged
cbf_d_safe: float = 0.12           # slightly above min_separation_dist (0.10 m)
cbf_gamma: float = 1.0             # barrier decay rate
rew_scale_cbf_intervention: float = 0.0  # set negative to penalize CBF reliance
```

### Integration Point

`_pre_physics_step` in `drone_swarm_env.py`, between action stacking and thrust
mapping:

```python
if self.cfg.cbf_enabled:
    all_actions, cbf_rate = apply_cbf_safety(
        all_actions, pos_w, lin_vel_w, self.cfg
    )
    self.extras["log"]["cbf_intervention_rate"] = cbf_rate
```

### Tensor Shapes

| Tensor | Shape |
| :--- | :--- |
| `actions` (input/output) | `[num_envs, num_agents, 4]` |
| `pos_w` | `[num_envs, num_agents, 3]` |
| `lin_vel_w` | `[num_envs, num_agents, 3]` |
| `cbf_intervention_mask` | `[num_envs, num_agents]` bool |

---

## 3B. SwarmRaft Consensus (L3)

### Design

A simplified Raft-style consensus adapted for swarm simulation:

1. **Heartbeat:** Each alive agent resets a counter each step. Missing heartbeats
   (counter exceeds `raft_heartbeat_timeout`) flag the agent as lost.
2. **Leader election:** Deterministic — the lowest-index alive agent per environment
   is the leader. No message rounds needed; the simulation's shared tensor makes
   this instantaneous and split-brain-free.
3. **Formation redistribution:** Leader recomputes circular formation slots for the
   remaining alive agents using the same chord-length formula as `_reset_idx`, then
   writes new targets into `_desired_pos_w`.
4. **Cooldown:** A `raft_replan_cooldown` guard prevents thrashing when multiple
   agents are lost in quick succession.

### Config Parameters (`GGSwarmMarlEnvCfg`)

```python
raft_enabled: bool = False          # False keeps Phase 2 behavior unchanged
raft_tick_interval: int = 10        # physics steps between SwarmRaft ticks
raft_heartbeat_timeout: int = 50    # steps before declaring agent lost
raft_replan_cooldown: int = 100     # min steps between formation replans
```

### Observation Space

SwarmRaft adds **2 dims** to each agent's observation (obs: 12 → 14):

| Dim | Meaning |
| :--- | :--- |
| 13 | `is_leader` — 1.0 if this agent is the current leader, else 0.0 |
| 14 | `num_alive_frac` — fraction of original agents still alive (0–1) |

To preserve Phase 2 checkpoint compatibility, the obs expansion is **only active
when `raft_enabled = True`**. The registered task `Template-GGSwarm-Marl-Direct-v0`
keeps 12-dim obs. A new registration `Template-GGSwarm-Marl-Phase3-v0` uses 14-dim
obs for Phase 3 training runs.

### Tensor Shapes

| Tensor | Shape |
| :--- | :--- |
| `heartbeat` | `[num_envs, num_agents]` int |
| `agent_alive` | `[num_envs, num_agents]` bool |
| `raft_replan_counter` | `[num_envs]` int |

---

## 3C. MINCO Trajectory Smoother (L5)

### Design

Maintains a per-agent action history buffer and applies a causal exponential moving
average (EMA) to smooth the raw policy output:

```
a_smooth[t] = α · a_raw[t] + (1 − α) · a_smooth[t−1]
```

where `α = minco_smoothing_alpha` (lower = more smoothing). This is the **MVP**
implementation. The buffer also stores position history to enable an upgrade to a
true minimum-snap polynomial fit (5th-order) once EMA is validated.

### Config Parameters (`GGSwarmMarlEnvCfg`)

```python
minco_enabled: bool = False         # False keeps Phase 2 behavior unchanged
minco_buffer_size: int = 5          # action history window (for polynomial upgrade)
minco_smoothing_alpha: float = 0.3  # EMA coefficient (lower = smoother)
```

### Integration Point

`_pre_physics_step`, after CBF filter, before thrust mapping:

```python
if self.cfg.minco_enabled:
    all_actions = self._minco.smooth(all_actions)
```

### Tensor Shapes

| Tensor | Shape |
| :--- | :--- |
| `action_buffer` | `[num_envs, num_agents, buffer_size, 4]` |
| `smooth_actions` (output) | `[num_envs, num_agents, 4]` |

---

## Implementation Plan

### Step 1: CBF (Priority 1, ~2 days)

1. Implement `cbf_safety.py` — pure-torch, fully unit-tested.
2. Add config params (`cbf_enabled`, `cbf_d_safe`, `cbf_gamma`) to env cfg.
3. Integrate into `_pre_physics_step` behind `cbf_enabled` flag.
4. Log `cbf_intervention_rate` to `extras["log"]`.
5. Verify with Phase 2 checkpoint: CBF should intervene ~0% when policy is well-trained.

### Step 2: SwarmRaft (Priority 2, ~3–4 days)

1. Implement `swarm_raft.py` — pure-torch, unit-tested.
2. Add config params to env cfg.
3. Register `Template-GGSwarm-Marl-Phase3-v0` with 14-dim obs.
4. Integrate SwarmRaft tick into `_get_observations` (gated by `raft_enabled`).
5. Add `is_leader` / `num_alive_frac` dims to obs construction when enabled.
6. Add `scripts/run.py phase3 train/play/eval` commands.

### Step 3: MINCO (Priority 3, ~2 days)

1. Implement `minco_trajectory.py` — pure-torch, unit-tested.
2. Add config params and `MincoSmoother` instance to env `__init__`.
3. Integrate into `_pre_physics_step` after CBF, behind `minco_enabled` flag.
4. Reset buffer on env reset in `_reset_idx`.

### Step 4: Evaluation

1. Implement `scripts/eval_phase3.py` with metrics for P3.1–P3.4.
2. Add `python scripts/run.py phase3 eval` command.
3. Run baseline comparison: Phase 2 checkpoint with Phase 3 features toggled.

---

## Evaluation Procedure

### Metrics

| Metric | Target | Maps to |
| :--- | :--- | :--- |
| Collision rate (events/episode) | **0** | O1, P3.1 |
| Gap-fill latency (s) | **< 2.0** | O3, P3.2 |
| Velocity std reduction | **≥ 20%** | O2, P3.3 |
| Mean formation error (m) | **< 0.5** | P3.4 |

### Procedure

- **Checkpoint:** `best_agent.pt` from Phase 2 training (or Phase 3 retrain if
  obs space expanded).
- **Episodes:** 10 nominal + 10 with one agent killed mid-episode.
- **CBF test:** Run with `cbf_enabled=True`; verify zero collisions.
- **SwarmRaft test:** Kill agent 0 at step 50; measure steps until formation error
  returns below 0.5 m and compute wall-clock latency.
- **MINCO test:** Compare `std(‖lin_vel_b‖)` over 10 episodes with
  `minco_enabled=False` vs. `True`.

---

## Risks and Mitigations

| Risk | Mitigation |
| :--- | :--- |
| CBF over-correction causes oscillation | Tune `cbf_gamma`; add deadband around barrier threshold |
| SwarmRaft obs expansion invalidates Phase 2 checkpoint | Keep 12-dim task registered; Phase 3 task is separate |
| MINCO EMA over-smooths and slows response | Start with `alpha=0.6` (light smoothing); measure latency impact |
| Formation redistribution causes transient collisions during gap-fill | CBF is active during redistribution to prevent collisions |
| `raft_replan_cooldown` too short → thrashing | Default 100 steps (~1 s at 100 Hz); increase if oscillation observed |

---

## Dependencies

- `torch` — all computations
- No new third-party packages required
- Phase 2 training result (`best_agent.pt`) as evaluation baseline
