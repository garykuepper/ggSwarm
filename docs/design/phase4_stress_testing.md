# Phase 4: Stress Testing (Weeks 12–13, Apr 8 – Apr 14)

Phase 4 validates the full GNSC stack under adversarial conditions. The Phase 3
mechanisms (CBF, SwarmRaft, MINCO) are proven in isolation during Phase 3; Phase 4
subjects them to compound failures, cluttered environments, and increasing swarm
scale to ensure the proposal's quantitative targets hold in realistic scenarios.

---

## Objectives

| ID | Objective | Success Criteria |
| :--- | :--- | :--- |
| P4.1 | Swarm recovers from in-episode agent loss | Formation re-syncs in **< 2.0 s** across ≥ 90% of kill events (O3) |
| P4.2 | Zero collisions in obstacle environments | **0** inter-agent or agent–obstacle collisions over 100 evaluation episodes |
| P4.3 | Swarm scales to 20+ agents with acceptable performance | Formation error **< 0.5 m** and VRAM **< 20 GB** at 20 agents |
| P4.4 | Produce Testing Report data | Metrics table covering nominal, agent-loss, sparse, and dense obstacle scenarios |

Aligns with proposal **Milestone M3 (Week 14, Apr 14):** "Mission success validation."

> **Proposal project-level targets** (< 0.1 m formation error, 0 collisions, 2 s recovery)
> are the *cumulative* targets.  Phase 4 determines whether the full stack meets them
> and identifies residual gaps to address in Phase 5 polish.

---

## Architecture Changes

Phase 4 extends the Phase 3 stack without modifying any of its core modules. All
additions are environment-level (obstacle spawning, kill mechanism) and evaluation
tooling (scale benchmarking, metrics aggregation).

### New Files

| File | Purpose |
| :--- | :--- |
| `drone_swarm_env_cfg.py` (extended) | `agent_loss_enabled`, `obstacle_enabled`, `obstacle_count`, `obstacle_radius` params |
| `drone_swarm_env_cfg_showcase.py` | Pre-built Phase 4/5 scenario configs inheriting from Phase 3 |
| `scripts/bench_scale.py` | Sweeps `num_agents` and records VRAM / steps-per-second / formation error |
| `scripts/eval_phase3.py` (extended) | Phase 4 agent-loss and obstacle scenario metrics added to the same script |

---

## 4A. Simulated Agent Loss

### Design

At a configurable random interval, one agent per environment is forcibly terminated
by setting its body position outside the episode bounds (triggering the existing
out-of-bounds done condition). SwarmRaft detects the missing heartbeat and replans
the formation within `raft_heartbeat_timeout` steps.

### Config Parameters (`GGSwarmMarlEnvCfg`)

```python
agent_loss_enabled: bool = False            # False keeps nominal Phase 3 behavior
agent_loss_interval_min: int = 200          # earliest step at which a kill fires
agent_loss_interval_max: int = 500          # latest step at which a kill fires
```

A single kill per episode is the default. More complex multi-kill sequences are
handled by running multiple episodes with fresh `_kill_step` samples.

### Key Tensors

| Tensor | Shape | Description |
| :--- | :--- | :--- |
| `_kill_step` | `[num_envs]` int | Randomly sampled step for the kill event |
| `_kill_agent_idx` | `[num_envs]` int | Randomly sampled agent index to kill |
| `_force_terminated` | `[num_envs, num_agents]` bool | Mask applied in `_get_dones` |

### Integration Point

`_get_dones` in `drone_swarm_env.py`:

```python
if self.cfg.agent_loss_enabled:
    kill_now = (self.episode_length_buf == self._kill_step)
    self._force_terminated[kill_now, self._kill_agent_idx[kill_now]] = True
    out_of_bounds |= self._force_terminated
```

### Metrics

| Metric | Target |
| :--- | :--- |
| Gap-fill latency (s) | < 2.0 s (P4.1 / O3) |
| Formation error during transition (m) | < 1.0 m peak |
| Collision events during transition | 0 |

Gap-fill latency is measured from the step of the kill event to the step where the
mean formation error first returns below 0.5 m.

---

## 4B. Obstacle Environments

### Design

Static cylinder obstacles are procedurally placed in `_setup_scene` using Isaac Lab's
`RigidObjectCfg`. Their XY positions are sampled on a grid (dense) or uniformly at
random (sparse) around each environment origin. The existing
`apply_cbf_obstacle_safety` function in `cbf_safety.py` is activated automatically
when `obstacle_enabled = True` and `_obstacle_positions` is populated.

### Config Parameters (`GGSwarmMarlEnvCfg`)

```python
obstacle_enabled: bool = False
obstacle_count: int = 10                    # obstacles per environment
obstacle_radius: float = 0.15              # cylinder radius (m)
obstacle_height: float = 2.0              # cylinder height (m)
obstacle_field_size: float = 2.0          # half-side of procedural placement zone (m)
cbf_obstacle_d_safe: float = 0.20         # safety radius around each obstacle (m)
```

### Scenario Presets (`drone_swarm_env_cfg_showcase.py`)

| Scenario | `obstacle_count` | Placement | Task ID |
| :--- | :--- | :--- | :--- |
| Nominal (no obstacles) | 0 | — | `Template-GGSwarm-Marl-Phase3-v0` |
| Sparse | 5–10 | Random | `Template-GGSwarm-Marl-Phase4-v0` |
| Dense | 20–30 | Random | `GGS-ClutteredForest-v0` |
| Urban Canyon | 12–16 | Grid | `GGS-UrbanCanyon-v0` |

### CBF Extension

The obstacle barrier function mirrors the inter-agent barrier:

```
h_obs(x) = ‖p_i − p_obs‖² − r_obs²
```

`apply_cbf_obstacle_safety` in `cbf_safety.py` processes each `(agent, obstacle)`
pair independently after the inter-agent CBF pass.

---

## 4C. Scale Testing

### Design

`scripts/bench_scale.py` instantiates the Phase 3 environment at increasing `num_agents`
values, runs for a fixed number of steps (e.g., 500), and records:

- **Mean formation error** — does spatial reasoning degrade?
- **VRAM usage** — `torch.cuda.max_memory_allocated()` after warmup
- **Steps per second** — throughput measured via wall-clock timer
- **CBF computation time** — fraction of step time in `apply_cbf_safety` (scales O(N²))

### Scale Points

| `num_agents` | Expected bottleneck |
| :--- | :--- |
| 3 | Baseline |
| 5 | — |
| 10 | GATv2 graph growth |
| 15 | CBF pairwise cost visible |
| 20 | Proposal showcase target (O4) |
| 25 | Beyond proposal scope; VRAM limit check |

Results are written to `logs/bench_scale_<timestamp>.csv` and printed as a markdown
table to stdout.

---

## Implementation Plan

### Step 1: Agent Loss Mechanism (~1 day)

1. Add `agent_loss_enabled`, `agent_loss_interval_min/max` to `GGSwarmMarlEnvCfg`.
2. Add kill-step sampling and `_force_terminated` buffer to `drone_swarm_env.py __init__`.
3. Integrate forced termination in `_get_dones` behind the flag.
4. Reset kill state in `_reset_idx`.
5. Add `python scripts/run.py phase4 eval` command wrapping `eval_phase3.py` with agent-loss config.

### Step 2: Obstacle Environments (~2 days)

1. Add obstacle config params to `GGSwarmMarlEnvCfg`.
2. Implement procedural cylinder spawning in `_setup_scene` (gated by `obstacle_enabled`).
3. Store `_obstacle_positions` and pass to `apply_cbf_obstacle_safety` in `_pre_physics_step`.
4. Create `drone_swarm_env_cfg_showcase.py` with `GGSwarmClutteredForestCfg` and `GGSwarmUrbanCanyonCfg`.
5. Register `GGS-ClutteredForest-v0` and `GGS-UrbanCanyon-v0` task IDs.

### Step 3: Scale Benchmarking (~1 day)

1. Implement `scripts/bench_scale.py` with sweep loop, VRAM and timing measurements.
2. Add `python scripts/run.py phase4 bench` command.
3. Run benchmark locally and on GCE; record results in a table in the Testing Report.

### Step 4: 100-Episode Evaluation Suite (~1 day)

1. Extend `scripts/eval_phase3.py` with agent-loss and obstacle scenario support.
2. Define four scenario configs: nominal, kill, sparse-obstacle, dense-obstacle.
3. Run 25 episodes per scenario (100 total) and aggregate metrics.
4. Write `logs/phase4_eval_<timestamp>.json` with the full metrics table.

---

## Evaluation Procedure

### Scenario Matrix (100 episodes)

| Scenario | Episodes | Agent Loss | Obstacles | Key Metric |
| :--- | :--- | :--- | :--- | :--- |
| Nominal | 25 | No | No | Formation error < 0.5 m |
| Agent Kill | 25 | Yes (1 per ep.) | No | Gap-fill latency < 2.0 s |
| Sparse Obstacles | 25 | No | 5–10 | 0 collisions |
| Dense Obstacles | 25 | No | 20–30 | 0 collisions, formation error < 0.5 m |

### Pass Criteria (all must hold for Phase 4 completion)

| Criterion | Threshold | Proposal Objective |
| :--- | :--- | :--- |
| Inter-agent collision rate | **0** events / 100 episodes | O1 |
| Agent–obstacle collision rate | **0** events / 100 episodes | O1 |
| Gap-fill latency (median) | **< 2.0 s** | O3 |
| Formation error, nominal (mean) | **< 0.5 m** | O1 (P3.4 carried forward) |
| VRAM at 20 agents | **< 20 GB** | O4 prerequisite |

---

## Risks and Mitigations

| Risk | Mitigation |
| :--- | :--- |
| CBF oscillation near obstacles during gap-fill | Increase `cbf_activation_margin`; tune `cbf_obstacle_d_safe` |
| SwarmRaft replan collision during formation transition | CBF remains active during all transitions; cooldown prevents rapid replanning |
| Scale benchmark exceeds VRAM at 20 agents | Reduce `num_envs` per env instance; use headless mode; profile GATv2 head count |
| Procedural obstacles spawn inside agent spawn zone | Add exclusion radius around `env_origin` for obstacle sampling |
| 100-episode eval suite too slow for iteration | Cache environment resets; reduce episode length to 300 steps for benchmarking |

---

## Dependencies

- Phase 3 complete: `cbf_safety.py`, `swarm_raft.py`, `minco_trajectory.py` all integrated and unit-tested
- Isaac Lab `RigidObjectCfg` for cylinder spawning
- `torch.cuda.max_memory_allocated()` for VRAM measurement
- Phase 3 checkpoint (`best_agent.pt`) as evaluation policy
