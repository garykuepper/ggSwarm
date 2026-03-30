# Phase 4: Stress Testing

**Timeline:** Apr 8 -- Apr 14 (Week 14)  |  **Gate:** M3 -- Mission success validation

## 1. Goals

| ID | Goal | Success Criteria |
| :--- | :--- | :--- |
| P4.1 | Polygon-mode SwarmRaft demo | Octagon → heptagon transition visible on drone kill; formation re-syncs in < 2.0 s |
| P4.2 | Steady-state hover | Drones hold position once formation converges; velocity penalties tuned to minimize drift |
| P4.3 | Zero inter-agent collisions | 0 collisions over 100 evaluation episodes (with CBF) |
| P4.4 | Swarm scales to 10+ agents | Formation maintained with train-8 / deploy-N via K-nearest |
| P4.5 | Obstacle environment testing | Benchmark swarm navigation in cluttered environments |
| P4.6 | Testing Report data produced | Metrics covering nominal, agent-loss, scale, and obstacle scenarios |

## 2. Tasks

### P4.1 Polygon-Mode SwarmRaft

Phase 3 implemented SwarmRaft in cloud mode (KNN cohesion, no rigid slots).
Phase 4 upgrades to polygon mode for a compelling visual demo:

- Switch `formation_mode="polygon"` with `dropout_enabled=True`
- Add **dynamic slot recomputation** on dropout: when a drone dies, recompute
  `_formation_offsets` for A-1 alive agents (new polygon with fewer vertices)
- Reassign `_desired_pos_w` for surviving drones to new slot positions
- Train from scratch with polygon formation + dropout
- Target: octagon (8) → heptagon (7) → hexagon (6) visible transition

### P4.2 Steady-State Hover

Drones currently micro-drift because velocity penalties are weak (-0.05).
Tune to stop drones once formation converges:

- Increase `lin_vel_reward_scale` from -0.05 to -0.2 or higher
- Increase `ang_vel_reward_scale` from -0.05 to -0.2 or higher
- Alternatively: conditional velocity penalty (only penalize heavily after
  formation converged) to avoid slowing initial goal-seeking
- Retrain and verify: drones should hover still once in position

### P4.3 Collision Testing

Validate Phase 3's CBF + collision termination under stress:

- Dense formation scenarios (target_spacing 0.3m)
- 100-episode evaluation runs
- Verify zero collisions with CBF + MINCO-CBF sync

### P4.4 Scale Benchmarking

Sweep `--num_agents` (8, 10, 15, 20) using the K-nearest deployment:

- All use the same checkpoint (trained with 8 agents)
- Record: formation error, KNN distances, VRAM usage, steps/second
- Verify KNN-based cohesion scales correctly (no centroid dependency)

### P4.5 Obstacle Environments

Benchmark swarm navigation in cluttered environments:

- Add static obstacles to the terrain
- Verify CBF prevents obstacle collisions
- Measure mission success rate

### P4.6 Evaluation Suite

Run 25 episodes each across scenarios:

- Nominal (8 agents, cloud hover)
- Polygon formation (8 agents, polygon hover)
- Agent loss (8 agents, kill 1 at random step)
- Scale (10, 15 agents, cloud hover)
- Dense formation (8 agents, target_spacing 0.3m)

## 3. Design Integration

Phase 4 builds on Phase 3's complete L1-L5 stack:

```text
Phase 3 Stack (GNN + MINCO + CBF + SwarmRaft + Collision Detection)
    |
    +-- P4.1: Polygon SwarmRaft (new formation_mode + dynamic offsets)
    +-- P4.2: Velocity Penalty Tuning (config change + retrain)
    +-- P4.3: CBF Collision Stress Test (eval only)
    +-- P4.4: Scale Benchmark (eval only, deploy-N)
    +-- P4.5: Obstacle Environments (terrain config)
    +-- P4.6: Evaluation Suite (data collection)
    |
    v
Testing Report Data → M3 Gate (Apr 14)
```

### Key Phase 3 Components Available

| Component | Status | File |
| :--- | :--- | :--- |
| GATv2 GNN (L2) | Done | `gnn_policy.py` |
| MINCO min-jerk (L3) | Done | `minco.py` |
| SwarmRaft dropout (L3) | Done | `ggswarm_env.py` |
| CBF safety shield (L4) | Done | `cbf.py` |
| Virtual collision detection | Done | `ggswarm_env.py` |
| KNN-based cohesion | Done | `ggswarm_env.py` |
| MINCO-CBF state sync | Done | `ggswarm_env.py` |

### Commands

```powershell
# Nominal evaluation (8 agents)
python scripts/skrl/play.py --task ggswarm-v0 --num_agents 8 --num_envs 8 `
  --policy gnn --checkpoint <path> --trajectories --play_length 500

# Scale test (15 agents, same checkpoint)
python scripts/skrl/play.py --task ggswarm-v0 --num_agents 15 --num_envs 15 `
  --policy gnn --checkpoint <path> --trajectories

# Polygon formation with dropout
python scripts/skrl/play.py --task ggswarm-v0 --num_agents 8 --num_envs 8 `
  --policy gnn --checkpoint <path> --trajectories --play_length 1000

# TensorBoard
tensorboard --logdir logs/skrl/ggswarm
```

### Pass Criteria

| Criterion | Threshold | Proposal Objective |
| :--- | :--- | :--- |
| Inter-agent collision rate | 0 / 100 episodes | O1 |
| Gap-fill latency (median) | < 2.0 s (100 steps) | O3 |
| Formation error, nominal | < 0.5 m | O1 |
| Scale test (15 agents) | Formation maintained | O4 |
| Velocity jitter reduction | ≥ 20% vs raw GNN | O2 |
| Steady-state hover drift | < 0.05 m/s mean velocity | New |

## 4. Results

Phase 4 has not started. Scheduled: Apr 8 -- Apr 14.

---

## See Also

- [Phase 3: Muscle Refinement](phase3_muscle_refinement.md)
- [Architecture](../design/architecture.md)
