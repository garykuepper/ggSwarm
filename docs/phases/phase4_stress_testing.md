# Phase 4: Stress Testing

**Timeline:** Apr 8 -- Apr 14 (Week 14)  |  **Gate:** M3 -- Mission success validation

## 1. Goals

| ID | Goal | Success Criteria |
| :--- | :--- | :--- |
| P4.1 | Swarm recovers from agent loss | Formation re-syncs in < 2.0 s across >= 90% of kill events |
| P4.2 | Zero inter-agent collisions | 0 collisions over 100 evaluation episodes (with CBF) |
| P4.3 | Swarm scales to 10+ agents | Formation maintained with train-3 / deploy-N via K-nearest |
| P4.4 | Testing Report data produced | Metrics covering nominal, agent-loss, and scale scenarios |

## 2. Tasks

All work extends the Phase 3 stack via config flags in `GgswarmEnvCfg`.
No core env rewrites needed.

**Agent loss testing** — use Phase 3's agent loss recovery (P3.4). Run
play with `--kill_drone` flag to simulate mid-episode agent death. Measure
gap-fill latency (steps from kill to formation error returning below 0.5m).

**Collision testing** — use Phase 3's CBF safety shield (P3.3). Run dense
formation scenarios (target_spacing 0.3m) and verify zero collisions.

**Scale benchmarking** — sweep `--num_agents` (3, 6, 10, 15, 20) using
Phase 2C's K-nearest deployment. Record formation error, VRAM usage,
and steps/second at each scale. All use the same checkpoint (trained
with 3 agents).

**Evaluation suite** — run 25 episodes each across scenarios:

- Nominal (3 agents, formation hover)
- Agent loss (3 agents, kill at random step)
- Scale (6, 10 agents, formation hover)
- Dense formation (3 agents, target_spacing 0.3m)

## 3. Design Integration

Phase 4 introduces no architectural changes. It validates Phase 3
components under stress.

```text
Phase 3 Stack (GNN + EMA + CBF + Agent Loss)
    |
    +-- Agent Loss Testing (P3.4 validation)
    +-- CBF Collision Testing (P3.3 validation)
    +-- Scale Benchmark (Phase 2C validation)
    +-- 100-Episode Eval Suite
    |
    v
Testing Report Data
```

### Commands

```powershell
# Nominal evaluation
python scripts/skrl/play.py --task ggswarm-v0 --num_agents 3 --num_envs 3 `
  --policy gnn --checkpoint <path> --trajectories --play_length 500

# Scale test (10 agents, same checkpoint)
python scripts/skrl/play.py --task ggswarm-v0 --num_agents 10 --num_envs 10 `
  --policy gnn --checkpoint <path> --trajectories

# Agent loss test
python scripts/skrl/play.py --task ggswarm-v0 --num_agents 3 --num_envs 3 `
  --policy gnn --checkpoint <path> --kill_drone 1 --trajectories

# TensorBoard
tensorboard --logdir logs/skrl/ggswarm
```

### Pass Criteria

| Criterion | Threshold | Proposal Objective |
| :--- | :--- | :--- |
| Inter-agent collision rate | 0 / 100 episodes | O1 |
| Gap-fill latency (median) | < 2.0 s | O3 |
| Formation error, nominal | < 0.5 m | O1 |
| Scale test (10 agents) | Formation maintained | O4 |

## 4. Results

Phase 4 has not started.

---

## See Also

- [Phase 3: Muscle Refinement](phase3_muscle_refinement.md)
- [Architecture](../design/architecture.md)
