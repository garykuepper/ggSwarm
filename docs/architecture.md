# Architecture: ggSwarm Decentralized Drone Coordination

## 1. Overview

ggSwarm is a decentralized formation control framework for large-scale Unmanned
Aerial Vehicle (UAV) swarms, built on the NVIDIA Isaac Lab simulation platform.
It follows the **Graph Neural Swarm Control (GNSC)** 5-Layer model with a
**Centralized Training, Decentralized Execution (CTDE)** workflow.

## 2. GNSC 5-Layer Model Mapping

| Layer | Responsibility | Implementation Component | Phase |
| :--- | :--- | :--- | :--- |
| **L1: Local Sensing** | LiDAR/IMU data collection | `GGSwarmMarlEnv` perception buffers (12-dim obs) | ✅ Phase 1 |
| **L2: GNN Messaging** | Spatial awareness / GNN | Distance-based adjacency matrix → GATv2 policy | ✅ Phase 2 |
| **L3: Consensus** | Formation alignment | `SwarmRaft` logic | ⬜ Phase 3 |
| **L4: Safety Shield** | Collision avoidance | Control Barrier Functions (CBF) | ⬜ Phase 3 |
| **L5: Execution** | Trajectory following | Thrust/moment force application → MINCO (Phase 4) | ✅ Phase 1 (basic) |

## 3. Data Flow

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  L1: Sensing │───▶│ L2: Adjacency│───▶│ L2: GATv2    │───▶│ L5: Force│
│  12-dim obs  │    │  Matrix      │    │  Policy      │    │  Control │
│  per agent   │    │  [N×N]       │    │  (MAPPO/PPO) │    │  4-dim   │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────┘
```

1. **Perception:** Each agent gathers local state (lin_vel, ang_vel, gravity, rel_pos_to_goal).
2. **Adjacency:** A distance-based graph (threshold 2.0m) defines message-passing edges.
3. **Policy:** GATv2 processes graph-structured state to output control actions.
4. **Optimization:** (Phase 4) MINCO refines actions into smooth trajectories.
5. **Control:** Propeller forces and torques are applied via `permanent_wrench_composer`.

**Phase 2 prerequisite baseline:** `GGS-Hover-v0` trains a single drone to hold
its spawn pose before multi-agent formation tuning. This task keeps the same
observation/action interfaces but disables formation objectives and introduces an
explicit ground-hit penalty to prevent reward artifacts while grounded.

## 4. Training Pipeline

| Component | Technology |
| :--- | :--- |
| Framework | NVIDIA Isaac Lab 2.3 / Isaac Sim 5.1 |
| RL Library | SKRL (MAPPO agent) |
| Policy | GATv2Conv (PyTorch Geometric) |
| Optimizer | PPO with KL-adaptive learning rate |
| Compute | Local RTX 3070 (dev) / Cloud GPU (heavy training) |

Single-agent MARL with MAPPO (e.g. `GGS-Hover-v0`): SKRL's sequential trainer uses a
code path that does not populate `infos['shared_states']`; `scripts/skrl/train.py`
injects them from `DirectMARLEnv.state()` when `num_agents == 1` so the centralized
critic receives valid inputs.

## 5. Key Files

| File | Purpose |
| :--- | :--- |
| `drone_swarm_env.py` | MARL environment (scene, physics, obs, rewards, resets) |
| `drone_swarm_env_cfg.py` | Environment configuration (agents, spaces, params) |
| `drone_hover_env.py` | Hover-only baseline env (`GGS-Hover-v0`) with spawn-hold reward |
| `drone_hover_env_cfg.py` | Hover config (single-agent + ground-hit penalty params) |
| `agents/skrl_mappo_cfg.yaml` | SKRL MAPPO hyperparameters |
| `agents/skrl_mappo_hover_cfg.yaml` | SKRL MAPPO hyperparameters for hover baseline |
| `agents/skrl_gnn_policy.py` | GATv2 GNN policy wrapper (PyG bridge) |
| `scripts/skrl/train.py` | Training entry point |
| `scripts/skrl/play.py` | Evaluation / playback entry point |
| `scripts/eval_hover.py` | Hover baseline metrics and pass/fail evaluation |
| `scripts/run.py` | Unified helper CLI for runs (hover/phase2/debug) |

---

*Note: This document is maintained as a project rule. All structural changes must be reflected here.*
