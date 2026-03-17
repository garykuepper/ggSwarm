# Architecture: ggSwarm Decentralized Drone Coordination

## 1. Overview

ggSwarm is a decentralized formation control framework for large-scale Unmanned Aerial Vehicle (UAV) swarms, built on the NVIDIA Isaac Lab simulation platform. It follows the **Graph Neural Swarm Control (GNSC)** 5-Layer model.

## 2. GNSC 5-Layer Model Mapping

| Layer | Responsibility | Implementation Component |
| :--- | :--- | :--- |
| **L1: Local Sensing** | LiDAR/IMU data collection | `GgswarmMarlEnv` perception buffers |
| **L2: GNN Messaging** | Spatial awareness / GNN | `GATv2` policy networks |
| **L3: Consensus** | Formation alignment | `SwarmRaft` logic |
| **L4: Safety Shield** | Collision avoidance | Control Barrier Functions (CBF) |
| **L5: Execution** | Trajectory following | MINCO Optimization / Low-level PWM |

## 3. Data Flow

1. **Perception:** Each agent gathers local state (pose, velocity) and relative neighbor states.
2. **Adjacency:** A distance-based K-hop graph is constructed to define message-passing edges.
3. **Policy:** GNN processes the graph-structured state to output high-level control intents.
4. **Optimization:** MINCO refines intents into smooth trajectories.
5. **Control:** Actuators track the optimized paths.

---

*Note: This document is maintained as a project rule. All structural changes must be reflected here.*
