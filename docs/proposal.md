# Project Proposal: Decentralized Drone Swarm Formation Control

**Student:** Gary Kuepper
**Advisors:** Eric Tao
**Term:** Spring 2026

---

## 1. Executive Summary

This project introduces a **decentralized coordination framework** for large UAV swarms to eliminate single points of failure and high latency inherent in centralized systems. By integrating **Graph Neural Networks (GNN)** for spatial reasoning and **Minimum Control (MINCO)** trajectory optimization, the system achieves robust, fault-tolerant behavior. The architecture features a "brain" (Graph Attention Network) for scalable awareness and "muscles" (trajectory optimization) for smooth maneuvers. **SwarmRaft**—a decentralized adaptation of the Raft consensus algorithm—ensures autonomous recovery after agent failure. Implementation occurs in **NVIDIA Isaac Lab**, utilizing GPU-accelerated simulation to train thousands of agents in complex environments like forests and urban canyons.

---

## 2. Problem Statement

* **Centralization Risks:** Traditional methods rely on centralized planning, creating a single point of failure.
* **Scalability:** Control and coordination become increasingly difficult as swarm size and task complexity grow.
* **Robustness:** Reliability decreases when communication is interrupted or agents are lost.

---

## 3. Proposed Solution: GNSC 5-Layer Architecture

The project employs a **Centralized Training, Decentralized Execution (CTDE)** workflow using the five-layer Graph Neural Swarm Control (GNSC) model:

* **L1 - Local Sensing:** Agents collect LiDAR and IMU perception data locally within Isaac Lab.
* **L2 - GNN Message Passing:** Information is aggregated within a K-hop neighborhood using **GATv2** to establish spatial awareness.
* **L3 - Distributed Consensus:** **SwarmRaft** logic (a peer-to-peer consensus mechanism) allows agents to align on global formation objectives through local peer interactions without a central leader.
* **L4 - Runtime Safety Shields:** **Control Barrier Functions (CBFs)** enforce hard safety constraints to ensure zero inter-agent collisions.
* **L5 - Mission Execution:** The swarm executes high-level commands, such as "Move to Goal" or "Change Formation Shape".

---

## 4. Goals and Objectives

### Project Goals

* **G1:** Establish a scalable coordination policy using graph-based learning.
* **G2:** Achieve mathematically optimal flight paths for swarm agents.
* **G3:** Ensure mission continuity despite individual drone failures or signal loss.
* **G4:** Create professional-grade visual validation of swarm autonomy.

### Technical Objectives

* **O1:** Maintain a mean formation error of **< 0.1m** during steady-state flight.
* **O2:** Reduce velocity jitter by at least **20%** via MINCO optimization.
* **O3:** Re-sync the swarm and fill gaps within **2.0s** of a simulated failure using SwarmRaft.
* **O4:** Produce an HD demonstration video of **20+ agents** navigating a cluttered forest.

---

## 5. Implementation Details

* **Platform:** NVIDIA Isaac Lab 2.3 and Isaac Sim 5.1.
* **RL Library:** SKRL using Proximal Policy Optimization (PPO).
* **Software Stack:** `PyTorch` 2.5+ and `OpenUSD`.
* **Compute:** Local RTX 3070 (8GB VRAM) for development; cloud-based **NVIDIA Brev (A100 80GB)** for large-scale multi-agent training to ensure convergence within the project timeline.

---

## 6. Methodology: Reward Shaping

To achieve stable formation control, the reinforcement learning agent is trained using a multi-objective reward function $R$:

$$R = w_{pos} \cdot R_{pos} + w_{vel} \cdot R_{vel} + w_{ang\_vel} \cdot R_{ang\_vel} + w_{alive} \cdot R_{alive} + w_{term} \cdot R_{term}$$

* **Formation Error ($R_{pos}$):** A Gaussian-shaped reward based on the Euclidean distance to the desired formation coordinate.
* **Stability ($R_{vel}, R_{ang\_vel}$):** Penalties on linear and angular velocity jitter to encourage smooth hovering.
* **Resilience ($R_{alive}$):** A constant positive bonus for each timestep the agent remains within safety bounds.
* **Termination ($R_{term}$):** A significant negative penalty for collisions or exceeding altitude limits ($0.1m < z < 3.0m$).

---

## 7. Risks and Mitigations

* **VRAM Saturation:** Mitigation involves using **headless training** locally and offloading heavy workloads to the cloud.
* **GNN Over-smoothing:** Deep graph layers can cause identical node features; mitigation involves restricting message passing to **3-hops**.
* **Algorithm Divergence:** Complex shapes may fail to converge; mitigation involves **curriculum learning**.

---

## 7. Timeline and Milestones

### Timeline and Milestones

| # | Weeks | Dates | Phase | Activity | Milestone |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 5–6 | Feb 5 – Feb 17 | Foundation | Install NVIDIA Isaac Lab; configure simulated multirotor assets; finalize graph connectivity logic. | - |
| 2 | 7–8 | Feb 18 – Mar 3 | Brain Development | Train the GATv2 policy using **Multi-Agent PPO (MAPPO)**; test basic formation keeping in empty space. | **M1 (Week 8):** GNN policy training |
| 3 | 9–11 | Mar 4 – Mar 24 | Muscle Refinement | Integrate MINCO trajectory optimization as a post-processing layer; implement SwarmRaft consensus logic. | **M2 (Week 11, by 3/24):** Logic integration |
| 4 | 12–13 | Mar 25 – Apr 7 | Stress Testing | Conduct simulated agent loss tests; benchmark swarm navigation in high-density obstacle environments. | - |
| 5 | 14–15 | Apr 8 – Apr 21 | Showcase Prep | Finalize RTX Tiled Rendering; record HD demonstration; compile results into the final Testing Report. | **M3 (Week 14, by 4/14):** Mission success validation; **M4 (Week 15, by 4/21):** HD showcase + Testing Report + Presentation Draft |
| 6 | 16 | Apr 22 – Apr 24 | Delivery | Present at Capstone Festival; submit Portfolio and Learning Journals. | Final Presentation due 4/24/26 |

---

## 8. Final Deliverables

1. **Technical Repository:** GATv2 training pipeline, MINCO scripts, and environment configs.
2. **Isaac Lab USD Stage:** Photorealistic "Cluttered Forest" and "Urban Canyon" training environments.
3. **Visual Showcase:** HD video of shape transitions (Hexagon, Circle), obstacle negotiation, and fault recovery.
4. **Testing Report:** Comprehensive analysis of stability, recovery latency, and collision rates across 100+ episodes.
