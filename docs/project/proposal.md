# Project Proposal: Decentralized Drone Swarm Formation Control

**Student:** Gary Kuepper
**Advisors:** Eric Tao
**Term:** Spring 2026

> **Edit Policy:** This is the authoritative source document for the project.
> All revisions must preserve the original text via ~~strikethrough~~ and add
> new text inline with a `[Revised]` tag and date. Do not delete original content.

---

## 1. Executive Summary

This project introduces a **decentralized coordination framework** for large UAV
swarns to eliminate single points of failure and high latency inherent in
centralized systems. By integrating **Graph Neural Networks (GNN)** for
spatial reasoning and **Minimum Control (MINCO)** trajectory optimization, the
system achieves robust, fault-tolerant behavior. The architecture features a
"brain" (Graph Attention Network) for scalable awareness and "muscles"
(trajectory optimization) for smooth maneuvers. **SwarmRaft**—a decentralized
adaptation of the Raft consensus algorithm—ensures autonomous recovery after
agent failure. Implementation occurs in **NVIDIA Isaac Lab**, utilizing
GPU-accelerated simulation to train thousands of agents in complex
environments like forests and urban canyons.

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
* **Compute:** Local RTX 3070 (8GB VRAM) for development; ~~cloud-based **NVIDIA `Brev` (A100 80GB)**~~ **GCE NVIDIA L4 (24GB VRAM)** `[Revised 2026-03-28]` for large-scale multi-agent training to ensure convergence within the project timeline.

> **`[Revised 2026-03-28]` Brev A100 → GCE L4:** Switched to Google Compute
> Engine with NVIDIA L4 GPU for cost efficiency. L4's 24GB VRAM is sufficient
> for 4096-env training runs.

---

## 6. Methodology: Reward Shaping

To achieve stable formation control, the reinforcement learning agent is trained using a multi-objective reward function $R$:

~~$$R = w_{pos} \times R_{pos} + w_{vel} \times R_{vel} + w_{ang\_vel} \times R_{ang\_vel} + w_{alive} \times R_{alive} + w_{term} \times R_{term}$$~~

~~* **Formation Error ($R_{pos}$):** A Gaussian-shaped reward based on the Euclidean distance to the desired formation coordinate.~~
~~* **Stability ($R_{vel}, R_{ang\_vel}$):** Penalties on linear and angular velocity jitter to encourage smooth hovering.~~
~~* **Resilience ($R_{alive}$):** A constant positive bonus for each timestep the agent remains within safety bounds.~~
~~* **Termination ($R_{term}$):** A significant negative penalty for collisions or exceeding altitude limits ($0.1m < z < 3.0m$).~~

`[Revised 2026-03-28]` The reward function was restructured during implementation
to better support the CTDE multi-agent architecture:

$$R = R_{hover} + R_{formation}$$

**Hover reward** (always active, per-drone):

* **Goal proximity ($R_{goal}$):** Tanh-mapped distance to goal: $15.0 \times (1 - \tanh(d / 0.8)) \times dt$. In cloud mode, rewards the group centroid reaching the goal rather than individual drones.
* **Velocity penalties ($R_{vel}$):** $-0.05 \times \|v_{lin}\|^2 \times dt$ and $-0.05 \times \|v_{ang}\|^2 \times dt$.

**Formation reward** (curriculum-scaled, cloud mode):

* **Centroid-to-goal ($R_{centroid}$):** Shared reward for group centroid reaching target position.
* **Cohesion ($R_{cohesion}$):** Tanh-mapped distance to group centroid, scale 3.0.
* **Spacing ($R_{spacing}$):** Threshold penalty for nearest neighbor below 0.50m or above 1.0m.

Termination is handled by episode reset (z < 0.05m or z > 2.0m), not reward penalty.
CBF safety shields (L4) enforce hard collision constraints separately from the reward.

---

## 7. Risks and Mitigations

* **VRAM Saturation:** Mitigation involves using **headless training** locally and offloading heavy workloads to the cloud.
* **GNN Over-smoothing:** Deep graph layers can cause identical node features; mitigation involves restricting message passing to **3-hops**.
* **Algorithm Divergence:** Complex shapes may fail to converge; mitigation involves **curriculum learning**.

---

## 7. Timeline and Milestones

### Timeline and Milestones

| Weeks | Dates | Phase | Activity | Milestone |
| :--- | :--- | :--- | :--- | :--- |
| 5–6 | Feb 5 – Feb 17 | 1. Foundation | Install NVIDIA Isaac Lab; configure simulated multirotor assets; finalize graph connectivity logic. | - |
| 7–8 | Feb 18 – Mar 3 | 2. Brain Development | Train GATv2 with ~~MAPPO~~ **PPO (CTDE)**[^1]; test formation in empty space. | **M1 (Week 8):** GNN policy training |
| 9–11 | Mar 4 – Mar 24 | 3. Muscle Refinement | Integrate MINCO trajectory optimization as a post-processing layer; implement SwarmRaft consensus logic. | **M2 (Week 11, by 3/24):** Logic integration |
| 12–13 | Mar 25 – Apr 7 | 4. Stress Testing | Conduct simulated agent loss tests; benchmark swarm navigation in high-density obstacle environments. | - |
| 14–15 | Apr 8 – Apr 21 | 5. Showcase Prep | Finalize RTX Tiled Rendering; record HD demo; compile results. | **M3 (Week 14, by 4/14):** Mission success validation; **M4 (Week 15):** HD showcase + Testing Report |
| 16 | Apr 22 – Apr 24 | 6. Delivery | Present at Capstone Festival; submit Portfolio and Learning Journals. | Final Presentation due 4/24/26 |

[^1]: `[Revised 2026-03-28]` **MAPPO → PPO (CTDE):** SKRL's MAPPO integration was problematic; switched to Centralized Training, Decentralized Execution — one shared PPO policy across all drones, each executing independently.

---

## 8. Final Deliverables

1. **Technical Repository:** GATv2 training pipeline, MINCO scripts, and environment configs.
2. **Isaac Lab USD Stage:** Photorealistic "Cluttered Forest" and "Urban Canyon" training environments.
3. **Visual Showcase:** HD video of shape transitions (Hexagon, Circle), obstacle negotiation, and fault recovery.
4. **Testing Report:** Comprehensive analysis of stability, recovery latency, and collision rates across 100+ episodes.
