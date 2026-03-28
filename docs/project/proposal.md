# Decentralized Formation Control for Drone Swarms

**CST489/499 Online Capstone**
**Student:** Gary Kuepper
**Advisors:** Eric Tao, Brian Robertson
**Term:** Spring 2026

> **Edit Policy:** This is the authoritative source document for the project.
> All revisions must preserve the original text via ~~strikethrough~~ and add
> new text inline with a `[Revised]` tag and date. Do not delete original content.

---

## 1. Executive Summary

This capstone project introduces a decentralized coordination framework to overcome
the single point of failure and high latency of centralized control in large UAV
swarms, crucial for high-stakes applications. The system achieves robust,
fault-tolerant behavior by integrating **Graph Neural Networks** for spatial reasoning
with advanced **Minimum Control (MINCO)** trajectory optimization for smooth flight
dynamics.

The architecture comprises the "brain," a Graph Attention Network that enables
scalable, permutation-invariant spatial awareness via local message passing, and the
"muscles," which use trajectory optimization to ensure dynamically feasible maneuvers
with reduced velocity jitter. **SwarmRaft**, a decentralized consensus logic, further
ensures autonomous re-synchronization and recovery after agent failure.

Implementation occurs in **NVIDIA Isaac Lab**, using GPU-accelerated simulation for
parallel training across thousands of environments, including cluttered forests and
urban canyons. Performance targets include a mean formation error below 0.1m,
re-synchronization within two seconds of failure, a success rate over 95% in dense
environments, and end-to-end decision latency under 90ms.

The final output will be a technical repository (PyTorch, OpenUSD) and a visual
showcase of complex shape transitions and autonomous fault recovery. This work
demonstrates that decentralized, self-organizing intelligence is the necessary
evolution for resilient autonomous systems.

---

## 2. Introduction/Background

This capstone project focuses on the design and simulation of a decentralized drone
swarm formation control system. The purpose of this proposal is to outline the goals
of the project, the technological problem it addresses, and the proposed solution.
This document also discusses related work in the field, identifies key stakeholders,
and examines ethical and legal considerations associated with the project.

### Project Name and Description

The project, *Decentralized Drone Swarm Formation Control*, is an individual capstone
project with no external client. The product is a software-based simulation that
models a group of autonomous agents capable of forming geometric formations without
centralized control. The project is intended for computer science students,
researchers, and engineers interested in decentralized control of autonomous agents.
Its purpose is to explore modern decentralized coordination strategies in which swarm
behavior emerges using only local agent information.

### Problem and/or Issue in Technology

Drone swarms are being explored for coordinated surveillance, reconnaissance, disaster
response, agriculture field monitoring, and large-scale aerial displays. Research
efforts have demonstrated autonomous coordination among 100+ drones, and defense
programs are developing swarms with hundreds of units capable of cooperative tasks.
However, these applications underscore the technological challenge of scaling control
and coordination as swarm size and task complexity increase.

### Solution to the Problem

This capstone project focuses on developing a decentralized drone swarm formation
system within a software simulation environment. The core principle involves
individual agents computing their actions autonomously, relying solely on local
information to facilitate formation-keeping and transition. This approach allows
coordination to emerge from decentralized interactions, responding to high-level
formation commands without requiring a central controller. Utilizing a simulation will
enable efficient evaluation and refinement of the system, keeping the project feasible
within the capstone's time constraints.

---

## 3. Environmental Scan/Literature Review

Research into controlling unmanned aerial vehicle (UAV) swarm formations has
established various coordination strategies, such as leader–follower, virtual
structure, behavior-based, and consensus-based approaches (Bu, Yan, & Yang, 2024).
These foundational techniques have proven effective in small-scale and controlled
settings. However, a major limitation of many traditional methods is their reliance
on centralized planning, thus a single point of failure. This introduces scalability
challenges and reduces robustness, particularly as the swarm grows or communication
reliability decreases (Bu et al., 2024).

To enhance scalability and robustness, recent research has shifted toward
decentralized and distributed control strategies. Decentralized approaches empower
individual agents to make decisions based only on local information, thereby
minimizing single points of failure and reducing the need for extensive global
communication. Prior simulation-based studies have demonstrated that decentralized
coordination can successfully maintain formation integrity and adapt to dynamic events
like agent loss or communication interference (Jiang et al., 2022; Wang, Li, & Chen,
2024).

Evaluating swarm coordination techniques often relies on simulation-based
experimentation, primarily due to the high cost, complexity, and safety concerns
associated with large-scale physical drone deployments. Simulation environments offer
a rapid means for researchers to test decentralized formation control strategies and
assess performance across diverse conditions without being limited by hardware (Xia,
Liu, & Sun, 2024). Accordingly, this proposed project will adopt a simulation-based
methodology to investigate decentralized swarm formation control, focusing
specifically on generating flexible formations and executing transitions driven by
high-level commands, rather than following predefined trajectories.

---

## 4. Stakeholders

The primary stakeholder for this project is the student developer, who will gain
experience in decentralized control, simulation design, and research-oriented software
development. Capstone faculty are also key stakeholders, as they evaluate the
project's technical quality, documentation, and adherence to course objectives. These
stakeholders benefit from a well-defined and feasible project that demonstrates
applied computer science concepts within an academic research context.

Additional stakeholders include computer science students, researchers, and engineers
who may reference the project or its results for educational or exploratory purposes.
These stakeholders stand to gain insight into decentralized swarm coordination
techniques and simulation-based experimentation. Because the project is limited to a
software-based simulation and does not involve real-world deployment, the potential
risks or losses for stakeholders are minimal.

### Ethical Considerations

Although the project is implemented entirely in simulation, ethical considerations
include the responsible presentation of decentralized swarm technologies, which may
have dual-use implications such as military or surveillance applications. To mitigate
these concerns, the project will be clearly framed as an academic research exercise.
No personal data will be collected, and the simulation will not model real individuals
or sensitive environments. Clear documentation of assumptions, limitations, and
intended use will help ensure the project is interpreted appropriately by stakeholders.

### Legal Considerations

Legal considerations for this project primarily involve intellectual property and
software licensing. All third-party libraries and tools used will comply with their
respective open-source licenses, and proper attribution will be provided where
required. The project's code, visualizations, and documentation will be original
works or appropriately cited if adapted from existing sources. No proprietary datasets
or copyrighted materials will be used without permission, ensuring compliance with
academic and legal standards.

---

## 5. Project Goals and Objectives

### Project Goals

| ID | Goal |
| :--- | :--- |
| **G1** | Establish a scalable coordination policy using graph-based learning. |
| **G2** | Achieve mathematically optimal flight paths for swarm agents. |
| **G3** | Ensure mission continuity despite individual drone failures or signal loss. |
| **G4** | Create a professional-grade visual validation of swarm autonomy. |

### Technical Objectives

| ID | Objective |
| :--- | :--- |
| **O1** | Implement a GATv2 control policy in Isaac Lab that maintains a mean formation error of less than 0.1m during steady-state flight. |
| **O2** | Integrate a MINCO trajectory optimization module to reduce velocity jitter by at least 20% compared to raw neural network outputs. |
| **O3** | Implement SwarmRaft consensus logic to automatically re-sync the swarm and fill gaps within 2.0s of a simulated drone failure. |
| **O4** | Produce a professional HD demonstration video of a swarm (20+ simulated agents) navigating a cluttered forest environment in NVIDIA Isaac Lab. |

---

## 6. Final Deliverables

1. **Technical Repository:** A Git-based repository containing the GATv2 training pipeline (Python/PyTorch), MINCO optimization scripts, and environment configuration files.
2. **Isaac Lab USD Stage:** A configured simulation stage utilizing OpenUSD, featuring photorealistic "Cluttered Forest" and "Urban Canyon" training environments.
3. **Visual Showcase:** A high-definition video demonstration showing the swarm forming geometric shapes (Hexagon, Circle), negotiating obstacles, and executing fault recovery.
4. **Technical Testing Report:** A comprehensive analysis of geometry stability, recovery latency, and collision rates across 100+ simulated episodes.

---

## 7. Approach/Methodology: GNSC 5-Layer Architecture

This project employs a **Centralized Training, Decentralized Execution (CTDE)**
reinforcement learning workflow within the Graph Neural Swarm Control (GNSC) 5-layer
architecture:

1. **(L1) Local Sensing:** ~~Simulated perception data (LiDAR/IMU) is collected
   locally for each agent in the Isaac Lab environment.~~ Each agent observes its
   own body-frame state and K-nearest neighbor relative positions.
   `[Revised 2026-03-28]`
2. **(L2) GNN Message Passing:** Information is aggregated within a ~~K-hop
   neighborhood~~ fully-connected within-group graph using GATv2 to establish
   spatial awareness. Sparse K-nearest edges will be used when scaling to 20+
   agents. `[Revised 2026-03-28]`
3. **(L3) Distributed Consensus:** Agents align on global formation objectives
   through local peer interactions.
4. **(L4) Runtime Safety Shields:** Control Barrier Functions (CBFs) are applied to
   enforce hard safety constraints, ensuring zero inter-agent collisions.
5. **(L5) Mission Execution:** The swarm executes high-level commands, such as
   "Move to Goal" or "Change Formation Shape."

> **`[Revised 2026-03-28]` L1 Local Sensing:** The implementation uses an abstract
> 12D observation vector (body-frame velocities, projected gravity, goal direction)
> plus K-nearest neighbor relative positions (6D for K=2). This is sensor-agnostic —
> the observation could come from any sensing modality. LiDAR may be added in Phase 4
> for obstacle avoidance if needed.
>
> **`[Revised 2026-03-28]` L2 GNN Message Passing:** For A=8 drones per group,
> fully-connected within-group edges (56 per group) are used. 2 GATv2 layers with
> residual connections and LayerNorm give full group coverage. Will transition to
> sparse K-nearest edges when scaling to 20+ agents.

---

## 8. Methodology: Reward Shaping

To achieve stable formation control, the reinforcement learning agent is trained
using a multi-objective reward function:

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

## 9. Timeline/Resources

| Weeks | Phase | Activity |
| :---: | :---: | :--- |
| 5–6 | Foundation | Install NVIDIA Isaac Lab; configure simulated multirotor assets; finalize graph connectivity logic. |
| 7–8 | Brain Development | Train the GATv2 policy using Proximal Policy Optimization (PPO); test basic formation keeping in empty space. |
| 9–10 | Muscle Refinement | Integrate MINCO trajectory optimization as a post-processing layer; implement SwarmRaft consensus logic. |
| 11–12 | Stress Testing | Conduct simulated agent loss tests; benchmark swarm navigation in high-density obstacle environments. |
| 13–15 | Showcase Prep | Finalize RTX Tiled Rendering; record HD demonstration; compile results into the final Testing Report. |
| 16 | Delivery | Present at Capstone Festival; submit Portfolio and Learning Journals. |

### Milestones

* **M1 (Week 8):** Successful training of a GNN coordination policy for basic formation holding.
* **M2 (Week 10):** Integration of trajectory refinement and fault-tolerant consensus.
* **M3 (Week 14):** Validation of mission success rates (>95%) in cluttered environments.
* **M4 (Week 15):** Completion of the high-fidelity 1080p visual showcase.

### Resources Needed

* **Local Compute:** Desktop with **NVIDIA RTX 3070 (8GB VRAM)** for code development and small-scale testing.
* **Cloud Compute:** ~~**NVIDIA Brev** or **AWS Batch** (g6.2xlarge instances)~~ **GCE NVIDIA L4 (24GB VRAM)** `[Revised 2026-03-28]` for high-environment-count training and showcase rendering.
* **Software Stack:** NVIDIA Isaac Sim 5.1, Isaac Lab 2.3, PyTorch 2.5+, and the SKRL library.

> **`[Revised 2026-03-28]` Brev/AWS → GCE L4:** Switched to Google Compute Engine
> with NVIDIA L4 GPU for cost efficiency. L4's 24GB VRAM is sufficient for 4096-env
> training runs.

---

## 10. Platform

**NVIDIA Isaac Lab** is the chosen platform because it is the current industry
standard for robot learning, offering massive GPU-parallel experience collection
(>10^5 environments) and a unified API for multi-agent reinforcement learning.

---

## 11. Risks and Dependencies

### Risks

* **Hardware VRAM Saturation:** The RTX 3070 8GB VRAM is insufficient for complex swarms. *Mitigation:* Use **headless training** locally; offload heavy workloads to **cloud deployment**.
* **GNN Over-smoothing:** Deep graph layers can lead to identical node features.
  *Mitigation:* ~~Restrict message passing to 3-hops.~~ Use 2 GATv2 layers with
  residual connections and LayerNorm. `[Revised 2026-03-28]`
* **Algorithm Divergence:** RL may fail to converge on complex shapes. *Mitigation:* Utilize **curriculum learning**.

### Dependencies

* **Driver Compatibility:** Isaac Lab 2.x requires specific NVIDIA production branch drivers (e.g., version 535+ for Linux or 537+ for Windows).

---

## 12. Testing Plan

The system will be verified using simulated benchmarks:

* **Geometry Stability Test:** Measure the Euclidean distance error between intended waypoints and actual drone positions. *Target: < 0.1m*.
* **Fault Recovery Test:** Programmatically "kill" 20% of the swarm; measure the time required to fill the gaps using SwarmRaft. *Target: < 2.0s*.
* **Obstacle Negotiation:** Run 100 episodes through randomized forests. *Target: > 95%* success rate.
* **Decision Latency:** Measure end-to-end latency from sensor input to motor command. *Target: < 90ms*.

---

## 13. References

Bu, Y., Yan, Y., & Yang, Y. (2024). Advancement challenges in UAV swarm formation
control: A comprehensive review. *Drones, 8*(7), 320.
https://doi.org/10.3390/drones8070320

Jiang, Z., Wang, H., Liu, Y., & Zhang, Q. (2022). Distributed coordinated control
scheme of UAV swarm based on heterogeneous roles. *Journal of Systems Engineering
and Electronics, 33*(2), 456–467.
https://doi.org/10.23919/JSEE.2022.000040

Ma, N., & Zhang, H. (2024). Consensus-based distributed formation control.
*Transactions of the Institute of Measurement and Control, 46*(4), 789–801.
https://doi.org/10.1177/01423312231171815

Wang, Z., Li, X., & Chen, J. (2024). A decentralized decision-making algorithm for
UAV swarms under communication constraints. *Expert Systems with Applications, 233*,
121933.
https://doi.org/10.1016/j.eswa.2023.121933

Xia, B., Liu, Y., & Sun, Q. (2024). Decentralized control strategies for UAV swarms:
A multi-layered approach. *Drones, 8*(8), 350.
https://doi.org/10.3390/drones8080350
