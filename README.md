# ggSwarm: Decentralized Formation Control for Drone Swarms

This project develops a **decentralized coordination framework** for Unmanned Aerial Vehicle (UAV) swarms to eliminate single points of failure and high latency inherent in centralized systems. The system is built within **NVIDIA Isaac Lab**, utilizing GPU-accelerated reinforcement learning to enable autonomous, resilient swarm behavior.

## Core Architecture (GNSC 5-Layer Model)

The project utilizes a **Centralized Training, Decentralized Execution (CTDE)** workflow organized into five functional layers:

1.  **Local Sensing:** Agents collect LiDAR and IMU data within the simulation.
2.  **GNN Message Passing:** A **GATv2 (Graph Attention Network)** serves as the "brain," enabling spatial awareness via local message passing within a K-hop neighborhood.
3.  **Distributed Consensus:** Agents utilize **SwarmRaft** logic to align on global objectives and re-synchronize autonomously after agent failures.
4.  **Runtime Safety Shields:** **Control Barrier Functions (CBFs)** enforce hard constraints to prevent inter-agent collisions.
5.  **Mission Execution:** The "muscles" of the system use **Minimum Control (MINCO) trajectory optimization** to execute smooth, feasible maneuvers.

## Technical Objectives & Performance Targets

*   **Scalability:** Maintain a mean formation error of **< 0.1m** during steady-state flight.
*   **Fluidity:** Reduce velocity jitter by at least **20%** through MINCO optimization.
*   **Resilience:** Achieve autonomous re-synchronization and gap-filling within **2.0 seconds** of an agent failure.
*   **Efficiency:** Maintain end-to-end decision latency under **90ms**.
*   **Robustness:** Maintain a **> 95% success rate** while navigating cluttered environments like forests or urban canyons.

## Development Stack

*   **Simulation:** NVIDIA Isaac Sim 5.1 / Isaac Lab 2.3.
*   **Learning:** PyTorch 2.5+ using Proximal Policy Optimization (PPO) via the SKRL library.
*   **Environment:** OpenUSD-based stages featuring complex, high-density obstacles.

---

## Getting Started

### Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/garykuepper/ggSwarm.git
    cd ggSwarm
    ```

2.  **Environment Setup**:
    Ensure you have Isaac Lab installed and the `env_isaaclab` conda environment active.
    ```bash
    conda activate env_isaaclab
    ```

3.  **Install the Extension**:
    ```bash
    python -m pip install -e source/ggSwarm
    ```

### Running the Training

To list the available environments, run:
```bash
python scripts/environments/list_envs.py
```

To start training with SKRL, run:
```bash
python scripts/skrl/train.py --task=Template-Gg-Swarm-Marl-Direct-v0
```

---

## Citation

If you use this work in your research, please cite:
(TBD)