# ggSwarm: Decentralized Formation Control for Drone Swarms

This project develops a **decentralized coordination framework** for Unmanned Aerial Vehicle (UAV) swarms to eliminate single points of failure and high latency inherent in centralized systems. The system is built within **NVIDIA Isaac Lab**, utilizing GPU-accelerated reinforcement learning to enable autonomous, resilient swarm behavior.

## Core Architecture (GNSC 5-Layer Model)

The project utilizes a **Centralized Training, Decentralized Execution (CTDE)** workflow organized into five functional layers:

1. Local Sensing: Agents collect LiDAR and IMU data within the simulation.
2. GNN Message Passing: A **GATv2 (Graph Attention Network)** serves as the "brain," enabling spatial awareness via local message passing within a K-hop neighborhood.
3. Distributed Consensus: Agents utilize **SwarmRaft** logic to align on global objectives and re-synchronize autonomously after agent failures.
4. Runtime Safety Shields: **Control Barrier Functions (CBFs)** enforce hard constraints to prevent inter-agent collisions.
5. Mission Execution: The "muscles" of the system use **Minimum Control (MINCO) trajectory optimization** to execute smooth, feasible maneuvers.

## Technical Objectives & Performance Targets

* Scalability: Maintain a mean formation error of **< 0.1m** during steady-state flight.
* Fluidity: Reduce velocity jitter by at least **20%** through MINCO optimization.
* Resilience: Achieve autonomous re-synchronization and gap-filling within **2.0 seconds** of an agent failure.
* Efficiency: Maintain end-to-end decision latency under **90ms**.
* Robustness: Maintain a **> 95% success rate** while navigating cluttered environments like forests or urban canyons.

## Development Stack

* Simulation: NVIDIA Isaac Sim 5.1 / Isaac Lab 2.3.
* **Learning:**
* **Goal**: Develop a decentralized drone swarm capable of coordinated movement and formation control.
* **Layers**:
  * Layer 1: Simulated Multirotor (`Crazyflie`)
  * Layer 2: Graph Connectivity (Distance-based)
  * Layer 3: GATv2 Policy (SKRL/PPO)
  * Layer 4: Consensus Mechanism (Phase 3)
  * Layer 5: Mission Planning (Phase 4)

## Installation

```bash
# Clone the repository
git clone https://github.com/garykuepper/ggSwarm.git
cd ggSwarm
```

## Setup Environment

```bash
# Create the conda environment
conda create -n env_isaaclab python=3.10
conda activate env_isaaclab
```

## Install Dependencies

```bash
# Install Isaac Lab (ensure ISAACLAB_PATH is set)
pip install -e source/ggSwarm
```

### Running the Phase 1 Demo

To verify the Phase 1 Foundation is working (spawning 20 drones and calculating graph connectivity), run the Phase 1 demonstration script:

```bash
..\IsaacLab\isaaclab.bat -p scripts/phase1_demo.py --task=Template-Ggswarm-Marl-Direct-v0
```

*Note: For a detailed breakdown of the Phase 1 codebase, please read [Phase 1 Documentation](docs/phase1_foundation.md).*

### Running the Training (Phase 2)

To start training with SKRL, run (from `ggSwarm` directory):

```bash
..\IsaacLab\isaaclab.bat -p scripts/skrl/train.py --task=Template-Ggswarm-Marl-Direct-v0
```

---

## Citation

If you use this work in your research, please cite:
(TBD)