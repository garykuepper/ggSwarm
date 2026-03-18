# ggSwarm: Decentralized Formation Control for Drone Swarms

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3-76B900?logo=nvidia&logoColor=white)](https://isaac-sim.github.io/IsaacLab/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/isaac-sim)
[![skrl](https://img.shields.io/badge/skrl-1.1.0-blueviolet?logo=github&logoColor=white)](https://skrl.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)](#)

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

## Development Phases

| Phase | Weeks | Focus | Status |
| :--- | :--- | :--- | :--- |
| **1. Foundation** | 5–6 | Isaac Lab env, drone spawning, graph connectivity | ✅ Complete |
| **2. Brain Development** | 7–8 | GATv2 policy training with PPO/MAPPO | 🔄 In Progress |
| **3. Muscle Refinement** | 9–10 | MINCO optimization, SwarmRaft consensus | ⬜ Planned |
| **4. Stress Testing** | 11–12 | Agent failure tests, obstacle environments | ⬜ Planned |
| **5. Showcase Prep** | 13–15 | HD rendering, final validation, demo video | ⬜ Planned |

*Detailed plans: [Phase 1](docs/phase1_foundation.md) · [Phase 2](docs/phase2_brain_development.md)*

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

To verify the Phase 1 Foundation is working (spawning drones and calculating graph connectivity), run the Phase 1 demonstration script:

```bash
..\IsaacLab\isaaclab.bat -p scripts/phase1_demo.py --task=Template-Ggswarm-Marl-Direct-v0
```

*Note: For a detailed breakdown of the Phase 1 codebase, please read [Phase 1 Documentation](docs/phase1_foundation.md).*

### Running the Training (Phase 2)

To start training with SKRL, run (from `ggSwarm` directory):

```bash
..\IsaacLab\isaaclab.bat -p scripts/skrl/train.py --task=Template-Ggswarm-Marl-Direct-v0 --algorithm=MAPPO
```

*Note: For the full Phase 2 plan including GATv2 integration and reward tuning, see [Phase 2 Documentation](docs/phase2_brain_development.md).*

## Code Quality

This project uses **Ruff** for linting and formatting, and **pre-commit** to ensure code standards.

### Setting up pre-commit

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install
```

### Running manual checks

```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files

# Run ruff directly
ruff check .
ruff format .
```

---

## Citation

If you use this work in your research, please cite:
(TBD)
