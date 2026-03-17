# Phase 1: Foundation (Documentation)

Phase 1 of the ggSwarm project focuses on building the foundational Isaac Lab environment, setting up the multirotor assets, and calculating the graph connectivity (adjacency matrix) used by the swarm.

## Core Environment Files

The primary environment source code for Phase 1 is located in the task registry:

* **`ggswarm_marl_env.py`**: The main environment class (`GgswarmMarlEnv`). Handles spawning the drones, applying physics forces (thrust/moments), capturing observations, calculating the distance-based adjacency matrix, and distributing rewards/penalties.
* **`ggswarm_marl_env_cfg.py`**: The configuration class (`GgswarmMarlEnvCfg`). Sets the `num_agents` to 20, configures the `CRAZYFLIE_CFG` drone asset, and sets the observation/action space bounds.
* **`__init__.py`**: Registers the environment under the name `Template-Ggswarm-Marl-Direct-v0`.

*Path:* `source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/`

## Phase 1 Execution & Proof

To execute the Phase 1 environment and demonstrate that the drones are successfully loaded and calculating graph connections, a standalone demonstration script is provided.

**Script:** `scripts/phase1_demo.py` (formerly `random_agent.py`)

### How to Run

1. Open your terminal and activate your Isaac Lab environment:

   ```powershell
   conda activate env_isaaclab
   ```

2. Navigate to your `ggSwarm` directory and run the launcher:

   ```powershell
   ..\IsaacLab\isaaclab.bat -p scripts\phase1_demo.py --task=Template-Ggswarm-Marl-Direct-v0
   ```

**What it does:**
This script spawns 20 `Crazyflie` drones and commands them with random physics actions. Concurrently, it prints the live calculation of the adjacency matrix to the terminal (e.g., "[Phase 1 Evidence] Drones connected..."). This provides visual confirmation of the Isaac Sim 3D rendering alongside mathematical proof of the L2 Graph Connectivity layer forming in real-time.
