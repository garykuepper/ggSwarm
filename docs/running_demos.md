# Running ggSwarm Demonstrations

This document provides instructions on how to run the various simulation and training demonstrations for the ggSwarm project.

> **Important:** All commands assume you have activated your virtual environment (run `.\env_isaaclab\Scripts\activate` in each new terminal session) and are running from the `ggSwarm` project root directory.

## Phase 1: Foundation Demo

This demo validates the foundational environment: spawning the drone swarm, tracking state, and computing the distance-based adjacency matrices (Graph Connectivity) without any intelligent policy acting on it.

```powershell
python scripts/phase1_demo.py --task=Template-GGSwarm-Marl-Direct-v0
```

## Phase 2: Brain Training & GNN Policy

Phase 2 development introduces the **Graph Attention Network (GATv2)** policy to enable local message passing among the swarm.

### 1. Training the Swarm

To start training the drone swarm with the decentralized GATv2 model using MAPPO, run:

```powershell
python scripts\skrl\train.py --task=Template-GGSwarm-Marl-Direct-v0 --algorithm=MAPPO --headless --ml_framework torch --gnn
```

* This command runs headless for faster simulation.
* Tensorboard logs and checkpoints are automatically saved to `logs/skrl/ggswarm_marl/`.

### 2. Monitoring Training

You can monitor the curriculum rewards and losses via TensorBoard:

```powershell
tensorboard --logdir=logs/skrl/ggswarm_marl
```

### 3. Evaluating the Trained Swarm (Playback)

To watch the agents fly using the latest trained model checkpoint in the Isaac Sim GUI:

```powershell
python scripts\skrl\play.py --task=Template-GGSwarm-Marl-Direct-v0 --algorithm=MAPPO --ml_framework torch --gnn
```

## Phase 2 Utility Shortcuts

For convenience, you can use the centralized `run_phase2.py` script to handle training, playback, and monitoring:

* **Train**: `python scripts/run_phase2.py train`
* **Play**: `python scripts/run_phase2.py play`
* **Monitor**: `python scripts/run_phase2.py monitor`
