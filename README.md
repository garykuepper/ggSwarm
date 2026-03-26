# ggSwarm: Decentralized Formation Control for Drone Swarms

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3-76B900?logo=nvidia&logoColor=white)](https://isaac-sim.github.io/IsaacLab/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/isaac-sim)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

`ggSwarm` develops a decentralized UAV swarm controller in NVIDIA Isaac Lab using
CTDE (Centralized Training, Decentralized Execution), graph-based policy learning
with GATv2, and staged milestone delivery.

> Developed with [Claude Code](https://claude.com/claude-code)

## Table of Contents

- [Project Navigation](#project-navigation)
- [Quickstart](#quickstart)
- [Run Demos and Training](#run-demos-and-training)
- [Schedule and Milestones](#schedule-and-milestones)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Navigation

### Design

- [Architecture (GNSC 5-Layer Model)](docs/design/architecture.md) — system design source of truth
- [Proposal and project scope](docs/project/proposal.md)

### Phases

- [Phase 1: Foundation](docs/phases/phase1_foundation.md) — Isaac Lab setup, env implementation
- [Phase 2: Brain Development](docs/phases/phase2_brain_development.md) — GATv2 policy, formation control
- [Phase 3: Muscle Refinement](docs/phases/phase3_muscle_refinement.md) — CBF safety, SwarmRaft, MINCO
- [Phase 4: Stress Testing](docs/phases/phase4_stress_testing.md) — agent loss, obstacles, scale
- [Phase 5: Showcase Prep](docs/phases/phase5_showcase_prep.md) — HD demo, Testing Report
- [Phase 6: Delivery](docs/phases/phase6_delivery.md) — Capstone Festival, submissions

### Operations

- [Commands reference](docs/ops/commands.md) — train, play, eval, video
- [Training workflow](docs/ops/training_workflow.md) — 8-step GCE training cycle
- [Post-training assessment](docs/ops/post_train_analysis.md) — metrics and decision matrix
- [GCS sync workflow](docs/ops/gce_results_sync.md) — push/pull logs from GCE

### Status

- [Weekly updates](docs/status/weekly_updates.md)
- [Changelog](docs/status/changelog.md)
- [Run history](docs/status/run_history.md)

## Quickstart

> Isaac Lab is expected to be installed inside your Python virtual environment.
> No sibling `IsaacLab` repository clone is required for this project setup.

1. Clone the repo:

   ```powershell
   git clone https://github.com/garykuepper/ggSwarm.git
   cd ggSwarm
   ```

2. Create and activate a Python 3.11 virtual environment:

   ```powershell
   py -3.11 -m venv env_isaaclab
   env_isaaclab\Scripts\activate
   python -m pip install --upgrade pip
   ```

3. Install Isaac Sim and Isaac Lab into the active virtual environment:
   - Isaac Sim pip package: [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
   - Isaac Lab install docs: [Isaac Lab Installation](https://isaac-sim.github.io/IsaacLab/)

4. Install this package:

   ```powershell
   pip install -e source/ggSwarm
   ```

5. If you encounter Windows HDF5 DLL conflicts, pin `h5py`:

   ```powershell
   pip install "h5py>=3.9.0,<3.12" --force-reinstall
   ```

## Run Demos and Training

Full command reference: [`docs/ops/commands.md`](docs/ops/commands.md)

```powershell
# Hover baseline
python scripts/run.py hover train --headless
python scripts/run.py hover play --checkpoint <path>

# Phase 2B formation
python scripts/run.py phase2b train --headless --checkpoint <phase2a_best>
python scripts/run.py phase2b play --video --video_prefix p2b-3 --checkpoint <path>

# Phase 2C perturbation
python scripts/run.py phase2c train --headless --checkpoint <phase2b_best>

# Debug
python scripts/run.py debug smoke --task Template-GGSwarm-Marl-Formation-v0 --iterations 1 --gnn --headless
```

## Schedule and Milestones

| Phase | Timeline | Gate |
| :--- | :--- | :--- |
| 1. Foundation | Feb 5 – Feb 17 | - |
| 2. Brain Development | Feb 25 – Mar 25 | M1: Formation control |
| 3. Muscle Refinement | Mar 25 – Apr 7 | M2: Logic integration |
| 4. Stress Testing | Apr 8 – Apr 14 | M3: Mission validation |
| 5. Showcase Prep | Apr 14 – Apr 21 | M4: HD showcase |
| 6. Delivery | Apr 22 – Apr 24 | **Final: Apr 24** |

Full timeline: [`docs/project/proposal.md`](docs/project/proposal.md#7-timeline-and-milestones)
| Progress: [`docs/status/weekly_updates.md`](docs/status/weekly_updates.md)

## Troubleshooting

- `ModuleNotFoundError: No module named 'isaacsim'`
  - Ensure the venv is active: `env_isaaclab\Scripts\activate`
- `ModuleNotFoundError: No module named 'isaaclab'`
  - Isaac Lab is not installed in the active venv
- Long path errors on Windows
  - Enable long path support (`LongPathsEnabled = 1`)
- `ImportError: DLL load failed while importing _errors` (`h5py`)
  - Run: `pip install "h5py>=3.9.0,<3.12" --force-reinstall`

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
