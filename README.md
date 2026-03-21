# ggSwarm: Decentralized Formation Control for Drone Swarms

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3-76B900?logo=nvidia&logoColor=white)](https://isaac-sim.github.io/IsaacLab/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/isaac-sim)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

`ggSwarm` develops a decentralized UAV swarm controller in NVIDIA Isaac Lab using
CTDE, graph-based policy learning, and staged milestone delivery. This README is
kept intentionally concise: quick setup, run links, and project navigation.

## Table of Contents

- [Project Navigation](#project-navigation)
- [Quickstart](#quickstart)
- [Run Demos and Training](#run-demos-and-training)
- [Schedule and Milestones](#schedule-and-milestones)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Project Navigation

- Proposal and project scope: [`docs/project/proposal.md`](docs/project/proposal.md)
- Architecture source of truth: [`docs/design/architecture.md`](docs/design/architecture.md)
- Commands reference (train, play, eval, video): [`docs/ops/commands.md`](docs/ops/commands.md)
- GCE training + monitoring: [`docs/ops/gce_training_and_monitoring.md`](docs/ops/gce_training_and_monitoring.md)
- GCS sync workflow: [`docs/ops/gce_results_sync.md`](docs/ops/gce_results_sync.md)
- Phase notes:
  - [`docs/design/phase1_foundation.md`](docs/design/phase1_foundation.md)
  - [`docs/design/phase2_brain_development.md`](docs/design/phase2_brain_development.md)
- Status and change tracking:
  - [`docs/status/weekly_updates.md`](docs/status/weekly_updates.md)
  - [`docs/status/changelog.md`](docs/status/changelog.md)

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

Use the dedicated run guide for current commands and workflows:

- [`docs/ops/commands.md`](docs/ops/commands.md)

Quick unified helper examples:

```powershell
# Hover baseline
python scripts/run.py hover train --headless
python scripts/run.py hover play --checkpoint <path_to_checkpoint>
python scripts/run.py hover eval --num_episodes 10
python scripts/run.py hover monitor

# Phase 2 formation
python scripts/run.py phase2 train --headless
python scripts/run.py phase2 eval --num_episodes 10
python scripts/run.py phase2 monitor

# Debug utilities
python scripts/run.py debug smoke --task GGS-Hover-v0 --iterations 1 --headless
python scripts/run.py debug latest-checkpoint --family hover
```

## Schedule and Milestones

The schedule and milestone plan is maintained in the proposal:

- [`docs/project/proposal.md` timeline section](docs/project/proposal.md#7-timeline-and-milestones)

Progress and timeline changes are tracked over time in:

- [`docs/status/weekly_updates.md`](docs/status/weekly_updates.md)

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
