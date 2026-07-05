[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3-76B900?logo=nvidia&logoColor=white)](https://isaac-sim.github.io/IsaacLab/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/isaac-sim)
![Capstone](https://img.shields.io/badge/Capstone-Shipped%20Apr%202026-green)
![Live](https://img.shields.io/badge/ggSwarm%20Live-Active-blue)

![ggSwarm — A Decentralized Drone Swarm](docs/assets/banner.jpg)

A learned, graph-attention-based multi-drone formation policy — and the
program building it out from a sim-only capstone into a real-hardware
drone-show product.

## Two programs in one repo

### ggSwarm Capstone (v1, shipped April 2026)

A simulation-only multi-agent RL drone swarm. Tested whether a single
GATv2 + PPO policy can replace hand-designed multi-agent coordination
logic. **Yes, within tested scope:** 8 drones hold triangle, polygon,
and letter-G formations with <0.3 m steady-state error, recover from
mid-episode dropout within 2 s (SwarmRaft), navigate cylinder forests,
and the same policy generalizes to 20 agents without retraining.

Frozen at tag `v1.0.0-capstone` on the `capstone` branch. All
documentation, phase write-ups, testing report, and run history live
under [`docs/capstone/`](docs/capstone/).

![ggSwarm highlight](docs/assets/ggswarm-highlight.gif)

### ggSwarm Live (active)

Real-hardware research program. Takes the capstone policy out of an
idealized, centralized simulation and makes it work decentralized,
under real aerodynamics, and eventually on real drones. Two phases:
**Phase 1 (sim)** — proper decentralization (no anchors, no central
coordinator) and downwash/aero physics fidelity — then **Phase 2
(hardware)** — a goal list, not a detailed plan, for getting the policy
flying on real drones. See
[`docs/ggswarm_live/vision.md`](docs/ggswarm_live/vision.md).

A separate drone-light-show project (its own algorithm, developed to
fund this research) is **not** part of ggSwarm; the earlier combined
plan is preserved at
[`docs/ggswarm_live/archive/`](docs/ggswarm_live/archive/) for reference.

## Table of Contents

- [Two programs in one repo](#two-programs-in-one-repo)
  - [ggSwarm Capstone (v1, shipped April 2026)](#ggswarm-capstone-v1-shipped-april-2026)
  - [ggSwarm Live (active)](#ggswarm-live-active)
- [Project Navigation](#project-navigation)
  - [ggSwarm Live](#ggswarm-live)
  - [Capstone (frozen)](#capstone-frozen)
- [Quickstart](#quickstart)
- [Training and Playback (capstone-era sim)](#training-and-playback-capstone-era-sim)
- [License](#license)

## Project Navigation

Top-level docs splitter: [`docs/README.md`](docs/README.md).

### ggSwarm Live

- [Program overview](docs/ggswarm_live/README.md)
- [Vision](docs/ggswarm_live/vision.md)
- Phases:
  [0 capstone baseline](docs/ggswarm_live/phases/phase0_capstone_baseline.md) ·
  [1 sim: decentralization + downwash](docs/ggswarm_live/phases/phase1_sim.md) ·
  [2 hardware transfer](docs/ggswarm_live/phases/phase2_hardware.md)
- [Backlog](docs/ggswarm_live/backlog.md) — loose, unscheduled ideas
- [References](docs/ggswarm_live/references.md)
- [Archive](docs/ggswarm_live/archive/) — earlier 18-phase plan, kept for reference
- Status: [changelog](docs/ggswarm_live/status/changelog.md) ·
  [log](docs/ggswarm_live/status/log.md)

### Capstone (frozen)

- [Capstone README](docs/capstone/README.md)
- [Architecture](docs/capstone/design/architecture.md) ·
  [Assumptions](docs/capstone/design/assumptions.md) ·
  [Tensor contracts](docs/capstone/design/tensor_contracts.md)
- [Proposal](docs/capstone/project/proposal.md) ·
  [Testing Report](docs/capstone/project/testing_report.md)
- Phases:
  [1](docs/capstone/phases/phase1_foundation.md) ·
  [2](docs/capstone/phases/phase2_brain_development.md) ·
  [3](docs/capstone/phases/phase3_muscle_refinement.md) ·
  [4](docs/capstone/phases/phase4_stress_testing.md) ·
  [5](docs/capstone/phases/phase5_showcase_prep.md) ·
  [6](docs/capstone/phases/phase6_delivery.md)
- [Concepts](docs/capstone/concepts.md) — ML / RL / GNN glossary
- Status: [weekly updates](docs/capstone/status/weekly_updates.md) ·
  [changelog](docs/capstone/status/changelog.md) ·
  [run history](docs/capstone/status/run_history.md)

## Quickstart

The current installable code is the capstone simulation stack
(`ggswarm-v0` Isaac Lab task). The ggSwarm Live hardware stack adds PX4
+ companion-computer pieces over time, on top of this same repo.

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
   pip install -e source/ggswarm
   ```

5. If you encounter Windows HDF5 DLL conflicts, pin `h5py`:

   ```powershell
   pip install "h5py>=3.9.0,<3.12" --force-reinstall
   ```

## Training and Playback (capstone-era sim)

The capstone training and playback workflow still runs from this branch
against `ggswarm-v0`. The capstone tree is frozen but the code path is
not — Phase 1 of ggSwarm Live (shared-scene multi-drone training) builds
on it directly.

```powershell
# Train (local, headless)
env_isaaclab/Scripts/python.exe scripts/skrl/train.py --headless `
  --task ggswarm-v0 --num_envs 512 --max_iterations 500 --log_subdir p2a

# Play a checkpoint (GUI)
python scripts/skrl/play.py --task ggswarm-v0 `
  --checkpoint logs/skrl/ggswarm/p2a/<run>/checkpoints/best_agent.pt

# Record video (NVENC H.264)
python scripts/skrl/play.py --task ggswarm-v0 `
  --checkpoint <path> --video --video_prefix p2a-1

# Generate trajectory diagnostic plots
python scripts/skrl/play.py --task ggswarm-v0 `
  --checkpoint <path> --trajectories

# TensorBoard
tensorboard --logdir logs/skrl/ggswarm
```

For capstone-era training operations (GCE launches, log-sync workflow,
post-training analysis), see
[`docs/capstone/ops/`](docs/capstone/ops/).

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

> Developed with [Claude Code](https://claude.com/claude-code)

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
