# ggSwarm: Decentralized Formation Control for Drone Swarms

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3-76B900?logo=nvidia&logoColor=white)](https://isaac-sim.github.io/IsaacLab/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/isaac-sim)
[![SKRL](https://img.shields.io/badge/skrl-1.1.0-blueviolet?logo=github&logoColor=white)](https://skrl.readthedocs.io/)
[![Machine Learning](https://img.shields.io/badge/pytorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

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

| Phase | Weeks | Dates | Focus | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1. Foundation** | 5–6 | Feb 5 – Feb 17 | Isaac Lab env, drone spawning, graph connectivity | ✅ Complete |
| **2. Brain Development** | 7–8 | Feb 18 – Mar 3 | GATv2 policy training with MAPPO | 🔄 In Progress |
| **3. Muscle Refinement** | 9–11 | Mar 4 – Mar 24 | MINCO optimization, SwarmRaft consensus | ⬜ Planned |
| **4. Stress Testing** | 12–13 | Mar 25 – Apr 7 | Agent failure tests, obstacle environments | ⬜ Planned |
| **5. Showcase Prep** | 14–15 | Apr 8 – Apr 21 | HD rendering, final validation, demo video | ⬜ Planned |

*Detailed plans: [Phase 1](docs/phase1_foundation.md) · [Phase 2](docs/phase2_brain_development.md)*

## Development Stack

* Simulation: NVIDIA Isaac Sim 5.1 / Isaac Lab 2.3.
* **Learning:**
* **Goal**: Develop a decentralized drone swarm capable of coordinated movement and formation control.
* **Layers**:
  * Layer 1: Simulated Multirotor (`Crazyflie`)
  * Layer 2: Graph Connectivity (Distance-based)
  * Layer 3: GATv2 Policy (SKRL/MAPPO)
  * Layer 4: Consensus Mechanism (Phase 3)
  * Layer 5: Mission Planning (Phase 4)

## Prerequisites

Before installing, ensure the following requirements are met:

* **Python 3.11** — Isaac Sim 5.1 requires Python 3.11 exactly.
  Download from [python.org](https://www.python.org/downloads/) if needed.
* **NVIDIA GPU Driver** — Version ≥ 552.86 (CUDA 12) required on Windows.
  Download the latest from [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/).
* **Windows Long Path Support** — Required to avoid installation errors.
  Enable via the registry: `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`
  or run `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force` in an elevated PowerShell.
* **Isaac Lab** — Clone to a sibling directory of `ggSwarm` (i.e., `../IsaacLab`):

  ```powershell
  git clone https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
  ```

  The resulting layout must be:

  ```text
  isaaclab\
  ├── IsaacLab\      ← cloned Isaac Lab repo
  └── ggSwarm\       ← this project
  ```

  All run commands use `..\IsaacLab\isaaclab.bat` and rely on this sibling structure.

## Installation

### Step 1 — Clone ggSwarm

```powershell
git clone https://github.com/garykuepper/ggSwarm.git
cd ggSwarm
```

### Step 2 — Create a Python 3.11 Virtual Environment

```powershell
# Use the py launcher if python3.11 is not your default
py -3.11 -m venv env_isaaclab

# Activate the environment
env_isaaclab\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

> **Important:** Run `env_isaaclab\Scripts\activate` at the start of **every new terminal session**
> before calling `isaaclab.bat`. The script delegates to whatever Python is on your active `PATH`,
> so without the `venv` active it will not find `isaacsim`.

### Step 3 — Install Isaac Sim

Isaac Sim 5.1 is distributed as a pip package from NVIDIA's `PyPI` index:

```powershell
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

> **Note:** This download is large (~10 GB). Ensure you have sufficient disk space.

### Step 4 — Install `PyTorch` (CUDA 12.8)

```powershell
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

### Step 5 — Install Isaac Lab Extensions

From the `ggSwarm` directory, invoke the Isaac Lab installer (installs Isaac Lab
source extensions and the `skrl` learning framework):

```powershell
..\IsaacLab\isaaclab.bat --install skrl
```

After this completes, **pin `h5py` to a version compatible with Isaac Sim's bundled
HDF5 DLLs**. Without this, Isaac Sim's HDF5 1.12.x DLLs conflict with the
`h5py 3.16+` wheel (which bundles HDF5 2.x), causing a fatal Windows DLL crash:

```powershell
pip install "h5py>=3.9.0,<3.12" --force-reinstall
```

### Step 6 — Install ggSwarm Package

```powershell
pip install -e source/ggSwarm
```

### Running the Demonstrations & Training

Detailed instructions on how to run the Phase 1 environments, invoke the Phase 2 MAPPO/GATv2 training scripts, and evaluate the swarm's flight policy have been moved to a dedicated guide.

👉 **[View the Guide to Running Demonstrations](docs/running_demos.md)**

## Troubleshooting

| Error | Cause | Fix |
| :--- | :--- | :--- |
| `isaaclab.bat: not recognized` | Script is not on your `PATH` | Always use the relative path: `..\IsaacLab\isaaclab.bat` |
| `ModuleNotFoundError: No module named 'isaacsim'` | venv not activated | Run `env_isaaclab\Scripts\activate` first |
| `ModuleNotFoundError: No module named 'isaaclab'` | Isaac Lab not installed | Run Step 5 (`isaaclab.bat --install skrl`) |
| Long path errors during install | Windows path limit (260 chars) | Enable long path support (see Prerequisites) |
| `ModuleNotFoundError: No module named 'pxr'` | Isaac Lab imports before `AppLauncher` starts | All `isaaclab`/task imports must come **after** `app_launcher = AppLauncher(args_cli)`. See `scripts/phase1_demo.py` for the correct pattern. |
| `ImportError: DLL load failed while importing _errors` on `h5py` | `h5py 3.16+` bundles HDF5 2.x DLLs that conflict with Isaac Sim's HDF5 1.12.x DLLs | Run `pip install "h5py>=3.9.0,<3.12" --force-reinstall` (Step 5 covers this) |

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Citation

If you use this work in your research, please cite:
(TBD)
