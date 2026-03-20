# Running ggSwarm Demonstrations

This document provides instructions on how to run the various simulation and training demonstrations for the ggSwarm project.

> **Important:** All commands assume you have activated your virtual environment (run `.\env_isaaclab\Scripts\activate` in each new terminal session) and are running from the `ggSwarm` project root directory.

## Unified Run Helper (Recommended)

For day-to-day runs, ggSwarm provides a small, unified helper CLI:

- `scripts/run.py`

It exposes consistent subcommands for:

- **hover**: `train`, `play`, `eval`, `monitor` (task: `GGS-Hover-v0`)
- **phase2**: `train`, `play`, `eval`, `monitor` (task: `Template-GGSwarm-Marl-Direct-v0`)
- **debug**: `smoke`, `latest-checkpoint`

### Common Commands

```powershell
# Hover baseline (single drone)
python scripts/run.py hover train --headless
python scripts/run.py hover play
python scripts/run.py hover eval --num_episodes 10
python scripts/run.py hover monitor

# Phase 2 formation (swarm)
python scripts/run.py phase2 train --headless
python scripts/run.py phase2 play
python scripts/run.py phase2 eval --num_episodes 10
python scripts/run.py phase2 monitor
```

### Phase 2 Altitude-First Validation

For recovery runs, evaluate both formation and altitude stability:

- `mean_formation_error_m` should trend down toward the Phase 2 target.
- `mean_altitude_error_m` should remain low.
- `ground_hit_rate` should remain near zero.
- `airborne_ratio` should remain high.

```powershell
python scripts/run.py phase2 eval --num_agents 3 --num_episodes 10
```

If needed, adjust the airborne threshold margin used during eval:

```powershell
python scripts/eval_phase2.py --task Template-GGSwarm-Marl-Direct-v0 --algorithm MAPPO --ml_framework torch --num_agents 3 --num_episodes 10 --airborne_height_margin 0.2
```

### Default Progress and ETA Reporting

Training progress reporting is enabled by default for both `hover train` and
`phase2 train`. The trainer prints periodic lines with:

- steps completed vs total timesteps
- elapsed time
- rolling steps/second (over a recent time window)
- ETA to completion

Use `--no_progress` only when you want quiet output.

```powershell
# Default: progress/ETA enabled
python scripts/run.py hover train --headless

# Optional: hide periodic progress lines
python scripts/run.py hover train --headless --no_progress
```

Optional tuning flags:

- `--progress_interval_s` controls how often progress lines print (default `10`).
- `--eta_window_s` controls the rolling window used for throughput and ETA
  smoothing (default `120`).

## Phase 2: Brain Training & GNN Policy

Phase 2 development introduces the **Graph Attention Network (GATv2)** policy to enable local message passing among the swarm.

All Phase 2 training, playback, evaluation, and monitoring are accessed via
`scripts/run.py`:

```powershell
python scripts/run.py phase2 train --headless
python scripts/run.py phase2 play
python scripts/run.py phase2 eval --num_episodes 10
python scripts/run.py phase2 monitor
```

## Unit Tests (Pure Torch Contracts)

These tests validate the core adjacency and reward contract logic without launching Isaac Sim.

```powershell
python -m pytest -q
```
