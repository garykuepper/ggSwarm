# ggSwarm Commands Reference

This document provides instructions on how to train, evaluate, play, and record videos with ggSwarm.

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

### Pause and Resume Phase 2 Training

Pause a running training job in its terminal with `Ctrl+C`.

Before restarting, print the newest Phase 2 checkpoint path:

```powershell
$ckpt = (Get-ChildItem "logs\skrl\ggswarm_marl" -Recurse -File -Include best_agent.pt,agent_*.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName); $ckpt
```

Resume training from that checkpoint:

```powershell
python scripts/run.py phase2 train --headless --checkpoint "$ckpt"
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

### High-Quality Video Recording

When using `play --video`, ggSwarm now defaults to:

- `rendering_mode=quality` (unless explicitly overridden)
- preferred encoder `hevc_nvenc` (GPU NVENC), with automatic fallback to CPU
  codecs if NVENC is unavailable
- single-env recording for stable, non-flashing output

Examples:

```powershell
# Default high-quality video path (quality rendering + NVENC preferred)
python scripts/run.py phase2 play --video --video_length 1500

# Explicitly choose rendering mode and codec controls
python scripts/run.py phase2 play --video --video_length 1500 --rendering_mode quality --video_codec hevc_nvenc --video_preset p5 --video_ffmpeg_params "-rc vbr -cq 19 -b:v 0"

# Force CPU fallback codec if needed
python scripts/run.py phase2 play --video --video_codec libx265 --video_preset slow --video_bitrate 8M
```

### Eval and assess video (headless)

Phase eval (`scripts/eval.py` via `run.py … eval`) and post-training assess can record the same style of clip under the **training run directory** (`videos/eval/`), not under `videos/play/`.

Replace ``<run>`` below with your real timestamp folder (e.g. ``2026-03-22_04-32-04_mappo_torch``). Literal ``<run>`` in the path is invalid on Windows and is rejected before Isaac Sim starts.

```powershell
python scripts/run.py hover-stability eval --headless --video --checkpoint logs/skrl/ggswarm_marl/<run>/checkpoints/best_agent.pt

python scripts/run.py hover-stability assess --run_dir logs/skrl/ggswarm_marl/<run> --video --no_sync
```

Optional: `--video_length`, `--rendering_mode`, `--video_codec`, etc. (same semantics as `play --video`).

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

## Google Compute Engine (remote training)

For SSH, `nohup`/`tmux`, tailing `[PROGRESS]` lines, and TensorBoard over an SSH
tunnel, see [`gce_training_and_monitoring.md`](gce_training_and_monitoring.md).

For syncing runs to your PC via GCS, see [`gce_results_sync.md`](gce_results_sync.md).

## Unit Tests (Pure Torch Contracts)

These tests validate the core adjacency and reward contract logic without launching Isaac Sim. They run quickly on any machine with PyTorch installed.

### Running Tests

```powershell
# Run all Tier 1 unit tests (fast, CPU-only, always runs)
python -m pytest tests/ -q

# Run only specific test class or function
python -m pytest tests/test_contract_logic.py::TestComputeAdjacencyMatrix -v

# Run with verbose output
python -m pytest tests/ -v
```

### Test Structure

- **Tier 1 (Unit Tests)**: `tests/test_contract_logic.py` – Tests pure torch functions (`compute_adjacency_matrix`, `compute_marl_rewards`, `compute_hover_rewards`, etc.). Runs anywhere without Isaac Sim.
- **Tier 2 (Integration Tests)**: `tests/test_env_smoke.py` – Tests full environment with Isaac Sim (marked `@pytest.mark.isaacsim_ci`, skipped by default). Only runs with `python -m pytest tests/ -m isaacsim_ci` on GPU machines.

### Test Coverage

- **29 unit tests** covering:
  - Curriculum alpha scheduling
  - Adjacency matrix computation (connectivity, diagonal constraint, batching)
  - Graph edge_index conversion and batched node index shifting
  - MARL rewards (position, formation, cohesion, separation, termination)
  - Hover rewards (goal tracking, ground hit penalty, height gating)
  - Shape validation and error handling
