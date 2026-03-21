# Training and monitoring on Google Compute Engine

Operational guide for running ggSwarm training on a **Compute Engine** VM and checking progress from your **local** machine. For copying runs off the VM, see [`gce_results_sync.md`](gce_results_sync.md) for the GCS mirror workflow.

> **Instance (verify before use):** `isaacsim` in `us-central1-a`. Confirm with `gcloud compute instances list`. Maintainer notes: [`.cursor/quick-start.md`](../../.cursor/quick-start.md).

## Layout on the VM

Typical layout (home directory may differ):

- Repo: `~/ggSwarm`
- Virtualenv: `~/ggSwarm/env_isaaclab`
- Training logs (Phase 2): `~/ggSwarm/logs/skrl/ggswarm_marl/<timestamp>_mappo_torch/`
- Optional transcript log: e.g. `~/train_phase2.log` if you redirect `nohup` output

## Train commands

For full command reference (including `play`, `eval`, `record video`), see [`commands.md`](commands.md).

Common invocations:

```bash
python scripts/run.py phase2 train --headless
python scripts/run.py hover train --headless
```

## Start training

1. SSH:

   ```bash
   gcloud compute ssh isaacsim --zone=us-central1-a
   ```

2. From the repo root, activate the environment:

   ```bash
   cd ~/ggSwarm
   source env_isaaclab/bin/activate
   ```

3. Choose your run mode:

   **Interactive (survives current SSH session only):**

   ```bash
   python scripts/run.py phase2 train --headless
   ```

   **Detached (survives logout and disconnects; recommended for long runs):**

   ```bash
   nohup python scripts/run.py phase2 train --headless > ~/train_phase2.log 2>&1 &
   ```

   (Adjust the log path as needed. Backgrounded process runs even after you disconnect.)

Resume and checkpoints follow [`commands.md`](commands.md) — pause with `Ctrl+C`, then restart with `--checkpoint` pointing to the latest `agent_*.pt` or `best_agent.pt`.

## Monitor training

### A. Text progress (`[PROGRESS]` lines)

Training emits periodic `[PROGRESS]` lines (steps, ETA, throughput). If you used `nohup` to `~/train_phase2.log`, from **your local** shell:

```powershell
gcloud compute ssh isaacsim --zone=us-central1-a --command="grep PROGRESS ~/train_phase2.log | tail -5"
```

Adjust the path if your transcript lives elsewhere.

### B. TensorBoard on the VM (recommended for curves)

**Easiest way:** Use the helper script on your **local** machine:

```powershell
.\scripts\cloud\monitor_training.ps1
```

This opens an SSH tunnel to the VM and automatically opens `http://localhost:6006` in your browser. Keep the terminal open while monitoring. Press `Ctrl+C` to close the tunnel.

To monitor a different family:

```powershell
.\scripts\cloud\monitor_training.ps1 -Family hover
```

**Manual (no script):** If you prefer, run these steps in two separate terminals:

1. Open an SSH tunnel (keep this terminal open):

   ```powershell
   gcloud compute ssh isaacsim --zone=us-central1-a -- -N -L 6006:127.0.0.1:6006
   ```

2. Open `http://localhost:6006` in your browser.

**Sanity check (non-interactive):** To list scalar tags without starting the web server:

```bash
./env_isaaclab/bin/tensorboard --logdir logs/skrl/ggswarm_marl --inspect
```

This confirms event files and tags (e.g. `Episode / Total timesteps`) while a training job writes to the same log root.

### C. TensorBoard on your PC after syncing logs

If you sync `logs/skrl/ggswarm_marl` to your laptop (e.g. via `gsutil rsync` as in [`gce_results_sync.md`](gce_results_sync.md)), run:

```powershell
python scripts/run.py phase2 monitor
```

This only shows runs present under **local** `logs\skrl\ggswarm_marl` and does **not** SSH to GCP by itself; point it at logs that exist on your machine.
