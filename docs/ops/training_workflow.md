# ggSwarm Training Workflow: Train → Evaluate → Adjust → Retrain

This document describes the iterative workflow for training, evaluating, and improving the ggSwarm drone formation control policy. This cycle is critical for efficiently using cloud compute resources and converging to a performant solution.

> **Post-training analysis:** For the full assessment procedure (sync, convergence check,
> TensorBoard checklist, assess command, decision matrix, changelog template) see
> **[`docs/ops/post_train_analysis.md`](post_train_analysis.md)**.
>
> **Cross-run scorecard:** [`docs/status/run_history.md`](../status/run_history.md) —
> fill in a row there after every run before changing any config (Rule 23).

---

## Workflow Overview

```
1. TRAIN       → Run training on GCE VM (headless), push logs to GCS
2. SYNC        → Pull results to local machine
3. INSPECT     → View reward curves and diagnostics in TensorBoard
4. EVAL        → Run enhanced evaluation with detailed metrics
5. DIAGNOSE    → Identify failure modes from metrics
6. ADJUST      → Modify reward scales, curriculum, hyperparameters
7. LOG         → Record changes in changelog.md for reproducibility
8. RETRAIN     → Deploy new config to GCE and iterate
```

---

## Step 0: Deploy Updated Config to VM

Before launching a new training run, ensure the latest code (with any reward tuning, curriculum changes, or L4 optimizations) is deployed to the VM.

### Resolve Git Conflicts on VM

The VM may have stale local edits to cloud scripts. Resolve these by discarding the local changes and pulling the latest remote:

```bash
# SSH into VM
gcloud compute ssh isaacsim --zone=us-central1-a --project=gg-swarm

# Navigate to repo
cd ~/ggSwarm

# Discard stale local cloud script edits
git checkout -- scripts/cloud/

# Pull latest changes from remote
git pull
```

If the pull still fails, use the escape hatch:

```bash
git reset --hard origin/phase2
```

### Verify Config Values Landed

After pulling, confirm critical parameters are present:

```bash
# Check reward tuning
grep -E "rew_scale_upright|rew_scale_ang_vel|num_envs" \
  source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env_cfg.py

# Check L4 optimization (rollouts, mini_batches)
grep -E "rollouts|mini_batches" \
  source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/agents/skrl_mappo_cfg.yaml
```

Expected values for current run:
- `rew_scale_upright: float = 3.0` (aggressive uprightness to prevent flipping)
- `rew_scale_ang_vel = -0.25` (strong angular velocity penalty)
- `num_envs=128` (L4 GPU scaling, 4x more parallel experience)
- `rollouts: 64` (larger rollout buffer for stabler advantage estimates)
- `mini_batches: 8` (more gradient updates per rollout)

### Launch Training (Detached)

Start training in a detached process so it survives SSH disconnection. The `train_and_push.sh` wrapper runs training and auto-uploads to GCS on completion or interrupt:

```bash
# Set GCS target
export GGSWARM_GCS_URI=gs://gg-swarm-training-logs

# Launch training (runs in background)
nohup ./scripts/cloud/train_and_push.sh phase2 train --headless \
  > ~/train_phase2.log 2>&1 &

# Note the PID for later reference
```

The `train_and_push.sh` script:
1. Runs `python scripts/run.py phase2 train --headless` for 300k timesteps
2. On success or interrupt (Ctrl+C), calls `gcloud storage rsync` to push logs to GCS
3. Exits with the training process's exit code

### Monitor Training Progress

From your local machine, monitor the VM's training log:

```powershell
# Check last 50 lines of training log (shows progress and errors)
gcloud compute ssh isaacsim --zone=us-central1-a --project=gg-swarm `
  --command="tail -50 ~/train_phase2.log"

# Or tail in real-time (keep command running)
gcloud compute ssh isaacsim --zone=us-central1-a --project=gg-swarm `
  --command="tail -f ~/train_phase2.log"
```

Look for `[PROGRESS]` lines showing steps completed, throughput (steps/sec), and ETA. Early training should show:
- `[PROGRESS] 0/300,000` → training started
- `28-32 steps/sec` on L4 GPU (typical throughput with 128 envs)

If you see errors (e.g. `OutOfMemory`, `CUDA error`), stop training with Ctrl+C and reduce `num_envs` to 64 in the config, then redeploy and retrain.

---

## Step 1: TRAIN on GCE

For deployment, launching training, and monitoring on the VM, see **[Step 0: Deploy Updated Config to VM](#step-0-deploy-updated-config-to-vm)** above.

Once training completes, results are automatically uploaded to GCS by `train_and_push.sh`. To manually push or check uploads:

```bash
# On VM, push logs to GCS (if using nohup instead of train_and_push.sh)
export GGSWARM_GCS_URI=gs://gg-swarm-training-logs
gcloud storage rsync --recursive --exclude='videos/.*' logs/skrl/ggswarm_marl "$GGSWARM_GCS_URI/logs/skrl/ggswarm_marl"
```

---

## Step 2: SYNC Results Locally

### Pull from GCS

> **Windows note (shell-syntax rule):** `gcloud storage rsync` is broken on Windows.
> Use `gcloud storage cp` instead.

```powershell
# Copy a specific run directory from GCS to local (replace <timestamp> with actual value)
gcloud storage cp -r gs://gg-swarm-training-logs/logs/skrl/ggswarm_marl/<timestamp>_mappo_torch logs/skrl/ggswarm_marl/

# List available runs in GCS to find the timestamp
gcloud storage ls gs://gg-swarm-training-logs/logs/skrl/ggswarm_marl/
```

Results land in: `logs/skrl/ggswarm_marl/<timestamp>_mappo_torch/`

---

## Step 3: INSPECT Training Progress

### View TensorBoard Scalars

```bash
# Launch TensorBoard
tensorboard --logdir logs/skrl/ggswarm_marl

# Open http://localhost:6006 in browser
```

**Key metrics to look for:**

| Metric | Good Sign | Bad Sign |
|:---|:---|:---|
| `Episode/Mean Reward` | Steady increase, then plateau | Flat or declining |
| `Episode/Mean Length` | Stays near max (for formation goal) | Drops to zero (crashes) |
| `rew_alive` (per-term) | Increasing, stays high | Decreasing or zero |
| `rew_upright` (per-term) | High and stable | Declining (flipping) |
| `rew_vel`, `rew_ang_vel` | Negative but small magnitude | Large negative (wild motion) |
| `rew_formation` | Increases during curriculum | Never increases (formation not learned) |

### Playback Videos

```bash
# If videos were recorded during training
python scripts/run.py phase2 play --checkpoint "logs/skrl/ggswarm_marl/<run>/checkpoints/best_agent.pt"

# Watch for:
# - Do drones stay airborne?
# - Are they close together (formation)?
# - Do they flip/crash?
# - Is motion smooth or jittery?
```

---

## Step 4: EVAL - Run Automated Assessment

Use the unified `assess` subcommand — it runs convergence check, best-checkpoint eval,
and prints a PASS / WARN / FAIL scorecard automatically.

```powershell
# Phase 2A (hover-stability)
python scripts/run.py hover-stability assess --run_dir "logs/skrl/ggswarm_marl/<run>" --num_episodes 5

# Phase 2B (formation)
python scripts/run.py phase2b assess --run_dir "logs/skrl/ggswarm_marl/<run>" --num_episodes 5

# Add --progression to also sweep all intermediate checkpoints (~3 min extra)
python scripts/run.py hover-stability assess --run_dir "logs/skrl/ggswarm_marl/<run>" --num_episodes 5 --progression
```

For the full assessment procedure, TensorBoard checklist, decision matrix, and
changelog template, see **[`docs/ops/post_train_analysis.md`](post_train_analysis.md)**.

---

## Step 5: DIAGNOSE - Metric Interpretation Guide

### Formation Not Learning

**Symptoms:**
- `mean_formation_error_m` stays > 1.0m (target: < 0.5m)
- `separation_event_rate` remains high (target: ~0)

**Possible causes & fixes:**
- Formation reward is too weak → increase `rew_scale_formation` (default 1.0, try 2.0)
- Curriculum starts too late → move `curriculum_start_step` earlier (80k → 50k)
- Hover not mastered → check airborne_ratio and mean_roll first

### Drones Flipping/Tumbling

**Symptoms:**
- `mean_roll_deg` or `mean_pitch_deg` > 30°
- `orientation_violation_rate` > 0.1
- `mean_episode_survival_steps` drops during training

**Possible causes & fixes:**
- Uprightness reward too weak → increase `rew_scale_upright` (1.0 → 2.0)
- Angular velocity penalty too weak → increase `rew_scale_ang_vel` magnitude (0.02 → 0.15)
- Initial policy is too exploratory → reduce `initial_log_std` in config (make it more negative)
- Spawn yaw range too wide → reduce `spawn_yaw_range` (0.3 → 0.1)

### Altitude Oscillation

**Symptoms:**
- `altitude_std_m` is large (> 0.5m)
- Drones creep above max_height or below min_height
- `ground_hit_rate` increasing

**Possible causes & fixes:**
- Velocity penalty too weak → increase `rew_scale_vel` magnitude (0.1 → 0.15)
- Thrust control unstable → check moment_scale and thrust_to_weight in config
- Curriculum transitioning too abruptly → smooth with `curriculum_pos_floor` (keep it > 0.3)

### Poor Airborne Ratio

**Symptoms:**
- `airborne_ratio` < 0.7
- Ground crashes increasing
- `mean_episode_survival_steps` < max_episode_length

**Possible causes & fixes:**
- Alive bonus too weak → increase `rew_scale_alive` (0.5 → 1.0)
- Termination penalty too weak → increase `rew_scale_terminated` magnitude (10 → 15)
- Position reward not motivating hover → check if `curriculum_pos_floor` is high enough
- Initial spawn height too low → adjust spawn logic or min_height threshold

---

## Step 6: ADJUST Configuration

### Quick Config Changes

Edit [`drone_swarm_env_cfg.py`](../../source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env_cfg.py):

```python
# Example: boost uprightness to prevent flipping
rew_scale_upright = 2.0  # was 1.0

# Example: give hover more time before formation kicks in
curriculum_start_step = 80000  # was 50000
curriculum_end_step = 250000    # was 200000
curriculum_pos_floor = 0.4      # was 0.3

# Example: stronger angular velocity penalty
rew_scale_ang_vel = -0.15  # was -0.02
```

### NVIDIA L4 Training Optimization

If training on L4 GPU (24GB VRAM):

```python
# In drone_swarm_env_cfg.py
scene: InteractiveSceneCfg = InteractiveSceneCfg(
    num_envs=128,  # was 32 - 4x more parallel experience
    env_spacing=5.0,
    replicate_physics=True,
)

# In skrl_mappo_cfg.yaml
agent:
  rollouts: 64      # was 32
  mini_batches: 8   # was 4
```

Benefits:
- More diverse experience per gradient update (4x more parallel envs)
- Stabler advantage estimates (larger rollout buffer)
- ~2x wall-clock speedup from more GPU utilization

Monitor VRAM: `nvidia-smi`. If > 85% usage, reduce `num_envs` back to 64.

### Hyperparameter Tuning (Advanced)

| Problem | Parameter | Direction | Notes |
|:---|:---|:---|:---|
| Policy too deterministic | `entropy_loss_scale` | ↑ | 0.01 → 0.02 (explore more) |
| Policy too random (noisy actions) | `initial_log_std` | ↓ | -1.0 → -1.5 (tighter actions) |
| Training unstable | `learning_rate` | ↓ | 1e-4 → 5e-5 (slower updates) |
| Convergence too slow | `learning_rate` | ↑ | 1e-4 → 2e-4 (faster updates) |
| Gradients exploding | `grad_norm_clip` | ↓ | 1.0 → 0.5 (tighter clip) |

---

## Step 7: LOG Changes

Before retraining, update [`docs/status/changelog.md`](../../docs/status/changelog.md) with **all** modifications and rationale:

**Template:**

```markdown
## [Date] - Training Run V2

### Changes
- Increased `rew_scale_upright` from 1.0 to 2.0 (fix tumbling observed in eval)
- Increased `rew_scale_ang_vel` magnitude from 0.02 to 0.15 (stronger spin penalty)
- Delayed curriculum start from 50k to 80k (give hover more time to stabilize)
- Scaled num_envs from 32 to 128 on L4 GPU (better gradient estimates, 4x more data/step)

### Rationale
- Previous run showed mean_roll=45°, mean_pitch=40°, indicating severe flipping
- orientation_violation_rate was 0.8 (80% of steps violating 45° threshold)
- Mean episode survival only 2.5 seconds (out of 10s max)
- These changes target stability-first training; formation will come later

### Expected Impact
- Expect airborne_ratio > 0.9 after stabilization
- Mean roll/pitch should be < 15° in next eval
- Effective throughput increases 4x from num_envs scaling
```

---

## Step 8: RETRAIN

### Deploy Updated Config

After making adjustments (Step 6) and documenting them (Step 7), deploy the new config to the VM and launch retraining.

**Option A: Push changes via git, then follow Step 0 on VM (Recommended)**

```bash
# Locally: commit and push config changes
git add source/ docs/
git commit -m "Adjust reward config: [specific changes]"
git push origin phase2

# Then follow Step 0 on the VM to pull and launch
```

**Option B: Edit directly on VM (Quick iteration)**

```bash
# SSH into VM and edit config interactively
ssh user@vm_ip
nano source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env_cfg.py

# Then launch training (see Step 0 for canonical command)
export GGSWARM_GCS_URI=gs://gg-swarm-training-logs
nohup ./scripts/cloud/train_and_push.sh phase2 train --headless \
  > ~/train_phase2.log 2>&1 &
```

For detailed deployment, conflict resolution, verification, and monitoring steps, see **[Step 0: Deploy Updated Config to VM](#step-0-deploy-updated-config-to-vm)**.

### Resume Interrupted Training

Training automatically saves checkpoints every ~10k steps. If interrupted (network loss, manual Ctrl+C), resume from the latest checkpoint:

```bash
# Find the latest checkpoint
ls -1t logs/skrl/ggswarm_marl/<run>/checkpoints/ | head -5

# Resume training (automatically continues from checkpoint step)
export GGSWARM_GCS_URI=gs://gg-swarm-training-logs
nohup ./scripts/cloud/train_and_push.sh phase2 train --headless \
  --checkpoint "logs/skrl/ggswarm_marl/<run>/checkpoints/agent_150000.pt" \
  > ~/train_phase2_resume.log 2>&1 &
```

---

## Training Workflow Checklist

Before committing to a long run, verify:

- [ ] Reward config changes are documented in changelog.md
- [ ] GPU memory headroom checked (`nvidia-smi` shows < 90% usage during init)
- [ ] Checkpoint directory has write permissions
- [ ] GCS bucket access verified (`gsutil ls gs://gg-swarm-training-logs/`)
- [ ] Expected training duration estimated (300k steps ~2-4 hours on L4 with 128 envs)
- [ ] Backup of config saved (git commit before retraining)
- [ ] Tensorboard write_interval is reasonable (1000 steps, i.e., log every ~30 seconds)
- [ ] entropy_loss_scale is non-zero (unless deliberately disabled)

---

## Troubleshooting

### Training Process Issues

**Q: Training hangs or is very slow (< 100 steps/sec)**
- Check GPU utilization: `nvidia-smi` should show > 80%
- If low, check if visualization is enabled (disable with `--headless`)
- Reduce `num_envs` if VRAM is full

**Q: VRAM OOM (Out of Memory)**
- Reduce `num_envs` (try 64 or 32)
- Or reduce `rollouts` (try 32 or 16)
- Watch `nvidia-smi` throughout training

**Q: Checkpoint not found on resume**
- Verify checkpoint path: `ls logs/skrl/ggswarm_marl/<run>/checkpoints/agent_*.pt`
- Use full absolute path, not relative

### Evaluation Issues

**Q: Eval script crashes with "Could not determine max_episode_length"**
- Ensure environment config has `episode_length_s` set (should be 10.0)
- Check that env is initialized correctly

**Q: Per-term reward logging not appearing in TensorBoard**
- Verify `self.extras["log"]` is set in `_get_rewards()`
- TensorBoard needs 1000+ steps to display new metrics (write_interval)

---

## Success Criteria by Phase

### Phase 2A — Hover-Stability Gate (must pass before Phase 2B)

| Metric | Gate | Rationale |
| :--- | :--- | :--- |
| `survival_steps` | > 500 | Agents stay airborne long enough to learn |
| `airborne_ratio` | > 0.9 | Drones reliably hold altitude |
| `ground_hit_rate` | < 0.05 | Minimal ground contact |
| `mean_roll_deg` | < 15° | Level flight before adding formation pressure |
| `orientation_violation_rate` | < 0.1 | Few severe tilt events |

### Phase 2B — Formation Gate (must pass before Phase 3)

| Metric | Gate | Rationale |
| :--- | :--- | :--- |
| `mean_formation_error_m` | < 0.5 m | Agents within target spacing |
| `airborne_ratio` | > 0.9 | Stability maintained under formation pressure |
| `mean_roll_deg` | < 15° | Level flight maintained |
| `ground_hit_rate` | < 0.05 | No regression in stability |
| `survival_steps` | > 500 | Episodes long enough to measure formation |

---

## References

- **Architecture:** [`docs/design/architecture.md`](../../docs/design/architecture.md)
- **Phase 2 Goals:** [`docs/design/phase2_brain_development.md`](../../docs/design/phase2_brain_development.md)
- **Evaluation Script:** [`scripts/eval_phase2.py`](../../scripts/eval_phase2.py)
- **Checkpoint Analyzer:** [`scripts/analyze_checkpoints.py`](../../scripts/analyze_checkpoints.py)
- **Environment Config:** [`source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env_cfg.py`](../../source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env_cfg.py)
- **SKRL Config:** [`source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/agents/skrl_mappo_cfg.yaml`](../../source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/agents/skrl_mappo_cfg.yaml)
