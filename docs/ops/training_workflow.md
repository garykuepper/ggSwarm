# ggSwarm Training Workflow: Train → Evaluate → Adjust → Retrain

This document describes the iterative workflow for training, evaluating, and improving the ggSwarm drone formation control policy. This cycle is critical for efficiently using cloud compute resources and converging to a performant solution.

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

## Step 1: TRAIN on GCE

### Launch Training on VM

```bash
# SSH into GCE VM
gcloud compute ssh isaacsim --zone us-central1-a

# Navigate to project
cd /path/to/ggSwarm

# Start training (runs in background, output to log file)
./scripts/cloud/train_and_push.sh

# OR manually:
python scripts/run.py phase2 train --headless --max_iterations 300000 2>&1 | tee training.log
```

### Monitor Training

While training runs:

```bash
# Monitor GPU/CPU usage
watch -n 1 nvidia-smi

# Or stream TensorBoard over SSH tunnel (from local machine)
ssh -L 6006:localhost:6006 user@vm_ip
# Then open http://localhost:6006 in browser
tensorboard --logdir logs/skrl/ggswarm_marl
```

### Push to GCS

Once training completes (or periodically):

```bash
# Push logs to GCS
gsutil -m rsync -r -d logs/skrl/ggswarm_marl gs://gg-swarm-training-logs/

# Or use the helper script
scripts/cloud/push_results_to_gcs.sh
```

---

## Step 2: SYNC Results Locally

### Pull from GCS

```bash
# List available runs first
python scripts/cloud/list_checkpoints.py --family marl

# Pull latest run
python scripts/cloud/pull_results_from_gcs.py --family marl --latest 1

# Or specific run
python scripts/cloud/pull_results_from_gcs.py --family marl --dry-run  # preview
python scripts/cloud/pull_results_from_gcs.py --family marl
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

## Step 4: EVAL - Run Enhanced Evaluation

### Quick Evaluation (Latest Checkpoint)

```bash
# Evaluate latest checkpoint with enhanced metrics
python scripts/run.py phase2 eval --num_episodes 10 \
    --checkpoint "logs/skrl/ggswarm_marl/<run>/checkpoints/best_agent.pt"
```

Output includes:
- **Formation metrics:** mean_formation_error_m, separation_event_rate
- **Stability metrics:** mean_roll_deg, mean_pitch_deg, orientation_violation_rate
- **Survival metrics:** mean_episode_survival_steps, altitude_std_m
- **Motion quality:** mean_speed_mps, ground_hit_rate, airborne_ratio

### Checkpoint Progression Analysis

```bash
# Analyze how metrics change over the training run
python scripts/analyze_checkpoints.py \
    --run_dir "logs/skrl/ggswarm_marl/<run>" \
    --interval 50000 \
    --num_episodes 3 \
    --output_csv "checkpoint_progression.csv"
```

This produces a CSV table showing each metric at 50k, 100k, 150k, etc. steps. Use this to identify:
- When orientation degradation starts (increasing roll/pitch)
- When formation learning kicks in (formation_error drops after curriculum_start_step)
- If policy improves monotonically or degrades at certain points

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

```bash
# Option A: Update config locally, then push to VM
git add docs/
git commit -m "Update reward config: boost uprightness and ang_vel penalties"
git push origin main

# On VM:
git pull
python scripts/run.py phase2 train --headless --max_iterations 300000

# Option B: SSH and edit directly on VM
ssh user@vm_ip
# Edit config via ssh
nano source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env_cfg.py
# Retrain
```

### Monitor with Checkpointing

```bash
# Training automatically saves at 10k, 20k, 30k, ... steps
# If interrupted, resume with:
python scripts/run.py phase2 train --headless \
    --checkpoint "logs/skrl/ggswarm_marl/<run>/checkpoints/agent_150000.pt"
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

### Phase 2 (Current)

| Metric | Target | Rationale |
|:---|:---|:---|
| mean_formation_error_m | < 0.5 m | Agents within target spacing |
| separation_event_rate | < 0.1 | Few collision risks |
| airborne_ratio | > 0.9 | Drones stay airborne |
| mean_roll_deg | < 15° | Level flight |
| mean_pitch_deg | < 15° | Level flight |
| mean_episode_survival_steps | > 9 s (out of 10) | Rarely crash before timeout |
| ground_hit_rate | < 0.05 | Minimal ground contact |

---

## References

- **Architecture:** [`docs/design/architecture.md`](../../docs/design/architecture.md)
- **Phase 2 Goals:** [`docs/design/phase2_brain_development.md`](../../docs/design/phase2_brain_development.md)
- **Evaluation Script:** [`scripts/eval_phase2.py`](../../scripts/eval_phase2.py)
- **Checkpoint Analyzer:** [`scripts/analyze_checkpoints.py`](../../scripts/analyze_checkpoints.py)
- **Environment Config:** [`source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env_cfg.py`](../../source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/drone_swarm_env_cfg.py)
- **SKRL Config:** [`source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/agents/skrl_mappo_cfg.yaml`](../../source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/agents/skrl_mappo_cfg.yaml)
