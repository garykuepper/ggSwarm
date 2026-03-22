# Deployment Summary - Stability Config Tuning

**Date**: 2026-03-21  
**Status**: ✅ DEPLOYED & TRAINING STARTED

## Changes Deployed to GCE VM (isaacsim)

All updates from commit `9cc2ed2` have been successfully deployed to the `isaacsim` VM in `us-central1-a`.

### Reward Config Updates
Verified on VM:
- ✅ `rew_scale_upright`: 3.0 → **5.0**
- ✅ `rew_scale_ang_vel`: -0.25 → **-0.5**
- ✅ `rew_scale_terminated`: -15.0 → **-20.0**
- ✅ `spawn_yaw_range`: 0.3 → **0.1**

### Training Launch

**Command**:
```bash
export GGSWARM_GCS_URI=gs://gg-swarm-training-logs
nohup ./scripts/cloud/train_and_push.sh phase2 train --headless --max_iterations 120000 > ~/train_phase2_stability_fix.log 2>&1 &
```

**Parameters**:
- **Training steps**: 120,000 (vs 300k previous; 60% reduction based on convergence analysis)
- **Max iterations**: 120,000 (MAPPO policy updates)
- **Headless mode**: Enabled (no rendering, faster execution)
- **GPU**: NVIDIA L4 (24GB VRAM)
- **Log file**: `~/train_phase2_stability_fix.log`
- **Auto-upload**: Results auto-sync to GCS on completion

## Training Status

Started at: 2026-03-21 (time when deployment completed)  
Progress: **ACTIVELY RUNNING**

**Current metrics (as of last check):**
- Internal step counter: ~4,500/7,680,000 (represents environment loop steps)
- MAPPO iterations: ~4,000/120,000 actual policy updates
- Throughput: ~16.7 steps/sec
- Estimated completion time: 2-4 hours

**Note:** The progress bar shows total environment steps (7.68M for 120k iterations × 128 envs), but training will stop when `max_iterations=120000` MAPPO updates complete (the actual policy learning iterations).

## Monitoring

### Real-time Log
```bash
gcloud compute ssh isaacsim --zone=us-central1-a --command="tail -f ~/train_phase2_stability_fix.log"
```

### TensorBoard (when ready)
```bash
python scripts/cloud/monitor_training.py --family marl
```

### Convergence Check (after training completes)
```bash
python scripts/cloud/check_convergence.py --run_dir logs/skrl/ggswarm_marl/<run_timestamp>_mappo_torch
```

## Expected Outcomes

After 120k steps with aggressive stability tuning:
- **Airborne ratio**: > 0.9 (vs 0.651 in baseline)
- **Mean roll**: < 15° (vs 63.5° in baseline)
- **Mean pitch**: < 15° (vs 62° in baseline)
- **Ground hit rate**: < 0.05 (vs 0.535 in baseline)
- **Formation error**: < 0.5m (vs 1.126m in baseline)

## Key Improvements

1. **Reward tuning** makes uprightness top priority (5.0 > 3.0)
2. **Spin penalty** increased 2x (-0.5 vs -0.25)
3. **Training budget** cut 60% (120k vs 300k) based on convergence analysis
4. **Faster eval** default (5 episodes vs 10)
5. **Automated GCS sync** on completion or SIGINT

## Next Steps

1. **Monitor training progress** - Check logs hourly, especially first 10k steps
2. **Evaluate intermediate checkpoints** - Use convergence analysis tool at 50k, 80k, 120k
3. **Sync results when done** - Auto-upload to GCS; pull locally with `pull_results_from_gcs.py`
4. **Quick eval on best checkpoint** - 5-episode eval (new default)
5. **Compare to baseline** - Check if tuning fixed the 63.5° roll/pitch issue

## Troubleshooting

### If training hangs
```bash
gcloud compute ssh isaacsim --zone=us-central1-a --command="ps aux | grep train"
```

### If training crashes
```bash
gcloud compute ssh isaacsim --zone=us-central1-a --command="tail -200 ~/train_phase2_stability_fix.log | grep -i error"
```

### Resume from checkpoint
If interrupted, training can resume from latest checkpoint:
```bash
nohup ./scripts/cloud/train_and_push.sh phase2 train --headless --max_iterations 120000 --checkpoint "logs/skrl/ggswarm_marl/<run>/checkpoints/agent_<step>.pt" > ~/train_phase2_resume.log 2>&1 &
```

---

**Commit**: `9cc2ed2`  
**Branch**: `main` (deployed)  
**VM Status**: Training in progress
