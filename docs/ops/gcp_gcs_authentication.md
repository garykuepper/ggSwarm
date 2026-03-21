# GCP GCS Authentication Setup for ggSwarm Training Results

## Problem

The VM's service account doesn't have proper scopes to write to GCS, causing `push_results_to_gcs.sh` to fail with "403 Provided scope(s) are not authorized".

## Solution

Choose one of the following approaches:

### Option 1: Update VM Scopes (Recommended - Permanent Fix)

The VM's service account needs the `cloud-platform` scope to access GCS. This was already applied, but requires restarting the VM for changes to take effect.

**Status:**
- ✅ VM scopes have been updated to `cloud-platform`
- ✅ VM service account has `storage.objectAdmin` role on the GCS bucket
- ❌ Restart required for new scopes to apply

**Next Steps:**
1. Stop the VM again (it was just restarted, give it a few minutes)
2. Let it fully boot
3. Test the push with: `gcloud compute ssh isaacsim --zone=us-central1-a --command="cd ~/ggSwarm && ./scripts/cloud/push_results_to_gcs.sh"`

### Option 2: Automatic Sync on Training Completion (For Future Runs)

Edit the training script to automatically sync results after training completes:

1. On the VM, edit `~/ggSwarm/scripts/run.py`
2. After the training loop completes, add:
   ```bash
   ./scripts/cloud/push_results_to_gcs.sh
   ```

Or use the provided `train_and_push.sh` wrapper that handles this automatically.

### Option 3: Quick Manual Sync (Workaround)

Until the scope issue is resolved, use the manual sync script:

```powershell
# On your PC
.\scripts\cloud\manual_sync_results.ps1 -Family marl
```

This copies results directly from VM to your PC, bypassing GCS.

## Current Status

For the current training run (2026-03-20_22-30-39):
- ✅ Checkpoints are already on your PC
- ✅ Ready for evaluation/playback
- ⏳ GCS sync: still needs authentication fix

## For Future Runs

To make automatic sync work seamlessly:

1. **After VM scopes take full effect**, GCS push will work automatically
2. **Then** you can use the helper scripts as designed:
   ```powershell
   # On VM after training
   ./scripts/cloud/push_results_to_gcs.sh
   
   # On PC to pull
   .\scripts\cloud\pull_results_from_gcs.ps1
   
   # On PC to list & select checkpoints
   .\scripts\cloud\list_checkpoints.ps1
   ```

## Troubleshooting

**If scopes still don't work after VM restart:**

Try using service account key credentials:
```bash
# On VM, create service account key
export GOOGLE_APPLICATION_CREDENTIALS=~/.gcs-key.json
gsutil -m rsync -r -x 'videos/.*' logs/skrl/ggswarm_marl gs://gg-swarm-training-logs/logs/skrl/ggswarm_marl
```

**Check current VM scopes:**
```bash
gcloud compute instances describe isaacsim --zone=us-central1-a --format="value(serviceAccounts[0].scopes)"
```

Should include: `https://www.googleapis.com/auth/cloud-platform`

