# GCS mirror for training results

Operational guide for syncing training logs and checkpoints between your **Compute Engine VM** and your **local PC** via Google Cloud Storage (GCS). This doc describes the `gcloud storage rsync` workflow for bandwidth-efficient, resumable transfers.

**Context:** Training on the VM writes to `logs/skrl/...` (see [`gce_training_and_monitoring.md`](gce_training_and_monitoring.md) for how to run and monitor training). This guide covers moving those artifacts to GCS, then pulling them locally.

**GCS Project & Bucket:**
- **GCP Project**: `gg-swarm`
- **GCS Bucket**: `gs://gg-swarm-training-logs`

## Script defaults

Defaults when optional arguments are omitted:

| Script | Defaults | Notes |
| ------ | -------- | ----- |
| [`push_results_to_gcs.sh`](../../scripts/cloud/push_results_to_gcs.sh) | Family: `marl`, Bucket: `$GGSWARM_GCS_BUCKET` or `gs://gg-swarm-training-logs` | Run on VM. Pass `hover` as first arg for hover baseline. Videos always excluded. |
| [`pull_results_from_gcs.py`](../../scripts/cloud/pull_results_from_gcs.py) | Family: `marl`, Bucket: `$GGSWARM_GCS_BUCKET` or `gs://gg-swarm-training-logs`, No `--dry-run` (real sync) | Run on PC/Linux/Mac. Pass `--dry-run` to preview only. Videos always excluded. |
| [`list_checkpoints.py`](../../scripts/cloud/list_checkpoints.py) | Family: `marl`, Latest: `10` | Run on PC/Linux/Mac. Lists newest checkpoints first, capped at N rows. |

## Workflow

### 1. Push from VM after training

On the **VM** (`isaacsim`), after training completes:

```bash
cd ~/ggSwarm
./scripts/cloud/push_results_to_gcs.sh
```

For hover baseline (non-default family):

```bash
./scripts/cloud/push_results_to_gcs.sh hover
```

**What happens:** Checkpoints, params, and TensorBoard events upload to GCS; videos are excluded. Resumable: re-run to sync only new/changed files.

### 2. Pull to your local machine

On your **local** machine (Windows, Linux, or Mac):

```bash
cd /path/to/ggSwarm
python scripts/cloud/pull_results_from_gcs.py
```

For hover baseline:

```bash
python scripts/cloud/pull_results_from_gcs.py --family hover
```

**What happens:** Training runs mirror to `logs/skrl/ggswarm_marl/<timestamp>_mappo_torch/` (same structure as VM). Videos are not pulled. Resumable: re-run to pull only new/changed files.

### 3. List and choose a checkpoint

On your **local** machine:

```bash
python scripts/cloud/list_checkpoints.py
```

Override family or row count:

```bash
python scripts/cloud/list_checkpoints.py --family hover --latest 20
```

Output shows the newest checkpoints first with paths and file sizes. The latest checkpoint path is printed at the bottom for easy copy-paste.

### 4. Play or evaluate

Run your selected checkpoint:

```bash
python scripts/run.py phase2 play --checkpoint "logs/skrl/ggswarm_marl/2026-03-20_22-30-39_mappo_torch/checkpoints/best_agent.pt"
```

Or evaluate:

```bash
python scripts/run.py phase2 eval --num_episodes 10 --checkpoint "logs/skrl/ggswarm_marl/2026-03-20_22-30-39_mappo_torch/checkpoints/best_agent.pt"
```

## Operational tips

### Dry-run (preview changes)

See what would be synced without actually transferring files:

```bash
# Local dry-run
python scripts/cloud/pull_results_from_gcs.py --dry-run
```

Or with raw `gcloud storage`:

```bash
# VM dry-run
gcloud storage rsync --recursive --exclude='videos/.*' --dry-run logs/skrl/ggswarm_marl gs://gg-swarm-training-logs/logs/skrl/ggswarm_marl

# Local dry-run
gcloud storage rsync --recursive --dry-run gs://gg-swarm-training-logs/logs/skrl/ggswarm_marl logs/skrl/ggswarm_marl
```

The `--dry-run` flag previews changes without committing them.

### Periodic push during long training

For periodic backups to GCS during a long training run, use a separate `tmux` window on the VM:

```bash
while true; do
  ./scripts/cloud/push_results_to_gcs.sh
  sleep 1800  # Every 30 minutes
done
```

### Cost and retention

GCS charges for storage and egress. Set a lifecycle rule to auto-delete old logs:

```bash
gcloud storage buckets update gs://gg-swarm-training-logs --add-lifecycle-delete-age=30
```

This deletes objects older than 30 days. Document the policy in your operational playbook.

### Multiple training families

To sync the `hover` baseline separately:

```bash
# VM
./scripts/cloud/push_results_to_gcs.sh hover

# Local
python scripts/cloud/pull_results_from_gcs.py --family hover
```

Both families can coexist in GCS and locally without conflict.

## Your VM

To confirm your instance, run locally:

```powershell
gcloud compute instances list
```

Maintainer's current setup:

|| Field | Value |
|| ----- | ----- |
|| Name | `isaacsim` |
|| Zone | `us-central1-a` |
|| Repo path | `~/ggSwarm` |
|| Logs | `logs/skrl/ggswarm_marl/<timestamp>_mappo_torch/` |

Verify IPs with `gcloud compute instances list` — external IPs can change if the VM is recreated.

## Troubleshooting

**`gcloud storage: command not found` or `ModuleNotFoundError`**
- Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install).

**`AccessDenied` when pushing from VM**
- Ensure the service account (or logged-in user) has `storage.objectAdmin` or at least `storage.objects.create` and `storage.objects.delete` on the bucket.
- On VM: `gcloud auth list` to verify the active account.

**`AccessDenied` when pulling on local machine**
- Ensure `gcloud auth login` succeeded and your account has read access to the bucket.
- Run: `gcloud auth list` and `gcloud storage ls gs://gg-swarm-training-logs` to verify.

## Integration with training/monitoring

For the full GCE workflow:

1. **Train on VM:** [`gce_training_and_monitoring.md`](gce_training_and_monitoring.md) — start jobs, tail progress, view TensorBoard.
2. **Sync to GCS:** This doc — helper scripts or raw `gcloud storage rsync`.
3. **Work locally:** Run `play`, `eval`, `monitor` on synced checkpoints.

See also: [`commands.md`](commands.md) for checkpoint management and [`../../README.md`](../../README.md) for project navigation.
