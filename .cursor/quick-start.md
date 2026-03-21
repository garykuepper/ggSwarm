# Quick Start: Common Agent Tasks

Quick reference for common ggSwarm development tasks.

## Task 1: Add a New Reward Term to Phase 2
- Add parameter to GGSwarmMarlEnvCfg with default 0.0
- Compute term in _compute_rewards() 
- Document in docs/status/changelog.md
- See Rule 5 in project-rules.mdc for example

## Task 2: Debug Why Agents Are not Hovering
- Monitor: python scripts/run.py hover monitor
- Check: rew_scale_pos greater_equal 1.0, rew_scale_alive greater 0.1, rew_scale_ground_hit less_equal -10.0
- Fix learning rate to 5e-5 if oscillating

## Task 3: Resume Interrupted Training
Find latest checkpoint then resume training from it.

## Task 4: Inspect a Checkpoint Policy
Load checkpoint and inspect weight norms - GNN weights should be greater 0.1.

## Task 5: Tune Curriculum Learning
Configure curriculum_start_step, curriculum_end_step, alpha parameters.

## GCP: training VM and gcloud (maintainer environment)

For sync, SSH, or “where did training run?” questions, assume:

- **gcloud**: Already authenticated on this Windows machine; `gcloud` / `gsutil` are expected to work from the repo (e.g. after `env_isaaclab\Scripts\activate`).
- **Compute Engine instance** (verify before using IPs — they can change if the VM is recreated):

  | Name     | Zone           | Machine        | External IP   |
  | -------- | -------------- | -------------- | ------------- |
  | isaacsim | us-central1-a  | g2-standard-4  | 34.69.135.96  |

  Internal IP (VPC): `10.128.0.2`. Check live: `gcloud compute instances list`.

- **Repo on VM**: The project is **git-cloned on `isaacsim`**; training logs are under `logs/skrl/...` relative to that clone unless overridden.
- **SSH**: Prefer `gcloud compute ssh isaacsim --zone=us-central1-a`.
- **GCP Project**: `gg-swarm`
- **GCS Bucket**: `gs://gg-swarm-training-logs` (created for training results)

**Helper scripts** for sync workflow (defaults: **marl** when family omitted; bucket `GGSWARM_GCS_BUCKET` / `gs://gg-swarm-training-logs`; videos never synced):
- VM push: `./scripts/cloud/push_results_to_gcs.sh` or `... hover`
- PC pull: `.\scripts/cloud/pull_results_from_gcs.ps1` or `-Family hover`; add `-DryRun` to preview only
- List checkpoints: `.\scripts/cloud/list_checkpoints.ps1` (up to **10** newest); optional `-Latest N`

Full runbook: [`docs/ops/gce_training_and_monitoring.md`](../docs/ops/gce_training_and_monitoring.md) and [`docs/ops/gce_results_sync.md`](../docs/ops/gce_results_sync.md).

See .cursor/debugging-guide.md for comprehensive troubleshooting.
