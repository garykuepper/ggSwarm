#!/bin/bash
# Push ggSwarm training results to GCS (excludes videos)
# Usage: ./push_results_to_gcs.sh [marl|hover]

set -e

FAMILY="${1:-marl}"
GCS_BUCKET="${GGSWARM_GCS_BUCKET:-gs://gg-swarm-training-logs}"
LOCAL_LOG_DIR="logs/skrl/ggswarm_$FAMILY"
REMOTE_PATH="$GCS_BUCKET/logs/skrl/ggswarm_$FAMILY"

if [[ ! -d "$LOCAL_LOG_DIR" ]]; then
    echo "Error: $LOCAL_LOG_DIR not found"
    exit 1
fi

echo "[INFO] Pushing $LOCAL_LOG_DIR to $REMOTE_PATH (excluding videos)"
gsutil -m rsync -r -x "^videos/.*" "$LOCAL_LOG_DIR" "$REMOTE_PATH"
echo "[INFO] Push complete."
