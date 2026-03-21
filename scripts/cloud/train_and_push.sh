#!/bin/bash
# Optional training wrapper for the VM: runs a training job and auto-uploads to GCS on completion or signal.
# This allows you to "fire and forget" a training run; if the VM is interrupted, best-effort upload happens.
#
# Usage:
#   ./train_and_push.sh phase2 --headless
#   ./train_and_push.sh hover --headless --num_agents 3
#
# The script will:
#   1. Run: python scripts/run.py <args>
#   2. On success/failure/SIGINT: attempt gsutil rsync push
#   3. Exit with the training exit code.

set -e

# Require GGSWARM_GCS_URI for safety.
if [[ -z "$GGSWARM_GCS_URI" ]]; then
    echo "Error: GGSWARM_GCS_URI not set. Set it to gs://bucket/prefix and retry."
    exit 1
fi

# Trap SIGINT (Ctrl+C) and SIGTERM to attempt upload on interrupt.
TRAINING_PID=""
on_interrupt() {
    echo ""
    echo "[INFO] Interrupted. Attempting to push logs to GCS before exit..."
    if [[ -n "$TRAINING_PID" ]]; then
        kill "$TRAINING_PID" 2>/dev/null || true
        wait "$TRAINING_PID" 2>/dev/null || true
    fi
    push_to_gcs
    exit 130
}
trap on_interrupt SIGINT SIGTERM

push_to_gcs() {
    # Infer family from arguments (hacky but acceptable for optional script).
    FAMILY="marl"
    for arg in "$@"; do
        if [[ "$arg" == "hover" ]]; then
            FAMILY="hover"
            break
        fi
    done

    LOCAL_LOG_DIR="logs/skrl/ggswarm_$FAMILY"
    REMOTE_PATH="$GGSWARM_GCS_URI/logs/skrl/ggswarm_$FAMILY"

    echo "[INFO] Pushing $LOCAL_LOG_DIR to $REMOTE_PATH (videos excluded)..."
    gsutil -m rsync -r -x "videos/.*" "$LOCAL_LOG_DIR" "$REMOTE_PATH"
    echo "[INFO] Push complete."
}

# Run training.
echo "[INFO] Starting training: python scripts/run.py $@"
cd ~/ggSwarm
source env_isaaclab/bin/activate
python scripts/run.py "$@" &
TRAINING_PID=$!

# Wait for training to complete.
if wait "$TRAINING_PID"; then
    TRAIN_EXIT_CODE=0
    echo "[INFO] Training completed successfully."
else
    TRAIN_EXIT_CODE=$?
    echo "[WARNING] Training exited with code $TRAIN_EXIT_CODE"
fi

# Push to GCS after training finishes.
push_to_gcs

exit "$TRAIN_EXIT_CODE"
