#!/bin/bash
# Bash script for syncing ggSwarm logs to/from GCS (Linux/Mac, useful on VM).
# Usage:
#   ./sync_gcs.sh push --family marl
#   ./sync_gcs.sh pull --family marl --include-videos
#   ./sync_gcs.sh pull --family hover -n (dry-run)

set -e

ACTION=""
FAMILY="marl"
INCLUDE_VIDEOS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        push|pull)
            ACTION="$1"
            shift
            ;;
        --family)
            FAMILY="$2"
            shift 2
            ;;
        --include-videos)
            INCLUDE_VIDEOS=1
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$ACTION" ]]; then
    echo "Usage: $0 {push|pull} [--family {marl|hover}] [--include-videos] [-n]"
    exit 1
fi

if [[ -z "$GGSWARM_GCS_URI" ]]; then
    echo "Error: GGSWARM_GCS_URI environment variable not set."
    echo "Set it to gs://bucket-name/prefix and retry."
    exit 1
fi

# Determine local and remote paths.
LOCAL_LOG_DIR="logs/skrl/ggswarm_$FAMILY"
REMOTE_PATH="$GGSWARM_GCS_URI/logs/skrl/ggswarm_$FAMILY"

# Build gcloud storage rsync arguments.
RSYNC_ARGS=(--recursive)

if [[ $INCLUDE_VIDEOS -eq 0 ]]; then
    RSYNC_ARGS+=(--exclude='videos/.*')
fi

if [[ $DRY_RUN -eq 1 ]]; then
    RSYNC_ARGS+=(--dry-run)
    echo "[DRY-RUN] No files will be modified."
fi

# Perform action.
if [[ "$ACTION" == "push" ]]; then
    echo "Pushing $LOCAL_LOG_DIR to $REMOTE_PATH"
    gcloud storage rsync "${RSYNC_ARGS[@]}" "$LOCAL_LOG_DIR" "$REMOTE_PATH"
    echo "Push complete."
else
    echo "Pulling $REMOTE_PATH to $LOCAL_LOG_DIR"
    gcloud storage rsync "${RSYNC_ARGS[@]}" "$REMOTE_PATH" "$LOCAL_LOG_DIR"
    echo "Pull complete."
fi
