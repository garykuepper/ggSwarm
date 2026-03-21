#!/bin/bash
# Setup GCS authentication on the VM for training result uploads
# Run this once on the VM to configure permanent credentials

set -e

echo "[INFO] Setting up GCS authentication for ggSwarm training uploads..."

# Check if we're on the VM
if ! command -v gcloud &> /dev/null; then
    echo "[ERROR] gcloud not found. Make sure you're running this on the VM."
    exit 1
fi

# Check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    echo "[ERROR] gsutil not found. Install Google Cloud SDK first."
    exit 1
fi

# Option 1: Use the service account key if provided
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "[INFO] Using existing credentials from \$GOOGLE_APPLICATION_CREDENTIALS"
    gsutil -m rsync -r -x "videos/.*" ~/ggSwarm/logs/skrl/ggswarm_marl gs://gg-swarm-training-logs/logs/skrl/ggswarm_marl
    echo "[SUCCESS] Upload complete!"
    exit 0
fi

# Option 2: Try default compute service account scopes
echo "[INFO] Attempting to use compute service account..."
if gsutil -m rsync -r -x "videos/.*" ~/ggSwarm/logs/skrl/ggswarm_marl gs://gg-swarm-training-logs/logs/skrl/ggswarm_marl; then
    echo "[SUCCESS] Upload successful using default compute service account!"
    exit 0
else
    echo "[WARNING] Default service account lacks permissions."
    echo ""
    echo "To fix this permanently, you have two options:"
    echo ""
    echo "OPTION 1 (Recommended): Update VM scopes (requires VM restart)"
    echo "  - Stop the VM: gcloud compute instances stop isaacsim --zone=us-central1-a"
    echo "  - Update scopes: gcloud compute instances set-service-account isaacsim"
    echo "      --service-account 843023359729-compute@developer.gserviceaccount.com"
    echo "      --scopes https://www.googleapis.com/auth/cloud-platform"
    echo "      --zone=us-central1-a"
    echo "  - Start the VM: gcloud compute instances start isaacsim --zone=us-central1-a"
    echo ""
    echo "OPTION 2: Use a service account key file"
    echo "  - Create key: gcloud iam service-accounts keys create ~/gcs-key.json"
    echo "      --iam-account=843023359729-compute@developer.gserviceaccount.com"
    echo "  - Set env var: export GOOGLE_APPLICATION_CREDENTIALS=~/gcs-key.json"
    echo "  - Then run: ./scripts/cloud/push_results_to_gcs.sh"
    exit 1
fi
