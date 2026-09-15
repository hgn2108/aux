#!/usr/bin/env bash
# Deploy the demo to Google Cloud Run.
#
# Cloud Run bills only while a request is being served and scales to zero in between, so a
# portfolio demo sits inside the always-free allowance: 180,000 vCPU-seconds and 360,000
# GiB-seconds a month. At 16GiB and 4 CPU memory is the binding constraint, leaving roughly
# six hours of actual serving a month. A demo uses far less, since a visit lasts minutes.
#
#   ./scripts/deploy_cloudrun.sh YOUR_PROJECT_ID
#
# Prerequisites: gcloud installed and authenticated, billing enabled on the project.

set -euo pipefail

PROJECT="${1:?usage: deploy_cloudrun.sh PROJECT_ID [REGION]}"
REGION="${2:-us-central1}"
SERVICE="aux"

# gcloud can print a reauthentication notice and still exit 0, so pipefail catches nothing
# and the script reports a successful deploy that never happened. Check for a usable token
# before doing anything else.
if ! gcloud auth print-access-token >/dev/null 2>&1; then
    echo "gcloud credentials have expired. Run: gcloud auth login" >&2
    exit 1
fi

echo "==> project $PROJECT, region $REGION"
gcloud config set project "$PROJECT" >/dev/null
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com --quiet

# Built on Cloud Build rather than locally: the image carries ~4GB of model weights, and
# pushing that from a laptop is slow. cloudbuild.yaml reuses the previous image's layers,
# so only the first build pays for the weights.
echo "==> building (15-25 minutes the first time, ~2 minutes after a code-only change)"
gcloud builds submit --config cloudbuild.yaml

echo "==> deploying"
gcloud run deploy "$SERVICE" \
    --image "gcr.io/$PROJECT/$SERVICE" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    `# 8Gi was not enough: loading MuQ-MuLan holds the checkpoint and the constructed` \
    `# model at once and peaked at 8.3GB, so the container was killed mid-load and` \
    `# restarted forever. Cloud Run requires at least 4 CPU to allow 16Gi.` \
    --memory 16Gi \
    --cpu 4 \
    --timeout 900 \
    --concurrency 8 \
    `# Scale to zero when idle: this is what keeps it inside the free tier.` \
    --min-instances 0 \
    `# One instance caps the bill and removes the need for session affinity, since` \
    `# Streamlit holds a WebSocket that must stay on the instance that opened it.` \
    --max-instances 1 \
    `# CPU is throttled to zero between requests by default, which would suspend model` \
    `# loading partway through a cold start.` \
    --no-cpu-throttling \
    --set-env-vars AUX_PUBLIC=1

echo
URL=$(gcloud run services describe "$SERVICE" --region "$REGION" \
      --format='value(status.url)')
if [ -z "$URL" ]; then
    echo "deploy reported success but the service has no URL" >&2
    exit 1
fi
echo "$URL"
