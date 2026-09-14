#!/usr/bin/env bash
# Deploy the demo to Google Cloud Run.
#
# Cloud Run bills only while a request is being served and scales to zero in between, so a
# portfolio demo sits inside the always-free allowance: 180,000 vCPU-seconds and 360,000
# GiB-seconds a month. At the settings below that is roughly 12 hours of actual serving.
#
#   ./scripts/deploy_cloudrun.sh YOUR_PROJECT_ID
#
# Prerequisites: gcloud installed and authenticated, billing enabled on the project.

set -euo pipefail

PROJECT="${1:?usage: deploy_cloudrun.sh PROJECT_ID [REGION]}"
REGION="${2:-us-central1}"
SERVICE="aux"

echo "==> project $PROJECT, region $REGION"
gcloud config set project "$PROJECT" >/dev/null
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com --quiet

# Built on Cloud Build rather than locally: the image carries ~4GB of model weights, and
# pushing that from a laptop is slow.
echo "==> building (this takes 15-25 minutes the first time)"
gcloud builds submit --tag "gcr.io/$PROJECT/$SERVICE" --timeout=3600s

echo "==> deploying"
gcloud run deploy "$SERVICE" \
    --image "gcr.io/$PROJECT/$SERVICE" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --memory 8Gi \
    --cpu 2 \
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
gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
