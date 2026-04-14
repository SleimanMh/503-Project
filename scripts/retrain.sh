#!/usr/bin/env bash
# ── Retraining pipeline ──────────────────────────────────────────────────────
# Runs the full ML training pipeline inside the ml-service container and
# hot-reloads the models without restarting the service.
#
# Usage (from project root):
#   bash scripts/retrain.sh                 # default: container named ml-service-1
#   ML_CONTAINER=proj-test-ml-service-1 bash scripts/retrain.sh
#   ML_SERVICE_URL=http://localhost:8001 bash scripts/retrain.sh
#
# For AWS ECS: call this from an EventBridge-triggered Lambda or run-task.
# ---------------------------------------------------------------------------
set -euo pipefail

ML_CONTAINER=${ML_CONTAINER:-"proj-test-ml-service-1"}
ML_SERVICE_URL=${ML_SERVICE_URL:-"http://localhost:8001"}
DATA_PATH=${DATA_PATH:-"/app/data/acn_sessions.csv"}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-"artifacts/"}

echo "=== EV Charging — ML Retraining Pipeline ==="
echo "Container : $ML_CONTAINER"
echo "ML URL    : $ML_SERVICE_URL"
echo "Data      : $DATA_PATH"
echo "Artifacts : $ARTIFACTS_DIR"
echo "Started   : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

# ── Step 1: Prepare datasets ─────────────────────────────────────────────────
echo "--- Step 1/4: Preparing datasets (prepare_data.py) ---"
docker exec "$ML_CONTAINER" python -m training.prepare_data \
    --data-path "$DATA_PATH" \
    --output-dir "$ARTIFACTS_DIR"

# ── Step 2: Train departure model (M1) ───────────────────────────────────────
echo ""
echo "--- Step 2/4: Training M1 departure model (HistGBDT Q10/Q50/Q90) ---"
docker exec "$ML_CONTAINER" python -m training.train_departure \
    --artifacts-dir "$ARTIFACTS_DIR"

# ── Step 3: Train energy model (M3) ──────────────────────────────────────────
echo ""
echo "--- Step 3/4: Training M3 energy estimator (HistGBDT) ---"
docker exec "$ML_CONTAINER" python -m training.train_energy \
    --artifacts-dir "$ARTIFACTS_DIR"

# ── Step 4: Train arrivals model (M4A) ───────────────────────────────────────
echo ""
echo "--- Step 4/4: Training M4A arrivals forecast (HistGBDT Poisson) ---"
docker exec "$ML_CONTAINER" python -m training.train_arrivals \
    --artifacts-dir "$ARTIFACTS_DIR"

# ── Step 5: Hot-reload models in running service ──────────────────────────────
echo ""
echo "--- Step 5/4: Hot-reloading models via POST /retrain ---"
HTTP_STATUS=$(curl -s -o /tmp/retrain_response.json -w "%{http_code}" \
    -X POST "${ML_SERVICE_URL}/retrain")

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "Hot-reload successful:"
    cat /tmp/retrain_response.json
else
    echo "WARNING: Hot-reload returned HTTP $HTTP_STATUS"
    cat /tmp/retrain_response.json || true
    echo "Models saved to disk — restart ml-service to pick up new models."
fi

echo ""
echo "=== Retraining complete: $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
