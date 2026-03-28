"""ML monitoring router — exposes model health and evaluation metrics."""

import logging

import httpx
from fastapi import APIRouter

from app.config import ML_SERVICE_URL, ML_TIMEOUT_S

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml", tags=["ml-monitoring"])


@router.get("/status")
async def ml_status():
    """Return ML service health and model loading status."""
    try:
        async with httpx.AsyncClient(timeout=ML_TIMEOUT_S) as client:
            resp = await client.get(f"{ML_SERVICE_URL}/health")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning(f"ML service unreachable: {e}")
        return {"status": "unreachable", "error": str(e)}


@router.get("/metrics")
async def ml_metrics():
    """Return ML model evaluation metrics, feature info, and metadata."""
    try:
        async with httpx.AsyncClient(timeout=ML_TIMEOUT_S) as client:
            resp = await client.get(f"{ML_SERVICE_URL}/model-metrics")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning(f"ML metrics unavailable: {e}")
        return {"error": str(e)}


@router.get("/drift-status")
async def ml_drift_status(window: int = 200):
    """
    Return live model drift metrics.

    Computes rolling MAE and PSI (Population Stability Index) over the last
    `window` collected sessions to detect when the model has drifted and
    retraining is needed.
    """
    try:
        async with httpx.AsyncClient(timeout=ML_TIMEOUT_S) as client:
            resp = await client.get(
                f"{ML_SERVICE_URL}/drift-status",
                params={"window": window},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning(f"Drift status unavailable: {e}")
        return {"error": str(e), "drift_detected": False, "drift_level": "unknown"}


@router.post("/retrain")
async def ml_retrain(force: bool = False):
    """
    Trigger model retraining on collected session data.

    Use force=true to retrain even if fewer than 500 new sessions are available.
    Retraining runs synchronously in the ml-service (~1-3 minutes).
    """
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{ML_SERVICE_URL}/retrain",
                params={"force": force},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"Retrain request failed: {e}")
        return {"error": str(e), "retrained": False}
