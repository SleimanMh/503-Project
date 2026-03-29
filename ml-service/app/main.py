"""
ML Service — Container 2 (Internal)
Serves demand forecasting (Model 1) and departure prediction (Model 2).
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import timezone
from pathlib import Path

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

from app.config import MODEL_VERSION, ARTIFACTS_DIR, FEEDBACK_CSV, COLLECTED_SESSIONS_CSV, ACN_DATA_PATH
from app.models.demand_forecast import load_demand_models, predict_demand
from app.models.departure_prediction import load_departure_model, predict_departure
from app.schemas import (
    DemandForecastRequest,
    DemandForecastResponse,
    DemandForecastPoint,
    DeparturePredictionRequest,
    DeparturePredictionResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_demand_loaded = False
_departure_loaded = False

# ── Prometheus custom metrics ────────────────────────────────────────────
DEPARTURE_PREDICTIONS_TOTAL = Counter(
    "ml_departure_predictions_total",
    "Total departure predictions served",
)
DEMAND_FORECASTS_TOTAL = Counter(
    "ml_demand_forecasts_total",
    "Total demand forecast requests served",
)
DEPARTURE_LATENCY = Histogram(
    "ml_departure_prediction_seconds",
    "Departure prediction latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
DEMAND_LATENCY = Histogram(
    "ml_demand_forecast_seconds",
    "Demand forecast latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
MODEL_LOADED = Gauge(
    "ml_model_loaded",
    "Whether an ML model is loaded (1=yes, 0=no)",
    ["model_name"],
)
RETRAIN_TOTAL = Counter(
    "ml_retrain_total",
    "Total retraining runs triggered",
    ["status"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _demand_loaded, _departure_loaded
    try:
        load_demand_models()
        _demand_loaded = True
        MODEL_LOADED.labels(model_name="demand_forecast").set(1)
        logger.info("Demand forecasting models loaded")
    except FileNotFoundError:
        MODEL_LOADED.labels(model_name="demand_forecast").set(0)
        logger.warning("Demand model not found — /forecast will be unavailable")

    try:
        load_departure_model()
        _departure_loaded = True
        MODEL_LOADED.labels(model_name="departure_prediction").set(1)
        logger.info("Departure prediction model loaded")
    except FileNotFoundError:
        MODEL_LOADED.labels(model_name="departure_prediction").set(0)
        logger.warning("Departure model not found — /predict-departure will be unavailable")

    yield


app = FastAPI(
    title="EV Charging ML Service",
    description="Internal ML inference for demand forecasting and departure prediction",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# Prometheus auto-instrumentation — exposes /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_version=MODEL_VERSION,
        demand_model_loaded=_demand_loaded,
        departure_model_loaded=_departure_loaded,
    )


@app.post("/forecast", response_model=DemandForecastResponse)
async def forecast(request: DemandForecastRequest):
    t0 = time.perf_counter()
    current = request.current_time.replace(tzinfo=timezone.utc) if request.current_time.tzinfo is None else request.current_time
    results = predict_demand(
        current_time=current,
        horizon_windows=request.horizon_windows,
        recent_arrivals=request.recent_arrivals,
        recent_kwh=request.recent_kwh,
    )
    DEMAND_LATENCY.observe(time.perf_counter() - t0)
    DEMAND_FORECASTS_TOTAL.inc()
    return DemandForecastResponse(
        model_version=MODEL_VERSION,
        forecast=[DemandForecastPoint(**r) for r in results],
    )


@app.post("/predict-departure", response_model=DeparturePredictionResponse)
async def predict_departure_endpoint(request: DeparturePredictionRequest):
    t0 = time.perf_counter()
    arrival = request.arrival_time.replace(tzinfo=timezone.utc) if request.arrival_time.tzinfo is None else request.arrival_time
    result = predict_departure(
        arrival_time=arrival,
        site_id=request.site_id,
        cluster_id=request.cluster_id,
        user_mean_stay=request.user_historical_mean_stay_min,
        station_mean_stay=request.station_historical_mean_stay_min,
        requested_energy_kwh=request.requested_energy_kwh,
    )
    DEPARTURE_LATENCY.observe(time.perf_counter() - t0)
    DEPARTURE_PREDICTIONS_TOTAL.inc()
    return DeparturePredictionResponse(
        model_version=MODEL_VERSION,
        predicted_stay_duration_min=result["predicted_stay_duration_min"],
        predicted_departure_time=result["predicted_departure_time"],
    )


@app.get("/model-metrics")
async def model_metrics():
    """Return model evaluation metrics and metadata for monitoring."""
    eval_path = ARTIFACTS_DIR / "eval_report.json"
    eval_data = {}
    if eval_path.exists():
        eval_data = json.loads(eval_path.read_text())

    return {
        "model_version": MODEL_VERSION,
        "demand_model_loaded": _demand_loaded,
        "departure_model_loaded": _departure_loaded,
        "evaluation": eval_data,
        "models": {
            "demand_forecast": {
                "type": "XGBoost",
                "targets": ["arrival_count", "total_kwh"],
                "input_features": [
                    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                    "is_weekend", "month_sin", "month_cos",
                    "lag_1", "lag_2", "lag_48",
                    "lag_kwh_1", "lag_kwh_48",
                    "rolling_6", "rolling_48", "rolling_kwh_6",
                ],
                "description": "Predicts future EV arrivals and energy demand per 30-min window",
            },
            "departure_prediction": {
                "type": "XGBoost",
                "target": "stay_duration_minutes",
                "input_features": [
                    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                    "is_weekend", "month_sin", "month_cos",
                    "site_encoded", "cluster_encoded",
                    "user_mean_stay", "station_mean_stay",
                    "requested_energy_kwh",
                ],
                "description": "Predicts how long each EV will stay based on arrival patterns",
            },
        },
    }


@app.get("/drift-status")
async def drift_status(window: int = 200):
    """
    Return live model drift metrics computed from collected feedback data.

    Compares rolling prediction error (MAE) and stay-duration distribution
    shift (PSI) against the training baseline.  Use this to decide when to
    trigger retraining.

    Query param:
      window (int, default 200) — number of recent sessions to compute rolling metrics over.
    """
    from training.drift_monitor import compute_drift_metrics
    window = max(10, min(window, 2000))
    return compute_drift_metrics(FEEDBACK_CSV, window=window)


@app.post("/retrain")
async def trigger_retrain(force: bool = False):
    """
    Trigger model retraining on collected live session data.

    Retraining runs synchronously (may take 1-3 minutes).  The models are
    hot-swapped in-memory after training completes.

    Query param:
      force (bool, default False) — retrain even if fewer than 500 new sessions.
    """
    global _demand_loaded, _departure_loaded
    from training.retrain import run_retrain

    result = run_retrain(
        acn_data_path=ACN_DATA_PATH,
        collected_path=COLLECTED_SESSIONS_CSV,
        artifacts_dir=ARTIFACTS_DIR,
        force=force,
    )

    if result.get("retrained"):
        RETRAIN_TOTAL.labels(status="success").inc()
        # Hot-reload the updated model files
        try:
            load_departure_model()
            _departure_loaded = True
            MODEL_LOADED.labels(model_name="departure_prediction").set(1)
            logger.info("Departure model hot-reloaded after retraining")
        except Exception as e:
            logger.error(f"Failed to reload departure model: {e}")

        try:
            load_demand_models()
            _demand_loaded = True
            MODEL_LOADED.labels(model_name="demand_forecast").set(1)
            logger.info("Demand model hot-reloaded after retraining")
        except Exception as e:
            logger.error(f"Failed to reload demand model: {e}")
    else:
        RETRAIN_TOTAL.labels(status="skipped").inc()

    return result
