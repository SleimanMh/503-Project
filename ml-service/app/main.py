"""
ML Service — Container 2 (Internal)
Serves demand forecasting, departure prediction (M1 — HistGBDT + Q10/Q50/Q90),
energy prediction (M3), and arrivals forecast (M4A).
"""

import csv
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

from app.config import MODEL_VERSION, ARTIFACTS_DIR, FEEDBACK_CSV, COLLECTED_SESSIONS_CSV, ACN_DATA_PATH
from app.models.demand_forecast import load_demand_models, predict_demand
from app.models.departure_prediction import load_departure_model, predict_departure
from app.models.energy_prediction import load_energy_model, predict_energy
from app.models.arrivals_prediction import load_arrivals_model, predict_arrivals
from app.models.user_arrival import predict_arrival_window
from app.user_profiles import init_db as init_user_profiles_db, get_profile, update_profile
from app.schemas import (
    DemandForecastRequest,
    DemandForecastResponse,
    DemandForecastPoint,
    DeparturePredictionRequest,
    DeparturePredictionResponse,
    EnergyPredictionRequest,
    EnergyPredictionResponse,
    ArrivalsPredictionRequest,
    ArrivalsPredictionResponse,
    ArrivalsPredictionSlot,
    HealthResponse,
    UserSessionPredictionRequest,
    UserSessionPredictionResponse,
    CollectSessionRequest,
    CollectSessionResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_demand_loaded = False
_departure_loaded = False
_energy_loaded = False
_arrivals_loaded = False

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
    global _demand_loaded, _departure_loaded, _energy_loaded, _arrivals_loaded

    # Download latest models from S3 (if MODELS_S3_BUCKET is configured).
    # This means every pod restart picks up the newest retrained models
    # without requiring a new Docker image build or deployment.
    from training.s3_store import download_models_from_s3
    download_models_from_s3(ARTIFACTS_DIR)
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
        logger.info("Departure prediction models loaded (Q10/Q50/Q90)")
    except FileNotFoundError:
        MODEL_LOADED.labels(model_name="departure_prediction").set(0)
        logger.warning("Departure model not found — /predict-departure will be unavailable")

    try:
        load_energy_model()
        _energy_loaded = True
        MODEL_LOADED.labels(model_name="energy_prediction").set(1)
        logger.info("Energy prediction models loaded (M3)")
    except FileNotFoundError:
        MODEL_LOADED.labels(model_name="energy_prediction").set(0)
        logger.warning("Energy model not found — /predict-energy will return defaults")

    try:
        load_arrivals_model()
        _arrivals_loaded = True
        MODEL_LOADED.labels(model_name="arrivals_forecast").set(1)
        logger.info("Arrivals forecast models loaded (M4A)")
    except FileNotFoundError:
        MODEL_LOADED.labels(model_name="arrivals_forecast").set(0)
        logger.warning("Arrivals model not found — /predict-arrivals will be unavailable")

    init_user_profiles_db()
    logger.info("User profiles DB initialised")

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
        energy_model_loaded=_energy_loaded,
        arrivals_model_loaded=_arrivals_loaded,
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
    # Support legacy field name alias
    user_mean_stay = request.user_mean_stay_min or request.user_historical_mean_stay_min
    result = predict_departure(
        arrival_time=arrival,
        site_id=request.site_id,
        cluster_id=request.cluster_id,
        user_n_sessions=request.user_n_sessions,
        user_mean_stay=user_mean_stay,
        user_p10_duration=request.user_p10_duration_min,
        user_p90_duration=request.user_p90_duration_min,
        user_cv_duration=request.user_cv_duration,
        user_mean_kwh=request.user_mean_kwh,
        user_p90_kwh=request.user_p90_kwh,
        user_kwh_cv=request.user_kwh_cv,
        user_days_since_last_visit=request.user_days_since_last_visit,
        user_station_affinity=request.user_station_affinity,
        station_mean_stay=request.station_historical_mean_stay_min,
        requested_energy_kwh=request.requested_energy_kwh,
    )
    DEPARTURE_LATENCY.observe(time.perf_counter() - t0)
    DEPARTURE_PREDICTIONS_TOTAL.inc()
    return DeparturePredictionResponse(
        model_version=MODEL_VERSION,
        predicted_stay_q10_min=result["predicted_stay_q10_min"],
        predicted_stay_q50_min=result["predicted_stay_q50_min"],
        predicted_stay_q90_min=result["predicted_stay_q90_min"],
        predicted_stay_duration_min=result["predicted_stay_duration_min"],
        predicted_departure_time=result["predicted_departure_time"],
    )


ENERGY_PREDICTIONS_TOTAL = Counter(
    "ml_energy_predictions_total",
    "Total energy predictions served",
)
ENERGY_LATENCY = Histogram(
    "ml_energy_prediction_seconds",
    "Energy prediction latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)
ARRIVALS_FORECASTS_TOTAL = Counter(
    "ml_arrivals_forecasts_total",
    "Total arrivals forecast requests served",
)
ARRIVALS_LATENCY = Histogram(
    "ml_arrivals_forecast_seconds",
    "Arrivals forecast latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)


@app.post("/predict-energy", response_model=EnergyPredictionResponse)
async def predict_energy_endpoint(request: EnergyPredictionRequest):
    """Predict EV energy need (M3) — used when battery SoC is unavailable."""
    t0 = time.perf_counter()
    arrival = request.arrival_time.replace(tzinfo=timezone.utc) if request.arrival_time.tzinfo is None else request.arrival_time
    result = predict_energy(
        arrival_time=arrival,
        site_id=request.site_id,
        cluster_id=request.cluster_id,
        user_n_sessions=request.user_n_sessions,
        user_mean_kwh=request.user_mean_kwh,
        user_p90_kwh=request.user_p90_kwh,
        user_kwh_cv=request.user_kwh_cv,
        user_days_since_last_visit=request.user_days_since_last_visit,
        station_mean_kwh=request.station_mean_kwh,
    )
    ENERGY_LATENCY.observe(time.perf_counter() - t0)
    ENERGY_PREDICTIONS_TOTAL.inc()
    return EnergyPredictionResponse(
        model_version=MODEL_VERSION,
        predicted_kwh=result["predicted_kwh"],
        predicted_kwh_q10=result["predicted_kwh_q10"],
        predicted_kwh_q90=result["predicted_kwh_q90"],
    )


@app.post("/predict-arrivals", response_model=ArrivalsPredictionResponse)
async def predict_arrivals_endpoint(request: ArrivalsPredictionRequest):
    """Forecast future EV arrivals per 15-min slot (M4A)."""
    t0 = time.perf_counter()
    current = request.current_time.replace(tzinfo=timezone.utc) if request.current_time.tzinfo is None else request.current_time
    slots = predict_arrivals(
        current_time=current,
        horizon_slots=request.horizon_slots,
        recent_arrivals=request.recent_arrivals,
    )
    ARRIVALS_LATENCY.observe(time.perf_counter() - t0)
    ARRIVALS_FORECASTS_TOTAL.inc()
    return ArrivalsPredictionResponse(
        model_version=MODEL_VERSION,
        predictions=[ArrivalsPredictionSlot(**s) for s in slots],
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
                "type": "HistGradientBoostingRegressor",
                "target": "log(stay_duration_minutes)",
                "quantiles": [0.10, 0.50, 0.90],
                "description": "Predicts EV stay duration (Q10/Q50/Q90) with cold-start support",
            },
            "energy_prediction": {
                "type": "HistGradientBoostingRegressor",
                "target": "log(kwh_delivered)",
                "quantiles": [0.10, 0.50, 0.90],
                "description": "Predicts EV energy need (M3) when battery SoC is unavailable",
            },
            "arrivals_forecast": {
                "type": "HistGradientBoostingRegressor",
                "target": "n_arrivals per 15-min slot",
                "quantiles": [0.10, 0.50, 0.90],
                "description": "Forecasts future EV arrivals per 15-min slot (M4A)",
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

        try:
            load_energy_model()
            _energy_loaded = True
            MODEL_LOADED.labels(model_name="energy_prediction").set(1)
            logger.info("Energy model hot-reloaded after retraining")
        except Exception as e:
            logger.error(f"Failed to reload energy model: {e}")

        try:
            load_arrivals_model()
            _arrivals_loaded = True
            MODEL_LOADED.labels(model_name="arrivals_forecast").set(1)
            logger.info("Arrivals model hot-reloaded after retraining")
        except Exception as e:
            logger.error(f"Failed to reload arrivals model: {e}")
    else:
        RETRAIN_TOTAL.labels(status="skipped").inc()

    return result


# ── User-centric prediction endpoints ─────────────────────────────────────

USER_SESSION_PREDICTIONS_TOTAL = Counter(
    "ml_user_session_predictions_total",
    "Total user-centric session predictions served",
)
USER_SESSION_LATENCY = Histogram(
    "ml_user_session_prediction_seconds",
    "User session prediction latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)
SESSIONS_COLLECTED_TOTAL = Counter(
    "ml_sessions_collected_total",
    "Total sessions collected for retraining",
)


@app.post("/predict-user-session", response_model=UserSessionPredictionResponse)
async def predict_user_session(request: UserSessionPredictionRequest):
    """Proactive user-centric prediction.

    Given only a user_id and target day of week, predict:
      • When the user will likely arrive (arrival hour window)
      • How long they will stay (Q10/Q50/Q90)
      • How much energy they will need (Q10/mean/Q90)

    The arrival time is NOT provided by the caller — the system predicts it
    from the user's historical charging patterns (M5 — statistical).
    Falls back to station-level defaults for unknown / anonymous users.

    Use this endpoint to pre-schedule charger capacity *before* the car
    physically arrives, enabling proactive load management.
    """
    t0 = time.perf_counter()

    profile = get_profile(request.user_id)
    known_user = profile is not None

    # M5 — predict arrival window from user history + day of week
    arrival = predict_arrival_window(day_of_week=request.day_of_week, profile=profile)

    # Build a representative datetime anchored to the next matching weekday
    now = datetime.now(timezone.utc)
    days_ahead = (request.day_of_week - now.weekday()) % 7
    base_date = now + timedelta(days=days_ahead)
    mean_h_int = int(arrival["mean_hour"])
    mean_m_int = int((arrival["mean_hour"] % 1) * 60)
    representative_arrival = base_date.replace(
        hour=mean_h_int, minute=mean_m_int, second=0, microsecond=0
    )

    # Build user kwargs from profile (all optional — HistGBDT handles NaN)
    user_kwargs: dict = {}
    if profile:
        user_kwargs = {
            "user_n_sessions":            float(profile.get("n_sessions", 0)),
            "user_mean_stay":             profile.get("mean_stay_min"),
            "user_p10_duration":          profile.get("p10_duration_min"),
            "user_p90_duration":          profile.get("p90_duration_min"),
            "user_cv_duration":           profile.get("cv_duration"),
            "user_mean_kwh":              profile.get("mean_kwh"),
            "user_p90_kwh":               profile.get("p90_kwh"),
            "user_kwh_cv":                profile.get("kwh_cv"),
            "user_days_since_last_visit": profile.get("days_since_last"),
            "user_station_affinity":      None,  # dict — not a scalar feature
        }

    # M1 — departure / stay prediction
    dep = predict_departure(
        arrival_time=representative_arrival,
        site_id=request.site_id,
        cluster_id=request.cluster_id,
        **user_kwargs,
    )

    # M3 — energy prediction (subset of user kwargs)
    energy_keys = {
        "user_n_sessions", "user_mean_kwh", "user_p90_kwh",
        "user_kwh_cv", "user_days_since_last_visit",
    }
    eng = predict_energy(
        arrival_time=representative_arrival,
        site_id=request.site_id,
        cluster_id=request.cluster_id,
        **{k: v for k, v in user_kwargs.items() if k in energy_keys},
    )

    USER_SESSION_LATENCY.observe(time.perf_counter() - t0)
    USER_SESSION_PREDICTIONS_TOTAL.inc()

    return UserSessionPredictionResponse(
        model_version=MODEL_VERSION,
        user_id=request.user_id,
        known_user=known_user,
        user_n_sessions=int(profile["n_sessions"]) if profile else 0,
        predicted_arrival_mean_hour=arrival["mean_hour"],
        predicted_arrival_q10_hour=arrival["q10_hour"],
        predicted_arrival_q90_hour=arrival["q90_hour"],
        arrival_data_source=arrival["data_source"],
        predicted_stay_q10_min=dep["predicted_stay_q10_min"],
        predicted_stay_q50_min=dep["predicted_stay_q50_min"],
        predicted_stay_q90_min=dep["predicted_stay_q90_min"],
        predicted_kwh=eng["predicted_kwh"],
        predicted_kwh_q10=eng["predicted_kwh_q10"],
        predicted_kwh_q90=eng["predicted_kwh_q90"],
    )


@app.post("/collect-session", response_model=CollectSessionResponse)
async def collect_session(request: CollectSessionRequest):
    """Record a completed charging session.

    Two things happen:
      1. If user_id is known, the user's profile is updated incrementally
         (Welford online mean) so their next prediction improves immediately.
      2. The session is appended to collected_sessions.csv in ACN format
         so it feeds the next retraining run.

    This is the data collection path — every real plug-in / unplug event
    should call this endpoint.  Anonymous sessions (user_id=None) are
    still collected for aggregate model retraining.
    """
    profile_updated = False
    try:
        if request.user_id is not None:
            update_profile(
                user_id=request.user_id,
                session={
                    "connection_time": request.connection_time,
                    "disconnect_time":  request.disconnect_time,
                    "kwh_delivered":    request.kwh_delivered,
                    "station_id":       request.station_id,
                },
            )
            profile_updated = True

        # Append to collected_sessions.csv in ACN format
        COLLECTED_SESSIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
        file_exists = COLLECTED_SESSIONS_CSV.exists()
        with open(COLLECTED_SESSIONS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "user_id", "station_id", "site_id", "cluster_id",
                "connection_time", "disconnect_time", "kwh_delivered",
            ])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "user_id":         request.user_id or "",
                "station_id":      request.station_id,
                "site_id":         request.site_id or "",
                "cluster_id":      request.cluster_id or "",
                "connection_time": request.connection_time.isoformat(),
                "disconnect_time": request.disconnect_time.isoformat(),
                "kwh_delivered":   request.kwh_delivered,
            })

        SESSIONS_COLLECTED_TOTAL.inc()
        return CollectSessionResponse(
            accepted=True,
            user_profile_updated=profile_updated,
            message="Session collected",
        )
    except Exception as exc:
        logger.error("Failed to collect session: %s", exc)
        return CollectSessionResponse(
            accepted=False,
            user_profile_updated=False,
            message=str(exc),
        )
