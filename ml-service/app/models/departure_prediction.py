"""Departure prediction inference — loads HistGBDT quantile models and predicts stay duration."""

import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from app.config import ARTIFACTS_DIR
from training.prepare_data import DEPARTURE_FEATURE_COLS

_models: dict = {}   # tag → fitted model  (q10, q50, q90)


def load_departure_model():
    """Load the three quantile departure models from disk."""
    global _models
    for tag in ("q10", "q50", "q90"):
        path = ARTIFACTS_DIR / f"departure_{tag}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Departure model not found: {path}")
        _models[tag] = joblib.load(path)


def _cyclical(value: float, period: float) -> tuple[float, float]:
    angle = 2 * np.pi * value / period
    return float(np.sin(angle)), float(np.cos(angle))


def predict_departure(
    arrival_time: datetime,
    site_id: str = "0002",
    cluster_id: str = "0039",
    # User history — None for anonymous / cold-start sessions
    user_n_sessions:            Optional[float] = None,
    user_mean_stay:             Optional[float] = None,   # minutes
    user_p10_duration:          Optional[float] = None,
    user_p90_duration:          Optional[float] = None,
    user_cv_duration:           Optional[float] = None,
    user_mean_kwh:              Optional[float] = None,
    user_p90_kwh:               Optional[float] = None,
    user_kwh_cv:                Optional[float] = None,
    user_days_since_last_visit: Optional[float] = None,
    user_station_affinity:      Optional[float] = None,
    # Station context
    station_mean_stay:          Optional[float] = None,   # minutes; defaults to dataset mean
    # Energy context
    requested_energy_kwh:       Optional[float] = None,
) -> dict:
    """
    Predict Q10 / Q50 / Q90 stay duration for an arriving EV session.

    Returns:
      predicted_stay_q10_min  — optimistic departure (car leaves early)
      predicted_stay_q50_min  — median planning horizon
      predicted_stay_q90_min  — conservative window (LP presence boundary)
      predicted_departure_time — ISO datetime at Q50

    User history fields may be None for anonymous / first-visit sessions.
    HistGBDT handles NaN natively — no imputation required.
    """
    if not _models:
        raise RuntimeError("Departure models not loaded — call load_departure_model() first.")

    hour  = arrival_time.hour + arrival_time.minute / 60
    dow   = arrival_time.weekday()
    month = arrival_time.month

    hour_sin,  hour_cos  = _cyclical(hour,  24)
    dow_sin,   dow_cos   = _cyclical(dow,    7)
    month_sin, month_cos = _cyclical(month, 12)

    is_weekend    = float(dow >= 5)
    is_summer     = float(month in (6, 7, 8, 9))
    slot_of_day   = float(int(hour * 4))   # 0–95
    arrival_regime = 0.0 if hour < 7 else (1.0 if hour < 13 else 2.0)

    try:
        site_encoded    = float(int(site_id))
    except (ValueError, TypeError):
        site_encoded    = 0.0
    try:
        cluster_encoded = float(int(cluster_id))
    except (ValueError, TypeError):
        cluster_encoded = 0.0

    # Station prior: fall back to ACN overall mean (340 min) when unknown
    st_mean = float(station_mean_stay) if station_mean_stay is not None else 340.9

    def _nan_or(v: Optional[float]) -> float:
        return float(v) if v is not None else np.nan

    # Build feature dict in DEPARTURE_FEATURE_COLS order
    feat: dict[str, float] = {
        "hour_sin":                  hour_sin,
        "hour_cos":                  hour_cos,
        "dow_sin":                   dow_sin,
        "dow_cos":                   dow_cos,
        "is_weekend":                is_weekend,
        "month_sin":                 month_sin,
        "month_cos":                 month_cos,
        "is_summer":                 is_summer,
        "slot_of_day":               slot_of_day,
        "arrival_regime":            arrival_regime,
        "site_encoded":              site_encoded,
        "cluster_encoded":           cluster_encoded,
        "user_n_sessions":           _nan_or(user_n_sessions),
        "user_mean_stay":            _nan_or(user_mean_stay),
        "user_p10_duration":         _nan_or(user_p10_duration),
        "user_p90_duration":         _nan_or(user_p90_duration),
        "user_cv_duration":          _nan_or(user_cv_duration),
        "user_mean_kwh":             _nan_or(user_mean_kwh),
        "user_p90_kwh":              _nan_or(user_p90_kwh),
        "user_kwh_cv":               _nan_or(user_kwh_cv),
        "user_days_since_last_visit":_nan_or(user_days_since_last_visit),
        "user_station_affinity":     _nan_or(user_station_affinity),
        "is_anonymous":              0.0 if user_n_sessions is not None else 1.0,
        "station_mean_stay":         st_mean,
        "requested_energy_kwh":      _nan_or(requested_energy_kwh),
    }

    X = np.array([[feat[f] for f in DEPARTURE_FEATURE_COLS]], dtype=float)

    # Predict in log(duration_min) space, back-transform with exp()
    q10_min = float(np.exp(_models["q10"].predict(X)[0]))
    q50_min = float(np.exp(_models["q50"].predict(X)[0]))
    q90_min = float(np.exp(_models["q90"].predict(X)[0]))

    # Enforce monotonicity (quantile crossing can occasionally occur)
    q10_min = min(q10_min, q50_min)
    q90_min = max(q90_min, q50_min)

    departure_time = arrival_time + timedelta(minutes=q50_min)

    return {
        "predicted_stay_q10_min": round(q10_min, 1),
        "predicted_stay_q50_min": round(q50_min, 1),
        "predicted_stay_q90_min": round(q90_min, 1),
        # Legacy field kept for backward compatibility with old callers
        "predicted_stay_duration_min": round(q50_min, 1),
        "predicted_departure_time": departure_time,
    }

    """
    Predict how long a vehicle will stay given arrival context.
    Returns predicted stay duration in minutes and departure time.
    """
    if _model is None:
        raise RuntimeError("Departure model not loaded")

    hour = arrival_time.hour + arrival_time.minute / 60
    dow = arrival_time.weekday()
    month = arrival_time.month

    hour_sin, hour_cos = _cyclical(hour, 24)
    dow_sin, dow_cos = _cyclical(dow, 7)
    month_sin, month_cos = _cyclical(month, 12)
    is_weekend = 1 if dow >= 5 else 0

    # Encode site/cluster as integers (fallback to 0 for unknown)
    try:
        site_encoded = int(site_id)
    except ValueError:
        site_encoded = 0
    try:
        cluster_encoded = int(cluster_id)
    except ValueError:
        cluster_encoded = 0

    # Default historical averages if not provided
    if user_mean_stay is None:
        user_mean_stay = 300.0  # ~5 hours default
    if station_mean_stay is None:
        station_mean_stay = 300.0
    # Use 9 kWh as a neutral default (ACN dataset mean) when unknown
    if requested_energy_kwh is None:
        requested_energy_kwh = 9.0

    features = np.array([[
        hour_sin, hour_cos, dow_sin, dow_cos, is_weekend,
        month_sin, month_cos,
        site_encoded, cluster_encoded,
        user_mean_stay, station_mean_stay,
        requested_energy_kwh,
    ]])

    predicted_min = float(_model.predict(features)[0])
    predicted_min = max(15.0, predicted_min)  # minimum 15 minutes

    return {
        "predicted_stay_duration_min": predicted_min,
        "predicted_departure_time": arrival_time + timedelta(minutes=predicted_min),
    }
