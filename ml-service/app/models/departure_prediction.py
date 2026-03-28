"""Departure prediction inference — loads trained XGBoost model and predicts stay duration."""

import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from app.config import ARTIFACTS_DIR


_model = None


def load_departure_model():
    global _model
    model_path = ARTIFACTS_DIR / "departure_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Departure model not found at {model_path}")
    with open(model_path, "rb") as f:
        _model = pickle.load(f)
    return _model


def _cyclical(value: float, period: float) -> tuple[float, float]:
    angle = 2 * np.pi * value / period
    return float(np.sin(angle)), float(np.cos(angle))


def predict_departure(
    arrival_time: datetime,
    site_id: str = "0002",
    cluster_id: str = "0039",
    user_mean_stay: float | None = None,
    station_mean_stay: float | None = None,
    requested_energy_kwh: float | None = None,
) -> dict:
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
