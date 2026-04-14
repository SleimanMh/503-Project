"""Energy need estimator (M3) inference — HistGBDT quantile models."""

import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import ARTIFACTS_DIR
from training.prepare_data import ENERGY_FEATURE_COLS

_models: dict = {}   # tag → fitted model  (mean, q10, q90)


def load_energy_model():
    """Load the three energy models from disk."""
    global _models
    for tag in ("mean", "q10", "q90"):
        path = ARTIFACTS_DIR / f"energy_{tag}.joblib"
        if tag == "mean":
            path = ARTIFACTS_DIR / "energy_model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Energy model not found: {path}")
        _models[tag] = joblib.load(path)


def _cyclical(value: float, period: float) -> tuple[float, float]:
    angle = 2 * np.pi * value / period
    return float(np.sin(angle)), float(np.cos(angle))


def predict_energy(
    arrival_time: datetime,
    site_id: str = "0002",
    cluster_id: str = "0039",
    user_n_sessions:            Optional[float] = None,
    user_mean_kwh:              Optional[float] = None,
    user_p90_kwh:               Optional[float] = None,
    user_kwh_cv:                Optional[float] = None,
    user_days_since_last_visit: Optional[float] = None,
    station_mean_kwh:           Optional[float] = None,
) -> dict:
    """
    Predict Q10 / mean / Q90 kWh need for an arriving EV session.

    Returns:
      predicted_kwh      — mean estimate (best point prediction)
      predicted_kwh_q10  — optimistic lower bound
      predicted_kwh_q90  — conservative upper bound

    User history fields may be None for anonymous sessions.
    """
    if not _models:
        raise RuntimeError("Energy models not loaded — call load_energy_model() first.")

    hour  = arrival_time.hour + arrival_time.minute / 60
    dow   = arrival_time.weekday()
    month = arrival_time.month

    hour_sin,  hour_cos  = _cyclical(hour,  24)
    dow_sin,   dow_cos   = _cyclical(dow,    7)
    month_sin, month_cos = _cyclical(month, 12)

    is_weekend    = float(dow >= 5)
    is_summer     = float(month in (6, 7, 8, 9))
    arrival_regime = 0.0 if hour < 7 else (1.0 if hour < 13 else 2.0)

    try:
        site_encoded    = float(int(site_id))
    except (ValueError, TypeError):
        site_encoded    = 0.0
    try:
        cluster_encoded = float(int(cluster_id))
    except (ValueError, TypeError):
        cluster_encoded = 0.0

    def _nan_or(v: Optional[float]) -> float:
        return float(v) if v is not None else np.nan

    feat: dict[str, float] = {
        "hour_sin":                  hour_sin,
        "hour_cos":                  hour_cos,
        "dow_sin":                   dow_sin,
        "dow_cos":                   dow_cos,
        "is_weekend":                is_weekend,
        "month_sin":                 month_sin,
        "month_cos":                 month_cos,
        "is_summer":                 is_summer,
        "arrival_regime":            arrival_regime,
        "site_encoded":              site_encoded,
        "cluster_encoded":           cluster_encoded,
        "user_n_sessions":           _nan_or(user_n_sessions),
        "user_mean_kwh":             _nan_or(user_mean_kwh),
        "user_p90_kwh":              _nan_or(user_p90_kwh),
        "user_kwh_cv":               _nan_or(user_kwh_cv),
        "user_days_since_last_visit":_nan_or(user_days_since_last_visit),
        "is_anonymous":              0.0 if user_n_sessions is not None else 1.0,
        "station_mean_kwh":          float(station_mean_kwh) if station_mean_kwh is not None else 8.5,
    }

    X = np.array([[feat[f] for f in ENERGY_FEATURE_COLS]], dtype=float)

    # Predict in log(kWh) space, back-transform with clip(min=0.1 kWh)
    mean_kwh = float(np.exp(_models["mean"].predict(X)[0]))
    q10_kwh  = float(np.exp(_models["q10"].predict(X)[0]))
    q90_kwh  = float(np.exp(_models["q90"].predict(X)[0]))

    # Enforce monotonicity
    q10_kwh = min(q10_kwh, mean_kwh)
    q90_kwh = max(q90_kwh, mean_kwh)

    return {
        "predicted_kwh":     round(max(0.1, mean_kwh), 3),
        "predicted_kwh_q10": round(max(0.1, q10_kwh),  3),
        "predicted_kwh_q90": round(max(0.1, q90_kwh),  3),
    }
