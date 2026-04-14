"""Arrivals forecast (M4A) inference — HistGBDT quantile models, 15-min slots."""

import numpy as np
import joblib
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.config import ARTIFACTS_DIR
from training.prepare_data import ARRIVALS_FEATURE_COLS

_models: dict = {}   # tag → fitted model  (q10, q50, q90)


def load_arrivals_model():
    """Load the three arrivals models from disk."""
    global _models
    for tag in ("q10", "q50", "q90"):
        path = ARTIFACTS_DIR / f"arrivals_{tag}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Arrivals model not found: {path}")
        _models[tag] = joblib.load(path)


def _cyclical(value: float, period: float) -> tuple[float, float]:
    angle = 2 * np.pi * value / period
    return float(np.sin(angle)), float(np.cos(angle))


def predict_arrivals(
    current_time: datetime,
    horizon_slots: int = 8,
    recent_arrivals: Optional[list[float]] = None,
) -> list[dict]:
    """
    Forecast arrivals per 15-min slot for the next `horizon_slots` slots.

    Args:
        current_time:    UTC timestamp for the first slot
        horizon_slots:   Number of 15-min slots to forecast (default = 8 = 2 hours)
        recent_arrivals: Recent arrivals history (most-recent-first, used for lag features).
                         Pass an empty list or None if no history is available.

    Returns list of dicts with keys:
        slot_index, slot_start, arrivals_q10, arrivals_q50, arrivals_q90
    """
    if not _models:
        raise RuntimeError("Arrivals models not loaded — call load_arrivals_model() first.")

    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    history = list(recent_arrivals or [])   # most-recent-first
    results = []

    for i in range(horizon_slots):
        slot_start = current_time + timedelta(minutes=15 * i)
        hour  = slot_start.hour + slot_start.minute / 60
        dow   = slot_start.weekday()
        month = slot_start.month

        hour_sin,  hour_cos  = _cyclical(hour,  24)
        dow_sin,   dow_cos   = _cyclical(dow,    7)
        month_sin, month_cos = _cyclical(month, 12)

        is_weekend = float(dow >= 5)
        is_summer  = float(month in (6, 7, 8, 9))
        slot_of_day = float((slot_start.hour * 60 + slot_start.minute) // 15)

        def _lag(n: int) -> float:
            """Return arrival count from n slots back (0 if no history)."""
            if n <= len(history):
                return float(history[n - 1])
            return 0.0

        last_4 = [history[j] for j in range(min(4, len(history)))]
        rolling_mean = float(np.mean(last_4)) if last_4 else 0.0
        rolling_std  = float(np.std(last_4))  if last_4 else 0.0

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
            "lag_arrivals_4":            _lag(4),
            "lag_arrivals_16":           _lag(16),
            "lag_arrivals_96":           _lag(96),
            "lag_arrivals_672":          _lag(672),
            "rolling_arrivals_mean_1h":  rolling_mean,
            "rolling_arrivals_std_1h":   rolling_std,
        }

        X = np.array([[feat[f] for f in ARRIVALS_FEATURE_COLS]], dtype=float)

        q10 = max(0.0, float(_models["q10"].predict(X)[0]))
        q50 = max(0.0, float(_models["q50"].predict(X)[0]))
        q90 = max(0.0, float(_models["q90"].predict(X)[0]))
        q10 = min(q10, q50)
        q90 = max(q90, q50)

        results.append({
            "slot_index":    i,
            "slot_start":    slot_start.isoformat(),
            "arrivals_q10":  round(q10, 3),
            "arrivals_q50":  round(q50, 3),
            "arrivals_q90":  round(q90, 3),
        })

        # Autoregressive: feed Q50 prediction back as history for next slot
        history.insert(0, q50)
        if len(history) > 672:
            history = history[:672]

    return results
