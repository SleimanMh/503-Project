"""
Drift Monitor — detects when the departure prediction model has drifted
from its baseline performance using real collected feedback data.

Two signals are computed:
  1. Rolling MAE  — prediction error over the last `window` sessions
  2. PSI (Population Stability Index) — shift in stay-duration distribution
     vs. the original training distribution (baseline buckets from ACN data).

PSI interpretation:
  < 0.10  — no significant shift
  0.10–0.25 — moderate shift, worth monitoring
  > 0.25  — significant drift, retraining recommended
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Baseline stay-duration distribution from training data (ACN 2018-2019).
# Pre-computed bucket edges and expected proportions so the monitor works
# without loading the full training parquet at runtime.
_BASELINE_BUCKET_EDGES = [0, 60, 120, 180, 240, 300, 360, 480, 720, 1440]
_BASELINE_PROPORTIONS = [0.03, 0.08, 0.11, 0.12, 0.13, 0.14, 0.16, 0.12, 0.08, 0.03]

# Drift thresholds
_PSI_WARN = 0.10
_PSI_ALERT = 0.25
_MAE_MULTIPLIER_ALERT = 1.5  # rolling MAE > 1.5x baseline triggers alert

# Baseline MAE from training evaluation (departure_metrics.json)
_BASELINE_DEPARTURE_MAE = 121.3  # minutes


def _compute_psi(actual_values: np.ndarray, edges: list, expected: list) -> float:
    """
    Compute Population Stability Index between actual distribution and expected.

    PSI = Σ (actual% - expected%) * ln(actual% / expected%)
    """
    actual_counts, _ = np.histogram(actual_values, bins=edges + [np.inf])
    actual_pct = actual_counts / actual_counts.sum()

    psi = 0.0
    for a, e in zip(actual_pct, expected):
        a = max(a, 1e-6)  # avoid log(0)
        e = max(e, 1e-6)
        psi += (a - e) * np.log(a / e)
    return float(psi)


def compute_drift_metrics(
    feedback_path: Path,
    window: int = 200,
) -> dict:
    """
    Compute drift metrics from the departure feedback CSV.

    Returns a dict with:
      - sessions_total: total feedback rows available
      - sessions_in_window: rows used for rolling metrics
      - rolling_mae_min: MAE over last `window` sessions
      - baseline_mae_min: training baseline MAE
      - mae_ratio: rolling_mae / baseline_mae
      - psi_stay_duration: PSI of actual stay duration in window vs. training
      - drift_detected: bool — True if either signal exceeds alert threshold
      - drift_level: "none" | "warning" | "alert"
    """
    if not feedback_path.exists():
        return {
            "sessions_total": 0,
            "sessions_in_window": 0,
            "rolling_mae_min": None,
            "baseline_mae_min": _BASELINE_DEPARTURE_MAE,
            "mae_ratio": None,
            "psi_stay_duration": None,
            "drift_detected": False,
            "drift_level": "none",
            "message": "No feedback data collected yet.",
        }

    try:
        df = pd.read_csv(feedback_path)
        if df.empty or "prediction_error_min" not in df.columns:
            return {
                "sessions_total": 0,
                "sessions_in_window": 0,
                "rolling_mae_min": None,
                "baseline_mae_min": _BASELINE_DEPARTURE_MAE,
                "mae_ratio": None,
                "psi_stay_duration": None,
                "drift_detected": False,
                "drift_level": "none",
                "message": "Feedback file exists but contains no valid rows.",
            }
    except Exception as e:
        logger.error(f"Failed to read feedback CSV: {e}")
        return {"error": str(e), "drift_detected": False, "drift_level": "none"}

    total = len(df)
    recent = df.tail(window)
    n_window = len(recent)

    # Rolling MAE
    rolling_mae = float(recent["prediction_error_min"].abs().mean())
    mae_ratio = rolling_mae / _BASELINE_DEPARTURE_MAE

    # PSI on actual stay durations in this window
    actual_stays = recent["actual_stay_min"].values
    psi = _compute_psi(actual_stays, _BASELINE_BUCKET_EDGES, _BASELINE_PROPORTIONS)

    # Determine drift level
    if mae_ratio > _MAE_MULTIPLIER_ALERT or psi > _PSI_ALERT:
        drift_level = "alert"
        drift_detected = True
    elif mae_ratio > 1.2 or psi > _PSI_WARN:
        drift_level = "warning"
        drift_detected = False  # warning only, not a hard trigger
    else:
        drift_level = "none"
        drift_detected = False

    return {
        "sessions_total": total,
        "sessions_in_window": n_window,
        "rolling_mae_min": round(rolling_mae, 2),
        "baseline_mae_min": _BASELINE_DEPARTURE_MAE,
        "mae_ratio": round(mae_ratio, 3),
        "psi_stay_duration": round(psi, 4),
        "psi_threshold_warn": _PSI_WARN,
        "psi_threshold_alert": _PSI_ALERT,
        "drift_detected": drift_detected,
        "drift_level": drift_level,
        "message": (
            "Drift alert: retraining recommended."
            if drift_level == "alert"
            else "Warning: performance degrading, monitor closely."
            if drift_level == "warning"
            else "Model performance is stable."
        ),
    }
