"""User Arrival Pattern Prediction (M5 — statistical).

Predicts when a specific user is likely to arrive on a given day of week,
based on per-DOW Gaussian statistics stored in their profile.

No trained ML model is needed here — the profile store already maintains
an online estimate of (mean_hour, std_hour) per day.  Falls back to
station-level aggregate defaults for anonymous / cold-start users.

Design rationale:
  The system must NOT require the actual arrival time as input — this makes
  the predictor usable for *proactive* scheduling (e.g., pre-allocating
  charger capacity hours before the car arrives), unlike the reactive
  /predict-departure endpoint which requires a plug-in event timestamp.
"""

from typing import Optional

# Station-level default arrival windows derived from ACN dataset analysis.
# Format: {dow: (mean_hour, std_hour)}
# 0=Monday … 6=Sunday
_STATION_DEFAULTS: dict[int, tuple[float, float]] = {
    0: (9.2,  1.8),   # Monday
    1: (9.0,  1.7),   # Tuesday
    2: (9.1,  1.8),   # Wednesday
    3: (9.3,  1.9),   # Thursday
    4: (9.0,  2.1),   # Friday
    5: (10.5, 2.5),   # Saturday
    6: (11.0, 2.8),   # Sunday
}

# Minimum observations per DOW before we trust user-specific patterns
_MIN_DOW_OBSERVATIONS = 3


def predict_arrival_window(
    day_of_week: int,
    profile: Optional[dict] = None,
) -> dict:
    """Return predicted arrival hour statistics for a given day of week.

    Args:
        day_of_week  — 0=Monday … 6=Sunday
        profile      — user profile dict from user_profiles.get_profile()
                       If None or insufficient DOW history → station defaults.

    Returns dict with keys:
        mean_hour         — expected arrival hour (fractional, 0–24)
        std_hour          — standard deviation in hours
        q10_hour          — 10th-percentile (early arrival)
        q90_hour          — 90th-percentile (late arrival)
        data_source       — "user_history" | "station_default"
        n_observations    — number of sessions used for the estimate
    """
    if profile is not None:
        patterns = profile.get("arrival_patterns") or {}
        pat = patterns.get(str(day_of_week))
        if pat and int(pat.get("count", 0)) >= _MIN_DOW_OBSERVATIONS:
            return {
                "mean_hour":      float(pat["mean_hour"]),
                "std_hour":       float(pat["std_hour"]),
                "q10_hour":       float(pat["q10_hour"]),
                "q90_hour":       float(pat["q90_hour"]),
                "data_source":    "user_history",
                "n_observations": int(pat["count"]),
            }

    mean_h, std_h = _STATION_DEFAULTS.get(day_of_week, (9.0, 2.0))
    return {
        "mean_hour":      mean_h,
        "std_hour":       std_h,
        "q10_hour":       round(max(0.0, mean_h - 1.28 * std_h), 2),
        "q90_hour":       round(min(23.99, mean_h + 1.28 * std_h), 2),
        "data_source":    "station_default",
        "n_observations": 0,
    }
