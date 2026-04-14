"""Request/response schemas for the ML Service."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Demand Forecast ──────────────────────────────────────────────────────
class DemandForecastRequest(BaseModel):
    current_time: datetime = Field(..., description="Current UTC timestamp")
    horizon_windows: int = Field(default=6, ge=1, le=96, description="Number of 30-min windows to forecast")
    recent_arrivals: list[int] = Field(default=[], description="Recent arrival counts (most recent first) for lag features")
    recent_kwh: list[float] = Field(default=[], description="Recent kWh totals (most recent first) for lag features")


class DemandForecastPoint(BaseModel):
    window_start: datetime
    predicted_arrivals: float
    predicted_kwh: float


class DemandForecastResponse(BaseModel):
    model_version: str
    forecast: list[DemandForecastPoint]


# ── Departure Prediction (M1) — now returns Q10/Q50/Q90 ─────────────────
class DeparturePredictionRequest(BaseModel):
    arrival_time: datetime = Field(..., description="Vehicle arrival time (UTC)")
    site_id: str = Field(default="0002")
    cluster_id: str = Field(default="0039")
    # User history (all optional — None triggers cold-start path)
    user_n_sessions:            Optional[float] = Field(default=None)
    user_historical_mean_stay_min: Optional[float] = Field(default=None, description="Legacy alias for user_mean_stay")
    user_mean_stay_min:         Optional[float] = Field(default=None)
    user_p10_duration_min:      Optional[float] = Field(default=None)
    user_p90_duration_min:      Optional[float] = Field(default=None)
    user_cv_duration:           Optional[float] = Field(default=None)
    user_mean_kwh:              Optional[float] = Field(default=None)
    user_p90_kwh:               Optional[float] = Field(default=None)
    user_kwh_cv:                Optional[float] = Field(default=None)
    user_days_since_last_visit: Optional[float] = Field(default=None)
    user_station_affinity:      Optional[float] = Field(default=None)
    station_historical_mean_stay_min: Optional[float] = Field(default=None)
    requested_energy_kwh: Optional[float] = Field(
        default=None,
        description="Energy needed by the vehicle (kWh). Strong predictor of stay duration.",
    )


class DeparturePredictionResponse(BaseModel):
    model_version: str
    # Quantile predictions (back-transformed from log-space)
    predicted_stay_q10_min: float
    predicted_stay_q50_min: float
    predicted_stay_q90_min: float
    # Legacy field preserved for backward compatibility
    predicted_stay_duration_min: float
    predicted_departure_time: datetime


# ── Energy Prediction (M3) ───────────────────────────────────────────────
class EnergyPredictionRequest(BaseModel):
    arrival_time: datetime = Field(..., description="Vehicle arrival time (UTC)")
    site_id: str = Field(default="0002")
    cluster_id: str = Field(default="0039")
    user_n_sessions:            Optional[float] = Field(default=None)
    user_mean_kwh:              Optional[float] = Field(default=None)
    user_p90_kwh:               Optional[float] = Field(default=None)
    user_kwh_cv:                Optional[float] = Field(default=None)
    user_days_since_last_visit: Optional[float] = Field(default=None)
    station_mean_kwh:           Optional[float] = Field(default=None)


class EnergyPredictionResponse(BaseModel):
    model_version: str
    predicted_kwh:     float   # mean estimate — use as planning target
    predicted_kwh_q10: float   # optimistic lower bound
    predicted_kwh_q90: float   # conservative upper bound


# ── Arrivals Forecast (M4A) ──────────────────────────────────────────────
class ArrivalsPredictionRequest(BaseModel):
    current_time: datetime = Field(..., description="UTC timestamp of first slot")
    horizon_slots: int = Field(default=8, ge=1, le=96, description="Number of 15-min slots to forecast")
    recent_arrivals: list[float] = Field(
        default=[],
        description="Recent arrivals per 15-min slot (most recent first). Used for lag features.",
    )


class ArrivalsPredictionSlot(BaseModel):
    slot_index:   int
    slot_start:   str    # ISO datetime string
    arrivals_q10: float
    arrivals_q50: float
    arrivals_q90: float


class ArrivalsPredictionResponse(BaseModel):
    model_version: str
    predictions: list[ArrivalsPredictionSlot]


# ── Health ───────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model_version: str
    demand_model_loaded:    bool
    departure_model_loaded: bool
    energy_model_loaded:    bool = False
    arrivals_model_loaded:  bool = False


# ── User Session Prediction (user-centric, no arrival_time input) ─────────
class UserSessionPredictionRequest(BaseModel):
    """Proactive prediction request — caller provides only user identity and
    the target day of week.  The system does NOT require the current arrival
    time; it is *predicted* from the user's historical charging patterns.

    Use this endpoint to pre-schedule charger capacity before the car arrives.
    """
    user_id:     str = Field(..., description="User identifier (RFID / app token)")
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday … 6=Sunday")
    site_id:     str = Field(default="0002")
    cluster_id:  str = Field(default="0039")


class UserSessionPredictionResponse(BaseModel):
    model_version: str
    user_id:       str
    known_user:    bool = Field(..., description="False = cold-start / anonymous")
    user_n_sessions: int

    # Arrival window (M5 — statistical)
    predicted_arrival_mean_hour: float = Field(..., description="Expected arrival hour (0–24)")
    predicted_arrival_q10_hour:  float = Field(..., description="Early-arrival bound")
    predicted_arrival_q90_hour:  float = Field(..., description="Late-arrival bound")
    arrival_data_source: str = Field(..., description="user_history | station_default")

    # Stay duration (M1 — HistGBDT)
    predicted_stay_q10_min: float
    predicted_stay_q50_min: float
    predicted_stay_q90_min: float

    # Energy need (M3 — HistGBDT)
    predicted_kwh:     float
    predicted_kwh_q10: float
    predicted_kwh_q90: float


# ── Collect Session (ACN-format, feeds retraining + updates profile) ──────
class CollectSessionRequest(BaseModel):
    """Record a completed charging session.

    Schema mirrors ACN data so that collected sessions can be merged
    directly with the training dataset for the next retrain run.
    """
    user_id:         Optional[str]   = Field(default=None, description="None for anonymous")
    station_id:      str             = Field(..., description="Charging station identifier")
    connection_time: datetime        = Field(..., description="Plug-in timestamp (UTC)")
    disconnect_time: datetime        = Field(..., description="Unplug timestamp (UTC)")
    kwh_delivered:   float           = Field(..., gt=0, description="Energy delivered (kWh)")
    site_id:         Optional[str]   = Field(default=None)
    cluster_id:      Optional[str]   = Field(default=None)


class CollectSessionResponse(BaseModel):
    accepted:             bool
    user_profile_updated: bool
    message:              str
