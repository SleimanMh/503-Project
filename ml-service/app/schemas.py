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


# ── Departure Prediction ────────────────────────────────────────────────
class DeparturePredictionRequest(BaseModel):
    arrival_time: datetime = Field(..., description="Vehicle arrival time (UTC)")
    site_id: str = Field(default="0002")
    cluster_id: str = Field(default="0039")
    user_historical_mean_stay_min: Optional[float] = Field(default=None)
    station_historical_mean_stay_min: Optional[float] = Field(default=None)
    requested_energy_kwh: Optional[float] = Field(
        default=None,
        description="Energy needed by the vehicle (kWh). Strong predictor of stay duration.",
    )

class DeparturePredictionResponse(BaseModel):
    model_version: str
    predicted_stay_duration_min: float
    predicted_departure_time: datetime


# ── Health ───────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model_version: str
    demand_model_loaded: bool
    departure_model_loaded: bool
