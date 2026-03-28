"""Request/response schemas for the API Gateway."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class EVInput(BaseModel):
    ev_id: str
    battery_pct: float = Field(ge=0, le=100)
    target_pct: float = Field(ge=0, le=100)
    battery_capacity_kwh: float = Field(gt=0, le=200, default=60.0)
    max_charge_kw: float = Field(gt=0, le=350, default=7.2)
    arrival_time: Optional[datetime] = None
    planned_departure_time: Optional[datetime] = None


class SimulationRequest(BaseModel):
    vehicles: list[EVInput] = Field(..., min_length=1, max_length=200)
    transformer_capacity_kw: float = Field(default=500.0, gt=0)
    building_base_load_kw: float = Field(default=150.0, gt=0)
    simulation_duration_hours: float = Field(default=24.0, gt=0, le=168)
    time_step_minutes: int = Field(default=15, ge=5, le=60)
    compare_baseline: bool = Field(default=True, description="Also run FCFS baseline for comparison")
    chargers_section_a: int = Field(default=10, ge=1, le=100, description="EV chargers in LV Section A")
    chargers_section_b: int = Field(default=10, ge=1, le=100, description="EV chargers in LV Section B")
    feeder_length_m: float = Field(default=30.0, gt=0, le=500, description="LV feeder cable length per section (metres)")


class EVResult(BaseModel):
    ev_id: str
    energy_delivered_kwh: float
    energy_needed_kwh: float
    satisfaction_pct: float
    power_schedule_kw: list[float]


class TimeSeriesPoint(BaseModel):
    time_minutes: float
    ev_load_kw: float
    building_load_kw: float
    total_load_kw: float
    transformer_utilization_pct: float
    overload: bool


class GridMetrics(BaseModel):
    feasible: bool
    max_transformer_loading_pct: float
    min_bus_voltage_pu: float
    overload_minutes: float
    warnings: list[str]
    network_losses_kw: float = Field(default=0.0)
    bus_voltages_pu: dict[str, float] = Field(default={})


class MLMetrics(BaseModel):
    """Tracks how ML models contributed to this simulation run."""
    departure_predictions_used: int = Field(default=0, description="Vehicles whose departure was ML-predicted")
    departure_predictions_skipped: int = Field(default=0, description="Vehicles with user-provided departures")
    demand_forecast_windows: int = Field(default=0, description="Future windows in ML demand forecast")
    capacity_reserved_kwh: float = Field(default=0, description="Total energy capacity reserved by ML forecast")
    avg_predicted_stay_min: Optional[float] = Field(default=None, description="Mean ML-predicted stay duration")
    default_stay_min: float = Field(default=340.9, description="ACN per-hour mean used by no-ML baseline (overall average)")


class StrategyResult(BaseModel):
    strategy: str
    ev_results: list[EVResult]
    overall_satisfaction_pct: float
    peak_load_kw: float
    total_energy_delivered_kwh: float
    overload_slots: int
    time_series: list[TimeSeriesPoint]
    grid_validation: Optional[GridMetrics] = None


class SimulationResponse(BaseModel):
    run_id: str
    status: str
    simulation_duration_hours: float
    time_step_minutes: int
    num_vehicles: int
    ai_result: StrategyResult              # LP optimizer + ML departure + ML forecast
    no_ml_result: Optional[StrategyResult] = None  # LP optimizer + fixed assumptions (no ML)
    baseline_result: Optional[StrategyResult] = None  # FCFS + no ML
    ml_metrics: Optional[MLMetrics] = None  # ML contribution metrics
