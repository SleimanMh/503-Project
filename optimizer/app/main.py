"""
Optimizer + Grid Validation Service — Container 3 (Internal)
Receives vehicle state and forecasts, returns optimized charging schedule
validated against grid constraints.
"""

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter

from app.schemas import (
    OptimizationRequest,
    OptimizationResponse,
    GridValidationRequest,
    GridValidationResponse,
)
from app.optimizer.lp_scheduler import optimize_schedule
from app.optimizer.fcfs_baseline import fcfs_schedule
from app.grid.validator import validate_grid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Prometheus custom metrics ────────────────────────────────────────────
OPTIMIZATION_LATENCY = Histogram(
    "optimizer_solve_seconds",
    "Optimization solve latency in seconds",
    ["strategy"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)
OPTIMIZATION_REQUESTS = Counter(
    "optimizer_requests_total",
    "Total optimization requests",
    ["strategy"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Optimizer + Grid Validation service started")
    yield


app = FastAPI(
    title="EV Charging Optimizer + Grid Validation",
    description="Internal service: LP optimization and grid safety checks",
    version="1.0.0",
    lifespan=lifespan,
)

# Prometheus auto-instrumentation — exposes /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "optimizer"}


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize(request: OptimizationRequest):
    """Run optimization with the chosen strategy (optimal, fcfs, or greedy)."""
    import time as _time
    t0 = _time.perf_counter()

    if request.strategy == "fcfs":
        schedules = fcfs_schedule(
            vehicles=request.vehicles,
            num_slots=request.num_slots,
            slot_duration_hours=request.slot_duration_hours,
            transformer_capacity_kw=request.transformer_capacity_kw,
            base_load_per_slot_kw=request.base_load_per_slot_kw,
        )
    else:
        # "optimal" and "greedy" both use LP (greedy is a future heuristic)
        schedules = optimize_schedule(
            vehicles=request.vehicles,
            num_slots=request.num_slots,
            slot_duration_hours=request.slot_duration_hours,
            transformer_capacity_kw=request.transformer_capacity_kw,
            base_load_per_slot_kw=request.base_load_per_slot_kw,
            predicted_future_arrivals=request.predicted_future_arrivals or None,
        )

    OPTIMIZATION_LATENCY.labels(strategy=request.strategy).observe(_time.perf_counter() - t0)
    OPTIMIZATION_REQUESTS.labels(strategy=request.strategy).inc()

    # Compute aggregate metrics
    total_delivered = sum(s.energy_delivered_kwh for s in schedules)
    total_needed = sum(s.energy_needed_kwh for s in schedules)
    overall_sat = (total_delivered / total_needed * 100) if total_needed > 0 else 100.0

    # Compute per-slot totals
    num_slots = request.num_slots
    total_load_per_slot = []
    for k in range(num_slots):
        ev_load = sum(s.power_per_slot_kw[k] for s in schedules)
        total = ev_load + request.base_load_per_slot_kw[k]
        total_load_per_slot.append(total)

    peak_load = max(total_load_per_slot) if total_load_per_slot else 0
    overload_slots = sum(1 for l in total_load_per_slot if l > request.transformer_capacity_kw)
    utilization = [
        round(l / request.transformer_capacity_kw * 100, 2) if request.transformer_capacity_kw > 0 else 0
        for l in total_load_per_slot
    ]

    return OptimizationResponse(
        strategy=request.strategy,
        schedules=schedules,
        total_energy_delivered_kwh=round(total_delivered, 3),
        total_energy_needed_kwh=round(total_needed, 3),
        overall_satisfaction_pct=round(overall_sat, 1),
        peak_load_kw=round(peak_load, 2),
        overload_slots=overload_slots,
        transformer_utilization_per_slot=utilization,
    )


@app.post("/validate-grid", response_model=GridValidationResponse)
async def validate(request: GridValidationRequest):
    """Run grid validation (simplified power flow or OpenDSS when available)."""
    return validate_grid(request)
