"""Simulation API router."""

from fastapi import APIRouter

from app.schemas import SimulationRequest, SimulationResponse, RollingSimResponse
from app.services.orchestrator import run_simulation, run_rolling_simulation

router = APIRouter(prefix="/api/v1/simulation", tags=["simulation"])


@router.post("/run", response_model=SimulationResponse)
async def run(request: SimulationRequest):
    """Run a full simulation: ML forecast → optimize → grid validate."""
    return await run_simulation(request)


@router.post("/rolling", response_model=RollingSimResponse)
async def rolling(request: SimulationRequest):
    """
    Rolling-horizon simulation with per-event ML snapshots.

    Re-runs the LP at each EV arrival, calling ML demand forecast at each
    decision point to reserve capacity for predicted future arrivals.
    Returns all intermediate snapshots so the dashboard can replay the
    simulation step-by-step.
    """
    return await run_rolling_simulation(request)
