"""
API Gateway — Container 1 (External)
The only public-facing service. Orchestrates ML and optimizer calls.
"""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import ML_SERVICE_URL, OPTIMIZER_SERVICE_URL
from app.routers.simulation import router as simulation_router
from app.routers.ml_monitoring import router as ml_monitoring_router

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API Gateway starting")
    logger.info(f"  ML Service: {ML_SERVICE_URL}")
    logger.info(f"  Optimizer: {OPTIMIZER_SERVICE_URL}")
    yield
    logger.info("API Gateway shutting down")


app = FastAPI(
    title="EV Charging Optimization API",
    description="AI-Assisted EV Charging Optimization and Grid Compatibility System",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(simulation_router)
app.include_router(ml_monitoring_router)

# Prometheus metrics — exposes /metrics endpoint
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)


@app.get("/health")
async def health():
    """Check health of gateway and internal services."""
    ml_healthy = False
    optimizer_healthy = False

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{ML_SERVICE_URL}/health")
            ml_healthy = r.status_code == 200
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{OPTIMIZER_SERVICE_URL}/health")
            optimizer_healthy = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "healthy",
        "services": {
            "ml_service": "up" if ml_healthy else "down",
            "optimizer": "up" if optimizer_healthy else "down",
        },
    }
