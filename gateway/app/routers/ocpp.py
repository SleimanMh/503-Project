"""
OCPP 1.6J WebSocket endpoint for the API Gateway.

Mounts at /ocpp — each charge point connects to /ocpp/{cp_id}.
Also exposes GET /ocpp/sessions for the dashboard.
"""

import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.services.ocpp_handler import EVChargePointHandler, get_active_sessions
from app.services import ocpp_simulator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocpp", tags=["OCPP"])


class _WsAdapter:
    """Thin adapter so the ocpp library can use a FastAPI WebSocket."""

    def __init__(self, ws: WebSocket):
        self._ws = ws

    async def recv(self) -> str:
        return await self._ws.receive_text()

    async def send(self, data: str) -> None:
        await self._ws.send_text(data)


@router.websocket("/{cp_id}")
async def ocpp_endpoint(cp_id: str, websocket: WebSocket):
    """Accept an OCPP 1.6J charge point connection."""
    await websocket.accept(subprotocol="ocpp1.6")
    logger.info(f"[OCPP] Charge point connected: {cp_id}")

    adapter = _WsAdapter(websocket)
    handler = EVChargePointHandler(cp_id, adapter)

    try:
        await handler.start()
    except WebSocketDisconnect:
        logger.info(f"[OCPP] Charge point disconnected: {cp_id}")
    except Exception as exc:
        logger.error(f"[OCPP] Error for {cp_id}: {exc}")


@router.get("/sessions")
async def list_sessions():
    """Return all currently active OCPP sessions (for dashboard monitoring)."""
    sessions = get_active_sessions()
    return {"count": len(sessions), "sessions": sessions}


# ── Simulator control endpoints ───────────────────────────────────────────────

class SimulateRequest(BaseModel):
    fleet: int = 3
    realistic: bool = True
    speed_factor: float = 60.0
    meter_interval: int = 30
    seed: Optional[int] = None
    session_minutes: float = 30.0


@router.post("/simulate/start")
async def simulate_start(req: SimulateRequest):
    """Launch an OCPP fleet simulation from inside the container."""
    return await ocpp_simulator.start_simulation(
        fleet=req.fleet,
        realistic=req.realistic,
        speed_factor=req.speed_factor,
        meter_interval=req.meter_interval,
        seed=req.seed,
        session_minutes=req.session_minutes,
    )


@router.post("/simulate/stop")
async def simulate_stop():
    """Cancel the running OCPP simulation."""
    return await ocpp_simulator.stop_simulation()


@router.get("/simulate/status")
async def simulate_status():
    """Return current simulation state."""
    return ocpp_simulator.get_status()
