"""
OCPP fleet simulator — runs inside the gateway container.

Exposes run_fleet_simulation() which spawns OCPP charge-point sessions
as asyncio tasks against the gateway's own WebSocket server.
The gateway connects to itself via ws://localhost:8000/ocpp/{cp_id}.
"""

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import websockets
from ocpp.routing import on
from ocpp.v16 import ChargePoint as Cp16
from ocpp.v16 import call, call_result
from ocpp.v16.enums import Action, RegistrationStatus

logger = logging.getLogger(__name__)

# ── State accessible by the API router ────────────────────────────────────────
_sim_task:   Optional[asyncio.Task] = None
_sim_status: dict = {"running": False, "vehicles": [], "started_at": None, "error": None}


# ── Vehicle profile ───────────────────────────────────────────────────────────

@dataclass
class VehicleProfile:
    cp_id: str
    id_tag: str
    battery_capacity_kwh: float
    initial_soc_pct: float
    max_charge_kw: float
    stay_minutes: float
    arrival_offset_s: float

    @property
    def energy_needed_kwh(self) -> float:
        return self.battery_capacity_kwh * (1.0 - self.initial_soc_pct / 100.0)

    @property
    def charge_time_min(self) -> float:
        return (self.energy_needed_kwh / self.max_charge_kw) * 60.0

    @property
    def meter_start_wh(self) -> int:
        seed = abs(hash(self.cp_id)) % 50_000
        return seed + int(self.battery_capacity_kwh * self.initial_soc_pct * 10)


def _sample_realistic_profile(cp_id: str, seed: Optional[int] = None) -> VehicleProfile:
    rng = random.Random(seed)
    battery_kwh  = rng.choice([24, 30, 40, 50, 60, 75, 82, 100])
    initial_soc  = rng.uniform(30, 85)
    max_kw       = rng.choice([3.3, 6.6, 7.2, 9.6, 11.5])
    stay_min     = max(30.0, rng.lognormvariate(math.log(300), 0.7))
    offset_s     = rng.uniform(0, 1800)
    id_tag       = f"RFID_{cp_id}" if rng.random() > 0.78 else f"ANON_{cp_id[-3:]}"
    return VehicleProfile(
        cp_id=cp_id,
        id_tag=id_tag,
        battery_capacity_kwh=battery_kwh,
        initial_soc_pct=initial_soc,
        max_charge_kw=max_kw,
        stay_minutes=stay_min,
        arrival_offset_s=offset_s,
    )


# ── OCPP Charge Point client ──────────────────────────────────────────────────

class SimulatedChargePoint(Cp16):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_received = asyncio.Event()
        self.current_limit_w: float = 7200.0

    @on(Action.SetChargingProfile)
    async def on_set_charging_profile(self, connector_id, cs_charging_profiles, **kwargs):
        periods = cs_charging_profiles.get("chargingSchedule", {}).get("chargingSchedulePeriod", [])
        if periods:
            self.current_limit_w = float(periods[0]["limit"])
        self.profile_received.set()
        return call_result.SetChargingProfile(status="Accepted")


async def _run_session(url: str, profile: VehicleProfile, meter_interval_s: int, speed_factor: float):
    cp_id = profile.cp_id
    wait_s = profile.arrival_offset_s / speed_factor
    if wait_s > 0.5:
        await asyncio.sleep(wait_s)

    async with websockets.connect(
        f"{url}/ocpp/{cp_id}",
        subprotocols=["ocpp1.6"],
        ping_interval=30,
    ) as ws:
        cp = SimulatedChargePoint(cp_id, ws)
        asyncio.ensure_future(cp.start())

        await asyncio.sleep(0.5)
        boot = await cp.call(call.BootNotification(
            charge_point_model="EV-Simulator",
            charge_point_vendor="DashboardLaunch",
        ))
        assert boot.status == RegistrationStatus.accepted

        await asyncio.sleep(0.3)
        await cp.call(call.Authorize(id_tag=profile.id_tag))

        await asyncio.sleep(0.3)
        tx = await cp.call(call.StartTransaction(
            connector_id=1,
            id_tag=profile.id_tag,
            meter_start=profile.meter_start_wh,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))
        tx_id = tx.transaction_id

        try:
            await asyncio.wait_for(cp.profile_received.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning(f"[{cp_id}] No SetChargingProfile within 10 s — using default")

        stay_s        = profile.stay_minutes * 60
        charge_done_s = profile.charge_time_min * 60
        energy_dispensed_wh = 0.0
        energy_cap_wh = profile.energy_needed_kwh * 1000.0
        meter_start   = profile.meter_start_wh
        interval_s    = meter_interval_s / speed_factor
        session_start = time.monotonic()
        elapsed_s     = 0.0

        while elapsed_s < stay_s / speed_factor:
            await asyncio.sleep(interval_s)
            elapsed_s   = time.monotonic() - session_start
            sim_elapsed = elapsed_s * speed_factor

            if sim_elapsed < charge_done_s and energy_dispensed_wh < energy_cap_wh:
                actual_power_w = min(profile.max_charge_kw * 1000, cp.current_limit_w)
                delta_wh = actual_power_w * (meter_interval_s / 3600.0)
                energy_dispensed_wh = min(energy_dispensed_wh + delta_wh, energy_cap_wh)
            else:
                actual_power_w = 0.0

            meter_wh = meter_start + energy_dispensed_wh
            await cp.call(call.MeterValues(
                connector_id=1,
                meter_value=[{
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "sampled_value": [
                        {"value": f"{meter_wh:.1f}", "measurand": "Energy.Active.Import.Register",
                         "unit": "Wh", "context": "Sample.Periodic"},
                        {"value": f"{actual_power_w:.0f}", "measurand": "Power.Active.Import",
                         "unit": "W", "context": "Sample.Periodic"},
                    ],
                }],
                transaction_id=tx_id,
            ))

        final_meter = int(meter_start + energy_dispensed_wh)
        await cp.call(call.StopTransaction(
            meter_stop=final_meter,
            timestamp=datetime.now(timezone.utc).isoformat(),
            transaction_id=tx_id,
        ))
        logger.info(f"[{cp_id}] Session complete — {energy_dispensed_wh/1000:.2f} kWh delivered")

    # Update vehicle status in global state
    for v in _sim_status["vehicles"]:
        if v["cp_id"] == cp_id:
            v["done"] = True


async def _fleet_task(profiles: list, meter_interval_s: int, speed_factor: float):
    global _sim_status
    _sim_status["running"] = True
    url = "ws://localhost:8000"
    try:
        await asyncio.gather(
            *[_run_session(url, p, meter_interval_s, speed_factor) for p in profiles],
            return_exceptions=True,
        )
    except asyncio.CancelledError:
        logger.info("[Simulator] Fleet cancelled")
    except Exception as exc:
        logger.error(f"[Simulator] Fleet error: {exc}")
        _sim_status["error"] = str(exc)
    finally:
        _sim_status["running"] = False


# ── Public API used by the router ─────────────────────────────────────────────

def get_status() -> dict:
    return dict(_sim_status)


async def start_simulation(
    fleet: int = 3,
    realistic: bool = True,
    speed_factor: float = 60.0,
    meter_interval: int = 30,
    seed: Optional[int] = None,
    cp_prefix: str = "CP",
    session_minutes: float = 30.0,
) -> dict:
    global _sim_task, _sim_status

    if _sim_status["running"]:
        return {"started": False, "reason": "simulation already running"}

    rng = random.Random(seed)
    profiles = []
    for i in range(1, fleet + 1):
        cp_id = f"{cp_prefix}_{i:03d}"
        if realistic:
            p_seed = rng.randint(0, 99999) if seed is None else seed + i
            p = _sample_realistic_profile(cp_id, seed=p_seed)
        else:
            offset_s = (i - 1) * (30.0 / max(fleet - 1, 1)) if fleet > 1 else 0.0
            p = VehicleProfile(
                cp_id=cp_id,
                id_tag=f"RFID_{cp_id}",
                battery_capacity_kwh=60.0,
                initial_soc_pct=40.0,
                max_charge_kw=7.2,
                stay_minutes=session_minutes,
                arrival_offset_s=offset_s,
            )
        profiles.append(p)

    _sim_status = {
        "running": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
        "vehicles": [
            {
                "cp_id": p.cp_id,
                "battery_kwh": p.battery_capacity_kwh,
                "initial_soc": round(p.initial_soc_pct, 1),
                "max_kw": p.max_charge_kw,
                "stay_min": round(p.stay_minutes, 1),
                "done": False,
            }
            for p in profiles
        ],
    }

    loop = asyncio.get_event_loop()
    _sim_task = loop.create_task(_fleet_task(profiles, meter_interval, speed_factor))
    return {"started": True, "fleet": fleet, "speed_factor": speed_factor, "vehicles": _sim_status["vehicles"]}


async def stop_simulation() -> dict:
    global _sim_task, _sim_status
    if _sim_task and not _sim_task.done():
        _sim_task.cancel()
        try:
            await _sim_task
        except asyncio.CancelledError:
            pass
    _sim_status["running"] = False
    return {"stopped": True}
