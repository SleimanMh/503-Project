"""
OCPP 1.6J Central System handler.

Integrates OCPP messages with the ML + LP pipeline:
  - BootNotification / Heartbeat / Authorize → standard OCPP responses
  - StartTransaction → ML departure + energy prediction → LP schedule → SetChargingProfile
  - MeterValues → track live energy readings
  - StopTransaction → log actual energy delivered

WebSocket endpoint: ws://gateway:8000/ocpp/{charge_point_id}
Sub-protocol: ocpp1.6
"""

import asyncio
import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
from ocpp.routing import on
from ocpp.v16 import ChargePoint as Cp16
from ocpp.v16 import call, call_result
from ocpp.v16.enums import (
    Action,
    AuthorizationStatus,
    ChargingProfileKindType,
    ChargingProfilePurposeType,
    ChargingRateUnitType,
    RegistrationStatus,
)

from app.config import DATA_DIR, ML_SERVICE_URL, ML_TIMEOUT_S, OPTIMIZER_SERVICE_URL, OPTIMIZER_TIMEOUT_S

logger = logging.getLogger(__name__)

# ── Active session state ──────────────────────────────────────────────────────

@dataclass
class ActiveSession:
    transaction_id: int
    cp_id: str
    connector_id: int
    id_tag: str
    arrival_time: datetime
    meter_start_wh: float
    predicted_stay_min: float
    energy_needed_kwh: float
    schedule_kw: list[float]
    last_meter_wh: float = 0.0
    last_power_w: float = 0.0           # last known power reading from MeterValues
    done_charging_time: datetime | None = None   # when power first dropped to 0 (battery full)
    meter_stop_wh: float | None = None


# ── ACN-format session collection ────────────────────────────────────────────

_OCPP_SESSIONS_HEADER = [
    "session_id", "station_id", "site_id", "cluster_id",
    "connection_time", "disconnect_time", "done_charging",
    "kwh_delivered", "user_id",
    "duration_hr", "charge_time_hr", "idle_time_hr",
    "arrival_hour", "departure_hour",
    "day_of_week", "is_weekend", "month", "year",
    "avg_charge_kw", "predicted_stay_min",
]


def _write_ocpp_session(s: "ActiveSession", disconnect_time: datetime, meter_stop_wh: float) -> None:
    """
    Append a completed OCPP session to collected_sessions.csv in ACN format.

    disconnect_time  = timestamp from StopTransaction  (actual departure — driver unplugged)
    done_charging    = time power dropped to 0         (battery full / schedule ended)
    idle_time_hr     = disconnect_time − done_charging (car sat plugged-in but charged)

    The CSMS never predicted disconnect_time — it knew only predicted_stay_min.
    This gap between prediction and reality is what feeds model improvement.
    """
    kwh_delivered = max(0.0, (meter_stop_wh - s.meter_start_wh) / 1000.0)
    duration_hr   = (disconnect_time - s.arrival_time).total_seconds() / 3600.0

    # done_charging: if we tracked a power-drop event use it, else fall back to disconnect_time
    done_charging = s.done_charging_time if s.done_charging_time else disconnect_time
    charge_time_hr = (done_charging - s.arrival_time).total_seconds() / 3600.0
    idle_time_hr   = (disconnect_time - done_charging).total_seconds() / 3600.0

    avg_kw = (kwh_delivered / charge_time_hr) if charge_time_hr > 0 else 0.0

    row = {
        "session_id":        f"ocpp_{s.cp_id}_{s.transaction_id}",
        "station_id":        s.cp_id,
        "site_id":           "0002",
        "cluster_id":        "0039",
        "connection_time":   s.arrival_time.isoformat(),
        "disconnect_time":   disconnect_time.isoformat(),
        "done_charging":     done_charging.isoformat(),
        "kwh_delivered":     round(kwh_delivered, 4),
        "user_id":           s.id_tag,
        "duration_hr":       round(duration_hr, 4),
        "charge_time_hr":    round(charge_time_hr, 4),
        "idle_time_hr":      round(idle_time_hr, 4),
        "arrival_hour":      round(s.arrival_time.hour + s.arrival_time.minute / 60, 4),
        "departure_hour":    round(disconnect_time.hour + disconnect_time.minute / 60, 4),
        "day_of_week":       s.arrival_time.weekday(),
        "is_weekend":        1 if s.arrival_time.weekday() >= 5 else 0,
        "month":             s.arrival_time.month,
        "year":              s.arrival_time.year,
        "avg_charge_kw":     round(avg_kw, 4),
        "predicted_stay_min": round(s.predicted_stay_min, 2),
    }

    path = DATA_DIR / "ocpp_sessions.csv"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_OCPP_SESSIONS_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        logger.info(
            f"[OCPP] Session saved: {row['session_id']}  "
            f"kwh={kwh_delivered:.2f}  duration={duration_hr:.2f}h  "
            f"charge={charge_time_hr:.2f}h  idle={idle_time_hr:.2f}h"
        )
    except Exception as exc:
        logger.warning(f"[OCPP] Failed to write session to CSV: {exc}")


_active_sessions: dict[str, ActiveSession] = {}   # key = "{cp_id}:{connector_id}"
_active_handlers: dict[str, "EVChargePointHandler"] = {}  # key = cp_id → handler instance
_tx_counter: int = 0


def _next_tx_id() -> int:
    global _tx_counter
    _tx_counter += 1
    return _tx_counter


def get_active_sessions() -> list[dict]:
    """Return a snapshot of all active OCPP sessions (for the /ocpp/sessions endpoint)."""
    result = []
    for key, s in _active_sessions.items():
        elapsed_min = (datetime.now(timezone.utc) - s.arrival_time).total_seconds() / 60
        time_step_min = 15
        current_slot = min(int(elapsed_min / time_step_min), len(s.schedule_kw) - 1) if s.schedule_kw else 0
        current_power_kw = s.schedule_kw[current_slot] if s.schedule_kw else 0.0
        result.append({
            "session_key": key,
            "transaction_id": s.transaction_id,
            "cp_id": s.cp_id,
            "connector_id": s.connector_id,
            "id_tag": s.id_tag,
            "arrival_time": s.arrival_time.isoformat(),
            "predicted_stay_min": round(s.predicted_stay_min, 1),
            "energy_needed_kwh": round(s.energy_needed_kwh, 3),
            "meter_start_wh": round(s.meter_start_wh, 1),
            "last_meter_wh": round(s.last_meter_wh, 1),
            "schedule_kw": [round(p, 3) for p in s.schedule_kw],
            "current_power_kw": round(current_power_kw, 3),
        })
    return result


# ── ML + Optimizer helpers ────────────────────────────────────────────────────

async def _ml_predict_energy(arrival_time: datetime) -> float:
    """Call /predict-energy on the ML service. Returns predicted kWh (fallback 8.5)."""
    payload = {
        "arrival_datetime": arrival_time.isoformat(),
        "user_id": "ocpp_unknown",
    }
    try:
        async with httpx.AsyncClient(timeout=ML_TIMEOUT_S) as client:
            r = await client.post(f"{ML_SERVICE_URL}/predict-energy", json=payload)
            r.raise_for_status()
            return float(r.json()["energy_kwh_q50"])
    except Exception as exc:
        logger.warning(f"ML energy prediction failed: {exc} — using default 8.5 kWh")
        return 8.5


async def _ml_predict_departure(arrival_time: datetime, energy_kwh: float) -> float:
    """Call /predict-departure on the ML service. Returns predicted stay in minutes (fallback 180)."""
    payload = {
        "arrival_datetime": arrival_time.isoformat(),
        "energy_kwh": energy_kwh,
        "user_id": "ocpp_unknown",
    }
    try:
        async with httpx.AsyncClient(timeout=ML_TIMEOUT_S) as client:
            r = await client.post(f"{ML_SERVICE_URL}/predict-departure", json=payload)
            r.raise_for_status()
            return float(r.json()["stay_duration_min_q50"])
    except Exception as exc:
        logger.warning(f"ML departure prediction failed: {exc} — using default 180 min")
        return 180.0


async def _run_optimizer(
    ev_id: str,
    energy_kwh: float,
    max_kw: float,
    predicted_stay_min: float,
    time_step_min: int = 15,
    transformer_kw: float = 150.0,
) -> list[float]:
    """
    Call the LP optimizer with a single EV and return power_per_slot_kw list.
    Falls back to a flat schedule if the optimizer is unavailable.
    """
    num_slots = max(1, int(predicted_stay_min / time_step_min))
    arrival_slot = 0
    departure_slot = num_slots

    base_load = [30.0] * num_slots   # assume 30 kW static building load

    payload = {
        "vehicles": [{
            "ev_id": ev_id,
            "energy_needed_kwh": energy_kwh,
            "max_charge_kw": max_kw,
            "arrival_slot": arrival_slot,
            "departure_slot": departure_slot,
        }],
        "num_slots": num_slots,
        "slot_duration_hours": time_step_min / 60.0,
        "transformer_capacity_kw": transformer_kw,
        "base_load_per_slot_kw": base_load,
        "strategy": "optimal",
    }

    try:
        async with httpx.AsyncClient(timeout=OPTIMIZER_TIMEOUT_S) as client:
            r = await client.post(f"{OPTIMIZER_SERVICE_URL}/optimize", json=payload)
            r.raise_for_status()
            data = r.json()
            for sched in data["schedules"]:
                if sched["ev_id"] == ev_id:
                    return sched["power_per_slot_kw"]
        logger.warning("Optimizer returned no schedule for this EV — using flat schedule")
    except Exception as exc:
        logger.warning(f"Optimizer call failed: {exc} — using flat schedule")

    # Fallback: flat schedule up to max_kw
    slots_needed = max(1, int(energy_kwh / (max_kw * time_step_min / 60)))
    return [max_kw] * min(slots_needed, num_slots) + [0.0] * max(0, num_slots - slots_needed)


async def _reoptimize_fleet(
    transformer_kw: float = 150.0,
    time_step_min: int = 15,
) -> None:
    """
    Re-run LP optimization for ALL currently active sessions.
    Called on every StartTransaction and StopTransaction so transformer
    capacity is always shared fairly across the live fleet.
    Each connected charge point receives an updated SetChargingProfile.
    """
    if not _active_sessions:
        return

    now = datetime.now(timezone.utc)
    vehicles = []
    for session_key, s in list(_active_sessions.items()):
        elapsed_min = (now - s.arrival_time).total_seconds() / 60
        delivered_kwh = max(0.0, (s.last_meter_wh - s.meter_start_wh) / 1000.0)
        remaining_kwh = max(0.1, s.energy_needed_kwh - delivered_kwh)
        remaining_stay_min = max(float(time_step_min), s.predicted_stay_min - elapsed_min)
        num_remaining_slots = max(1, int(remaining_stay_min / time_step_min))
        vehicles.append({
            "ev_id": f"{s.cp_id}_{s.connector_id}",
            "energy_needed_kwh": remaining_kwh,
            "max_charge_kw": 7.2,
            "arrival_slot": 0,
            "departure_slot": num_remaining_slots,
            "_session_key": session_key,
        })

    if not vehicles:
        return

    num_slots = max(v["departure_slot"] for v in vehicles)
    base_load = [30.0] * num_slots
    optimizer_vehicles = [
        {k: v for k, v in veh.items() if not k.startswith("_")}
        for veh in vehicles
    ]
    payload = {
        "vehicles": optimizer_vehicles,
        "num_slots": num_slots,
        "slot_duration_hours": time_step_min / 60.0,
        "transformer_capacity_kw": transformer_kw,
        "base_load_per_slot_kw": base_load,
        "strategy": "optimal",
    }

    try:
        async with httpx.AsyncClient(timeout=OPTIMIZER_TIMEOUT_S) as client:
            r = await client.post(f"{OPTIMIZER_SERVICE_URL}/optimize", json=payload)
            r.raise_for_status()
            data = r.json()
        logger.info(f"[Fleet] Re-optimized {len(vehicles)} EV(s) across {num_slots} slots")
    except Exception as exc:
        logger.warning(f"[Fleet] Optimizer failed during fleet re-optimization: {exc}")
        return

    schedule_map = {
        sched["ev_id"]: sched["power_per_slot_kw"]
        for sched in data.get("schedules", [])
    }

    for veh in vehicles:
        ev_id = veh["ev_id"]
        session_key = veh["_session_key"]
        session = _active_sessions.get(session_key)
        if not session:
            continue
        new_schedule = schedule_map.get(ev_id)
        if not new_schedule:
            continue
        session.schedule_kw = new_schedule
        handler = _active_handlers.get(session.cp_id)
        if handler:
            asyncio.ensure_future(
                handler._push_charging_profile(
                    session.connector_id,
                    session.transaction_id,
                    new_schedule,
                    session.predicted_stay_min,
                )
            )
            logger.info(f"[Fleet] Pushed updated profile → {session.cp_id} ({len(new_schedule)} slots)")


def _to_charging_schedule_periods(schedule_kw: list[float], time_step_min: int = 15) -> list[dict]:
    """
    Convert a kW-per-slot list to OCPP ChargingSchedulePeriod objects (in Watts).
    Consecutive slots with the same power are merged into one period.
    """
    if not schedule_kw:
        return [{"startPeriod": 0, "limit": 0.0}]

    periods = []
    prev_w = None
    for i, kw in enumerate(schedule_kw):
        w = round(kw * 1000, 1)
        if w != prev_w:
            periods.append({"startPeriod": i * time_step_min * 60, "limit": w})
            prev_w = w
    return periods


# ── OCPP Charge Point Handler ─────────────────────────────────────────────────

class EVChargePointHandler(Cp16):
    """Central System handler for one connected charge point."""

    @on(Action.BootNotification)
    async def on_boot_notification(self, charge_point_model, charge_point_vendor, **kwargs):
        logger.info(f"[{self.id}] BootNotification: {charge_point_vendor}/{charge_point_model}")
        return call_result.BootNotification(
            current_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            interval=300,
            status=RegistrationStatus.accepted,
        )

    @on(Action.Heartbeat)
    async def on_heartbeat(self, **kwargs):
        return call_result.Heartbeat(
            current_time=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        )

    @on(Action.Authorize)
    async def on_authorize(self, id_tag, **kwargs):
        logger.info(f"[{self.id}] Authorize: {id_tag}")
        return call_result.Authorize(
            id_tag_info={"status": AuthorizationStatus.accepted}
        )

    @on(Action.StartTransaction)
    async def on_start_transaction(self, connector_id, id_tag, meter_start, timestamp, **kwargs):
        arrival_time = datetime.now(timezone.utc)
        tx_id = _next_tx_id()
        session_key = f"{self.id}:{connector_id}"

        logger.info(f"[{self.id}] StartTransaction connector={connector_id} tx={tx_id} tag={id_tag}")

        # Register handler instance so fleet re-optimization can push profiles to this CP
        _active_handlers[self.id] = self

        # ML predictions
        energy_kwh = await _ml_predict_energy(arrival_time)
        stay_min = await _ml_predict_departure(arrival_time, energy_kwh)

        # Flat fallback schedule so StartTransaction response is sent immediately
        _time_step = 15
        _num_slots = max(1, int(stay_min / _time_step))
        _slots_needed = max(1, int(energy_kwh / (7.2 * _time_step / 60)))
        fallback_schedule = (
            [7.2] * min(_slots_needed, _num_slots)
            + [0.0] * max(0, _num_slots - _slots_needed)
        )

        logger.info(
            f"[{self.id}] Session registered: energy={energy_kwh:.2f} kWh, "
            f"stay={stay_min:.0f} min — fleet re-optimization pending"
        )

        _active_sessions[session_key] = ActiveSession(
            transaction_id=tx_id,
            cp_id=self.id,
            connector_id=connector_id,
            id_tag=id_tag,
            arrival_time=arrival_time,
            meter_start_wh=float(meter_start),
            predicted_stay_min=stay_min,
            energy_needed_kwh=energy_kwh,
            schedule_kw=fallback_schedule,
            last_meter_wh=float(meter_start),
        )

        # Re-optimize ALL connected EVs together and push updated profiles to each
        asyncio.ensure_future(_reoptimize_fleet())

        return call_result.StartTransaction(
            transaction_id=tx_id,
            id_tag_info={"status": AuthorizationStatus.accepted},
        )

    async def _push_charging_profile(
        self,
        connector_id: int,
        transaction_id: int,
        schedule_kw: list[float],
        stay_min: float,
        time_step_min: int = 15,
    ):
        """Send SetChargingProfile to the CP after a short delay."""
        await asyncio.sleep(0.3)
        periods = _to_charging_schedule_periods(schedule_kw, time_step_min)
        total_duration_s = int(stay_min * 60)
        profile = call.SetChargingProfile(
            connector_id=connector_id,
            cs_charging_profiles={
                "chargingProfileId": transaction_id,
                "stackLevel": 0,
                "chargingProfilePurpose": ChargingProfilePurposeType.tx_profile,
                "chargingProfileKind": ChargingProfileKindType.absolute,
                "transactionId": transaction_id,
                "chargingSchedule": {
                    "chargingRateUnit": ChargingRateUnitType.watts,
                    "chargingSchedulePeriod": periods,
                    "duration": total_duration_s,
                },
            },
        )
        try:
            resp = await self.call(profile)
            logger.info(f"[{self.id}] SetChargingProfile → {resp.status} ({len(periods)} periods)")
        except Exception as exc:
            logger.error(f"[{self.id}] SetChargingProfile failed: {exc}")

    @on(Action.MeterValues)
    async def on_meter_values(self, connector_id, meter_value, transaction_id=None, **kwargs):
        session_key = f"{self.id}:{connector_id}"
        session = _active_sessions.get(session_key)

        for mv in meter_value:
            sampled = mv.get("sampled_value", [])
            for sv in sampled:
                measurand = sv.get("measurand", "Energy.Active.Import.Register")
                try:
                    value = float(sv["value"])
                except (ValueError, KeyError):
                    continue

                if measurand == "Energy.Active.Import.Register" and session:
                    session.last_meter_wh = value

                elif measurand == "Power.Active.Import" and session:
                    prev_power = session.last_power_w
                    session.last_power_w = value
                    # Detect transition: power was > 0, now dropped to 0 → battery full
                    if prev_power > 0 and value == 0.0 and session.done_charging_time is None:
                        session.done_charging_time = datetime.now(timezone.utc)
                        logger.info(
                            f"[{self.id}] Battery full (power→0)  "
                            f"done_charging={session.done_charging_time.isoformat()}"
                        )

        if session:
            logger.debug(f"[{self.id}] MeterValues: {session.last_meter_wh:.0f} Wh  {session.last_power_w:.0f} W")
        return call_result.MeterValues()

    @on(Action.StopTransaction)
    async def on_stop_transaction(self, meter_stop, timestamp, transaction_id, **kwargs):
        # Find and remove session
        session_key = None
        for k, s in _active_sessions.items():
            if s.transaction_id == transaction_id:
                session_key = k
                break

        if session_key:
            s = _active_sessions.pop(session_key)
            meter_stop_wh = float(meter_stop)
            delivered = (meter_stop_wh - s.meter_start_wh) / 1000.0
            satisfaction = min(delivered / s.energy_needed_kwh * 100, 100.0) if s.energy_needed_kwh > 0 else 100.0

            # disconnect_time = the actual departure (when the driver unplugged)
            # This is what the CSMS was trying to predict — now we know the truth.
            try:
                disconnect_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except Exception:
                disconnect_time = datetime.now(timezone.utc)

            # If power never dropped to 0 in MeterValues (very short session or
            # interval too long to catch the transition), treat disconnect as done_charging
            if s.done_charging_time is None and delivered > 0:
                s.done_charging_time = disconnect_time

            logger.info(
                f"[{self.id}] StopTransaction tx={transaction_id}: "
                f"{delivered:.3f} kWh  satisfaction={satisfaction:.0f}%  "
                f"predicted_stay={s.predicted_stay_min:.0f} min  "
                f"actual_stay={(disconnect_time - s.arrival_time).total_seconds()/60:.0f} min"
            )

            # Write full ACN-format record — this is real data for future retraining
            _write_ocpp_session(s, disconnect_time, meter_stop_wh)

            # Remove handler if this CP has no more active sessions
            if not any(sess.cp_id == self.id for sess in _active_sessions.values()):
                _active_handlers.pop(self.id, None)

            # Release freed capacity back to remaining connected EVs
            asyncio.ensure_future(_reoptimize_fleet())
        else:
            logger.warning(f"[{self.id}] StopTransaction for unknown tx={transaction_id}")

        return call_result.StopTransaction(
            id_tag_info={"status": AuthorizationStatus.accepted}
        )
