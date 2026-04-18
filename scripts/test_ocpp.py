"""
OCPP 1.6J Charge Point simulator — realistic multi-vehicle fleet mode.

Key design principle
--------------------
Departure time = when the DRIVER returns to the car (their schedule),
NOT when the battery is fully charged.  This matches real ACN data where
idle_time_hr averages 3-4 h.  The car may stop drawing power once full,
but it stays connected and the session ends only when the driver unplugs.

Session lifecycle per vehicle
-------------------------------
1. Arrival: BootNotification → Authorize → StartTransaction
   - Each vehicle arrives at a different time (staggered by arrival_offset_s)
   - meter_start = random odometer reading (not meaningful, just realistic)
2. Charging phase: MeterValues every meter_interval_s
   - Power = max_charge_kw until battery is full (initial_soc → 100%)
   - After full: power drops to 0 W (idle) but session continues
3. Departure: StopTransaction sent at stay_seconds from arrival
   - Completely independent of whether the car is fully charged

Single-vehicle mode (default):
  python scripts/test_ocpp.py --session-minutes 5 --cp-id CP_001

Fleet mode (multiple concurrent vehicles with staggered arrivals):
  python scripts/test_ocpp.py --fleet 4 --session-minutes 30 --spread-minutes 10

Realistic mode (random parameters from ACN-style distributions):
  python scripts/test_ocpp.py --fleet 5 --realistic
"""

import argparse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Vehicle profile ───────────────────────────────────────────────────────────

@dataclass
class VehicleProfile:
    """Everything known about an EV session before it starts."""
    cp_id: str
    id_tag: str
    battery_capacity_kwh: float   # total usable battery (kWh)
    initial_soc_pct: float        # state-of-charge at arrival (0-100)
    max_charge_kw: float          # on-board charger max AC power
    stay_minutes: float           # how long the driver stays (departure time)
    arrival_offset_s: float       # seconds to wait before connecting

    @property
    def energy_needed_kwh(self) -> float:
        return self.battery_capacity_kwh * (1.0 - self.initial_soc_pct / 100.0)

    @property
    def charge_time_min(self) -> float:
        """Theoretical full-charge time at max power."""
        return (self.energy_needed_kwh / self.max_charge_kw) * 60.0

    @property
    def idle_time_min(self) -> float:
        """Time spent plugged in but fully charged (mimics real ACN idle_time_hr)."""
        return max(0.0, self.stay_minutes - self.charge_time_min)

    @property
    def meter_start_wh(self) -> int:
        """Random-looking odometer reading at session start."""
        seed = abs(hash(self.cp_id)) % 50_000
        return seed + int(self.battery_capacity_kwh * self.initial_soc_pct * 10)


def _sample_realistic_profile(cp_id: str, seed: Optional[int] = None) -> VehicleProfile:
    """
    Sample vehicle parameters from distributions fitted to ACN data.

    ACN statistics (23,444 sessions, Caltech 2018-2019):
      - mean stay: 5.7 h  (log-normal, heavy right tail)
      - mean kwh:  9.0 kWh
      - initial SOC is not recorded but can be inferred
      - 78% anonymous users
    """
    rng = random.Random(seed)

    # Battery capacity: typical BEVs 2018-2026 (kWh)
    battery_kwh = rng.choice([24, 30, 40, 50, 60, 75, 82, 100])

    # SOC at arrival: most workplace chargers see 40-85% (driver didn't fully deplete)
    initial_soc = rng.uniform(30, 85)

    # Max AC charge rate (kW): Level 2, 3.3-11.5 kW range
    max_kw = rng.choice([3.3, 6.6, 7.2, 9.6, 11.5])

    # Stay duration — log-normal to get heavy right tail matching ACN
    # ACN mean=342 min (~5.7h), std~210 min → log-normal params
    mu = math.log(300)   # ~5 h median
    sigma = 0.7
    stay_min = max(30.0, rng.lognormvariate(mu, sigma))

    # Arrival offset: stagger vehicles by 0-30 minutes (Poisson-style arrivals)
    offset_s = rng.uniform(0, 1800)

    # RFID: anonymous 78% of time (None → use cp_id as tag)
    id_tag = f"RFID_{cp_id}" if rng.random() > 0.78 else f"ANON_{cp_id[-3:]}"

    return VehicleProfile(
        cp_id=cp_id,
        id_tag=id_tag,
        battery_capacity_kwh=battery_kwh,
        initial_soc_pct=initial_soc,
        max_charge_kw=max_kw,
        stay_minutes=stay_min,
        arrival_offset_s=offset_s,
    )


# ── OCPP Charge Point ─────────────────────────────────────────────────────────

class SimulatedChargePoint(Cp16):
    """CP that handles inbound SetChargingProfile from the Central System."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_received = asyncio.Event()
        self.current_limit_w: float = 7200.0   # updated by charging profile

    @on(Action.SetChargingProfile)
    async def on_set_charging_profile(self, connector_id, cs_charging_profiles, **kwargs):
        schedule = cs_charging_profiles.get("chargingSchedule", {})
        periods = schedule.get("chargingSchedulePeriod", [])
        duration = schedule.get("duration", "?")
        logger.info(
            f"[{self.id}] ← SetChargingProfile: {len(periods)} period(s), duration={duration} s"
        )
        for p in periods:
            logger.info(f"[{self.id}]   slot startPeriod={p['startPeriod']}s  limit={p['limit']} W")
        # Store the first period limit as current allowed power
        if periods:
            self.current_limit_w = float(periods[0]["limit"])
        self.profile_received.set()
        return call_result.SetChargingProfile(status="Accepted")


# ── Session runner ────────────────────────────────────────────────────────────

async def run_session(
    url: str,
    profile: VehicleProfile,
    meter_interval_s: int = 30,
    speed_factor: float = 1.0,
):
    """
    Run one complete OCPP session for a given vehicle profile.

    speed_factor > 1 compresses time (e.g. speed_factor=60 makes 1 real second
    represent 1 simulated minute) so you can test long sessions quickly.
    """
    cp_id = profile.cp_id
    id_tag = profile.id_tag

    # Wait for staggered arrival
    wait_s = profile.arrival_offset_s / speed_factor
    if wait_s > 0.5:
        logger.info(f"[{cp_id}] Arriving in {profile.arrival_offset_s:.0f} sim-s ...")
        await asyncio.sleep(wait_s)

    logger.info(
        f"[{cp_id}] Connecting  battery={profile.battery_capacity_kwh:.0f} kWh  "
        f"soc={profile.initial_soc_pct:.0f}%  need={profile.energy_needed_kwh:.1f} kWh  "
        f"max={profile.max_charge_kw:.1f} kW  stay={profile.stay_minutes:.0f} min  "
        f"(charge~{profile.charge_time_min:.0f} min + idle~{profile.idle_time_min:.0f} min)"
    )

    async with websockets.connect(
        f"{url}/ocpp/{cp_id}",
        subprotocols=["ocpp1.6"],
        ping_interval=30,
    ) as ws:
        cp = SimulatedChargePoint(cp_id, ws)
        asyncio.ensure_future(cp.start())

        # ── BootNotification ──────────────────────────────────────────────────
        await asyncio.sleep(0.5)
        boot_resp = await cp.call(call.BootNotification(
            charge_point_model="EV-Simulator",
            charge_point_vendor="TestVendor",
        ))
        logger.info(f"[{cp_id}] BootNotification → {boot_resp.status}")
        assert boot_resp.status == RegistrationStatus.accepted

        # ── Authorize ─────────────────────────────────────────────────────────
        await asyncio.sleep(0.3)
        auth_resp = await cp.call(call.Authorize(id_tag=id_tag))
        logger.info(f"[{cp_id}] Authorize({id_tag}) → {auth_resp.id_tag_info['status']}")

        # ── StartTransaction ──────────────────────────────────────────────────
        await asyncio.sleep(0.3)
        meter_start = profile.meter_start_wh
        tx_resp = await cp.call(call.StartTransaction(
            connector_id=1,
            id_tag=id_tag,
            meter_start=meter_start,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))
        tx_id = tx_resp.transaction_id
        logger.info(f"[{cp_id}] StartTransaction → tx={tx_id}  meterStart={meter_start} Wh")

        # ── Wait for SetChargingProfile ───────────────────────────────────────
        try:
            await asyncio.wait_for(cp.profile_received.wait(), timeout=10.0)
            logger.info(f"[{cp_id}] Charging schedule accepted")
        except asyncio.TimeoutError:
            logger.warning(f"[{cp_id}] No SetChargingProfile within 10 s — using default 7.2 kW")

        # ── MeterValues loop ──────────────────────────────────────────────────
        # Key insight: charging stops when battery is full, but the session
        # continues until the driver returns (stay_minutes from arrival).
        #
        # Phase 1 - Charging:  power = min(max_kw, profile_limit_kw)
        # Phase 2 - Idle:      power = 0 W  (fully charged, driver not back yet)
        # Phase 3 - Departure: StopTransaction at stay_minutes

        stay_s = profile.stay_minutes * 60
        charge_done_s = profile.charge_time_min * 60   # when battery hits 100%
        energy_dispensed_wh = 0.0
        energy_cap_wh = profile.energy_needed_kwh * 1000.0

        session_start = time.monotonic()
        interval_s = meter_interval_s / speed_factor

        logger.info(
            f"[{cp_id}] Session running  stay={profile.stay_minutes:.0f} min  "
            f"charge_done_at={profile.charge_time_min:.0f} min  speed_factor={speed_factor}x"
        )

        elapsed_s = 0.0
        while elapsed_s < stay_s / speed_factor:
            await asyncio.sleep(interval_s)
            elapsed_s = time.monotonic() - session_start
            sim_elapsed_s = elapsed_s * speed_factor   # simulated seconds

            # Determine current power (W): charging or idle
            if sim_elapsed_s < charge_done_s and energy_dispensed_wh < energy_cap_wh:
                actual_power_w = min(profile.max_charge_kw * 1000, cp.current_limit_w)
                delta_wh = actual_power_w * (meter_interval_s / 3600.0)
                energy_dispensed_wh = min(energy_dispensed_wh + delta_wh, energy_cap_wh)
                phase = "charging"
            else:
                actual_power_w = 0.0
                phase = "idle"

            meter_wh = meter_start + energy_dispensed_wh
            soc_pct = min(
                profile.initial_soc_pct + (energy_dispensed_wh / (profile.battery_capacity_kwh * 10)),
                100.0,
            )
            stay_pct = min(sim_elapsed_s / stay_s * 100, 100)

            await cp.call(call.MeterValues(
                connector_id=1,
                meter_value=[{
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "sampled_value": [
                        {
                            "value": f"{meter_wh:.1f}",
                            "measurand": "Energy.Active.Import.Register",
                            "unit": "Wh",
                            "context": "Sample.Periodic",
                        },
                        {
                            "value": f"{actual_power_w:.0f}",
                            "measurand": "Power.Active.Import",
                            "unit": "W",
                            "context": "Sample.Periodic",
                        },
                    ],
                }],
                transaction_id=tx_id,
            ))
            logger.info(
                f"[{cp_id}] MeterValues  {meter_wh:.0f} Wh  "
                f"power={actual_power_w/1000:.1f} kW  soc={soc_pct:.0f}%  "
                f"stay={stay_pct:.0f}%  [{phase}]"
            )

        # ── StopTransaction — driver unplugs at scheduled departure time ──────
        final_meter_wh = int(meter_start + energy_dispensed_wh)
        delivered_kwh = energy_dispensed_wh / 1000.0
        logger.info(
            f"[{cp_id}] → StopTransaction  meterStop={final_meter_wh} Wh  "
            f"delivered={delivered_kwh:.2f} kWh  idle={profile.idle_time_min:.0f} min"
        )
        stop_resp = await cp.call(call.StopTransaction(
            meter_stop=final_meter_wh,
            timestamp=datetime.now(timezone.utc).isoformat(),
            transaction_id=tx_id,
        ))
        logger.info(f"[{cp_id}] Session complete  stay={profile.stay_minutes:.0f} min  "
                    f"charged={delivered_kwh:.2f} kWh  status={stop_resp.id_tag_info['status']}")


# ── Fleet runner ──────────────────────────────────────────────────────────────

async def run_fleet(url: str, profiles: list, meter_interval_s: int, speed_factor: float):
    """Run all vehicle sessions concurrently."""
    tasks = [
        run_session(url, p, meter_interval_s, speed_factor)
        for p in profiles
    ]
    await asyncio.gather(*tasks, return_exceptions=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OCPP 1.6J CP Fleet Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single vehicle, 5-minute session (for quick testing)
  python scripts/test_ocpp.py --session-minutes 5

  # 4 vehicles with staggered arrivals, each 30-min stay
  python scripts/test_ocpp.py --fleet 4 --session-minutes 30 --spread-minutes 5

  # 5 vehicles with fully realistic random parameters (ACN-style)
  python scripts/test_ocpp.py --fleet 5 --realistic

  # Fast simulation: speed-factor 30 compresses 30 min into 1 real minute
  python scripts/test_ocpp.py --fleet 3 --realistic --speed-factor 30
        """,
    )
    parser.add_argument("--url", default="ws://localhost:8000")
    parser.add_argument("--fleet", type=int, default=1, help="Number of concurrent charge points")
    parser.add_argument("--cp-prefix", default="CP", help="Charge point ID prefix (e.g. CP_001)")
    parser.add_argument("--session-minutes", type=float, default=5,
                        help="Stay duration in minutes (ignored with --realistic)")
    parser.add_argument("--spread-minutes", type=float, default=2,
                        help="Max arrival spread across fleet (minutes)")
    parser.add_argument("--meter-interval", type=int, default=30,
                        help="MeterValues interval in seconds (real time)")
    parser.add_argument("--speed-factor", type=float, default=1.0,
                        help="Time compression: 60=1 real sec = 1 sim minute")
    parser.add_argument("--realistic", action="store_true",
                        help="Sample random realistic parameters for each vehicle")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    profiles = []
    for i in range(1, args.fleet + 1):
        cp_id = f"{args.cp_prefix}_{i:03d}"
        if args.realistic:
            seed = rng.randint(0, 99999) if args.seed is None else args.seed + i
            p = _sample_realistic_profile(cp_id, seed=seed)
        else:
            # Fixed parameters, staggered arrivals only
            offset_s = (i - 1) * (args.spread_minutes * 60 / max(args.fleet - 1, 1)) if args.fleet > 1 else 0
            p = VehicleProfile(
                cp_id=cp_id,
                id_tag=f"RFID_{cp_id}",
                battery_capacity_kwh=60.0,
                initial_soc_pct=40.0,
                max_charge_kw=7.2,
                stay_minutes=args.session_minutes,
                arrival_offset_s=offset_s,
            )
        profiles.append(p)

    logger.info(f"Starting fleet simulation: {len(profiles)} vehicle(s)  speed_factor={args.speed_factor}x")
    for p in profiles:
        logger.info(
            f"  {p.cp_id}  battery={p.battery_capacity_kwh:.0f} kWh  "
            f"soc={p.initial_soc_pct:.0f}%  need={p.energy_needed_kwh:.1f} kWh  "
            f"stay={p.stay_minutes:.0f} min  offset={p.arrival_offset_s:.0f} s"
        )

    asyncio.run(run_fleet(
        url=args.url,
        profiles=profiles,
        meter_interval_s=args.meter_interval,
        speed_factor=args.speed_factor,
    ))


if __name__ == "__main__":
    main()
