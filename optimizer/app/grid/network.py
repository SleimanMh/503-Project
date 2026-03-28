"""
Real distribution grid model for an EV charging hub using OpenDSS (dss-python).

OpenDSS is the industry-standard open-source distribution system simulator
developed by EPRI.  It supports unbalanced multi-phase power flow, accurate
transformer and cable modelling, and is widely used by utilities worldwide.

Network topology (North American commercial, 4.16 kV / 480 V):

    [HV Grid / Slack 4.16 kV]
              |
    [Transformer 4.16 kV -> 480 V  (transformer_kva kVA)]
              |
    [LV Main Busbar  480 V]
       /           \\          \\
[EV Section A]  [EV Section B]  [Building Bus]
 50% EV load     50% EV load    100% base load

Lines: 4/0 AWG aluminium underground cable (typical US parking garage)
  R = 0.328 ohm/km,  X = 0.098 ohm/km,  I_max = 230 A,  L = 30 m

Transformer (ANSI C57.12.20 padmount, 500 kVA rated):
  %R = 1.10,  %X ≈ 5.64  (from vk=5.75%, vkr=1.10%)

This module uses dss-python (cross-platform OpenDSS engine) and can be
combined with external OpenDSS .dss files exported from other tools.
"""

import math
import logging

from dss import DSS as _DSS_SINGLETON

logger = logging.getLogger(__name__)

# Power factors
EV_POWER_FACTOR = 0.97
BUILDING_POWER_FACTOR = 0.85


def _kvar(kw: float, pf: float) -> float:
    """Convert kW + power factor → kVAR."""
    pf = max(0.01, min(1.0, pf))
    return kw * math.tan(math.acos(pf))


def build_ev_charging_network(
    transformer_kva: float,
    feeder_length_m: float = 30.0,
    chargers_a: int = 10,
    chargers_b: int = 10,
) -> dict:
    """
    Build the 5-bus OpenDSS network for the EV charging hub.

    Parameters:
        transformer_kva — rated kVA of the step-down transformer
        feeder_length_m — cable run from LV busbar to each section (m)
        chargers_a      — number of EV chargers on Section A feeder
        chargers_b      — number of EV chargers on Section B feeder

    Returns dict with keys:
        "dss"             — the DSS engine handle
        "bus_names"       — label → OpenDSS bus name mapping
        "transformer_kva" — rated kVA (for loading calculation)
        "chargers_a"      — charger count section A
        "chargers_b"      — charger count section B
    """
    dss = _DSS_SINGLETON
    dss.Start(0)

    dss.Text.Command = "Clear"
    dss.Text.Command = (
        f"New Circuit.EVChargingHub "
        f"basekv=4.16  pu=1.0  phases=3  bus1=HVBus "
        f"Mvasc3=200  Mvasc1=210"
    )

    kva = max(10, transformer_kva)

    # ── Transformer (ANSI C57.12.20 padmount) ────────────────────────────
    pct_r = 1.10
    pct_x = math.sqrt(5.75**2 - 1.10**2)  # ≈ 5.64
    dss.Text.Command = (
        f"New Transformer.MainXfmr phases=3 windings=2 "
        f"buses=[HVBus, LVBusbar] "
        f"conns=[Delta, Wye] "
        f"kvs=[4.16, 0.48] "
        f"kvas=[{kva}, {kva}] "
        f"%Rs=[{pct_r/2}, {pct_r/2}] "
        f"XHL={pct_x} "
        f"%noloadloss=0.34 "
        f"%imag=0.50"
    )

    # ── Linecode for 4/0 AWG Al underground cable ─────────────────────────
    dss.Text.Command = (
        "New Linecode.AWG4_0AL nphases=3 "
        "R1=0.328  X1=0.098  R0=0.656  X0=0.196  "
        "Units=km  Normamps=230  Emergamps=276"
    )

    # ── LV feeder lines ──────────────────────────────────────────────────
    length_km = max(0.001, feeder_length_m / 1000.0)
    for name, bus2 in [("FeederA", "EVSectionA"),
                       ("FeederB", "EVSectionB"),
                       ("FeederBldg", "BuildingBus")]:
        dss.Text.Command = (
            f"New Line.{name} bus1=LVBusbar bus2={bus2} "
            f"Linecode=AWG4_0AL Length={length_km} Units=km"
        )

    # ── Loads (placeholder values — updated via set_loads before solve) ──
    dss.Text.Command = (
        "New Load.EVLoadA bus1=EVSectionA phases=3 "
        "kv=0.48 kw=0.001 kvar=0.0 model=1 conn=Wye"
    )
    dss.Text.Command = (
        "New Load.EVLoadB bus1=EVSectionB phases=3 "
        "kv=0.48 kw=0.001 kvar=0.0 model=1 conn=Wye"
    )
    dss.Text.Command = (
        "New Load.BuildingLoad bus1=BuildingBus phases=3 "
        "kv=0.48 kw=0.001 kvar=0.0 model=1 conn=Wye"
    )

    # Set voltage bases and calculate
    dss.Text.Command = "Set voltagebases=[4.16, 0.48]"
    dss.Text.Command = "Calcvoltagebases"

    bus_names = {
        "HV Bus":         "HVBus",
        "LV Main Busbar": "LVBusbar",
        "EV Section A":   "EVSectionA",
        "EV Section B":   "EVSectionB",
        "Building":       "BuildingBus",
    }

    return {"dss": dss, "bus_names": bus_names, "transformer_kva": kva,
            "chargers_a": max(1, chargers_a), "chargers_b": max(1, chargers_b)}


def set_loads(net: dict, ev_kw: float, building_kw: float) -> None:
    """
    Update the three loads in the OpenDSS network for one time step.
    EV load is split proportionally based on charger count per section.
    """
    dss = net["dss"]
    total_chargers = net["chargers_a"] + net["chargers_b"]
    frac_a = net["chargers_a"] / total_chargers
    frac_b = net["chargers_b"] / total_chargers

    ev_a_kw = ev_kw * frac_a
    ev_b_kw = ev_kw * frac_b

    dss.Text.Command = f"Edit Load.EVLoadA kw={ev_a_kw} kvar={_kvar(ev_a_kw, EV_POWER_FACTOR)}"
    dss.Text.Command = f"Edit Load.EVLoadB kw={ev_b_kw} kvar={_kvar(ev_b_kw, EV_POWER_FACTOR)}"
    dss.Text.Command = f"Edit Load.BuildingLoad kw={building_kw} kvar={_kvar(building_kw, BUILDING_POWER_FACTOR)}"


def solve_power_flow(net: dict) -> dict:
    """
    Run the OpenDSS power flow and extract results.

    Returns dict with:
        success              — whether the solve converged
        bus_voltages_pu      — {label: voltage_pu}
        transformer_loading_pct — %
        line_current_a       — max feeder current in amps
        total_losses_kw      — feeder + transformer losses
    """
    dss = net["dss"]
    bus_map = net["bus_names"]
    ckt = dss.ActiveCircuit

    dss.Text.Command = "Set mode=snapshot"
    dss.Text.Command = "Set controlmode=static"
    ckt.Solution.Solve()

    if not ckt.Solution.Converged:
        return {"success": False}

    # ── Bus voltages (average per-phase magnitude in p.u.) ────────────────
    bus_voltages = {}
    for label, dss_name in bus_map.items():
        ckt.SetActiveBus(dss_name)
        v_pu = ckt.ActiveBus.puVmagAngle  # [mag1, ang1, mag2, ang2, ...]
        if v_pu is not None and len(v_pu) >= 2:
            mags = [v_pu[i] for i in range(0, len(v_pu), 2)]
            bus_voltages[label] = round(sum(mags) / len(mags), 4)
        else:
            bus_voltages[label] = 1.0

    # ── Transformer loading ───────────────────────────────────────────────
    xfmr_kva = net["transformer_kva"]
    ckt.SetActiveElement("Transformer.MainXfmr")
    powers = ckt.ActiveCktElement.Powers  # [P1, Q1, P2, Q2, ...] per terminal/phase
    trafo_loading_pct = 0.0
    if powers is not None and len(powers) >= 6:
        n_phases = 3
        p_total = sum(abs(powers[i * 2]) for i in range(n_phases))
        q_total = sum(abs(powers[i * 2 + 1]) for i in range(n_phases))
        s_total = math.sqrt(p_total**2 + q_total**2)
        trafo_loading_pct = (s_total / xfmr_kva) * 100 if xfmr_kva > 0 else 0

    # ── Line currents ─────────────────────────────────────────────────────
    max_current_a = 0.0
    for line_name in ["FeederA", "FeederB", "FeederBldg"]:
        ckt.SetActiveElement(f"Line.{line_name}")
        currents = ckt.ActiveCktElement.CurrentsMagAng  # [mag1, ang1, ...]
        if currents is not None and len(currents) > 0:
            mags = [currents[i] for i in range(0, len(currents), 2)]
            if mags:
                max_current_a = max(max_current_a, max(mags))

    # ── Total losses ──────────────────────────────────────────────────────
    losses = ckt.Losses  # [P_watts, Q_vars]
    total_losses_kw = abs(losses[0]) / 1000.0 if losses is not None and len(losses) >= 1 else 0.0

    return {
        "success": True,
        "bus_voltages_pu": bus_voltages,
        "transformer_loading_pct": round(trafo_loading_pct, 2),
        "line_current_a": round(max_current_a, 2),
        "total_losses_kw": round(total_losses_kw, 2),
    }
