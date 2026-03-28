"""
Grid Validation — OpenDSS AC power flow.

Runs a full multi-phase power flow on the 5-bus radial EV charging
network defined in network.py using the OpenDSS engine (dss-python).

  • Transformer apparent-power loading (kVA, not just kW)
  • Real bus voltages from Newton-Raphson power flow
  • Actual feeder I²R losses
  • Per-phase line currents from the solved network

Falls back to simplified estimates if OpenDSS fails (e.g., divergence
at extreme loading).
"""

import logging
import math

from app.grid.network import build_ev_charging_network, set_loads, solve_power_flow
from app.schemas import GridValidationRequest, GridValidationResponse

logger = logging.getLogger(__name__)

# Voltage thresholds (ANSI C84.1 / IEEE Std 1547)
VOLTAGE_WARNING_PU = 0.95
VOLTAGE_CRITICAL_PU = 0.90
LINE_THERMAL_LIMIT_A = 230  # 4/0 AWG Al underground cable


def validate_grid(request: GridValidationRequest) -> GridValidationResponse:
    """
    Run AC power flow on the EV charging hub network and return grid metrics.

    1. Build circuit and run snapshot power flow at peak-load slot.
    2. Scale transformer loading for all other slots (loading ∝ kW at const. pf).
    3. Fall back to simplified formulas if OpenDSS diverges.
    """
    loads = request.total_load_per_slot_kw
    base = request.base_load_per_slot_kw or []
    slot_minutes = 15

    if not loads:
        return GridValidationResponse(
            feasible=True,
            max_transformer_loading_pct=0.0,
            min_bus_voltage_pu=1.0,
            max_line_current_a=0.0,
            overload_minutes=0.0,
            warnings=[],
            network_losses_kw=0.0,
            bus_voltages_pu={},
        )

    peak_total = max(loads)
    if peak_total <= 0:
        return GridValidationResponse(
            feasible=True,
            max_transformer_loading_pct=0.0,
            min_bus_voltage_pu=1.0,
            max_line_current_a=0.0,
            overload_minutes=0.0,
            warnings=[],
            network_losses_kw=0.0,
            bus_voltages_pu={},
        )

    peak_idx = loads.index(peak_total)
    bld_at_peak = base[peak_idx] if base and peak_idx < len(base) else 0.0
    ev_at_peak = max(0.0, peak_total - bld_at_peak)

    # ── Build network and run OpenDSS power flow at peak ──────────────────
    try:
        net = build_ev_charging_network(
            request.transformer_kva,
            feeder_length_m=getattr(request, "feeder_length_m", 30.0),
            chargers_a=getattr(request, "chargers_section_a", 10),
            chargers_b=getattr(request, "chargers_section_b", 10),
        )
        set_loads(net, ev_kw=ev_at_peak, building_kw=bld_at_peak)
        pf_result = solve_power_flow(net)
        pf_success = pf_result.get("success", False)
    except Exception as exc:
        logger.warning(f"OpenDSS power flow failed: {exc} — using simplified fallback")
        pf_success = False

    if not pf_success:
        return _simplified_fallback(request)

    # ── Extract peak-slot results ─────────────────────────────────────────
    trafo_loading_peak = pf_result["transformer_loading_pct"]
    bus_voltages_pu = pf_result["bus_voltages_pu"]
    min_vm_pu = min(bus_voltages_pu.values()) if bus_voltages_pu else 1.0
    max_line_current_a = pf_result["line_current_a"]
    total_losses = pf_result["total_losses_kw"]

    # ── Per-slot transformer loading (linear scaling from peak) ───────────
    slot_loadings = [
        trafo_loading_peak * (kw / peak_total) if peak_total > 0 else 0
        for kw in loads
    ]
    max_loading = max(slot_loadings)
    overload_slots = sum(1 for l in slot_loadings if l > 100.0)
    overload_minutes = overload_slots * slot_minutes

    # ── Build warnings ────────────────────────────────────────────────────
    warnings = []

    if max_loading > 80:
        warnings.append(
            f"Transformer loading reaches {max_loading:.1f}% (warning threshold: 80%)"
        )
    if max_loading > 100:
        warnings.append(
            f"OVERLOAD: Transformer exceeds 100% in {overload_slots} slots "
            f"({overload_minutes} min)"
        )

    if min_vm_pu < VOLTAGE_CRITICAL_PU:
        warnings.append(
            f"CRITICAL: Voltage drops to {min_vm_pu:.4f} p.u. "
            f"(ANSI Range B limit: {VOLTAGE_CRITICAL_PU} p.u.)"
        )
    elif min_vm_pu < VOLTAGE_WARNING_PU:
        warnings.append(
            f"Voltage drops to {min_vm_pu:.4f} p.u. "
            f"(warning threshold: {VOLTAGE_WARNING_PU} p.u.)"
        )

    if max_line_current_a > LINE_THERMAL_LIMIT_A:
        warnings.append(
            f"Line current {max_line_current_a:.0f} A exceeds thermal limit "
            f"{LINE_THERMAL_LIMIT_A} A"
        )

    if peak_total > 0 and total_losses / peak_total > 0.03:
        warnings.append(
            f"Network losses: {total_losses:.1f} kW "
            f"({total_losses / peak_total * 100:.1f}% of peak load)"
        )

    feasible = (
        max_loading <= 100.0
        and min_vm_pu >= VOLTAGE_CRITICAL_PU
        and max_line_current_a <= LINE_THERMAL_LIMIT_A
    )

    return GridValidationResponse(
        feasible=feasible,
        max_transformer_loading_pct=round(max_loading, 2),
        min_bus_voltage_pu=round(min_vm_pu, 4),
        max_line_current_a=round(max_line_current_a, 2),
        overload_minutes=overload_minutes,
        warnings=warnings,
        network_losses_kw=round(total_losses, 2),
        bus_voltages_pu=bus_voltages_pu,
    )


# ── Simplified fallback (used only if OpenDSS fails) ─────────────────────────

def _simplified_fallback(request: GridValidationRequest) -> GridValidationResponse:
    """Simplified formula-based estimate — only used if AC power flow diverges."""
    warnings = ["[Simplified model] OpenDSS did not converge — using formula estimates"]
    loads = request.total_load_per_slot_kw
    transformer_kw = request.transformer_kva * 0.95

    loading_per_slot = [(l / transformer_kw) * 100 for l in loads]
    max_loading = max(loading_per_slot) if loading_per_slot else 0
    overload_count = sum(1 for l in loading_per_slot if l > 100)

    if max_loading > 80:
        warnings.append(f"Transformer loading reaches {max_loading:.1f}%")
    if max_loading > 100:
        warnings.append(f"OVERLOAD: Transformer exceeds 100% in {overload_count} slots")

    v_nominal = 480
    r_feeder, x_feeder, pf = 0.05 * 0.5, 0.04 * 0.5, 0.95
    max_load_w = max(loads, default=0) * 1000
    max_current = max_load_w / (math.sqrt(3) * v_nominal) if v_nominal > 0 else 0
    v_drop = max_current * (r_feeder * pf + x_feeder * math.sqrt(1 - pf ** 2))
    min_voltage_pu = max(0.0, (v_nominal - v_drop) / v_nominal)

    if min_voltage_pu < 0.95:
        warnings.append(f"Voltage drops to {min_voltage_pu:.4f} p.u.")

    return GridValidationResponse(
        feasible=max_loading <= 100 and min_voltage_pu >= 0.95,
        max_transformer_loading_pct=round(max_loading, 2),
        min_bus_voltage_pu=round(min_voltage_pu, 4),
        max_line_current_a=round(max_current, 2),
        overload_minutes=overload_count * 15,
        warnings=warnings,
        network_losses_kw=0.0,
        bus_voltages_pu={},
    )
