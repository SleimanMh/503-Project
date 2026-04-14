"""
LP-based power allocation scheduler.

Solves the charging optimization problem from SYSTEM_DESIGN.md Section 5:

  maximize  Σ_i Σ_k  w_i · decay_k · x_{i,k} · Δt
  subject to:
      Σ_i x_{i,k}  ≤  P_max - base_load_k - reserved_k   for each slot k
      0 ≤ x_{i,k}  ≤  p_i^max                             for each (i,k)
      Σ_k x_{i,k} · Δt  ≤  e_i                            for each vehicle i
      Σ_k x_{i,k} · Δt  ≥  min_guarantee_i                for each vehicle i
      x_{i,k} = 0  if vehicle i is not present in slot k

Key features:
  - FRONT-LOADING: earlier slots in a vehicle's window get higher weight,
    so the LP delivers energy quickly and is robust to early departure.
  - MINIMUM GUARANTEE: every vehicle is guaranteed at least MIN_SAT_PCT
    of its requested energy (or its physical max if constrained).
  - MAX-MIN FAIRNESS: Phase 2 maximises the worst-off vehicle's
    satisfaction so no customer is neglected.

When ML predictions are provided, reserved_k is computed from predicted
future EV arrivals — the optimizer saves capacity for vehicles that haven't
arrived yet but are expected by the ML model.

Uses scipy.optimize.linprog (interior-point LP solver).
"""

import numpy as np
from scipy.optimize import linprog

from app.schemas import EVVehicle, EVSchedule, FutureArrival

# Every vehicle is guaranteed at least this fraction of its requested energy
# (or its physical maximum if the stay is too short).  This prevents any
# customer from leaving with nearly zero charge.
MIN_SAT_FRACTION = 0.20  # 20% minimum guarantee


def _build_reservation_profile(
    future_arrivals: list[FutureArrival],
    num_slots: int,
    slot_duration_hours: float,
) -> np.ndarray:
    """
    Convert ML-predicted future arrivals into a per-slot capacity reservation (kW).

    For each predicted arrival at slot s with N predicted EVs at avg_charge_kw:
    - Reserve N * avg_charge_kw in slots [s, s + slots_needed)
    - slots_needed estimated from predicted_kwh / (count * charge_rate * slot_hours)
    """
    reserved = np.zeros(num_slots)

    for fa in future_arrivals:
        if fa.predicted_count < 0.1 or fa.slot >= num_slots:
            continue

        power_kw = fa.predicted_count * fa.avg_charge_kw
        # Estimate how many slots these future EVs will need
        if fa.predicted_count > 0 and fa.avg_charge_kw > 0:
            energy_per_ev = fa.predicted_kwh / max(1, fa.predicted_count)
            slots_needed = max(1, int(np.ceil(
                energy_per_ev / (fa.avg_charge_kw * slot_duration_hours)
            )))
        else:
            slots_needed = 4  # default: 1 hour

        start = fa.slot
        end = min(num_slots, start + slots_needed)
        reserved[start:end] += power_kw

    return reserved


def optimize_schedule(
    vehicles: list[EVVehicle],
    num_slots: int,
    slot_duration_hours: float,
    transformer_capacity_kw: float,
    base_load_per_slot_kw: list[float],
    predicted_future_arrivals: list[FutureArrival] | None = None,
) -> list[EVSchedule]:
    """
    Two-phase LP for fair, front-loaded energy allocation:

    Phase 1 — maximize total energy delivered (urgency-weighted) with
              a FRONT-LOADING decay so earlier slots are preferred.
              Also enforces a MINIMUM GUARANTEE for every vehicle.
    Phase 2 — constrain total ≥ 95% of E*, then maximise the worst-off
              vehicle's satisfaction (max-min fairness).

    Front-loading rationale: ML may predict a 10-hour stay, but the vehicle
    might leave after 3 hours.  By delivering energy early, the system is
    robust against departure uncertainty.  The decay factor gives the first
    slot in a vehicle's window weight 1.0, and the last slot weight ~0.3.
    """
    n_vehicles = len(vehicles)
    n_vars = n_vehicles * num_slots  # x_{i,k} for each vehicle × slot

    # Build presence matrix: vehicle i is present in slot k?
    # Use departure_q90_slot as the presence window when provided — this
    # is the 90th-percentile departure from the ML quantile model, giving
    # the LP room to schedule all needed energy even if the car stays
    # longer than the median prediction.  Front-loading decay ensures
    # most energy is delivered in the Q50 (median) region anyway.
    presence = np.zeros((n_vehicles, num_slots), dtype=bool)
    for i, v in enumerate(vehicles):
        start = max(0, v.arrival_slot)
        # Q90 window takes priority; fall back to Q50 departure_slot
        end_slot = v.departure_q90_slot if v.departure_q90_slot is not None else v.departure_slot
        end = min(num_slots, end_slot)
        presence[i, start:end] = True

    # ── Front-loading decay per vehicle ──────────────────────────────────
    # For each vehicle, slots closer to arrival get higher weight.
    # decay(k) = 1.0 - 0.7 * (k - arrival) / (window - 1)
    # First slot = 1.0, last slot (Q90) = 0.3.
    # Since most energy should arrive by Q50 departure, the schedule is
    # naturally robust against early departure.
    DECAY_MIN = 0.3
    decay = np.ones((n_vehicles, num_slots))
    for i, v in enumerate(vehicles):
        start = max(0, v.arrival_slot)
        end_slot = v.departure_q90_slot if v.departure_q90_slot is not None else v.departure_slot
        end = min(num_slots, end_slot)
        window = end - start
        if window > 1:
            for k in range(start, end):
                progress = (k - start) / (window - 1)
                decay[i, k] = 1.0 - (1.0 - DECAY_MIN) * progress

    # ── ML Reservation: capacity reserved for predicted future arrivals ──
    RESERVATION_FRACTION = 0.10
    if predicted_future_arrivals:
        reserved = _build_reservation_profile(
            predicted_future_arrivals, num_slots, slot_duration_hours
        )
        reserved *= RESERVATION_FRACTION
        for k in range(num_slots):
            headroom = transformer_capacity_kw - base_load_per_slot_kw[k]
            reserved[k] = min(reserved[k], 0.5 * max(0, headroom))
    else:
        reserved = np.zeros(num_slots)

    # ── Shared constraint matrices ──
    A_ub_rows = []
    b_ub_vals = []

    # 1) Per-slot capacity: Σ_i x_{i,k} ≤ P_available_k
    for k in range(num_slots):
        row = np.zeros(n_vars)
        for i in range(n_vehicles):
            if presence[i, k]:
                row[i * num_slots + k] = 1.0
        A_ub_rows.append(row)
        available = transformer_capacity_kw - base_load_per_slot_kw[k] - reserved[k]
        b_ub_vals.append(max(0, available))

    # 2) Per-vehicle energy cap: Σ_k x_{i,k} · Δt ≤ e_i
    for i, v in enumerate(vehicles):
        row = np.zeros(n_vars)
        for k in range(num_slots):
            if presence[i, k]:
                row[i * num_slots + k] = slot_duration_hours
        A_ub_rows.append(row)
        b_ub_vals.append(v.energy_needed_kwh)

    # 3) Minimum energy guarantee: Σ_k x_{i,k} · Δt ≥ min_guarantee_i
    #    → rewrite as: −Σ_k x_{i,k} · Δt ≤ −min_guarantee_i
    #    The guarantee is MIN_SAT_FRACTION of needed energy, capped at
    #    what the vehicle can physically receive in its window.
    min_guarantees = []
    for i, v in enumerate(vehicles):
        max_e = float(presence[i].sum()) * slot_duration_hours * v.max_charge_kw
        physical_max = min(max_e, v.energy_needed_kwh)
        guarantee = min(MIN_SAT_FRACTION * v.energy_needed_kwh, physical_max)
        min_guarantees.append(guarantee)

        row = np.zeros(n_vars)
        for k in range(num_slots):
            if presence[i, k]:
                row[i * num_slots + k] = -slot_duration_hours
        A_ub_rows.append(row)
        b_ub_vals.append(-guarantee)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)

    # ── Bounds: 0 ≤ x_{i,k} ≤ p_i^max (or 0 if not present) ──
    bounds = []
    for i, v in enumerate(vehicles):
        for k in range(num_slots):
            if presence[i, k]:
                bounds.append((0, v.max_charge_kw))
            else:
                bounds.append((0, 0))

    # ════════════════════════════════════════════════════════════════
    #  PHASE 1 — maximise total energy (urgency + front-loading)
    #  Urgency capped at 2.0 so time-critical vehicles are prioritised
    #  but not at the total expense of others.
    #  Front-loading decay ensures energy is delivered early in the
    #  vehicle's stay, making the schedule robust to early departure.
    # ════════════════════════════════════════════════════════════════
    w1 = np.zeros(n_vars)
    for i, v in enumerate(vehicles):
        available_slots = max(1, presence[i].sum())
        urgency = v.energy_needed_kwh / (available_slots * slot_duration_hours * v.max_charge_kw + 1e-6)
        urgency = min(urgency, 2.0)
        w = 1.0 + 0.5 * urgency
        for k in range(num_slots):
            w1[i * num_slots + k] = w * decay[i, k] * slot_duration_hours if presence[i, k] else 0

    r1 = linprog(-w1, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs-ipm")
    if not r1.success:
        return _zero_schedules(vehicles, num_slots)

    best_total_energy = -r1.fun  # maximised value from phase 1

    # ════════════════════════════════════════════════════════════════
    #  PHASE 2 — max-min fairness
    #  Introduce a scalar variable t = minimum satisfaction fraction.
    #  Maximize t so the worst-off vehicle is as well-off as possible.
    #  This eliminates 0% allocations and produces realistic intermediate
    #  values (e.g. 47%, 63%, 81%) instead of 100%/0% extremes.
    #
    #  Augmented variable vector: [x_{0,0} ... x_{n-1,T-1}, t]
    #
    #  Constraints added:
    #    sat_i ≥ t  →  −Σ_k x_{i,k}·Δt/e_i + t ≤ 0   (for each vehicle)
    #    total energy ≥ 95% of E*                       (efficiency floor)
    #    t ≤ 1.0                                        (cap at full charge)
    #
    #  Vehicles physically unable to reach 10% satisfaction (e.g. ML
    #  predicted a 15-min stay for a 50 kWh battery) are excluded from
    #  the min-sat constraint so they don't drag t to ~0% for the fleet.
    # ════════════════════════════════════════════════════════════════

    # Per-vehicle physical ceiling within their presence window
    max_achievable = []
    for i, v in enumerate(vehicles):
        max_e = float(presence[i].sum()) * slot_duration_hours * v.max_charge_kw
        max_achievable.append(min(max_e, v.energy_needed_kwh))

    n_p2 = n_vars + 1                    # t is the last variable
    c_p2 = np.zeros(n_p2)
    c_p2[-1] = -1.0                      # maximize t (minimize −t)
    # Add small front-loading bonus to Phase 2 so the fairness solution
    # also prefers early delivery (tie-breaking when many solutions have
    # the same t value)
    for i, v in enumerate(vehicles):
        for k in range(num_slots):
            if presence[i, k]:
                c_p2[i * num_slots + k] = -1e-4 * decay[i, k]

    A_p2_rows, b_p2_vals = [], []

    # (a) Capacity + energy-cap constraints — same as Phase 1, t column = 0
    for row, bval in zip(A_ub_rows, b_ub_vals):
        A_p2_rows.append(np.append(row, 0.0))
        b_p2_vals.append(bval)

    # (b) sat_i ≥ t  for vehicles that can realistically be served
    for i, v in enumerate(vehicles):
        phys_sat = max_achievable[i] / v.energy_needed_kwh if v.energy_needed_kwh > 0 else 1.0
        if phys_sat < 0.10:          # skip physically-constrained vehicles
            continue
        row = np.zeros(n_p2)
        for k in range(num_slots):
            if presence[i, k]:
                row[i * num_slots + k] = -slot_duration_hours / v.energy_needed_kwh
        row[-1] = 1.0                # + t
        A_p2_rows.append(row)
        b_p2_vals.append(0.0)

    # (c) Total energy floor: −Σ x·Δt ≤ −0.95·E*
    tot = np.zeros(n_p2)
    for i in range(n_vehicles):
        for k in range(num_slots):
            if presence[i, k]:
                tot[i * num_slots + k] = slot_duration_hours
    A_p2_rows.append(-tot)
    b_p2_vals.append(-0.95 * best_total_energy)

    # (d) t ≤ 1.0
    row_t = np.zeros(n_p2)
    row_t[-1] = 1.0
    A_p2_rows.append(row_t)
    b_p2_vals.append(1.0)

    bounds_p2 = bounds + [(0.0, 1.0)]    # t ∈ [0, 1]

    r2 = linprog(
        c_p2,
        A_ub=np.array(A_p2_rows), b_ub=np.array(b_p2_vals),
        bounds=bounds_p2, method="highs-ipm",
    )

    # Use phase-2 result if feasible, fall back to phase-1
    x = (r2.x[:n_vars] if r2.success else r1.x).reshape(n_vehicles, num_slots)

    # ── Build response ──
    schedules = []
    for i, v in enumerate(vehicles):
        power_per_slot = x[i].tolist()
        energy = float(np.sum(x[i]) * slot_duration_hours)
        sat = min(100.0, (energy / v.energy_needed_kwh) * 100) if v.energy_needed_kwh > 0 else 100.0
        schedules.append(EVSchedule(
            ev_id=v.ev_id,
            power_per_slot_kw=power_per_slot,
            energy_delivered_kwh=round(energy, 3),
            energy_needed_kwh=v.energy_needed_kwh,
            satisfaction_pct=round(sat, 1),
        ))

    return schedules


def _zero_schedules(vehicles: list[EVVehicle], num_slots: int) -> list[EVSchedule]:
    return [
        EVSchedule(
            ev_id=v.ev_id,
            power_per_slot_kw=[0.0] * num_slots,
            energy_delivered_kwh=0.0,
            energy_needed_kwh=v.energy_needed_kwh,
            satisfaction_pct=0.0,
        )
        for v in vehicles
    ]
