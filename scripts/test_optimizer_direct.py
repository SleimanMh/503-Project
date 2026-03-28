"""Test the optimizer directly to isolate the bug."""
import requests
import json

# Direct optimizer test - single vehicle, arrival=0, departure=32
payload = {
    "vehicles": [
        {
            "ev_id": "T1",
            "energy_needed_kwh": 42,
            "max_charge_kw": 7.2,
            "arrival_slot": 0,
            "departure_slot": 32,
        }
    ],
    "num_slots": 32,
    "slot_duration_hours": 0.25,
    "transformer_capacity_kw": 200,
    "base_load_per_slot_kw": [40] * 32,
    "predicted_future_arrivals": [],
    "strategy": "optimal",
}

print("=== Direct optimizer call (arrival=0, departure=32) ===")
r = requests.post("http://localhost:8002/optimize", json=payload, timeout=10)
d = r.json()
s = d["schedules"][0]
delivered = s["energy_delivered_kwh"]
sat = s["satisfaction_pct"]
sched = s["power_per_slot_kw"]
nonzero = sum(1 for x in sched if x > 0.01)
print(f"  delivered: {delivered:.1f} kWh, satisfaction: {sat:.1f}%")
print(f"  non-zero slots: {nonzero}")
print(f"  schedule[:10]: {[round(x, 1) for x in sched[:10]]}")
print()

# Now test through gateway with arrival_time
print("=== Gateway call WITH arrival_time (09:00 UTC) ===")
from datetime import datetime, timezone
now = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)
gw_payload = {
    "vehicles": [
        {
            "ev_id": "T1",
            "battery_pct": 20,
            "target_pct": 90,
            "battery_capacity_kwh": 60,
            "max_charge_kw": 7.2,
            "arrival_time": now.isoformat(),
        },
    ],
    "transformer_capacity_kw": 200,
    "building_base_load_kw": 40,
    "simulation_duration_hours": 8,
    "time_step_minutes": 15,
    "compare_baseline": True,
}
r2 = requests.post("http://localhost:8000/api/v1/simulation/run", json=gw_payload, timeout=60)
d2 = r2.json()

ai = d2["ai_result"]["ev_results"][0]
print(f"  AI+ML: delivered={ai['energy_delivered_kwh']:.1f}, sat={ai['satisfaction_pct']:.1f}%")
print(f"  schedule[:10]: {[round(x, 1) for x in ai['power_schedule_kw'][:10]]}")
print()

# Gateway WITHOUT arrival_time
print("=== Gateway call WITHOUT arrival_time ===")
gw_payload2 = {
    "vehicles": [
        {
            "ev_id": "T1",
            "battery_pct": 20,
            "target_pct": 90,
            "battery_capacity_kwh": 60,
            "max_charge_kw": 7.2,
        },
    ],
    "transformer_capacity_kw": 200,
    "building_base_load_kw": 40,
    "simulation_duration_hours": 8,
    "time_step_minutes": 15,
    "compare_baseline": True,
}
r3 = requests.post("http://localhost:8000/api/v1/simulation/run", json=gw_payload2, timeout=60)
d3 = r3.json()

ai3 = d3["ai_result"]["ev_results"][0]
print(f"  AI+ML: delivered={ai3['energy_delivered_kwh']:.1f}, sat={ai3['satisfaction_pct']:.1f}%")
print(f"  schedule[:10]: {[round(x, 1) for x in ai3['power_schedule_kw'][:10]]}")
