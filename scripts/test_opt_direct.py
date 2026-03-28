"""Direct optimizer test from inside Docker network."""
import requests
import json

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

r = requests.post("http://optimizer:8002/optimize", json=payload, timeout=10)
d = r.json()
s = d["schedules"][0]
print("delivered:", s["energy_delivered_kwh"])
print("satisfaction:", s["satisfaction_pct"])
sched = s["power_per_slot_kw"]
nonzero = sum(1 for x in sched if x > 0.01)
print("non-zero slots:", nonzero)
print("schedule[:10]:", [round(x, 1) for x in sched[:10]])
