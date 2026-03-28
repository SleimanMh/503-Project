"""See per-vehicle satisfaction breakdown to understand 100%/0% pattern."""
import requests
import json
from datetime import datetime, timedelta, timezone

GATEWAY = "http://localhost:8000"
now = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)
if now > datetime.now(timezone.utc):
    now -= timedelta(days=1)

# 10 EVs, tight transformer = realistic contention
vehicles = []
for i in range(10):
    vehicles.append({
        "ev_id": f"EV-{i+1:02d}",
        "battery_pct": 15 + i * 5,
        "target_pct": 90,
        "battery_capacity_kwh": 60,
        "max_charge_kw": 7.2,
        "arrival_time": now.isoformat(),
    })

r = requests.post(
    f"{GATEWAY}/api/v1/simulation/run",
    json={
        "vehicles": vehicles,
        "transformer_capacity_kw": 100,
        "building_base_load_kw": 40,
        "simulation_duration_hours": 8,
        "time_step_minutes": 15,
        "compare_baseline": True,
    },
    timeout=60,
)
d = r.json()

for strat, key in [("AI+ML", "ai_result"), ("AI(noML)", "no_ml_result"), ("FCFS", "baseline_result")]:
    evs = d[key]["ev_results"]
    overall = d[key]["overall_satisfaction_pct"]
    print(f"=== {strat} === overall={overall:.1f}%")
    for e in evs:
        bar = "#" * int(e["satisfaction_pct"] / 5)
        print(f"  {e['ev_id']}: {e['satisfaction_pct']:5.1f}%  {e['energy_delivered_kwh']:5.1f}/{e['energy_needed_kwh']:5.1f} kWh  |{bar}")
    print()
