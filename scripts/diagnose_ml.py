"""Diagnose ML satisfaction values - realistic scenario with varied arrivals."""
import requests
import json
from datetime import datetime, timedelta, timezone

GATEWAY = "http://localhost:8000"
now = datetime.now(timezone.utc).replace(hour=8, minute=0, second=0, microsecond=0)
if now > datetime.now(timezone.utc):
    now -= timedelta(days=1)

# Realistic scenario: vehicles arrive at different times throughout the day,
# with different battery states and capacities --- like a real charging hub.
vehicles = [
    {"ev_id": "EV-01", "battery_pct": 15, "target_pct": 90, "battery_capacity_kwh": 60, "max_charge_kw": 7.2,
     "arrival_time": now.isoformat()},                                          # 08:00
    {"ev_id": "EV-02", "battery_pct": 25, "target_pct": 85, "battery_capacity_kwh": 40, "max_charge_kw": 7.2,
     "arrival_time": (now + timedelta(minutes=30)).isoformat()},                # 08:30
    {"ev_id": "EV-03", "battery_pct": 10, "target_pct": 90, "battery_capacity_kwh": 75, "max_charge_kw": 11.0,
     "arrival_time": (now + timedelta(minutes=45)).isoformat()},                # 08:45
    {"ev_id": "EV-04", "battery_pct": 50, "target_pct": 95, "battery_capacity_kwh": 60, "max_charge_kw": 7.2,
     "arrival_time": (now + timedelta(hours=1)).isoformat()},                   # 09:00
    {"ev_id": "EV-05", "battery_pct": 20, "target_pct": 80, "battery_capacity_kwh": 50, "max_charge_kw": 7.2,
     "arrival_time": (now + timedelta(hours=1, minutes=15)).isoformat()},       # 09:15
    {"ev_id": "EV-06", "battery_pct": 35, "target_pct": 90, "battery_capacity_kwh": 60, "max_charge_kw": 7.2,
     "arrival_time": (now + timedelta(hours=2)).isoformat()},                   # 10:00
    {"ev_id": "EV-07", "battery_pct": 5,  "target_pct": 85, "battery_capacity_kwh": 40, "max_charge_kw": 7.2,
     "arrival_time": (now + timedelta(hours=2, minutes=30)).isoformat()},       # 10:30
    {"ev_id": "EV-08", "battery_pct": 40, "target_pct": 90, "battery_capacity_kwh": 75, "max_charge_kw": 11.0,
     "arrival_time": (now + timedelta(hours=3)).isoformat()},                   # 11:00
    {"ev_id": "EV-09", "battery_pct": 30, "target_pct": 85, "battery_capacity_kwh": 60, "max_charge_kw": 7.2,
     "arrival_time": (now + timedelta(hours=3, minutes=45)).isoformat()},       # 11:45
    {"ev_id": "EV-10", "battery_pct": 20, "target_pct": 90, "battery_capacity_kwh": 50, "max_charge_kw": 7.2,
     "arrival_time": (now + timedelta(hours=4)).isoformat()},                   # 12:00
]

r = requests.post(
    f"{GATEWAY}/api/v1/simulation/run",
    json={
        "vehicles": vehicles,
        "transformer_capacity_kw": 80,
        "building_base_load_kw": 30,
        "simulation_duration_hours": 8,
        "time_step_minutes": 15,
        "compare_baseline": True,
    },
    timeout=60,
)
d = r.json()

ml_metrics = d.get("ml_metrics", {})
print("ML Metrics:")
print(json.dumps(ml_metrics, indent=2))
print()

for strat, key in [("AI+ML", "ai_result"), ("AI(noML)", "no_ml_result"), ("FCFS", "baseline_result")]:
    evs = d[key]["ev_results"]
    overall = d[key]["overall_satisfaction_pct"]
    zeros = sum(1 for e in evs if e["satisfaction_pct"] == 0)
    hundreds = sum(1 for e in evs if e["satisfaction_pct"] == 100)
    mid = sum(1 for e in evs if 0 < e["satisfaction_pct"] < 100)
    print(f"=== {strat} === overall={overall:.1f}%  (zeros={zeros}, mid={mid}, full={hundreds})")
    for e in evs:
        sat = e["satisfaction_pct"]
        delivered = e["energy_delivered_kwh"]
        needed = e["energy_needed_kwh"]
        bar = "#" * int(sat / 5) if sat > 0 else "X"
        print(f"  {e['ev_id']}: {sat:5.1f}%  {delivered:6.2f}/{needed:5.1f} kWh  |{bar}")
    print()

# Run multiple times to see variance from noise
print("=== Running 5 iterations to check variance ===")
for run in range(5):
    r2 = requests.post(
        f"{GATEWAY}/api/v1/simulation/run",
        json={
            "vehicles": vehicles,
            "transformer_capacity_kw": 80,
            "building_base_load_kw": 30,
            "simulation_duration_hours": 8,
            "time_step_minutes": 15,
            "compare_baseline": True,
        },
        timeout=60,
    )
    d2 = r2.json()
    ml_sat = d2["ai_result"]["overall_satisfaction_pct"]
    noml_sat = d2["no_ml_result"]["overall_satisfaction_pct"]
    fcfs_sat = d2["baseline_result"]["overall_satisfaction_pct"]
    ml_zeros = sum(1 for e in d2["ai_result"]["ev_results"] if e["satisfaction_pct"] == 0)
    ml_sats = [e["satisfaction_pct"] for e in d2["ai_result"]["ev_results"]]
    print(f"  Run {run+1}: AI+ML={ml_sat:.1f}% (zeros={ml_zeros}, sats={[f'{s:.0f}' for s in ml_sats]}), AI(noML)={noml_sat:.1f}%, FCFS={fcfs_sat:.1f}%")
