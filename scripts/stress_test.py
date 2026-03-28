"""Stress test: 20 EVs on a tight 80kW transformer to verify intermediate satisfaction."""
import requests
import json
from datetime import datetime, timedelta, timezone

GATEWAY = "http://localhost:8000"
now = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)
if now > datetime.now(timezone.utc):
    now -= timedelta(days=1)

# 20 EVs on a tight 80kW transformer with 40kW building load = only 40kW for EVs
vehicles = []
for i in range(20):
    vehicles.append({
        "ev_id": f"EV-{i+1:02d}",
        "battery_pct": 10 + (i % 5) * 10,  # 10-50% starting
        "target_pct": 90,
        "battery_capacity_kwh": 60,
        "max_charge_kw": 7.2,
        "arrival_time": now.isoformat(),
    })

# Test 1: tight transformer
print("=== STRESS TEST: 20 EVs, 80kW transformer, 40kW building ===")
for run in range(3):
    r = requests.post(
        f"{GATEWAY}/api/v1/simulation/run",
        json={
            "vehicles": vehicles,
            "transformer_capacity_kw": 80,
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
        sats = [e["satisfaction_pct"] for e in evs]
        min_s = min(sats)
        max_s = max(sats)
        zeros = sum(1 for s in sats if s == 0)
        low = sum(1 for s in sats if 0 < s < 20)
        print(f"  Run {run+1} {strat:10s}: overall={overall:5.1f}%  min={min_s:5.1f}%  max={max_s:5.1f}%  zeros={zeros}  low(<20)={low}")
    print()

# Test 2: show charger asymmetry effect
print("\n=== CHARGER ASYMMETRY TEST: 15A vs 5B ===")
r = requests.post(
    f"{GATEWAY}/api/v1/simulation/run",
    json={
        "vehicles": vehicles[:10],
        "transformer_capacity_kw": 100,
        "building_base_load_kw": 40,
        "simulation_duration_hours": 8,
        "time_step_minutes": 15,
        "compare_baseline": True,
        "chargers_section_a": 15,
        "chargers_section_b": 5,
    },
    timeout=60,
)
d = r.json()
gv = d["ai_result"].get("grid_validation", {})
bv = gv.get("bus_voltages_pu", {})
print(f"  Bus voltages: {json.dumps(bv, indent=4)}")
print(f"  Transformer loading: {gv.get('max_transformer_loading_pct')}%")
print(f"  Losses: {gv.get('network_losses_kw')} kW")
print(f"  Warnings: {gv.get('warnings')}")

# Symmetric for comparison
print("\n=== CHARGER SYMMETRIC TEST: 10A vs 10B ===")
r2 = requests.post(
    f"{GATEWAY}/api/v1/simulation/run",
    json={
        "vehicles": vehicles[:10],
        "transformer_capacity_kw": 100,
        "building_base_load_kw": 40,
        "simulation_duration_hours": 8,
        "time_step_minutes": 15,
        "compare_baseline": True,
        "chargers_section_a": 10,
        "chargers_section_b": 10,
    },
    timeout=60,
)
d2 = r2.json()
gv2 = d2["ai_result"].get("grid_validation", {})
bv2 = gv2.get("bus_voltages_pu", {})
print(f"  Bus voltages: {json.dumps(bv2, indent=4)}")
print(f"  Transformer loading: {gv2.get('max_transformer_loading_pct')}%")
print(f"  Losses: {gv2.get('network_losses_kw')} kW")

# Test 3: feeder length effect
print("\n=== FEEDER LENGTH TEST ===")
for length_m in [10, 30, 100, 200]:
    r3 = requests.post(
        f"{GATEWAY}/api/v1/simulation/run",
        json={
            "vehicles": vehicles[:10],
            "transformer_capacity_kw": 100,
            "building_base_load_kw": 40,
            "simulation_duration_hours": 8,
            "time_step_minutes": 15,
            "compare_baseline": True,
            "feeder_length_m": length_m,
        },
        timeout=60,
    )
    d3 = r3.json()
    gv3 = d3["ai_result"].get("grid_validation", {})
    bv3 = gv3.get("bus_voltages_pu", {})
    min_v = min(bv3.values()) if bv3 else 0
    print(f"  Feeder={length_m:3d}m: min_voltage={min_v:.4f} pu, losses={gv3.get('network_losses_kw')} kW, trafo={gv3.get('max_transformer_loading_pct')}%")
