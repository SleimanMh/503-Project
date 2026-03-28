"""Debug script: shows exactly what ML predicts and how it affects results."""
import requests
import json
from datetime import datetime, timedelta, timezone

GATEWAY = "http://localhost:8000"
now = datetime.now(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0)
if now > datetime.now(timezone.utc):
    now -= timedelta(days=1)

# Single vehicle: 20% -> 90% on 60 kWh battery = 42 kWh needed, 7.2 kW max
payload = {
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

r = requests.post(f"{GATEWAY}/api/v1/simulation/run", json=payload, timeout=60)
d = r.json()

ml_met = d.get("ml_metrics", {})
avg_stay = ml_met.get("avg_predicted_stay_min")
default_stay = ml_met.get("default_stay_min")
print(f"ML predicted stay:  {avg_stay:.0f} min ({avg_stay/60:.1f}h)" if avg_stay else "ML: N/A")
print(f"Default (no-ML):    {default_stay:.0f} min ({default_stay/60:.1f}h)")
print(f"Simulation horizon: 480 min (8h) = 32 slots")
print(f"ML dep slot:  {int(avg_stay/15) if avg_stay else '?'} / 32")
print(f"noML dep slot: {int(default_stay/15)} / 32")
print()

for strat_name, key in [("AI+ML", "ai_result"), ("AI(noML)", "no_ml_result"), ("FCFS", "baseline_result")]:
    strat = d[key]
    ev = strat["ev_results"][0]
    sched = ev["power_schedule_kw"]
    nonzero = [(i, round(s, 1)) for i, s in enumerate(sched) if s > 0]
    last = nonzero[-1][0] if nonzero else -1
    print(f"{strat_name}:")
    print(f"  satisfaction: {ev['satisfaction_pct']:.1f}%")
    print(f"  delivered: {ev['energy_delivered_kwh']:.1f} / {ev['energy_needed_kwh']:.1f} kWh")
    print(f"  slots with power: {len(nonzero)}, last at slot {last}")
    print(f"  schedule: {[round(s,1) for s in sched[:35]]}")
    print()

# KEY INSIGHT: if ML predicts ~450 min and default is 480 min, 
# both get 30-32 slots. With only 1 vehicle on a 200 kW transformer,
# both can easily deliver 42 kWh in ~24 slots (42/7.2*4=23.3 slots).
# So there's NO DIFFERENCE — the vehicle finishes before either departure.
#
# The ML advantage only shows when:
# 1. Resources are CONSTRAINED (many EVs competing for limited power)
# 2. ML predicts SHORTER stays → optimizer front-loads charge for those EVs  
# 3. No-ML defers charging → vehicle leaves with charge still pending
