"""Test ML predictions at different arrival hours to find where ML value is highest."""
import requests
from datetime import datetime, timedelta, timezone

ML_URL = "http://localhost:8000"  # Through gateway proxy

# We'll use the gateway's ML endpoints
# Actually, ML is behind the Docker network. Let's call the full simulation
# with different arrival times and see ML metrics.

now = datetime.now(timezone.utc)

for hour in [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
    arrival = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    # Ensure it's in the past
    if arrival > now:
        arrival -= timedelta(days=1)

    payload = {
        "vehicles": [
            {
                "ev_id": "T1",
                "battery_pct": 20,
                "target_pct": 90,
                "battery_capacity_kwh": 60,
                "max_charge_kw": 7.2,
                "arrival_time": arrival.isoformat(),
            },
        ],
        "transformer_capacity_kw": 200,
        "building_base_load_kw": 40,
        "simulation_duration_hours": 8,
        "time_step_minutes": 15,
        "compare_baseline": False,
    }

    r = requests.post(f"{ML_URL}/api/v1/simulation/run", json=payload, timeout=30)
    d = r.json()
    ml = d.get("ml_metrics", {})
    stay = ml.get("avg_predicted_stay_min")
    if stay:
        dep_slot = min(32, int(stay / 15))
        noml_slot = min(32, int(480 / 15))
        print(f"  Hour {hour:02d}:00 → ML stay={stay:6.0f} min ({stay/60:.1f}h)  dep_slot={dep_slot}/32  noML_slot={noml_slot}/32  diff={dep_slot-noml_slot:+d} slots")
    else:
        print(f"  Hour {hour:02d}:00 → ML unavailable")
