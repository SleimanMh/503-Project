"""Simulate the orchestrator's vehicle construction logic locally to find the bug."""
from datetime import datetime, timezone, timedelta
import math

DEFAULT_STAY_MINUTES = 480.0
ML_PREDICTED_STAY = 721.0  # What ML returns for 09:00 UTC arrival

# Simulation parameters
simulation_duration_hours = 8
time_step_minutes = 15
num_slots = int(simulation_duration_hours * 60 / time_step_minutes)  # 32
slot_duration_hours = time_step_minutes / 60  # 0.25

now_utc = datetime.now(timezone.utc)
arrival_time = now_utc.replace(hour=9, minute=0, second=0, microsecond=0)

print(f"now (UTC):       {now_utc}")
print(f"arrival_time:    {arrival_time}")
print(f"num_slots:       {num_slots}")
print()

# ---- Case 1: WITH arrival_time ----
minutes_from_now = max(0, (arrival_time - now_utc).total_seconds() / 60)
arrival_slot = int(minutes_from_now / time_step_minutes)
print(f"WITH arrival_time:")
print(f"  minutes_from_now:  {minutes_from_now:.1f}")
print(f"  arrival_slot:      {arrival_slot}")

# ML predicts 721 min stay
actual_stay = max(15, ML_PREDICTED_STAY)
actual_dep = min(num_slots, arrival_slot + max(1, int(actual_stay / time_step_minutes)))
print(f"  actual_stay:       {actual_stay:.0f} min")
print(f"  actual_dep:        {actual_dep}")

# No-ML
no_ml_stay = DEFAULT_STAY_MINUTES
no_ml_dep = min(num_slots, arrival_slot + max(1, int(no_ml_stay / time_step_minutes)))
print(f"  no_ml_dep:         {no_ml_dep}")

energy_needed = 60 * (90 - 20) / 100  # 42 kWh
ml_vehicle = {
    "ev_id": "T1",
    "energy_needed_kwh": round(max(0.1, energy_needed), 3),
    "max_charge_kw": 7.2,
    "arrival_slot": arrival_slot,
    "departure_slot": actual_dep,
}
no_ml_vehicle = {
    "ev_id": "T1",
    "energy_needed_kwh": round(max(0.1, energy_needed), 3),
    "max_charge_kw": 7.2,
    "arrival_slot": arrival_slot,
    "departure_slot": no_ml_dep,
}
print(f"  ml_vehicle:    {ml_vehicle}")
print(f"  no_ml_vehicle: {no_ml_vehicle}")
print()

# ---- Case 2: WITHOUT arrival_time ----
arrival_slot2 = 0  # default
# Without arrival_time, no ML prediction → DEFAULT_STAY
actual_stay2 = DEFAULT_STAY_MINUTES
actual_dep2 = min(num_slots, arrival_slot2 + max(1, int(actual_stay2 / time_step_minutes)))
print(f"WITHOUT arrival_time:")
print(f"  arrival_slot:  {arrival_slot2}")
print(f"  actual_dep:    {actual_dep2}")
print(f"  no_ml_dep:     {actual_dep2}")

# Base load
building_base_load_kw = 40
loads = []
for k in range(num_slots):
    hour = (k * time_step_minutes / 60) % 24
    if 7 <= hour <= 9:
        ramp = (hour - 7) / 2
        factor = 0.4 + 0.6 * ramp
    elif 9 < hour <= 17:
        factor = 1.0
    elif 17 < hour <= 20:
        factor = 1.0 - 0.5 * (hour - 17) / 3
    else:
        factor = 0.4
    variation = 0.05 * math.sin(2 * math.pi * hour / 24)
    load = building_base_load_kw * (factor + variation)
    loads.append(round(max(0, load), 2))

print(f"\nBase load profile[:10]:  {loads[:10]}")
print(f"Base load profile[-5:]:  {loads[-5:]}")
print(f"Max base load: {max(loads):.1f} kW")
print(f"Transformer capacity: 200 kW")
print(f"Available for EV (min): {200 - max(loads):.1f} kW")

# Check: the base load starts at hour 0 of simulation, not hour 0 of day
# This is critical — simulation hour 0 ≠ midnight
print(f"\nBASE LOAD STARTS AT HOUR 0 OF SIMULATION")
print(f"But the simulation could start at any UTC hour!")
print(f"Current UTC hour: {now_utc.hour}")
print(f"The base load sinusoid peaks at hours 9-17")
print(f"If simulation starts at hour {now_utc.hour}, first slot = hour {now_utc.hour}")
print()

# The base load function always starts at hour 0, regardless of wall clock!
# hour = (k * step_minutes / 60) % 24
# For k=0: hour = 0 which is MIDNIGHT → factor = 0.4
# This means base load is LOW at the start, not matching real time
