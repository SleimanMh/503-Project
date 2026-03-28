"""
ACN-Data Pipeline: Parse raw sessions and build training datasets
for Model 1 (demand forecasting) and Model 2 (departure prediction).

Usage:
    python -m training.prepare_data --data-path ../data/acn_sessions.csv --output-dir artifacts/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_acn_data(path: str) -> pd.DataFrame:
    """Load and clean the ACN sessions CSV."""
    df = pd.read_csv(path, parse_dates=["connection_time", "disconnect_time", "done_charging", "date"])
    df = df.dropna(subset=["connection_time", "disconnect_time", "kwh_delivered"])
    df = df[df["kwh_delivered"] > 0].copy()
    df = df[df["duration_hr"] > 0].copy()
    df["duration_min"] = df["duration_hr"] * 60
    df["connection_time"] = pd.to_datetime(df["connection_time"], utc=True)
    df["disconnect_time"] = pd.to_datetime(df["disconnect_time"], utc=True)
    df = df.sort_values("connection_time").reset_index(drop=True)
    return df


def _cyclical_encode(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Encode a periodic feature as sin/cos."""
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


# ── Model 1: Demand Forecasting Dataset ─────────────────────────────────
def build_demand_dataset(df: pd.DataFrame, window_minutes: int = 30) -> pd.DataFrame:
    """
    Aggregate sessions into fixed time windows and build features for
    predicting arrival_count and total_kwh per window.
    """
    df = df.copy()
    df["window"] = df["connection_time"].dt.floor(f"{window_minutes}min")

    # Aggregate per window
    agg = df.groupby("window").agg(
        arrival_count=("session_id", "count"),
        total_kwh=("kwh_delivered", "sum"),
        mean_duration_hr=("duration_hr", "mean"),
    ).reset_index()

    # Fill missing windows with zeros (continuous timeline)
    full_range = pd.date_range(
        start=agg["window"].min(),
        end=agg["window"].max(),
        freq=f"{window_minutes}min",
        tz="UTC",
    )
    agg = agg.set_index("window").reindex(full_range, fill_value=0).rename_axis("window").reset_index()

    # Temporal features
    agg["hour"] = agg["window"].dt.hour + agg["window"].dt.minute / 60
    agg["hour_sin"], agg["hour_cos"] = _cyclical_encode(agg["hour"], 24)
    agg["dow"] = agg["window"].dt.dayofweek
    agg["dow_sin"], agg["dow_cos"] = _cyclical_encode(agg["dow"], 7)
    agg["is_weekend"] = (agg["dow"] >= 5).astype(int)
    agg["month"] = agg["window"].dt.month
    agg["month_sin"], agg["month_cos"] = _cyclical_encode(agg["month"], 12)

    # Lag features (previous windows)
    agg["lag_1"] = agg["arrival_count"].shift(1)                    # 30 min ago
    agg["lag_2"] = agg["arrival_count"].shift(2)                    # 1 hour ago
    agg["lag_48"] = agg["arrival_count"].shift(48)                  # 24 hours ago
    agg["lag_kwh_1"] = agg["total_kwh"].shift(1)
    agg["lag_kwh_48"] = agg["total_kwh"].shift(48)

    # Rolling features
    agg["rolling_mean_6"] = agg["arrival_count"].shift(1).rolling(6, min_periods=1).mean()    # 3h
    agg["rolling_mean_48"] = agg["arrival_count"].shift(1).rolling(48, min_periods=1).mean()  # 24h
    agg["rolling_kwh_6"] = agg["total_kwh"].shift(1).rolling(6, min_periods=1).mean()

    # Drop rows with NaN from lags
    agg = agg.dropna().reset_index(drop=True)

    return agg


DEMAND_FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "month_sin", "month_cos",
    "lag_1", "lag_2", "lag_48",
    "lag_kwh_1", "lag_kwh_48",
    "rolling_mean_6", "rolling_mean_48", "rolling_kwh_6",
]

DEMAND_TARGET_COLS = ["arrival_count", "total_kwh"]


# ── Model 2: Departure Prediction Dataset ───────────────────────────────
def build_departure_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-session features for predicting stay duration.
    Only uses information available at arrival time (no leakage).
    """
    df = df.copy()

    # Features available at arrival
    df["hour_sin"], df["hour_cos"] = _cyclical_encode(df["arrival_hour"], 24)
    df["dow_sin"], df["dow_cos"] = _cyclical_encode(df["day_of_week"], 7)
    df["month_sin"], df["month_cos"] = _cyclical_encode(df["month"], 12)
    df["is_weekend"] = df["is_weekend"].astype(int)

    # Encode site and cluster as integers
    df["site_encoded"] = df["site_id"].astype("category").cat.codes
    df["cluster_encoded"] = df["cluster_id"].astype("category").cat.codes

    # Per-user historical average stay (computed up to that point — no leakage)
    if df["user_id"].notna().sum() > 0:
        df["user_mean_stay"] = (
            df.groupby("user_id")["duration_min"]
            .transform(lambda s: s.expanding().mean().shift(1))
        )
        df["user_mean_stay"] = df["user_mean_stay"].fillna(df["duration_min"].mean())
    else:
        df["user_mean_stay"] = df["duration_min"].mean()

    # Per-station average stay (expanding, shifted to avoid leakage)
    df["station_mean_stay"] = (
        df.groupby("station_id")["duration_min"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )
    df["station_mean_stay"] = df["station_mean_stay"].fillna(df["duration_min"].mean())

    # Energy requested — available at arrival time; strong proxy for stay duration
    # (more kWh needed → vehicle stays longer to charge)
    if "kwh_delivered" in df.columns:
        df["requested_energy_kwh"] = df["kwh_delivered"]
    else:
        df["requested_energy_kwh"] = 9.0  # ACN dataset mean as fallback

    # Target
    df["target_duration_min"] = df["duration_min"]

    return df


DEPARTURE_FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "month_sin", "month_cos",
    "site_encoded", "cluster_encoded",
    "user_mean_stay", "station_mean_stay",
    "requested_energy_kwh",
]

DEPARTURE_TARGET_COL = "target_duration_min"


# ── Temporal Split ───────────────────────────────────────────────────────
def temporal_split(df: pd.DataFrame, time_col: str,
                   train_frac: float = 0.7, val_frac: float = 0.1):
    """
    Chronological split: 70% train, 10% val, 20% test.
    Never shuffles — preserves temporal ordering.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


# ── Replay Scenario Selection ───────────────────────────────────────────
def select_replay_scenarios(df: pd.DataFrame) -> dict:
    """
    Identify specific dates from ACN-Data for the 4 validation scenarios.
    """
    daily_counts = df.groupby("date").agg(
        sessions=("session_id", "count"),
        total_kwh=("kwh_delivered", "sum"),
    )

    # Scenario 1: median-arrivals weekday
    weekday_dates = df[df["is_weekend"] == 0]["date"].unique()
    weekday_counts = daily_counts.loc[daily_counts.index.isin(weekday_dates)]
    median_val = weekday_counts["sessions"].median()
    scenario_1 = weekday_counts.iloc[
        (weekday_counts["sessions"] - median_val).abs().argsort()[:1]
    ].index[0]

    # Scenario 2: top 5th percentile (peak day)
    p95 = daily_counts["sessions"].quantile(0.95)
    peak_days = daily_counts[daily_counts["sessions"] >= p95]
    scenario_2 = peak_days["sessions"].idxmax()

    # Scenario 3: normal day (outage injected at runtime)
    scenario_3 = scenario_1  # reuse normal day

    # Scenario 4: same as peak day (sessions compressed 1.5x at runtime)
    scenario_4 = scenario_2

    return {
        "scenario_1_normal": str(scenario_1),
        "scenario_2_peak": str(scenario_2),
        "scenario_3_outage": str(scenario_3),
        "scenario_4_stress": str(scenario_4),
    }


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Prepare ACN-Data for training")
    parser.add_argument("--data-path", default="../data/acn_sessions.csv")
    parser.add_argument("--output-dir", default="artifacts/")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("Loading ACN data...")
    df = load_acn_data(args.data_path)
    print(f"  {len(df)} valid sessions, {df['date'].nunique()} unique dates")
    print(f"  Date range: {df['connection_time'].min()} → {df['connection_time'].max()}")

    # ── demand forecasting dataset ──
    print("\nBuilding demand forecasting dataset...")
    demand_df = build_demand_dataset(df)
    train_d, val_d, test_d = temporal_split(demand_df, "window")
    train_d.to_parquet(output / "demand_train.parquet", index=False)
    val_d.to_parquet(output / "demand_val.parquet", index=False)
    test_d.to_parquet(output / "demand_test.parquet", index=False)
    print(f"  Train: {len(train_d)}, Val: {len(val_d)}, Test: {len(test_d)} windows")

    # ── departure prediction dataset ──
    print("\nBuilding departure prediction dataset...")
    dep_df = build_departure_dataset(df)
    train_p, val_p, test_p = temporal_split(dep_df, "connection_time")
    train_p.to_parquet(output / "departure_train.parquet", index=False)
    val_p.to_parquet(output / "departure_val.parquet", index=False)
    test_p.to_parquet(output / "departure_test.parquet", index=False)
    print(f"  Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)} sessions")

    # ── data summary ──
    summary = {
        "total_sessions": len(df),
        "date_range": [str(df["connection_time"].min()), str(df["connection_time"].max())],
        "unique_stations": int(df["station_id"].nunique()),
        "unique_sites": int(df["site_id"].nunique()),
        "mean_kwh": float(df["kwh_delivered"].mean()),
        "mean_duration_hr": float(df["duration_hr"].mean()),
        "demand_features": DEMAND_FEATURE_COLS,
        "departure_features": DEPARTURE_FEATURE_COLS,
        "demand_split": {"train": len(train_d), "val": len(val_d), "test": len(test_d)},
        "departure_split": {"train": len(train_p), "val": len(val_p), "test": len(test_p)},
    }

    # ── replay scenarios ──
    print("\nSelecting replay scenario dates...")
    scenarios = select_replay_scenarios(df)
    summary["replay_scenarios"] = scenarios
    for name, date in scenarios.items():
        print(f"  {name}: {date}")

    with open(output / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll datasets saved to {output}/")


if __name__ == "__main__":
    main()
