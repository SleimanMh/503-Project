"""
ACN-Data Pipeline: Parse raw sessions and build training datasets for:
  M1  Departure prediction  (per-session, with user history + cold-start)
  M2  Demand forecasting    (aggregate time-series, 30-min windows)
  M3  Energy need estimator (per-session, log-kWh target)
  M4A Arrivals forecast     (aggregate time-series, 15-min slots)

Usage:
    python -m training.prepare_data --data-path ../data/acn_sessions.csv --output-dir artifacts/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLD_START_MASK_FRAC = 0.20   # fraction of identified-user rows to mask during training


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


def _compute_user_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-user expanding statistics available at each arrival time.

    Uses shift(1) everywhere so each row only sees sessions that occurred
    BEFORE the current one — no data leakage.
    Anonymous sessions (user_id == NaN) receive NaN for every user feature.
    """
    df = df.sort_values("connection_time").reset_index(drop=True)

    # Sequential position per user (0 = first session ever → no history)
    df["user_n_sessions"] = (
        df.groupby("user_id").cumcount().astype(float)
        .where(df["user_id"].notna(), other=np.nan)
    )

    # Expanding mean of duration (minutes), shifted so current row is excluded
    df["user_mean_stay"] = (
        df.groupby("user_id")["duration_min"]
        .transform(lambda s: s.expanding().mean().shift(1))
        .where(df["user_id"].notna(), other=np.nan)
    )

    df["user_p10_duration"] = (
        df.groupby("user_id")["duration_min"]
        .transform(lambda s: s.expanding().quantile(0.1).shift(1))
        .where(df["user_id"].notna(), other=np.nan)
    )

    df["user_p90_duration"] = (
        df.groupby("user_id")["duration_min"]
        .transform(lambda s: s.expanding().quantile(0.9).shift(1))
        .where(df["user_id"].notna(), other=np.nan)
    )

    df["user_cv_duration"] = (
        df.groupby("user_id")["duration_min"]
        .transform(lambda s: (s.expanding().std() / (s.expanding().mean() + 1e-6)).shift(1))
        .where(df["user_id"].notna(), other=np.nan)
    )

    # Energy history
    df["user_mean_kwh"] = (
        df.groupby("user_id")["kwh_delivered"]
        .transform(lambda s: s.expanding().mean().shift(1))
        .where(df["user_id"].notna(), other=np.nan)
    )

    df["user_p90_kwh"] = (
        df.groupby("user_id")["kwh_delivered"]
        .transform(lambda s: s.expanding().quantile(0.9).shift(1))
        .where(df["user_id"].notna(), other=np.nan)
    )

    df["user_kwh_cv"] = (
        df.groupby("user_id")["kwh_delivered"]
        .transform(lambda s: (s.expanding().std() / (s.expanding().mean() + 1e-6)).shift(1))
        .where(df["user_id"].notna(), other=np.nan)
    )

    # Days since last visit
    df["_prev_conn"] = df.groupby("user_id")["connection_time"].transform(lambda s: s.shift(1))
    df["user_days_since_last_visit"] = (
        ((df["connection_time"] - df["_prev_conn"]).dt.total_seconds() / 86400)
        .where(df["user_id"].notna(), other=np.nan)
    )
    df.drop(columns=["_prev_conn"], inplace=True)

    # Station affinity: fraction of past sessions at this station
    # _past_at_station = cumcount within (user, station) = sessions at station before this one
    # _past_total      = cumcount within user = total sessions before this one
    df["_past_total"] = df.groupby("user_id").cumcount()   # 0-indexed past count
    df["_past_at_stn"] = df.groupby(["user_id", "station_id"]).cumcount()
    df["user_station_affinity"] = np.where(
        df["user_id"].notna() & (df["_past_total"] > 0),
        df["_past_at_stn"] / df["_past_total"],
        np.nan,
    )
    df.drop(columns=["_past_total", "_past_at_stn"], inplace=True)

    # Anonymous flag (float so HistGBDT treats it as a numeric feature)
    df["is_anonymous"] = df["user_id"].isna().astype(float)

    return df


def apply_cold_start_mask(X: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Randomly mask user features to NaN for COLD_START_MASK_FRAC of
    identified-user rows.  Forces the model to learn good priors from
    temporal/station features alone — matches the 59% anonymous production scenario.
    Only call on TRAINING data, never on validation/test.
    """
    X = X.copy()
    user_cols = [
        "user_n_sessions", "user_mean_stay", "user_p10_duration",
        "user_p90_duration", "user_cv_duration", "user_mean_kwh",
        "user_p90_kwh", "user_kwh_cv", "user_days_since_last_visit",
        "user_station_affinity",
    ]
    identified = (X.get("is_anonymous", pd.Series(1.0, index=X.index)) == 0)
    mask = rng.random(len(X)) < COLD_START_MASK_FRAC
    rows_to_mask = identified & mask
    for col in user_cols:
        if col in X.columns:
            X.loc[rows_to_mask, col] = np.nan
    return X


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


# ── Model 2 (M1): Departure Prediction Dataset ───────────────────────────────
def build_departure_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-session features for predicting stay duration.
    Only uses information available at arrival time (no leakage).

    Target: log(duration_min) — log-space keeps predictions positive and
    handles the heavy right tail of overnight/long-stay sessions.
    """
    df = df.copy()

    # Temporal features
    df["hour_sin"], df["hour_cos"] = _cyclical_encode(df["arrival_hour"], 24)
    df["dow_sin"], df["dow_cos"] = _cyclical_encode(df["day_of_week"], 7)
    df["month_sin"], df["month_cos"] = _cyclical_encode(df["month"], 12)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8, 9]).astype(float)
    df["slot_of_day"] = (df["arrival_hour"] * 4).astype(int).clip(0, 95).astype(float)
    df["arrival_regime"] = pd.cut(
        df["arrival_hour"],
        bins=[-1, 7, 13, 25],
        labels=[0.0, 1.0, 2.0],
    ).astype(float)

    # Site/cluster encoding (integer codes; HistGBDT handles sparse categories well)
    df["site_encoded"] = df["site_id"].astype("category").cat.codes.astype(float)
    df["cluster_encoded"] = df["cluster_id"].astype("category").cat.codes.astype(float)

    # Per-user expanding history (no leakage)
    df = _compute_user_history_features(df)

    # Per-station average stay (expanding, shifted)
    df["station_mean_stay"] = (
        df.groupby("station_id")["duration_min"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )
    df["station_mean_stay"] = df["station_mean_stay"].fillna(df["duration_min"].mean())

    # Energy requested — available when user configures target SoC
    # (NaN when not provided; HistGBDT handles NaN natively)
    if "kwh_delivered" in df.columns:
        df["requested_energy_kwh"] = df["kwh_delivered"]
    else:
        df["requested_energy_kwh"] = np.nan

    # Log-space target
    df["log_duration_min"] = np.log(np.maximum(df["duration_min"], 1.0))

    return df


DEPARTURE_FEATURE_COLS = [
    # Temporal
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "month_sin", "month_cos", "is_summer", "slot_of_day", "arrival_regime",
    # Site context
    "site_encoded", "cluster_encoded",
    # User history (NaN-safe via HistGBDT)
    "user_n_sessions", "user_mean_stay", "user_p10_duration", "user_p90_duration",
    "user_cv_duration", "user_mean_kwh", "user_p90_kwh", "user_kwh_cv",
    "user_days_since_last_visit", "user_station_affinity", "is_anonymous",
    # Station prior
    "station_mean_stay",
    # Energy context
    "requested_energy_kwh",
]
DEPARTURE_TARGET_COL = "log_duration_min"   # back-transform with exp() at inference


# ── Model 3 (M3): Energy Need Dataset ────────────────────────────────────────
def build_energy_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-session features for predicting kWh delivered.
    Uses the same user-history features as departure, but targets
    log(kwh_delivered).  Call AFTER build_departure_dataset so that
    user history columns are already present.
    """
    df = df.copy()

    if "is_summer" not in df.columns:
        df["is_summer"] = df["month"].isin([6, 7, 8, 9]).astype(float)
    if "arrival_regime" not in df.columns:
        df["arrival_regime"] = pd.cut(
            df["arrival_hour"], bins=[-1, 7, 13, 25], labels=[0.0, 1.0, 2.0],
        ).astype(float)
    if "site_encoded" not in df.columns:
        df["site_encoded"] = df["site_id"].astype("category").cat.codes.astype(float)
    if "cluster_encoded" not in df.columns:
        df["cluster_encoded"] = df["cluster_id"].astype("category").cat.codes.astype(float)
    if "is_anonymous" not in df.columns:
        df = _compute_user_history_features(df)

    # Station mean kWh (expanding, shifted — no leakage)
    df["station_mean_kwh"] = (
        df.groupby("station_id")["kwh_delivered"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )
    df["station_mean_kwh"] = df["station_mean_kwh"].fillna(df["kwh_delivered"].mean())

    df["log_kwh_delivered"] = np.log(np.maximum(df["kwh_delivered"], 0.1))

    return df


ENERGY_FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "month_sin", "month_cos", "is_summer", "arrival_regime",
    "site_encoded", "cluster_encoded",
    "user_n_sessions", "user_mean_kwh", "user_p90_kwh", "user_kwh_cv",
    "user_days_since_last_visit", "is_anonymous",
    "station_mean_kwh",
]
ENERGY_TARGET_COL = "log_kwh_delivered"   # back-transform with exp() at inference


# ── Model 4A (M4A): Arrivals-per-slot Forecast ───────────────────────────────
def build_arrivals_dataset(df: pd.DataFrame, slot_minutes: int = 15) -> pd.DataFrame:
    """
    Aggregate sessions into 15-min slots and build time-series features for
    predicting n_arrivals per slot (Poisson-distributed count).

    Lag convention (each lag N → N slots back):
      lag_arrivals_4   =  4 * 15 min  =  1 hour back
      lag_arrivals_16  = 16 * 15 min  =  4 hours back
      lag_arrivals_96  = 96 * 15 min  = 24 hours back
      lag_arrivals_672 = 672 * 15 min =  7 days back
    """
    df = df.copy()
    df["slot"] = df["connection_time"].dt.floor(f"{slot_minutes}min")

    agg = df.groupby("slot").agg(
        n_arrivals=("session_id", "count"),
        total_kwh=("kwh_delivered", "sum"),
    ).reset_index()

    # Fill the full continuous timeline with zeros
    full_range = pd.date_range(
        start=agg["slot"].min(),
        end=agg["slot"].max(),
        freq=f"{slot_minutes}min",
        tz="UTC",
    )
    agg = (
        agg.set_index("slot")
        .reindex(full_range, fill_value=0)
        .rename_axis("slot")
        .reset_index()
    )

    # Temporal features
    agg["hour"] = agg["slot"].dt.hour + agg["slot"].dt.minute / 60
    agg["hour_sin"], agg["hour_cos"] = _cyclical_encode(agg["hour"], 24)
    dow = agg["slot"].dt.dayofweek
    agg["dow_sin"], agg["dow_cos"] = _cyclical_encode(dow, 7)
    month = agg["slot"].dt.month
    agg["month_sin"], agg["month_cos"] = _cyclical_encode(month, 12)
    agg["is_weekend"] = (dow >= 5).astype(int)
    agg["is_summer"] = month.isin([6, 7, 8, 9]).astype(int)
    agg["slot_of_day"] = ((agg["slot"].dt.hour * 60 + agg["slot"].dt.minute) // slot_minutes).astype(float)

    # Lag features (shift by N slots = shift N periods)
    arr = agg["n_arrivals"]
    agg["lag_arrivals_4"]   = arr.shift(4)
    agg["lag_arrivals_16"]  = arr.shift(16)
    agg["lag_arrivals_96"]  = arr.shift(96)
    agg["lag_arrivals_672"] = arr.shift(672)

    # Rolling statistics (1-hour window = 4 slots, based on past only)
    agg["rolling_arrivals_mean_1h"] = arr.shift(1).rolling(4, min_periods=1).mean()
    agg["rolling_arrivals_std_1h"]  = arr.shift(1).rolling(4, min_periods=1).std().fillna(0)

    agg = agg.dropna().reset_index(drop=True)
    return agg


ARRIVALS_FEATURE_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    "month_sin", "month_cos", "is_summer", "slot_of_day",
    "lag_arrivals_4", "lag_arrivals_16", "lag_arrivals_96", "lag_arrivals_672",
    "rolling_arrivals_mean_1h", "rolling_arrivals_std_1h",
]
ARRIVALS_TARGET_COL = "n_arrivals"


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

    # ── demand forecasting dataset (M4) ──
    print("\nBuilding demand forecasting dataset (M4)...")
    demand_df = build_demand_dataset(df)
    train_d, val_d, test_d = temporal_split(demand_df, "window")
    train_d.to_parquet(output / "demand_train.parquet", index=False)
    val_d.to_parquet(output / "demand_val.parquet", index=False)
    test_d.to_parquet(output / "demand_test.parquet", index=False)
    print(f"  Train: {len(train_d)}, Val: {len(val_d)}, Test: {len(test_d)} windows")

    # ── departure prediction dataset (M1) — includes user history features ──
    print("\nBuilding departure prediction dataset (M1) with user history...")
    dep_df = build_departure_dataset(df)
    train_p, val_p, test_p = temporal_split(dep_df, "connection_time")
    train_p.to_parquet(output / "departure_train.parquet", index=False)
    val_p.to_parquet(output / "departure_val.parquet", index=False)
    test_p.to_parquet(output / "departure_test.parquet", index=False)
    print(f"  Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)} sessions")
    anon_pct = (dep_df["is_anonymous"].sum() / len(dep_df) * 100)
    print(f"  Anonymous sessions: {anon_pct:.1f}%")

    # ── energy need dataset (M3) — built from enriched departure df ──
    print("\nBuilding energy need dataset (M3)...")
    # dep_df already has the user history columns computed correctly
    eng_df = build_energy_dataset(dep_df)
    train_e, val_e, test_e = temporal_split(eng_df, "connection_time")
    train_e.to_parquet(output / "energy_train.parquet", index=False)
    val_e.to_parquet(output / "energy_val.parquet", index=False)
    test_e.to_parquet(output / "energy_test.parquet", index=False)
    print(f"  Train: {len(train_e)}, Val: {len(val_e)}, Test: {len(test_e)} sessions")

    # ── arrivals forecast dataset (M4A) ──
    print("\nBuilding arrivals forecast dataset (M4A, 15-min slots)...")
    arr_df = build_arrivals_dataset(df)
    train_a, val_a, test_a = temporal_split(arr_df, "slot")
    train_a.to_parquet(output / "arrivals_train.parquet", index=False)
    val_a.to_parquet(output / "arrivals_val.parquet", index=False)
    test_a.to_parquet(output / "arrivals_test.parquet", index=False)
    print(f"  Train: {len(train_a)}, Val: {len(val_a)}, Test: {len(test_a)} slots")

    # ── data summary ──
    summary = {
        "total_sessions": len(df),
        "date_range": [str(df["connection_time"].min()), str(df["connection_time"].max())],
        "unique_stations": int(df["station_id"].nunique()),
        "unique_sites": int(df["site_id"].nunique()),
        "mean_kwh": float(df["kwh_delivered"].mean()),
        "mean_duration_hr": float(df["duration_hr"].mean()),
        "anonymous_pct": round(float(anon_pct), 1),
        "demand_features": DEMAND_FEATURE_COLS,
        "departure_features": DEPARTURE_FEATURE_COLS,
        "energy_features": ENERGY_FEATURE_COLS,
        "arrivals_features": ARRIVALS_FEATURE_COLS,
        "demand_split": {"train": len(train_d), "val": len(val_d), "test": len(test_d)},
        "departure_split": {"train": len(train_p), "val": len(val_p), "test": len(test_p)},
        "energy_split": {"train": len(train_e), "val": len(val_e), "test": len(test_e)},
        "arrivals_split": {"train": len(train_a), "val": len(val_a), "test": len(test_a)},
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
