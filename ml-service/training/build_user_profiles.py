"""Bootstrap user profiles DB from historical ACN sessions data.

Computes per-user rolling stats and per-DOW arrival patterns from the
full historical dataset, then writes them to the SQLite profiles DB.

After training the ML models, run this once to seed the profile store so
that known users get personalised predictions immediately on startup.

Usage:
    python -m training.build_user_profiles \\
        --data-path ../data/acn_sessions.csv \\
        --db-path   artifacts/user_profiles.db
"""

import argparse
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def build_profiles(data_path: str, db_path: str) -> None:
    df = pd.read_csv(
        data_path,
        parse_dates=["connection_time", "disconnect_time"],
    )
    # Keep only identified users with valid sessions
    df = df.dropna(subset=["connection_time", "disconnect_time", "kwh_delivered", "user_id"])
    df = df[df["kwh_delivered"] > 0].copy()
    df["connection_time"] = pd.to_datetime(df["connection_time"], utc=True)
    df["disconnect_time"]  = pd.to_datetime(df["disconnect_time"], utc=True)
    df["duration_min"] = (
        (df["disconnect_time"] - df["connection_time"]).dt.total_seconds() / 60
    )
    df = df[df["duration_min"] > 0].sort_values("connection_time").reset_index(drop=True)
    df["arrival_hour"] = df["connection_time"].dt.hour + df["connection_time"].dt.minute / 60.0
    df["dow"]          = df["connection_time"].dt.dayofweek

    logger.info(
        "Processing %d sessions for %d unique users",
        len(df), df["user_id"].nunique(),
    )

    profiles = []
    for uid, grp in df.groupby("user_id"):
        grp = grp.sort_values("connection_time").reset_index(drop=True)
        n = len(grp)

        # Per-DOW arrival patterns
        patterns: dict = {}
        for dow, day_grp in grp.groupby("dow"):
            hrs = day_grp["arrival_hour"].values
            patterns[str(int(dow))] = {
                "mean_hour": round(float(np.mean(hrs)), 3),
                "std_hour":  round(float(np.std(hrs)) if len(hrs) > 1 else 1.0, 3),
                "q10_hour":  round(float(np.percentile(hrs, 10)), 3),
                "q90_hour":  round(float(np.percentile(hrs, 90)), 3),
                "count":     int(len(hrs)),
            }

        # Station affinity
        station_affinity = grp["station_id"].value_counts(normalize=True).to_dict()

        # Time since last session
        last_row = grp.iloc[-1]
        prev_row = grp.iloc[-2] if n > 1 else None
        days_since = None
        if prev_row is not None:
            days_since = (
                (last_row["connection_time"] - prev_row["connection_time"]).total_seconds() / 86400
            )

        profile = {
            "user_id":          str(uid),
            "n_sessions":       n,
            "mean_stay_min":    round(float(grp["duration_min"].mean()), 2),
            "p10_duration_min": round(float(grp["duration_min"].quantile(0.10)), 2),
            "p90_duration_min": round(float(grp["duration_min"].quantile(0.90)), 2),
            "cv_duration":      round(
                float(grp["duration_min"].std() / (grp["duration_min"].mean() + 1e-6)), 4
            ),
            "mean_kwh":         round(float(grp["kwh_delivered"].mean()), 3),
            "p90_kwh":          round(float(grp["kwh_delivered"].quantile(0.90)), 3),
            "kwh_cv":           round(
                float(grp["kwh_delivered"].std() / (grp["kwh_delivered"].mean() + 1e-6)), 4
            ),
            "days_since_last":  round(float(days_since), 3) if days_since is not None else None,
            "station_affinity": station_affinity,
            "arrival_patterns": patterns,
            "last_arrival_ts":  last_row["connection_time"].isoformat(),
            "updated_at":       datetime.now(timezone.utc).isoformat(),
        }
        profiles.append(profile)

    logger.info("Built %d user profiles", len(profiles))

    # Write to SQLite
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id           TEXT PRIMARY KEY,
            n_sessions        INTEGER DEFAULT 0,
            mean_stay_min     REAL,
            p10_duration_min  REAL,
            p90_duration_min  REAL,
            cv_duration       REAL,
            mean_kwh          REAL,
            p90_kwh           REAL,
            kwh_cv            REAL,
            days_since_last   REAL,
            station_affinity  TEXT,
            arrival_patterns  TEXT,
            last_arrival_ts   TEXT,
            updated_at        TEXT
        )
    """)
    for p in profiles:
        con.execute("""
            INSERT OR REPLACE INTO user_profiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            p["user_id"], p["n_sessions"],
            p["mean_stay_min"], p["p10_duration_min"], p["p90_duration_min"],
            p["cv_duration"], p["mean_kwh"], p["p90_kwh"], p["kwh_cv"],
            p["days_since_last"],
            json.dumps(p["station_affinity"]),
            json.dumps(p["arrival_patterns"]),
            p["last_arrival_ts"], p["updated_at"],
        ))
    con.commit()
    con.close()
    logger.info("Profiles written to %s", db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build user profiles DB from ACN data")
    parser.add_argument("--data-path", required=True, help="Path to acn_sessions.csv")
    parser.add_argument("--db-path", default="artifacts/user_profiles.db",
                        help="Output SQLite DB path (default: artifacts/user_profiles.db)")
    args = parser.parse_args()
    build_profiles(args.data_path, args.db_path)
