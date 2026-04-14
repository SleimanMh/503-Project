"""User Profile Store — persists per-user charging statistics for inference.

Local:  SQLite (default) — path set by USER_PROFILES_DB env var
AWS:    DynamoDB          — table set by USER_PROFILES_TABLE env var

Backend selection: USER_PROFILES_BACKEND = "sqlite" | "dynamodb"

Each profile record stores:
  • Rolling session stats (mean stay, p10/p90 durations, mean kWh, …)
  • Per-DOW arrival hour distribution (mean, std, q10, q90, count)
  • Station affinity map {station_id: fraction}

Profiles are updated incrementally after every /collect-session call using
Welford's online algorithm — no full dataset scan required.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from app.config import USER_PROFILES_DB

logger = logging.getLogger(__name__)

_BACKEND = os.environ.get("USER_PROFILES_BACKEND", "sqlite").lower()
_TABLE   = os.environ.get("USER_PROFILES_TABLE", "ev-user-profiles")

# ── SQLite helpers ─────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(USER_PROFILES_DB))
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    """Create the profiles table if it doesn't exist (idempotent)."""
    USER_PROFILES_DB.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.execute("""
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
                station_affinity  TEXT,   -- JSON {station_id: fraction}
                arrival_patterns  TEXT,   -- JSON {0..6: {mean_hour,std_hour,q10,q90,count}}
                last_arrival_ts   TEXT,   -- ISO UTC
                updated_at        TEXT
            )
        """)


# ── Public API ─────────────────────────────────────────────────────────────

def get_profile(user_id: str) -> Optional[dict]:
    """Return the profile dict for user_id, or None if unknown."""
    if _BACKEND == "dynamodb":
        return _dynamo_get(str(user_id))
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?", (str(user_id),)
        ).fetchone()
    if row is None:
        return None
    profile = dict(row)
    for field in ("station_affinity", "arrival_patterns"):
        raw = profile.get(field)
        profile[field] = json.loads(raw) if raw else {}
    return profile


def update_profile(user_id: str, session: dict) -> None:
    """Incrementally update a user's profile with a newly completed session.

    session keys:
      connection_time  — datetime or ISO string (UTC)
      disconnect_time  — datetime or ISO string (UTC)
      kwh_delivered    — float
      station_id       — str (optional)
    """
    if _BACKEND == "dynamodb":
        _dynamo_update(str(user_id), session)
        return
    _sqlite_update(str(user_id), session)


# ── SQLite implementation ──────────────────────────────────────────────────

def _sqlite_update(uid: str, session: dict) -> None:
    conn_time = _to_dt(session["connection_time"])
    disc_time = _to_dt(session["disconnect_time"])
    duration_min = (disc_time - conn_time).total_seconds() / 60.0
    kwh = float(session["kwh_delivered"])
    station_id = str(session.get("station_id", "unknown"))
    dow = conn_time.weekday()
    arrival_hour = conn_time.hour + conn_time.minute / 60.0

    existing = get_profile(uid)

    if existing is None:
        profile = _new_profile(uid, conn_time, duration_min, kwh, station_id, dow, arrival_hour)
    else:
        profile = _incremental_update(existing, uid, conn_time, duration_min, kwh, station_id, dow, arrival_hour)

    _upsert_sqlite(profile)


def _new_profile(uid, conn_time, duration_min, kwh, station_id, dow, arrival_hour) -> dict:
    return {
        "user_id":          uid,
        "n_sessions":       1,
        "mean_stay_min":    round(duration_min, 2),
        "p10_duration_min": round(duration_min, 2),
        "p90_duration_min": round(duration_min, 2),
        "cv_duration":      0.0,
        "mean_kwh":         round(kwh, 3),
        "p90_kwh":          round(kwh, 3),
        "kwh_cv":           0.0,
        "days_since_last":  None,
        "station_affinity": {station_id: 1.0},
        "arrival_patterns": {
            str(dow): {
                "mean_hour": round(arrival_hour, 3),
                "std_hour":  1.0,
                "q10_hour":  round(max(0.0, arrival_hour - 1.28), 3),
                "q90_hour":  round(min(23.99, arrival_hour + 1.28), 3),
                "count":     1,
            }
        },
        "last_arrival_ts":  conn_time.isoformat(),
        "updated_at":       datetime.now(timezone.utc).isoformat(),
    }


def _incremental_update(existing, uid, conn_time, duration_min, kwh, station_id, dow, arrival_hour) -> dict:
    n = existing["n_sessions"]
    n_new = n + 1

    # Welford online mean for stay + kwh
    old_mean_stay = float(existing.get("mean_stay_min") or duration_min)
    new_mean_stay = old_mean_stay + (duration_min - old_mean_stay) / n_new
    old_mean_kwh = float(existing.get("mean_kwh") or kwh)
    new_mean_kwh = old_mean_kwh + (kwh - old_mean_kwh) / n_new

    # Station affinity: cumulative counts → renormalise
    affinity = dict(existing.get("station_affinity") or {})
    # Convert fractions back to approximate counts using n
    affinity = {k: v * n for k, v in affinity.items()}
    affinity[station_id] = affinity.get(station_id, 0.0) + 1
    total = sum(affinity.values())
    affinity = {k: round(v / total, 4) for k, v in affinity.items()}

    # Per-DOW arrival pattern (online Welford mean)
    patterns = dict(existing.get("arrival_patterns") or {})
    dow_key = str(dow)
    pat = patterns.get(dow_key, {
        "mean_hour": arrival_hour, "std_hour": 1.0,
        "q10_hour": max(0.0, arrival_hour - 1.28),
        "q90_hour": min(23.99, arrival_hour + 1.28),
        "count": 0,
    })
    c_old = int(pat.get("count", 0))
    c_new = c_old + 1
    m_old = float(pat.get("mean_hour", arrival_hour))
    m_new = m_old + (arrival_hour - m_old) / c_new
    std_new = max(0.5, abs(arrival_hour - m_new)) if c_new >= 5 else float(pat.get("std_hour", 1.0))
    patterns[dow_key] = {
        "mean_hour": round(m_new, 3),
        "std_hour":  round(std_new, 3),
        "q10_hour":  round(max(0.0, m_new - 1.28 * std_new), 3),
        "q90_hour":  round(min(23.99, m_new + 1.28 * std_new), 3),
        "count":     c_new,
    }

    last_ts = existing.get("last_arrival_ts")
    days_since = (conn_time - _to_dt(last_ts)).total_seconds() / 86400.0 if last_ts else None

    return {
        "user_id":          uid,
        "n_sessions":       n_new,
        "mean_stay_min":    round(new_mean_stay, 2),
        "p10_duration_min": existing.get("p10_duration_min"),
        "p90_duration_min": existing.get("p90_duration_min"),
        "cv_duration":      existing.get("cv_duration"),
        "mean_kwh":         round(new_mean_kwh, 3),
        "p90_kwh":          existing.get("p90_kwh"),
        "kwh_cv":           existing.get("kwh_cv"),
        "days_since_last":  round(days_since, 3) if days_since is not None else None,
        "station_affinity": affinity,
        "arrival_patterns": patterns,
        "last_arrival_ts":  conn_time.isoformat(),
        "updated_at":       datetime.now(timezone.utc).isoformat(),
    }


def _upsert_sqlite(p: dict) -> None:
    with _conn() as c:
        c.execute("""
            INSERT INTO user_profiles (
                user_id, n_sessions, mean_stay_min, p10_duration_min, p90_duration_min,
                cv_duration, mean_kwh, p90_kwh, kwh_cv, days_since_last,
                station_affinity, arrival_patterns, last_arrival_ts, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET
                n_sessions=excluded.n_sessions,
                mean_stay_min=excluded.mean_stay_min,
                p10_duration_min=excluded.p10_duration_min,
                p90_duration_min=excluded.p90_duration_min,
                cv_duration=excluded.cv_duration,
                mean_kwh=excluded.mean_kwh,
                p90_kwh=excluded.p90_kwh,
                kwh_cv=excluded.kwh_cv,
                days_since_last=excluded.days_since_last,
                station_affinity=excluded.station_affinity,
                arrival_patterns=excluded.arrival_patterns,
                last_arrival_ts=excluded.last_arrival_ts,
                updated_at=excluded.updated_at
        """, (
            p["user_id"], p["n_sessions"],
            p.get("mean_stay_min"), p.get("p10_duration_min"), p.get("p90_duration_min"),
            p.get("cv_duration"), p.get("mean_kwh"), p.get("p90_kwh"), p.get("kwh_cv"),
            p.get("days_since_last"),
            json.dumps(p.get("station_affinity") or {}),
            json.dumps(p.get("arrival_patterns") or {}),
            p.get("last_arrival_ts"), p.get("updated_at"),
        ))


# ── DynamoDB backend (AWS production) ─────────────────────────────────────

def _dynamo_get(user_id: str) -> Optional[dict]:
    try:
        import boto3
    except ImportError:
        logger.error("boto3 not installed — DynamoDB backend unavailable")
        return None
    ddb = boto3.resource("dynamodb")
    table = ddb.Table(_TABLE)
    resp = table.get_item(Key={"user_id": user_id})
    item = resp.get("Item")
    if item is None:
        return None
    for field in ("station_affinity", "arrival_patterns"):
        raw = item.get(field)
        if isinstance(raw, str):
            item[field] = json.loads(raw)
    # DynamoDB returns Decimal — convert to float
    return _decimal_to_float(item)


def _dynamo_update(user_id: str, session: dict) -> None:
    try:
        import boto3
    except ImportError:
        logger.error("boto3 not installed — DynamoDB backend unavailable")
        return
    existing = _dynamo_get(user_id) or {}
    conn_time = _to_dt(session["connection_time"])
    disc_time = _to_dt(session["disconnect_time"])
    duration_min = (disc_time - conn_time).total_seconds() / 60.0
    kwh = float(session["kwh_delivered"])
    station_id = str(session.get("station_id", "unknown"))
    dow = conn_time.weekday()
    arrival_hour = conn_time.hour + conn_time.minute / 60.0
    n = int(existing.get("n_sessions", 0))

    if n == 0:
        profile = _new_profile(user_id, conn_time, duration_min, kwh, station_id, dow, arrival_hour)
    else:
        profile = _incremental_update(existing, user_id, conn_time, duration_min, kwh, station_id, dow, arrival_hour)

    # DynamoDB stores all numerics; serialise JSON blobs as strings
    item = {k: v for k, v in profile.items() if v is not None}
    for field in ("station_affinity", "arrival_patterns"):
        if field in item and isinstance(item[field], dict):
            item[field] = json.dumps(item[field])

    ddb = boto3.resource("dynamodb")
    ddb.Table(_TABLE).put_item(Item=item)


# ── Utils ──────────────────────────────────────────────────────────────────

def _to_dt(val) -> datetime:
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(val).replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


def _decimal_to_float(obj):
    from decimal import Decimal
    if isinstance(obj, dict):
        return {k: _decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decimal_to_float(v) for v in obj]
    if isinstance(obj, Decimal):
        return float(obj)
    return obj
