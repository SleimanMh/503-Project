"""
Retraining pipeline — merges collected live sessions with the original ACN
training data and retrains both departure prediction and demand forecasting
models.

Trigger condition: at least RETRAIN_THRESHOLD new sessions collected.

Usage (inside the ml-service container):
    python -m training.retrain                          # auto-detect data paths
    python -m training.retrain --force                  # skip session-count check
    python -m training.retrain --artifacts-dir artifacts/

The script:
  1. Checks collected_sessions.csv row count.
  2. If >= RETRAIN_THRESHOLD (or --force flag), merges with ACN baseline data.
  3. Rebuilds demand + departure datasets via prepare_data pipeline.
  4. Retrains both XGBoost models (same hyperparameters as initial training).
  5. Overwrites the .pkl model files and updates metrics/eval_report.json.
  6. Archives the processed collected_sessions.csv with a timestamp.
"""

import argparse
import json
import logging
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import mlflow

from training.prepare_data import (
    build_demand_dataset,
    build_departure_dataset,
    temporal_split,
    DEMAND_FEATURE_COLS,
    DEMAND_TARGET_COLS,
    DEPARTURE_FEATURE_COLS,
    DEPARTURE_TARGET_COL,
)
from training.s3_store import upload_models_to_s3, upload_sessions_to_s3, upload_user_profiles_to_s3

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

RETRAIN_THRESHOLD = 500  # minimum new sessions before retraining


# ── Column mapping: collected_sessions.csv → ACN format ──────────────────────
# The collected CSV has the same columns as ACN but some derived columns
# (arrival_hour, day_of_week, etc.) may need to be computed for prepare_data.

def _load_collected_sessions(path: Path) -> pd.DataFrame:
    """Load collected simulation sessions and normalise to ACN schema."""
    df = pd.read_csv(path, parse_dates=["connection_time", "disconnect_time"])
    df = df.dropna(subset=["connection_time", "kwh_delivered"])
    df = df[df["kwh_delivered"] > 0].copy()
    df = df[df["duration_min"] > 0].copy()
    df["duration_hr"] = df["duration_min"] / 60
    df["connection_time"] = pd.to_datetime(df["connection_time"], utc=True)
    df["disconnect_time"] = pd.to_datetime(df["disconnect_time"], utc=True)

    # Ensure required columns exist (recompute if missing)
    df["arrival_hour"] = df["connection_time"].dt.hour + df["connection_time"].dt.minute / 60
    df["departure_hour"] = df["disconnect_time"].dt.hour + df["disconnect_time"].dt.minute / 60
    df["day_of_week"] = df["connection_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["connection_time"].dt.month
    df["year"] = df["connection_time"].dt.year

    # Fill columns that ACN has but collected data may not
    for col in ["space_id", "timezone", "charge_time_hr", "idle_time_hr"]:
        if col not in df.columns:
            df[col] = None
    if "avg_charge_kw" not in df.columns:
        df["avg_charge_kw"] = df["kwh_delivered"] / df["duration_hr"].replace(0, np.nan)

    return df.sort_values("connection_time").reset_index(drop=True)


def _load_ocpp_sessions(path: Path) -> pd.DataFrame:
    """Load OCPP-collected sessions (ocpp_sessions.csv) and normalise to ACN schema."""
    df = pd.read_csv(path, parse_dates=["connection_time", "disconnect_time"])
    df = df.dropna(subset=["connection_time", "kwh_delivered"])
    df = df[df["kwh_delivered"] > 0].copy()
    df = df[df["duration_hr"] > 0].copy()
    df["duration_min"] = df["duration_hr"] * 60
    df["connection_time"] = pd.to_datetime(df["connection_time"], utc=True)
    df["disconnect_time"] = pd.to_datetime(df["disconnect_time"], utc=True)
    df["arrival_hour"] = df["connection_time"].dt.hour + df["connection_time"].dt.minute / 60
    df["departure_hour"] = df["disconnect_time"].dt.hour + df["disconnect_time"].dt.minute / 60
    df["day_of_week"] = df["connection_time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["connection_time"].dt.month
    df["year"] = df["connection_time"].dt.year
    return df.sort_values("connection_time").reset_index(drop=True)


def _merge_datasets(acn_path: Path, collected_path: Path) -> pd.DataFrame:
    """Merge original ACN data with newly collected simulation sessions."""
    logger.info(f"Loading ACN baseline data: {acn_path}")
    acn = pd.read_csv(
        acn_path,
        parse_dates=["connection_time", "disconnect_time", "done_charging", "date"],
    )
    acn = acn.dropna(subset=["connection_time", "disconnect_time", "kwh_delivered"])
    acn = acn[acn["kwh_delivered"] > 0].copy()
    acn = acn[acn["duration_hr"] > 0].copy()
    acn["duration_min"] = acn["duration_hr"] * 60
    acn["connection_time"] = pd.to_datetime(acn["connection_time"], utc=True)
    acn["disconnect_time"] = pd.to_datetime(acn["disconnect_time"], utc=True)

    frames = [acn]
    total_collected = 0

    logger.info(f"Loading collected sessions: {collected_path}")
    if collected_path.exists():
        collected = _load_collected_sessions(collected_path)
        total_collected += len(collected)
        frames.append(collected)
    else:
        logger.info("  collected_sessions.csv not found, skipping")

    # Also load OCPP-native sessions if present
    ocpp_path = collected_path.parent / "ocpp_sessions.csv"
    if ocpp_path.exists():
        ocpp = _load_ocpp_sessions(ocpp_path)
        logger.info(f"  OCPP sessions: {len(ocpp)}")
        total_collected += len(ocpp)
        frames.append(ocpp)
    else:
        logger.info("  ocpp_sessions.csv not found, skipping")

    logger.info(f"  ACN sessions: {len(acn)}, collected sessions: {total_collected}")

    # Keep only columns that exist in all frames
    common_cols = list(set.intersection(*[set(f.columns) for f in frames]))
    merged = pd.concat([f[common_cols] for f in frames], ignore_index=True)
    merged = merged.sort_values("connection_time").reset_index(drop=True)
    logger.info(f"  Merged total: {len(merged)} sessions")
    return merged


def _train_departure(X_train, y_train, X_val, y_val, X_test, y_test) -> tuple:
    """Train departure prediction model with walk-forward CV."""
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
        m = xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
        )
        m.fit(X_train[tr_idx], y_train[tr_idx],
              eval_set=[(X_train[va_idx], y_train[va_idx])], verbose=False)
        cv_maes.append(mean_absolute_error(y_train[va_idx], m.predict(X_train[va_idx])))

    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_test)
    errors = np.abs(y_test - y_pred)
    metrics = {
        "cv_mae_mean_min": float(np.mean(cv_maes)),
        "test_mae_min": float(mean_absolute_error(y_test, y_pred)),
        "test_rmse_min": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "within_15min_pct": float((errors <= 15).mean() * 100),
        "within_30min_pct": float((errors <= 30).mean() * 100),
        "feature_importance": dict(zip(DEPARTURE_FEATURE_COLS, model.feature_importances_.tolist())),
    }
    return model, metrics


def _train_demand(X_train, y_train, X_val, y_val, X_test, y_test, target_name: str) -> tuple:
    """Train demand forecasting model for one target."""
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
        m = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=1.0, random_state=42,
        )
        m.fit(X_train[tr_idx], y_train[tr_idx],
              eval_set=[(X_train[va_idx], y_train[va_idx])], verbose=False)
        cv_maes.append(mean_absolute_error(y_train[va_idx], m.predict(X_train[va_idx])))

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        reg_lambda=1.0, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_test)
    metrics = {
        "cv_mae_mean": float(np.mean(cv_maes)),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "feature_importance": dict(zip(DEMAND_FEATURE_COLS, model.feature_importances_.tolist())),
    }
    return model, metrics


def run_retrain(
    acn_data_path: Path,
    collected_path: Path,
    artifacts_dir: Path,
    force: bool = False,
) -> dict:
    """
    Main retraining entry point.

    Returns a summary dict with model metrics and whether retraining ran.
    """
    # ── Check session count ───────────────────────────────────────────────
    has_collected = collected_path.exists()
    new_sessions = 0
    if has_collected:
        new_sessions = sum(1 for _ in open(collected_path)) - 1  # subtract header
    ocpp_path = collected_path.parent / "ocpp_sessions.csv"
    if ocpp_path.exists():
        new_sessions += sum(1 for _ in open(ocpp_path)) - 1
    if new_sessions > 0:
        logger.info(f"Collected sessions: {new_sessions} (threshold: {RETRAIN_THRESHOLD})")
    else:
        logger.info("No collected sessions found.")

    if not force and new_sessions < RETRAIN_THRESHOLD:
        return {
            "retrained": False,
            "reason": f"insufficient_data ({new_sessions}/{RETRAIN_THRESHOLD})",
            "new_sessions": new_sessions,
        }

    # ── Load / merge datasets ─────────────────────────────────────────────
    if has_collected and new_sessions > 0:
        merged = _merge_datasets(acn_data_path, collected_path)
    else:
        # Force-retrain on ACN baseline only (no collected data yet)
        logger.info(f"Retraining on ACN baseline only: {acn_data_path}")
        merged = pd.read_csv(
            acn_data_path,
            parse_dates=["connection_time", "disconnect_time", "done_charging", "date"],
        )
        merged = merged.dropna(subset=["connection_time", "disconnect_time", "kwh_delivered"])
        merged = merged[merged["kwh_delivered"] > 0].copy()
        merged = merged[merged["duration_hr"] > 0].copy()
        merged["duration_min"] = merged["duration_hr"] * 60
        merged["connection_time"] = pd.to_datetime(merged["connection_time"], utc=True)
        merged["disconnect_time"] = pd.to_datetime(merged["disconnect_time"], utc=True)
        merged = merged.sort_values("connection_time").reset_index(drop=True)
        logger.info(f"  Loaded {len(merged)} ACN sessions")

    # ── Rebuild training datasets ─────────────────────────────────────────
    logger.info("Rebuilding demand dataset...")
    demand_df = build_demand_dataset(merged)
    d_train, d_val, d_test = temporal_split(demand_df, "window")

    logger.info("Rebuilding departure dataset...")
    dep_df = build_departure_dataset(merged)
    p_train, p_val, p_test = temporal_split(dep_df, "connection_time")

    # ── Retrain departure model ───────────────────────────────────────────
    logger.info("Retraining departure prediction model...")
    dep_model, dep_metrics = _train_departure(
        p_train[DEPARTURE_FEATURE_COLS].values, p_train[DEPARTURE_TARGET_COL].values,
        p_val[DEPARTURE_FEATURE_COLS].values,   p_val[DEPARTURE_TARGET_COL].values,
        p_test[DEPARTURE_FEATURE_COLS].values,  p_test[DEPARTURE_TARGET_COL].values,
    )
    logger.info(f"  Departure MAE: {dep_metrics['test_mae_min']:.1f} min")

    # ── Retrain demand models ─────────────────────────────────────────────
    logger.info("Retraining demand forecasting models...")
    demand_models = {}
    demand_metrics = {}
    for target in DEMAND_TARGET_COLS:
        m, m_metrics = _train_demand(
            d_train[DEMAND_FEATURE_COLS].values, d_train[target].values,
            d_val[DEMAND_FEATURE_COLS].values,   d_val[target].values,
            d_test[DEMAND_FEATURE_COLS].values,  d_test[target].values,
            target,
        )
        demand_models[target] = m
        demand_metrics[target] = m_metrics
        logger.info(f"  Demand [{target}] MAE: {m_metrics['test_mae']:.3f}")

    # ── Save models ───────────────────────────────────────────────────────
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with open(artifacts_dir / "departure_model.pkl", "wb") as f:
        pickle.dump(dep_model, f)
    with open(artifacts_dir / "departure_metrics.json", "w") as f:
        json.dump(dep_metrics, f, indent=2)

    with open(artifacts_dir / "demand_model.pkl", "wb") as f:
        pickle.dump(demand_models, f)
    with open(artifacts_dir / "demand_metrics.json", "w") as f:
        json.dump(demand_metrics, f, indent=2)

    # Update eval report
    eval_report = {
        "demand_arrival_count_mae": demand_metrics["arrival_count"]["test_mae"],
        "demand_total_kwh_rmse": demand_metrics["total_kwh"]["test_rmse"],
        "departure_mae_min": dep_metrics["test_mae_min"],
        "departure_within_15min_pct": dep_metrics["within_15min_pct"],
        "acceptance_checks": {
            "demand_arrival_mae_pass": demand_metrics["arrival_count"]["test_mae"] <= 1.5,
            "demand_kwh_rmse_pass": demand_metrics["total_kwh"]["test_rmse"] <= 5.0,
            "departure_mae_pass": dep_metrics["test_mae_min"] <= 30,
            "departure_15min_pass": dep_metrics["within_15min_pct"] >= 60,
        },
        "retrained_at": datetime.now(timezone.utc).isoformat(),
        "sessions_used": len(merged),
        "new_sessions_added": new_sessions,
    }
    eval_report["all_pass"] = all(eval_report["acceptance_checks"].values())
    with open(artifacts_dir / "eval_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)

    # ── Archive processed collected sessions ──────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Upload raw sessions to S3 BEFORE archiving locally
    upload_sessions_to_s3(collected_path)

    archive_path = collected_path.parent / f"collected_sessions_processed_{ts}.csv"
    shutil.move(str(collected_path), str(archive_path))
    logger.info(f"Archived processed sessions to {archive_path}")

    # ── MLflow: log retraining run ────────────────────────────────────────
    mlflow.set_experiment("model-retraining")
    with mlflow.start_run(run_name=f"retrain-{ts}"):
        mlflow.log_params({
            "new_sessions": new_sessions,
            "total_sessions_used": len(merged),
            "retrain_trigger": "forced" if new_sessions < RETRAIN_THRESHOLD else "threshold",
        })
        mlflow.log_metrics({
            "departure_test_mae_min": dep_metrics["test_mae_min"],
            "departure_test_rmse_min": dep_metrics["test_rmse_min"],
            "departure_within_15min_pct": dep_metrics["within_15min_pct"],
        })
        for target, m_metrics in demand_metrics.items():
            mlflow.log_metrics({
                f"demand_{target}_test_mae": m_metrics["test_mae"],
                f"demand_{target}_test_rmse": m_metrics["test_rmse"],
            })
        mlflow.log_artifact(str(artifacts_dir / "eval_report.json"))
        mlflow.log_artifact(str(artifacts_dir / "departure_model.pkl"))
        mlflow.log_artifact(str(artifacts_dir / "demand_model.pkl"))
        logger.info("MLflow: retraining run logged.")

    logger.info("Retraining complete.")

    # ── Push new models to S3 so all pods pick them up on next restart ────
    upload_models_to_s3(artifacts_dir)
    # Also back up the user profiles DB
    from app.config import USER_PROFILES_DB
    upload_user_profiles_to_s3(USER_PROFILES_DB)

    return {
        "retrained": True,
        "new_sessions": new_sessions,
        "total_sessions_used": len(merged),
        "departure_mae_min": dep_metrics["test_mae_min"],
        "demand_arrival_mae": demand_metrics["arrival_count"]["test_mae"],
        "all_checks_pass": eval_report["all_pass"],
        "archived_to": str(archive_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Retrain EV ML models on collected data")
    parser.add_argument("--artifacts-dir", default="artifacts/")
    parser.add_argument("--acn-data-path", default=None,
                        help="Path to acn_sessions.csv (default: from DATA_DIR env)")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if fewer than RETRAIN_THRESHOLD sessions")
    args = parser.parse_args()

    from app.config import ARTIFACTS_DIR, COLLECTED_SESSIONS_CSV, ACN_DATA_PATH
    artifacts = Path(args.artifacts_dir) if args.artifacts_dir else ARTIFACTS_DIR
    acn_path = Path(args.acn_data_path) if args.acn_data_path else ACN_DATA_PATH

    result = run_retrain(
        acn_data_path=acn_path,
        collected_path=COLLECTED_SESSIONS_CSV,
        artifacts_dir=artifacts,
        force=args.force,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
