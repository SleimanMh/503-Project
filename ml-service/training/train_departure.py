"""
Train Model 2: Departure Time / Stay Duration Prediction (XGBoost).
Walk-forward cross-validation on ACN-Data per-session features.

Usage:
    python -m training.train_departure --artifacts-dir artifacts/
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import mlflow
import mlflow.xgboost

from training.prepare_data import DEPARTURE_FEATURE_COLS, DEPARTURE_TARGET_COL


def train_departure_model(artifacts_dir: str = "artifacts/"):
    artifacts = Path(artifacts_dir)

    train_df = pd.read_parquet(artifacts / "departure_train.parquet")
    val_df = pd.read_parquet(artifacts / "departure_val.parquet")
    test_df = pd.read_parquet(artifacts / "departure_test.parquet")

    X_train = train_df[DEPARTURE_FEATURE_COLS].values
    X_val = val_df[DEPARTURE_FEATURE_COLS].values
    X_test = test_df[DEPARTURE_FEATURE_COLS].values

    y_train = train_df[DEPARTURE_TARGET_COL].values
    y_val = val_df[DEPARTURE_TARGET_COL].values
    y_test = test_df[DEPARTURE_TARGET_COL].values

    print(f"Training departure prediction model")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Mean stay duration: {y_train.mean():.1f} min ({y_train.mean()/60:.1f} hr)")

    # ── Walk-forward cross-validation ──
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
        model_cv = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
        )
        model_cv.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[va_idx], y_train[va_idx])],
            verbose=False,
        )
        preds = model_cv.predict(X_train[va_idx])
        fold_mae = mean_absolute_error(y_train[va_idx], preds)
        cv_maes.append(fold_mae)
        print(f"  Fold {fold+1} MAE: {fold_mae:.1f} min")

    print(f"  CV Mean MAE: {np.mean(cv_maes):.1f} ± {np.std(cv_maes):.1f} min")

    # ── Train final model ──
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # ── Evaluate on test set ──
    y_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Percentage within ±15 and ±30 minutes
    errors = np.abs(y_test - y_pred)
    within_15 = (errors <= 15).mean() * 100
    within_30 = (errors <= 30).mean() * 100

    # ── Baselines ──
    # Fixed 4-hour assumption
    fixed_4h_pred = np.full_like(y_test, 240.0)
    fixed_4h_mae = mean_absolute_error(y_test, fixed_4h_pred)

    # Mean duration baseline
    mean_pred = np.full_like(y_test, y_train.mean())
    mean_mae = mean_absolute_error(y_test, mean_pred)

    print(f"\n  Test MAE: {test_mae:.1f} min")
    print(f"  Test RMSE: {test_rmse:.1f} min")
    print(f"  Within ±15 min: {within_15:.1f}%")
    print(f"  Within ±30 min: {within_30:.1f}%")
    print(f"  Fixed 4h baseline MAE: {fixed_4h_mae:.1f} min")
    print(f"  Mean baseline MAE: {mean_mae:.1f} min")
    print(f"  Improvement over fixed 4h: {(1 - test_mae/fixed_4h_mae)*100:.1f}%")

    metrics = {
        "cv_mae_mean_min": float(np.mean(cv_maes)),
        "cv_mae_std_min": float(np.std(cv_maes)),
        "test_mae_min": float(test_mae),
        "test_rmse_min": float(test_rmse),
        "within_15min_pct": float(within_15),
        "within_30min_pct": float(within_30),
        "fixed_4h_baseline_mae_min": float(fixed_4h_mae),
        "mean_baseline_mae_min": float(mean_mae),
        "improvement_over_fixed4h_pct": float((1 - test_mae / fixed_4h_mae) * 100),
        "feature_importance": dict(zip(
            DEPARTURE_FEATURE_COLS,
            model.feature_importances_.tolist()
        )),
    }

    # ── Save ──
    with open(artifacts / "departure_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(artifacts / "departure_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {artifacts}/departure_model.pkl")
    print(f"Metrics saved to {artifacts}/departure_metrics.json")

    # ── MLflow experiment tracking ──
    mlflow.set_experiment("departure-prediction")
    with mlflow.start_run(run_name="departure-xgboost"):
        mlflow.log_params({
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "cv_splits": 5,
        })
        mlflow.log_metrics({
            "cv_mae_mean_min": metrics["cv_mae_mean_min"],
            "cv_mae_std_min": metrics["cv_mae_std_min"],
            "test_mae_min": metrics["test_mae_min"],
            "test_rmse_min": metrics["test_rmse_min"],
            "within_15min_pct": metrics["within_15min_pct"],
            "within_30min_pct": metrics["within_30min_pct"],
            "improvement_over_fixed4h_pct": metrics["improvement_over_fixed4h_pct"],
        })
        mlflow.log_artifact(str(artifacts / "departure_model.pkl"))
        mlflow.log_artifact(str(artifacts / "departure_metrics.json"))
        print("MLflow: departure experiment logged.")

    # ── Acceptance check ──
    print(f"\n{'='*60}")
    print(f"Acceptance Check:")
    print(f"  MAE: {test_mae:.1f} min (target ≤ 30) {'✓' if test_mae <= 30 else '✗'}")
    print(f"  Within ±15 min: {within_15:.1f}% (target ≥ 60%) {'✓' if within_15 >= 60 else '✗'}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts/")
    args = parser.parse_args()
    train_departure_model(args.artifacts_dir)


if __name__ == "__main__":
    main()
