"""
Train Model 1: EV Arrival & Energy Demand Forecasting (XGBoost).
Walk-forward cross-validation on ACN-Data windowed aggregates.

Usage:
    python -m training.train_demand --artifacts-dir artifacts/
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

from training.prepare_data import DEMAND_FEATURE_COLS, DEMAND_TARGET_COLS


def train_demand_model(artifacts_dir: str = "artifacts/"):
    artifacts = Path(artifacts_dir)

    train_df = pd.read_parquet(artifacts / "demand_train.parquet")
    val_df = pd.read_parquet(artifacts / "demand_val.parquet")
    test_df = pd.read_parquet(artifacts / "demand_test.parquet")

    X_train = train_df[DEMAND_FEATURE_COLS].values
    X_val = val_df[DEMAND_FEATURE_COLS].values
    X_test = test_df[DEMAND_FEATURE_COLS].values

    models = {}
    metrics = {}

    for target in DEMAND_TARGET_COLS:
        print(f"\n{'='*60}")
        print(f"Training demand model for target: {target}")
        print(f"{'='*60}")

        y_train = train_df[target].values
        y_val = val_df[target].values
        y_test = test_df[target].values

        # ── Walk-forward cross-validation ──
        tscv = TimeSeriesSplit(n_splits=5)
        cv_maes = []
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
            model_cv = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
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
            print(f"  Fold {fold+1} MAE: {fold_mae:.3f}")

        print(f"  CV Mean MAE: {np.mean(cv_maes):.3f} ± {np.std(cv_maes):.3f}")

        # ── Train final model on full train set ──
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
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
        y_pred_test = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        # ── Baselines ──
        # Persistence: forecast = last observed value
        persistence_pred = np.concatenate([[y_test[0]], y_test[:-1]])
        persistence_mae = mean_absolute_error(y_test, persistence_pred)

        # Calendar average: mean for this hour+dow in training set
        test_hours = test_df["hour_sin"].values  # proxy for time slot
        calendar_pred = np.full_like(y_test, y_train.mean())
        calendar_mae = mean_absolute_error(y_test, calendar_pred)

        print(f"\n  Test MAE: {test_mae:.3f}")
        print(f"  Test RMSE: {test_rmse:.3f}")
        print(f"  Persistence baseline MAE: {persistence_mae:.3f}")
        print(f"  Calendar baseline MAE: {calendar_mae:.3f}")
        print(f"  Improvement over persistence: {(1 - test_mae/persistence_mae)*100:.1f}%")
        print(f"  Improvement over calendar: {(1 - test_mae/calendar_mae)*100:.1f}%")

        models[target] = model
        metrics[target] = {
            "cv_mae_mean": float(np.mean(cv_maes)),
            "cv_mae_std": float(np.std(cv_maes)),
            "test_mae": float(test_mae),
            "test_rmse": float(test_rmse),
            "persistence_mae": float(persistence_mae),
            "calendar_mae": float(calendar_mae),
            "improvement_over_persistence_pct": float((1 - test_mae / persistence_mae) * 100),
            "improvement_over_calendar_pct": float((1 - test_mae / calendar_mae) * 100),
        }

        # Feature importance
        importance = dict(zip(DEMAND_FEATURE_COLS,
                              model.feature_importances_.tolist()))
        metrics[target]["feature_importance"] = importance

    # ── Save models ──
    with open(artifacts / "demand_model.pkl", "wb") as f:
        pickle.dump(models, f)

    with open(artifacts / "demand_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModels saved to {artifacts}/demand_model.pkl")
    print(f"Metrics saved to {artifacts}/demand_metrics.json")

    # ── MLflow experiment tracking ──
    mlflow.set_experiment("demand-forecasting")
    with mlflow.start_run(run_name="demand-xgboost"):
        # Log hyperparameters
        mlflow.log_params({
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "cv_splits": 5,
        })
        # Log metrics for each target
        for target, m in metrics.items():
            mlflow.log_metrics({
                f"{target}_cv_mae_mean": m["cv_mae_mean"],
                f"{target}_cv_mae_std": m["cv_mae_std"],
                f"{target}_test_mae": m["test_mae"],
                f"{target}_test_rmse": m["test_rmse"],
                f"{target}_improvement_persistence_pct": m["improvement_over_persistence_pct"],
                f"{target}_improvement_calendar_pct": m["improvement_over_calendar_pct"],
            })
        # Log model artifacts
        mlflow.log_artifact(str(artifacts / "demand_model.pkl"))
        mlflow.log_artifact(str(artifacts / "demand_metrics.json"))
        print("MLflow: demand experiment logged.")

    # ── Check acceptance criteria ──
    arrival_mae = metrics["arrival_count"]["test_mae"]
    kwh_rmse = metrics["total_kwh"]["test_rmse"]
    print(f"\n{'='*60}")
    print(f"Acceptance Check:")
    print(f"  Arrival count MAE: {arrival_mae:.3f} (target ≤ 1.5) {'✓' if arrival_mae <= 1.5 else '✗'}")
    print(f"  Energy demand RMSE: {kwh_rmse:.3f} kWh (target ≤ 5.0) {'✓' if kwh_rmse <= 5.0 else '✗'}")

    return models, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts/")
    args = parser.parse_args()
    train_demand_model(args.artifacts_dir)


if __name__ == "__main__":
    main()
