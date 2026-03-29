"""
Evaluate both trained models against baselines and produce final report.

Usage:
    python -m training.evaluate --artifacts-dir artifacts/
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

import mlflow

from training.prepare_data import (
    DEMAND_FEATURE_COLS, DEMAND_TARGET_COLS,
    DEPARTURE_FEATURE_COLS, DEPARTURE_TARGET_COL,
)


def evaluate_models(artifacts_dir: str = "artifacts/"):
    artifacts = Path(artifacts_dir)
    report = {}

    # ── Demand model ──
    print("Evaluating demand forecasting model...")
    with open(artifacts / "demand_model.pkl", "rb") as f:
        demand_models = pickle.load(f)

    test_d = pd.read_parquet(artifacts / "demand_test.parquet")
    X_test = test_d[DEMAND_FEATURE_COLS].values

    for target in DEMAND_TARGET_COLS:
        y_test = test_d[target].values
        y_pred = demand_models[target].predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        report[f"demand_{target}_mae"] = float(mae)
        report[f"demand_{target}_rmse"] = float(rmse)
        print(f"  {target}: MAE={mae:.3f}, RMSE={rmse:.3f}")

    # ── Departure model ──
    print("\nEvaluating departure prediction model...")
    with open(artifacts / "departure_model.pkl", "rb") as f:
        dep_model = pickle.load(f)

    test_p = pd.read_parquet(artifacts / "departure_test.parquet")
    X_test_p = test_p[DEPARTURE_FEATURE_COLS].values
    y_test_p = test_p[DEPARTURE_TARGET_COL].values
    y_pred_p = dep_model.predict(X_test_p)

    dep_mae = mean_absolute_error(y_test_p, y_pred_p)
    dep_rmse = np.sqrt(mean_squared_error(y_test_p, y_pred_p))
    errors = np.abs(y_test_p - y_pred_p)
    within_15 = (errors <= 15).mean() * 100
    within_30 = (errors <= 30).mean() * 100

    report["departure_mae_min"] = float(dep_mae)
    report["departure_rmse_min"] = float(dep_rmse)
    report["departure_within_15min_pct"] = float(within_15)
    report["departure_within_30min_pct"] = float(within_30)

    print(f"  MAE: {dep_mae:.1f} min, RMSE: {dep_rmse:.1f} min")
    print(f"  Within ±15 min: {within_15:.1f}%, ±30 min: {within_30:.1f}%")

    # ── Overall pass/fail ──
    checks = {
        "demand_arrival_mae_pass": bool(report["demand_arrival_count_mae"] <= 1.5),
        "demand_kwh_rmse_pass": bool(report["demand_total_kwh_rmse"] <= 5.0),
        "departure_mae_pass": bool(dep_mae <= 30),
        "departure_15min_pass": bool(within_15 >= 60),
    }
    report["acceptance_checks"] = checks
    report["all_pass"] = all(checks.values())

    print(f"\n{'='*60}")
    print("Acceptance Checks:")
    for check, passed in checks.items():
        print(f"  {check}: {'✓ PASS' if passed else '✗ FAIL'}")
    print(f"\nOverall: {'ALL PASS ✓' if report['all_pass'] else 'SOME FAILED ✗'}")

    with open(artifacts / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # ── MLflow: log evaluation results ──
    mlflow.set_experiment("model-evaluation")
    with mlflow.start_run(run_name="eval-report"):
        eval_metrics = {k: v for k, v in report.items()
                        if isinstance(v, (int, float))}
        mlflow.log_metrics(eval_metrics)
        mlflow.log_artifact(str(artifacts / "eval_report.json"))
        mlflow.log_param("all_pass", report["all_pass"])
        for check, passed in checks.items():
            mlflow.log_param(check, passed)
        print("MLflow: evaluation report logged.")

    print(f"\nReport saved to {artifacts}/eval_report.json")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts/")
    args = parser.parse_args()
    evaluate_models(args.artifacts_dir)


if __name__ == "__main__":
    main()
