"""
Train M4A — Arrivals-per-slot Forecast (HistGradientBoostingRegressor).

Three heads on 15-min slot arrival counts:
  arrivals_q50.joblib  — Poisson mean (best estimate for reservation)
  arrivals_q10.joblib  — optimistic lower bound
  arrivals_q90.joblib  — conservative upper bound (capacity reservation)

The Q90 arrivals forecast is what the optimizer uses to reserve headroom
for cars that haven't arrived yet but are expected within the horizon.

Usage:
    python -m training.train_arrivals --artifacts-dir artifacts/
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from training.prepare_data import ARRIVALS_FEATURE_COLS, ARRIVALS_TARGET_COL

HGBT_PARAMS = dict(
    max_iter=600,
    max_depth=6,
    learning_rate=0.05,
    max_leaf_nodes=63,
    min_samples_leaf=30,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=40,
    random_state=42,
)


def train_arrivals_model(artifacts_dir: str = "artifacts/"):
    artifacts = Path(artifacts_dir)

    train_df = pd.read_parquet(artifacts / "arrivals_train.parquet")
    val_df   = pd.read_parquet(artifacts / "arrivals_val.parquet")
    test_df  = pd.read_parquet(artifacts / "arrivals_test.parquet")

    needed = ARRIVALS_FEATURE_COLS + [ARRIVALS_TARGET_COL]
    train_df = train_df[needed].dropna()
    val_df   = val_df[needed].dropna()
    test_df  = test_df[needed].dropna()

    X_train = train_df[ARRIVALS_FEATURE_COLS].values.astype(float)
    X_val   = val_df[ARRIVALS_FEATURE_COLS].values.astype(float)
    X_test  = test_df[ARRIVALS_FEATURE_COLS].values.astype(float)

    y_train = train_df[ARRIVALS_TARGET_COL].astype(float).values
    y_val   = val_df[ARRIVALS_TARGET_COL].astype(float).values
    y_test  = test_df[ARRIVALS_TARGET_COL].astype(float).values

    print(f"Training M4A Arrivals (HistGBDT, Poisson + quantile heads)")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)} slots")
    print(f"  Mean arrivals/slot (test): {y_test.mean():.3f}")
    print(f"  Non-zero slots: {(y_test > 0).mean()*100:.1f}%")

    models = {}
    for (loss, q, tag) in [
        ("poisson",  None, "q50"),   # Poisson for count mean — best point estimate
        ("quantile", 0.10, "q10"),
        ("quantile", 0.90, "q90"),
    ]:
        print(f"  Fitting {tag}...")
        kwargs = dict(loss=loss, **HGBT_PARAMS)
        if q is not None:
            kwargs["quantile"] = q
        m = HistGradientBoostingRegressor(**kwargs)
        m.fit(X_train, y_train)
        models[tag] = m

    preds_test = {tag: np.maximum(m.predict(X_test), 0) for tag, m in models.items()}

    test_mae  = float(mean_absolute_error(y_test, preds_test["q50"]))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, preds_test["q50"])))
    coverage  = float(
        ((y_test >= preds_test["q10"]) & (y_test <= preds_test["q90"])).mean() * 100
    )

    print(f"\n  Test MAE: {test_mae:.4f} arrivals/slot")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Q10–Q90 coverage: {coverage:.1f}%")

    metrics = {
        "test_mae":             round(test_mae, 4),
        "test_rmse":            round(test_rmse, 4),
        "q10_q90_coverage_pct": round(coverage, 2),
        "mean_arrivals_per_slot": round(float(y_test.mean()), 4),
        "n_features":           len(ARRIVALS_FEATURE_COLS),
        "train_rows":           len(X_train),
        "test_rows":            len(X_test),
    }

    joblib.dump(models["q50"], artifacts / "arrivals_q50.joblib")
    joblib.dump(models["q10"], artifacts / "arrivals_q10.joblib")
    joblib.dump(models["q90"], artifacts / "arrivals_q90.joblib")
    print("  Saved: arrivals_q50.joblib, arrivals_q10.joblib, arrivals_q90.joblib")

    with open(artifacts / "arrivals_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("arrivals-prediction")
        with mlflow.start_run(run_name="arrivals-histgbdt"):
            mlflow.log_params({"algorithm": "HistGradientBoostingRegressor", "loss_q50": "poisson"})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            for tag in ["q50", "q10", "q90"]:
                mlflow.log_artifact(str(artifacts / f"arrivals_{tag}.joblib"))
            mlflow.sklearn.log_model(
                models["q50"],
                artifact_path="arrivals_q50_model",
                registered_model_name="arrivals-predictor",
            )
        print("MLflow: arrivals experiment logged.")
    except Exception as _mlflow_err:
        print(f"MLflow logging skipped: {_mlflow_err}")

    return models, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts/")
    args = parser.parse_args()
    train_arrivals_model(args.artifacts_dir)


if __name__ == "__main__":
    main()
