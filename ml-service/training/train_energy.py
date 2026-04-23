"""
Train M3 — Energy Need Estimator (HistGradientBoostingRegressor).

Three heads trained on log(kwh_delivered):
  energy_model.joblib  — mean/squared-error (best point estimate)
  energy_q10.joblib    — optimistic energy need
  energy_q90.joblib    — conservative energy need (reserve buffer)

Anonymous sessions (59%) have NaN user features → HistGBDT handles natively.
Cold-start masking applied on 20% of identified-user rows.

Usage:
    python -m training.train_energy --artifacts-dir artifacts/
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

from training.prepare_data import ENERGY_FEATURE_COLS, ENERGY_TARGET_COL, apply_cold_start_mask

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


def train_energy_model(artifacts_dir: str = "artifacts/"):
    artifacts = Path(artifacts_dir)

    train_df = pd.read_parquet(artifacts / "energy_train.parquet")
    val_df   = pd.read_parquet(artifacts / "energy_val.parquet")
    test_df  = pd.read_parquet(artifacts / "energy_test.parquet")

    needed = ENERGY_FEATURE_COLS + [ENERGY_TARGET_COL]
    train_df = train_df[needed].dropna(subset=[ENERGY_TARGET_COL])
    val_df   = val_df[needed].dropna(subset=[ENERGY_TARGET_COL])
    test_df  = test_df[needed].dropna(subset=[ENERGY_TARGET_COL])

    X_train = train_df[ENERGY_FEATURE_COLS]
    X_val   = val_df[ENERGY_FEATURE_COLS]
    X_test  = test_df[ENERGY_FEATURE_COLS]

    y_train = train_df[ENERGY_TARGET_COL].values   # log(kWh)
    y_test  = test_df[ENERGY_TARGET_COL].values

    y_test_kwh = np.exp(y_test)

    print(f"Training M3 Energy (HistGBDT, log-kWh target)")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Mean kWh (test): {y_test_kwh.mean():.2f}")

    rng = np.random.default_rng(42)
    X_train_masked = apply_cold_start_mask(X_train, rng)

    models = {}
    for (loss, q, tag) in [
        ("squared_error", None, "mean"),
        ("quantile",     0.10,  "q10"),
        ("quantile",     0.90,  "q90"),
    ]:
        print(f"  Fitting {tag}...")
        kwargs = dict(loss=loss, **HGBT_PARAMS)
        if q is not None:
            kwargs["quantile"] = q
        m = HistGradientBoostingRegressor(**kwargs)
        m.fit(X_train_masked, y_train)
        models[tag] = m

    preds_test = {tag: np.exp(m.predict(X_test)).clip(0) for tag, m in models.items()}

    test_mae  = float(mean_absolute_error(y_test_kwh, preds_test["mean"]))
    test_rmse = float(np.sqrt(mean_squared_error(y_test_kwh, preds_test["mean"])))
    coverage  = float(
        ((y_test_kwh >= preds_test["q10"]) & (y_test_kwh <= preds_test["q90"])).mean() * 100
    )

    print(f"\n  Test MAE: {test_mae:.3f} kWh")
    print(f"  Test RMSE: {test_rmse:.3f} kWh")
    print(f"  Q10–Q90 coverage: {coverage:.1f}%")

    metrics = {
        "test_mae_kwh":         round(test_mae, 4),
        "test_rmse_kwh":        round(test_rmse, 4),
        "q10_q90_coverage_pct": round(coverage, 2),
        "n_features":           len(ENERGY_FEATURE_COLS),
        "train_rows":           len(X_train),
        "test_rows":            len(X_test),
    }

    joblib.dump(models["mean"], artifacts / "energy_model.joblib")
    joblib.dump(models["q10"],  artifacts / "energy_q10.joblib")
    joblib.dump(models["q90"],  artifacts / "energy_q90.joblib")
    print("  Saved: energy_model.joblib, energy_q10.joblib, energy_q90.joblib")

    with open(artifacts / "energy_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("energy-prediction")
        with mlflow.start_run(run_name="energy-histgbdt"):
            mlflow.log_params({"algorithm": "HistGradientBoostingRegressor", "target_space": "log(kWh)"})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            for tag in ["mean", "q10", "q90"]:
                mlflow.log_artifact(str(artifacts / f"energy{'_model' if tag == 'mean' else '_' + tag}.joblib"))
            mlflow.sklearn.log_model(
                models["mean"],
                artifact_path="energy_model",
                registered_model_name="energy-estimator",
            )
        print("MLflow: energy experiment logged.")
    except Exception as _mlflow_err:
        print(f"MLflow logging skipped: {_mlflow_err}")

    return models, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts/")
    args = parser.parse_args()
    train_energy_model(args.artifacts_dir)


if __name__ == "__main__":
    main()
