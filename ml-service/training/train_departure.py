"""
Train M1 â€” Departure Time Predictor (HistGradientBoostingRegressor).

Three quantile heads trained on log(duration_min):
  departure_q10.joblib  â€” optimistic departure (car leaves early)
  departure_q50.joblib  â€” median departure planning horizon
  departure_q90.joblib  â€” conservative window (LP presence boundary)

All outputs back-transformed with exp() at inference.

Key improvements over the previous XGBoost approach:
  â€¢ NaN-native: handles anonymous sessions (59%) without imputation
  â€¢ Cold-start masking: 20% of identified-user rows have user features
    masked to NaN during training â†’ model learns good temporal/station
    priors for the cold-start path
  â€¢ Log-space target: predictions always positive, handles heavy right tail

Usage:
    python -m training.train_departure --artifacts-dir artifacts/
"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from training.prepare_data import (
    DEPARTURE_FEATURE_COLS,
    DEPARTURE_TARGET_COL,
    apply_cold_start_mask,
)

# Shared HistGBDT hyperparameters â€” early stopping uses an internal 10% hold-out.
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


def train_departure_model(artifacts_dir: str = "artifacts/"):
    artifacts = Path(artifacts_dir)

    train_df = pd.read_parquet(artifacts / "departure_train.parquet")
    val_df   = pd.read_parquet(artifacts / "departure_val.parquet")
    test_df  = pd.read_parquet(artifacts / "departure_test.parquet")

    needed = DEPARTURE_FEATURE_COLS + [DEPARTURE_TARGET_COL]
    train_df = train_df[needed].dropna(subset=[DEPARTURE_TARGET_COL])
    val_df   = val_df[needed].dropna(subset=[DEPARTURE_TARGET_COL])
    test_df  = test_df[needed].dropna(subset=[DEPARTURE_TARGET_COL])

    X_train = train_df[DEPARTURE_FEATURE_COLS]
    X_val   = val_df[DEPARTURE_FEATURE_COLS]
    X_test  = test_df[DEPARTURE_FEATURE_COLS]

    y_train = train_df[DEPARTURE_TARGET_COL].values   # log(duration_min)
    y_val   = val_df[DEPARTURE_TARGET_COL].values
    y_test  = test_df[DEPARTURE_TARGET_COL].values

    # Back-transform targets for evaluation metrics
    y_val_min  = np.exp(y_val)
    y_test_min = np.exp(y_test)

    print(f"Training M1 Departure (HistGBDT, quantile loss, log-space target)")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Mean stay: {y_test_min.mean():.1f} min ({y_test_min.mean()/60:.1f} hr)")
    anon_pct = float(train_df["is_anonymous"].mean() * 100) if "is_anonymous" in train_df.columns else 0
    print(f"  Anonymous sessions in train: {anon_pct:.1f}%")

    # Apply cold-start masking to training data only
    rng = np.random.default_rng(42)
    X_train_masked = apply_cold_start_mask(X_train, rng)

    models = {}
    preds_val = {}

    for q, tag, loss in [
        (0.10, "q10", "quantile"),
        (0.50, "q50", "quantile"),
        (0.90, "q90", "quantile"),
    ]:
        print(f"  Fitting {tag} (quantile={q})...")
        m = HistGradientBoostingRegressor(loss=loss, quantile=q, **HGBT_PARAMS)
        m.fit(X_train_masked, y_train)
        # Back-transform: predictions are in log(min) space
        preds_val[tag] = np.exp(m.predict(X_val))
        models[tag] = m

    # â”€â”€ Evaluate on test set â”€â”€
    preds_test = {tag: np.exp(m.predict(X_test)) for tag, m in models.items()}

    test_mae   = mean_absolute_error(y_test_min, preds_test["q50"])
    test_rmse  = float(np.sqrt(mean_squared_error(y_test_min, preds_test["q50"])))
    errors_min = np.abs(y_test_min - preds_test["q50"])
    within_15  = float((errors_min <= 15).mean() * 100)
    within_30  = float((errors_min <= 30).mean() * 100)

    # Q10â€“Q90 interval coverage on test set
    coverage = float(
        ((y_test_min >= preds_test["q10"]) & (y_test_min <= preds_test["q90"])).mean() * 100
    )
    avg_interval_width = float(np.mean(preds_test["q90"] - preds_test["q10"]))

    # Baselines
    fixed_4h_mae = float(mean_absolute_error(y_test_min, np.full_like(y_test_min, 240.0)))
    mean_mae     = float(mean_absolute_error(y_test_min, np.full_like(y_test_min, y_train_orig := np.exp(y_train).mean())))

    print(f"\n  Test MAE (Q50): {test_mae:.1f} min ({test_mae/60:.2f} hr)")
    print(f"  Test RMSE:      {test_rmse:.1f} min")
    print(f"  Within Â±15 min: {within_15:.1f}%")
    print(f"  Within Â±30 min: {within_30:.1f}%")
    print(f"  Q10â€“Q90 coverage: {coverage:.1f}%  (avg width {avg_interval_width:.1f} min)")
    print(f"  Improvement over fixed-4h: {(1 - test_mae/fixed_4h_mae)*100:.1f}%")

    metrics = {
        "test_mae_min":           round(float(test_mae), 2),
        "test_rmse_min":          round(test_rmse, 2),
        "within_15min_pct":       round(within_15, 2),
        "within_30min_pct":       round(within_30, 2),
        "q10_q90_coverage_pct":   round(coverage, 2),
        "avg_interval_width_min": round(avg_interval_width, 2),
        "fixed_4h_baseline_mae":  round(fixed_4h_mae, 2),
        "mean_baseline_mae":      round(mean_mae, 2),
        "improvement_over_fixed4h_pct": round((1 - float(test_mae) / fixed_4h_mae) * 100, 1),
        "n_features": len(DEPARTURE_FEATURE_COLS),
        "train_rows": len(X_train),
        "val_rows":   len(X_val),
        "test_rows":  len(X_test),
        "anonymous_pct": round(anon_pct, 1),
    }

    # â”€â”€ Save models â”€â”€
    for tag, m in models.items():
        joblib.dump(m, artifacts / f"departure_{tag}.joblib")
    print(f"\n  Saved: departure_q10.joblib, departure_q50.joblib, departure_q90.joblib")

    with open(artifacts / "departure_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # â”€â”€ MLflow â”€â”€
    mlflow.set_tracking_uri("")
    mlflow.set_experiment("departure-prediction")
    with mlflow.start_run(run_name="departure-histgbdt-quantile"):
        mlflow.log_params({
            "algorithm": "HistGradientBoostingRegressor",
            "loss": "quantile",
            "quantiles": "0.10/0.50/0.90",
            "target_space": "log(duration_min)",
            "cold_start_mask_frac": 0.20,
            **{k: v for k, v in HGBT_PARAMS.items() if not callable(v)},
        })
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        for tag in ["q10", "q50", "q90"]:
            mlflow.log_artifact(str(artifacts / f"departure_{tag}.joblib"))

    print(f"Departure metrics: {artifacts}/departure_metrics.json")
    return models, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts/")
    args = parser.parse_args()
    train_departure_model(args.artifacts_dir)


if __name__ == "__main__":
    main()

