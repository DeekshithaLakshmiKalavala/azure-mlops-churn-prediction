"""
src/train.py
------------
Customer Churn Prediction — Model Training with MLflow Tracking
Tracks: parameters, metrics, artifacts, and registers the best model.
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.getenv("DATA_PATH",  "data/churn.csv")
MODEL_DIR   = os.getenv("MODEL_DIR",  "models")
MODEL_NAME  = os.getenv("MODEL_NAME", "churn-prediction-model")
MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "mlruns")   # local by default
EXPERIMENT  = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")
TEST_SIZE   = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_preprocess(path: str):
    """Load CSV, encode categoricals, return X, y, feature_names."""
    df = pd.read_csv(path)

    # Drop ID-like columns if present
    drop_cols = [c for c in df.columns if c.lower() in ("customerid", "rowid", "id")]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # Encode target  (supports 'Yes/No', 1/0, True/False)
    target_col = next(
        (c for c in df.columns if c.lower() in ("churn", "exited", "churned")),
        df.columns[-1],
    )
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].astype(str))
    X = df.drop(columns=[target_col])

    # Encode remaining categoricals
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.fillna(X.median(numeric_only=True))
    return X, y, list(X.columns)


def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


def train_and_log(model, model_params: dict, X_train, X_test, y_train, y_test,
                  scaler, feature_names: list, run_name: str):
    """Run one MLflow experiment: train → evaluate → log → register."""
    with mlflow.start_run(run_name=run_name):

        # ── Scale ─────────────────────────────────────────────────────────
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # ── Train ─────────────────────────────────────────────────────────
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = (
            model.predict_proba(X_test_s)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # ── Log params ────────────────────────────────────────────────────
        mlflow.log_params(model_params)
        mlflow.log_param("test_size",    TEST_SIZE)
        mlflow.log_param("random_seed",  RANDOM_SEED)
        mlflow.log_param("n_features",   len(feature_names))
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples",  X_test.shape[0])

        # ── Log metrics ───────────────────────────────────────────────────
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # ── Log model + scaler as artifacts ───────────────────────────────
        os.makedirs(MODEL_DIR, exist_ok=True)
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="model")

        # Log feature list for reproducibility
        feat_path = os.path.join(MODEL_DIR, "features.txt")
        with open(feat_path, "w") as f:
            f.write("\n".join(feature_names))
        mlflow.log_artifact(feat_path, artifact_path="model")

        # Log the sklearn model — enables mlflow models serve
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,   # auto-registers & versions
            input_example=pd.DataFrame([X_test[0]], columns=feature_names)
            if hasattr(X_test, "__getitem__")
            else None,
        )

        run_id = mlflow.active_run().info.run_id
        print(f"\n{'─'*55}")
        print(f"  Run: {run_name}  |  Run ID: {run_id[:8]}…")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v:.4f}")
        print(f"{'─'*55}")

        return metrics, run_id


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    print(f"\n🚀 MLflow experiment : {EXPERIMENT}")
    print(f"   Tracking URI       : {MLFLOW_URI}")
    print(f"   Data path          : {DATA_PATH}\n")

    # Load data
    X, y, feature_names = load_and_preprocess(DATA_PATH)
    X_arr = X.values
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # ── Candidate models ──────────────────────────────────────────────────
    candidates = [
        {
            "name":   "LogisticRegression",
            "model":  LogisticRegression(max_iter=500, random_state=RANDOM_SEED),
            "params": {"model_type": "LogisticRegression", "max_iter": 500},
        },
        {
            "name":   "RandomForest_100",
            "model":  RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
            "params": {"model_type": "RandomForestClassifier", "n_estimators": 100},
        },
        {
            "name":   "RandomForest_200",
            "model":  RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=5,
                random_state=RANDOM_SEED
            ),
            "params": {
                "model_type": "RandomForestClassifier",
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
            },
        },
    ]

    results = []
    for c in candidates:
        scaler = StandardScaler()
        metrics, run_id = train_and_log(
            model=c["model"],
            model_params=c["params"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            feature_names=feature_names,
            run_name=c["name"],
        )
        results.append({"name": c["name"], "run_id": run_id, **metrics})

    # ── Pick best model by ROC-AUC (or F1 if prob unavailable) ───────────
    sort_key = "roc_auc" if "roc_auc" in results[0] else "f1"
    best = max(results, key=lambda r: r[sort_key])

    print(f"\n🏆 Best model : {best['name']}")
    print(f"   {sort_key.upper()} : {best[sort_key]:.4f}")
    print(f"\n   View UI   : mlflow ui  (then open http://localhost:5000)")
    print(f"   Run ID    : {best['run_id']}\n")

    # Save best run_id for downstream use (CI/CD, deployment scripts)
    with open(os.path.join(MODEL_DIR, "best_run_id.txt"), "w") as f:
        f.write(best["run_id"])


if __name__ == "__main__":
    main()