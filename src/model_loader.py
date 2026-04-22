"""
src/model_loader.py
-------------------
Loads the latest Production model from the MLflow Model Registry.
Falls back to the local models/ directory if MLflow is unavailable.
"""

import logging
import os

import joblib
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "churn-prediction-model")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MODEL_DIR  = os.getenv("MODEL_DIR", "models")


def load_model_from_registry(stage: str = "Production"):
    """
    Load model from MLflow Model Registry.
    Tries 'Production' stage first, then falls back to latest version.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        model_uri = f"models:/{MODEL_NAME}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"✅ Loaded model '{MODEL_NAME}' @ stage='{stage}' from MLflow registry")
        return model
    except Exception as e:
        logger.warning(f"⚠️  Could not load '{stage}' stage: {e}. Trying latest version…")
        try:
            model_uri = f"models:/{MODEL_NAME}/latest"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✅ Loaded latest model version from MLflow registry")
            return model
        except Exception as e2:
            logger.warning(f"⚠️  MLflow registry unavailable: {e2}. Falling back to local model.")
            return None


def load_model_local():
    """Fallback: load the .pkl file directly from models/ directory."""
    candidates = [
        os.path.join(MODEL_DIR, "model.pkl"),
        os.path.join(MODEL_DIR, "churn_model.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            model = joblib.load(path)
            logger.info(f"✅ Loaded local model from {path}")
            return model
    raise FileNotFoundError(
        f"No model found in MLflow registry or local paths: {candidates}"
    )


def load_scaler():
    """Load the scaler saved alongside the model."""
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    logger.warning("No scaler found — predictions will use raw features.")
    return None


def load_feature_names():
    """Load the feature list saved during training."""
    feat_path = os.path.join(MODEL_DIR, "features.txt")
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            return [line.strip() for line in f if line.strip()]
    return None


def get_model_and_scaler():
    """
    Primary entry point used by the FastAPI app.
    Returns (model, scaler, feature_names).
    """
    model = load_model_from_registry() or load_model_local()
    scaler = load_scaler()
    features = load_feature_names()
    return model, scaler, features