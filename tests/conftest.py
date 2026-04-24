import sys
import os
import joblib
import numpy as np

# Add project root to path so "from app.main import app" works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Create a dummy model before any test imports app ──────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "churn_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

def create_dummy_model():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        X = np.random.rand(20, 19)
        y = np.random.randint(0, 2, 20)
        model = RandomForestClassifier(n_estimators=2, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)

    if not os.path.exists(SCALER_PATH):
        scaler = StandardScaler()
        scaler.fit(np.random.rand(20, 19))
        joblib.dump(scaler, SCALER_PATH)

create_dummy_model()