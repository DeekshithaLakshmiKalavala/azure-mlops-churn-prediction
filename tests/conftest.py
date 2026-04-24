import sys
import os
import joblib
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "churn_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# These match the exact sample_payload fields in test_api.py
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
ALL_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def create_dummy_model():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        # Build dummy data with same structure as real churn data
        dummy_data = {
            "gender":            ["Female", "Male"] * 10,
            "Partner":           ["Yes", "No"] * 10,
            "Dependents":        ["No", "Yes"] * 10,
            "PhoneService":      ["Yes", "No"] * 10,
            "MultipleLines":     ["No", "Yes"] * 10,
            "InternetService":   ["Fiber optic", "DSL"] * 10,
            "OnlineSecurity":    ["No", "Yes"] * 10,
            "OnlineBackup":      ["Yes", "No"] * 10,
            "DeviceProtection":  ["No", "Yes"] * 10,
            "TechSupport":       ["No", "Yes"] * 10,
            "StreamingTV":       ["Yes", "No"] * 10,
            "StreamingMovies":   ["Yes", "No"] * 10,
            "Contract":          ["Month-to-month", "Two year"] * 10,
            "PaperlessBilling":  ["Yes", "No"] * 10,
            "PaymentMethod":     ["Electronic check", "Mailed check"] * 10,
            "SeniorCitizen":     [0, 1] * 10,
            "tenure":            list(range(1, 21)),
            "MonthlyCharges":    [float(i * 5) for i in range(1, 21)],
            "TotalCharges":      [float(i * 50) for i in range(1, 21)],
        }
        df = pd.DataFrame(dummy_data)

        # Encode categoricals exactly as app/main.py should
        le = LabelEncoder()
        for col in CATEGORICAL_COLS:
            df[col] = le.fit_transform(df[col])

        X = df[ALL_COLS].values
        y = np.array([0, 1] * 10)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_scaled, y)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)


create_dummy_model()