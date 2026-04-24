import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "churn_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

app = FastAPI(title="Azure MLOps Churn Prediction API")

model  = None
scaler = None

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

def get_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model

def get_scaler():
    global scaler
    if scaler is None and os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    return scaler

class CustomerRequest(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def health_check():
    return {"message": "Churn Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(request: CustomerRequest):
    input_df = pd.DataFrame([request.model_dump()])

    # Encode categorical string columns to numbers
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        input_df[col] = le.fit_transform(input_df[col].astype(str))

    # Scale if scaler exists
    s = get_scaler()
    if s is not None:
        input_df = pd.DataFrame(s.transform(input_df), columns=input_df.columns)

    # Use get_model() not bare model variable
    m = get_model()
    prediction  = m.predict(input_df)[0]
    probability = m.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }