from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

sample_payload = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5,
    "TotalCharges": 1080.2
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Churn Prediction API is running"


def test_predict():
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "churn_probability" in data
    assert data["prediction"] in [0, 1]