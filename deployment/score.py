import json
import os
import joblib
import pandas as pd

model = None

def init():
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(model_dir, "churn_model.joblib")
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return {
            "prediction": int(prediction),
            "churn_probability": float(probability)
        }
    except Exception as e:
        return {"error": str(e)}