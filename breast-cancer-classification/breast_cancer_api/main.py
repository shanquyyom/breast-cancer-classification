from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load the trained model and scaler
model = joblib.load("model/breast_cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Define input schema
class CancerData(BaseModel):
    features: list  # 30 input features

@app.post("/predict")
def predict(data: CancerData):
    features = np.array(data.features).reshape(1, -1)
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    result = "Malignant" if prediction == 1 else "Benign"
    return {"prediction": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
