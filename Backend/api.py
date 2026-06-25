from fastapi import FastAPI
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("rf_model.pkl")

@app.get("/")
def home():
    return {"message": "Stock Prediction API is running"}

from pydantic import BaseModel

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)[0]

    result = "BUY" if prediction == 1 else "SELL"
    return {"prediction": result}