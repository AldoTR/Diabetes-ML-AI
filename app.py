from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title="Diabetes Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load(pathlib.Path('model/diabetes-prediction-model.joblib'))

class InputData(BaseModel):
    gender: int
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: int  
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

class OutputData(BaseModel):
    probability: float

@app.post('/predict', response_model=OutputData)
def predict(data: InputData):
    model_input = np.array([v for k, v in data.dict().items()]).reshape(1, -1)
    result = model.predict_proba(model_input)[:, 1][0]
    return {'probability': result}