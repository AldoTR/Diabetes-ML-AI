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
    gender:int=0
    age:float=54.0
    hypertension:int=0
    heart_disease:int=0
    smoking_history:int=0  
    bmi:float=27.32
    HbA1c_level:float=6.6
    blood_glucose_level:float=80

class OutputData(BaseModel):
    probability: float

@app.post('/predict', response_model=OutputData)
def predict(data: InputData):
    model_input = np.array([v for k, v in data.dict().items()]).reshape(1, -1)
    result = model.predict_proba(model_input)[:, 1][0]
    return {'probability': result}