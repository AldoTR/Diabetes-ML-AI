import requests

API_URL = 'http://127.0.0.1:8000/predict'

data = {
    "gender": 0,  
    "age": 54.0,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": 0,  
    "bmi": 27.32,
    "HbA1c_level": 6.6,
    "blood_glucose_level": 80
}

response = requests.post(API_URL, json=data)
print(response.json())