import requests
body = {
    "gender": 1,
    "age": 21.0,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": 0,
    "bmi": 26.1,
    "HbA1c_level": 5.0,
    "blood_glucose_level": 85,
    "diabetes": 0
    }
response = requests.post(url = 'http://127.0.0.1:8000/score',
              json = body)
print (response.json())
# output: {'score': 0.866490130600765}