import pandas as pd

file_path = 'original-diabetes-dataset-prediction.csv'
data = pd.read_csv(file_path)

def replace_gender(row):
    if row['gender'] == 'Female':
        return 0
    elif row['gender'] == 'Male':
        return 1
    else:
        return row['gender']

data['gender'] = data.apply(replace_gender, axis=1)

smoking_mapping = {
    'No Info': 0,
    'never': 1,
    'ever': 2,
    'former': 3,
    'not current': 4,
    'current': 5
}

data['smoking_history'] = data['smoking_history'].map(smoking_mapping)

print(data)

data.to_csv('diabetes-dataset-prediction-rebuilt.csv', index=False)

verification_data = pd.read_csv('diabetes-dataset-prediction-rebuilt.csv')
print(verification_data)



