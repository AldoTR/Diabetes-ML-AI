import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import pathlib

data = pd.read_csv(pathlib.Path('data/original-diabetes-dataset-prediction.csv'))

data['gender'] = data['gender'].map({'Female': 0, 'Male': 1})

smoking_map = {
    "No Info": 0,
    "never": 1,
    "ever": 2,
    "former": 3,
    'not current': 4,
    'current': 5
}

data['smoking_history'] = data['smoking_history'].map(smoking_map)

data.fillna(data.median(), inplace=True)

X = data.drop('diabetes', axis=1)
y = data['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

dump(clf, pathlib.Path('model/diabetes-prediction-model.joblib'))