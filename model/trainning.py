from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('data/diabetes-dataset-prediction-rebuilt.csv')

non_numeric_columns = df.select_dtypes(exclude='number').columns
encoder = OneHotEncoder(sparse=False)
encoded_columns = pd.DataFrame(encoder.fit_transform(df[non_numeric_columns]))
encoded_columns.columns = encoder.get_feature_names(non_numeric_columns)

df = pd.concat([df, encoded_columns], axis=1)
df = df.drop(non_numeric_columns, axis=1)

y = df.pop('diabetes')
X = df

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Entrenando el modelo...')
clf = RandomForestClassifier(n_estimators=9, max_depth=2, random_state=0)
clf.fit(X_train, y_train)

print('Guardando el modelo...')
dump(clf, 'model/diabetes-prediction-v1.joblib')