import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("../dataset/fire_dataset_advanced.csv")

# Features and target
X = df[['Temperature', 'Pressure', 'Humidity']]
y = df['Fire_Risk']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import joblib

joblib.dump(model, "../saved_models/fire_model.pkl")
print("Fire model saved")