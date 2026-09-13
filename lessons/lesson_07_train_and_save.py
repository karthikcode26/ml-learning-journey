"""
Lesson 7 — Train ONCE, save the model to a file
================================================
The key MLOps idea: SEPARATE training from predicting.

  - Training is expensive -> do it ONCE, then SAVE the trained model to disk.
  - Predicting is cheap  -> later, LOAD the saved model and predict instantly,
                             with NO retraining.

This script does the "train once and save" half. The saved file
(models/churn_model.joblib) is called a MODEL ARTIFACT.

Run from the project root (with .venv activated):

    python lessons/lesson_07_train_and_save.py
"""

import os
import joblib          # comes with scikit-learn; saves/loads Python objects
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# --- Train (same as Lesson 6: scale + logistic regression in a pipeline) ---
X = pd.read_csv("data/telco_X.csv")
y = pd.read_csv("data/telco_y.csv").squeeze()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print(f"Trained. Test accuracy: {model.score(X_test, y_test):.1%}")

# --- SAVE the trained model to a file ---------------------------------------
os.makedirs("models", exist_ok=True)
model_path = "models/churn_model.joblib"
joblib.dump(model, model_path)

size_kb = os.path.getsize(model_path) / 1024
print(f"\nSaved trained model -> {model_path}  ({size_kb:.1f} KB)")
print("This small file IS the model. No training data inside — just the")
print("learned numbers + the scaling info. Ship this file anywhere to predict.")

# Also save the feature column order, so predictions line up correctly later.
joblib.dump(list(X.columns), "models/feature_columns.joblib")
print("Saved feature column order -> models/feature_columns.joblib")
