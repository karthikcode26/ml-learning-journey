"""
MLOps Serving — Step 1: Train and save the model artifact
=========================================================
Before we can SERVE a model, we need a saved model to serve. This trains the
Random Forest churn model (our best from Project 01) and saves it as a file
(a "model artifact") that the API will load.

This is the TRAINING side — run occasionally, offline. The API (serving side)
never trains; it just loads what this produces.

Run from the 03-mlops-serving folder (with .venv activated):

    cd 03-mlops-serving
    python train_model.py

Prerequisite: the prepared churn data from Project 01, i.e.
    ../01-churn-prediction/data/telco_X.csv  and  telco_y.csv
(create them by running 01-churn-prediction/lessons/lesson_04_prepare_data.py)
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the prepared churn data from Project 01 (reuse — don't duplicate).
DATA_DIR = "../01-churn-prediction/data"
X = pd.read_csv(f"{DATA_DIR}/telco_X.csv")
y = pd.read_csv(f"{DATA_DIR}/telco_y.csv").squeeze()
print(f"Loaded churn data: {X.shape[0]} rows, {X.shape[1]} features")

# Train on ALL the data here (for a production model you use everything you have;
# we already validated accuracy with a train/test split back in Project 01).
model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
model.fit(X, y)
print(f"Trained Random Forest ({model.n_estimators} trees).")

# Save the model artifact + the feature column order (needed to align inputs).
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn_rf.joblib")
joblib.dump(list(X.columns), "models/feature_columns.joblib")

size_kb = os.path.getsize("models/churn_rf.joblib") / 1024
print(f"\nSaved artifact -> models/churn_rf.joblib ({size_kb:.0f} KB)")
print("Saved feature order -> models/feature_columns.joblib")
print("\nThis file is what the API will load. Training is now DONE and separate")
print("from serving — the API never retrains, it just uses this artifact.")
