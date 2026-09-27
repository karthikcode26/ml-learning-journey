"""
MLOps Serving — Step 2: The FastAPI prediction service
=======================================================
This is the SERVING side: a small web API that LOADS the saved model once, then
answers prediction requests over HTTP. Other systems (a website, a CRM, etc.)
send a customer's details and get back a churn prediction.

Key idea: the model is loaded ONCE at startup (fast), then reused for every
request. No retraining per request.

Run from the 03-mlops-serving folder (with .venv activated), AFTER train_model.py:

    cd 03-mlops-serving
    uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs  for an interactive test UI.
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --- Load the model artifact ONCE when the server starts ---
model = joblib.load("models/churn_rf.joblib")
feature_columns = joblib.load("models/feature_columns.joblib")

app = FastAPI(title="Churn Prediction API", version="1.0")


# --- Define the shape of the input a caller must send ---
# We accept the raw, human-friendly customer fields (a subset of the big
# one-hot feature list). Pydantic validates types automatically.
class Customer(BaseModel):
    tenure: int                 # months as a customer
    MonthlyCharges: float       # monthly bill ($)
    TotalCharges: float         # total billed so far ($)
    Contract: str               # "Month-to-month" | "One year" | "Two year"
    InternetService: str        # "DSL" | "Fiber optic" | "No"
    PaymentMethod: str          # e.g. "Electronic check"

    # A ready-to-use example shown in the /docs UI.
    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 2,
                "MonthlyCharges": 95.0,
                "TotalCharges": 190.0,
                "Contract": "Month-to-month",
                "InternetService": "Fiber optic",
                "PaymentMethod": "Electronic check",
            }
        }
    }


def to_feature_row(c: Customer) -> pd.DataFrame:
    """Turn the friendly input into the exact one-hot feature row the model
    expects. We build an all-zero row with the trained columns, then set the
    fields that apply. (This mirrors the one-hot encoding from Project 01.)"""
    row = {col: 0 for col in feature_columns}
    # numeric features (names match the trained columns)
    if "tenure" in row: row["tenure"] = c.tenure
    if "MonthlyCharges" in row: row["MonthlyCharges"] = c.MonthlyCharges
    if "TotalCharges" in row: row["TotalCharges"] = c.TotalCharges
    # categorical -> the matching one-hot column, e.g. "Contract_Two year"
    for prefix, value in [
        ("Contract", c.Contract),
        ("InternetService", c.InternetService),
        ("PaymentMethod", c.PaymentMethod),
    ]:
        colname = f"{prefix}_{value}"
        if colname in row:
            row[colname] = 1
        # if colname isn't a trained column, it's the dropped baseline -> stays 0
    return pd.DataFrame([row])[feature_columns]


@app.get("/")
def home():
    """A simple root so you can confirm the API is up."""
    return {"service": "Churn Prediction API", "docs": "/docs", "status": "ok"}


@app.get("/health")
def health():
    """A health check — used by monitoring/load balancers in real systems."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
def predict(customer: Customer):
    """Take one customer, return churn probability + a yes/no prediction."""
    X_row = to_feature_row(customer)
    prob = float(model.predict_proba(X_row)[0][1])   # P(churn)
    label = int(prob >= 0.5)
    return {
        "churn_probability": round(prob, 3),
        "will_churn": bool(label),
        "verdict": "likely to churn" if label else "likely to stay",
    }
