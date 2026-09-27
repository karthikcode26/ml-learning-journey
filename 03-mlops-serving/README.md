# Project 03 — MLOps: Serving a Model as a Web API

Turn a trained model into a **live prediction service** that other systems can
call over HTTP. This is the step that takes a model from "runs in a script" to
"used in production."

> Builds on Project 01 (churn) — we serve the Random Forest model here.

## The idea

```
 TRAIN (offline, occasional)          SERVE (online, constant)
   train_model.py  ──saves──►  models/churn_rf.joblib  ──loaded by──►  app.py (FastAPI)
                                                                          ↓
                                              POST /predict  →  { "churn_probability": 0.78 }
```

Training and serving are **separate**: `train_model.py` produces the model file;
`app.py` just loads it and answers requests (never retrains).

## Setup

Install the new serving dependencies (from the repo root):

```bash
source .venv/bin/activate
pip install -r requirements.txt        # now includes fastapi + uvicorn + joblib
```

## Step 1 — Create the model artifact

The API needs a saved model. First make sure the churn data exists, then train:

```bash
# one-time: create the prepared churn data (if not already there)
cd 01-churn-prediction && python lessons/lesson_04_prepare_data.py && cd ..

cd 03-mlops-serving
python train_model.py         # -> models/churn_rf.joblib
```

## Step 2 — Run the API

```bash
# still inside 03-mlops-serving
uvicorn app:app --reload
```

You'll see it start on `http://127.0.0.1:8000`.

## Step 3 — Test it

**Easiest: the built-in interactive docs.** Open in your browser:

```
http://127.0.0.1:8000/docs
```

FastAPI auto-generates a UI where you can fill in a customer and click "Execute".

**Or with curl** (in a second terminal):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 2, "MonthlyCharges": 95.0, "TotalCharges": 190.0,
       "Contract": "Month-to-month", "InternetService": "Fiber optic",
       "PaymentMethod": "Electronic check"}'
```

Expected response (roughly):

```json
{ "churn_probability": 0.7x, "will_churn": true, "verdict": "likely to churn" }
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | confirm the service is up |
| GET | `/health` | health check (for monitoring) |
| POST | `/predict` | send a customer, get a churn prediction |
| GET | `/docs` | interactive Swagger UI (auto-generated) |

> ML type: serving a **supervised** model. Next up: monitoring & drift, then
> feature engineering (Project 01).
