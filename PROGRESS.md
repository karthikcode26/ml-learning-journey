# Progress Log

A running checkpoint of the ML learning journey, so we can pick up easily.

## 📁 Repo structure (organized by problem/dataset)

```
01-churn-prediction/   ← supervised learning (churn dataset)
02-anomaly-detection/  ← unsupervised learning (transactions dataset)
03-mlops-serving/      ← serve the churn model as a web API
```

## ✅ Project 01 — Churn Prediction (Supervised) — COMPLETE

- L1 What ML is (rules-from-data; features vs. label vs. ignore).
- L2 Loaded real Telco Churn data (7,043 rows, 21 cols).
- L3 Explored data: hidden TotalCharges dirt; churners differ (short tenure, high charges).
- L4 Cleaned + one-hot encoded + split X/y (creates telco_X/telco_y.csv).
- L5/5b/5c/5d Logistic Regression: trained (~80%), inspected weights, built
  from scratch, saw .fit() builds 31 numbers.
- L6 Feature scaling + Pipeline.
- L7 Save/load model with joblib (train once, predict many).
- L8 Decision Tree — readable flowchart; SAW overfitting (depth=None: 99% train / 72% test!).
- L9 Random Forest — 100 trees vote; best result: 80.2% test, only 2.5% gap.

## ✅ Project 02 — Anomaly Detection (Unsupervised) — COMPLETE

- L1 Intuition (anomalies sit alone; mean/std/z-score).
- L2 Generated retail dataset (500 txns, ~4% anomalies).
- L3 z-score method + why it fails on relationship anomalies.
- L4a Isolation from scratch; L4b real IsolationForest; contamination dial.
- L5 Tuned contamination; precision/recall trade-off table.

## ⏳ Project 03 — MLOps Serving — BUILT, needs testing on laptop

- `train_model.py` — trains RF, saves models/churn_rf.joblib (model artifact).
- `app.py` — FastAPI service: GET /, GET /health, POST /predict, /docs UI.
- Pattern learned: training and serving are SEPARATE (load model once, reuse).
- TO DO when back: on laptop, `pip install -r requirements.txt`, run
  `python train_model.py` then `uvicorn app:app --reload`, test at /docs,
  paste the /predict response.

## 🧠 Big concepts owned

- Two types of ML: supervised (labels) vs. unsupervised (no labels).
- Overfitting = memorizing; tell = big train-test gap; always judge on unseen data.
- Ensembles (forests) = many varied models vote; errors cancel out.
- Random Forest (supervised, predict) vs. Isolation Forest (unsupervised, detect).
- Precision/recall trade-off (churn threshold ≈ anomaly contamination dial).
- Model artifact + separate train/serve = core MLOps.

## ⏭️ Next up (in order the user chose)

1. Finish testing Project 03 API on laptop (paste /predict result).
2. Optionally: monitoring/drift OR Dockerize the API (complete MLOps serving).
3. THEN Feature Engineering (Project 01) — create new features to boost accuracy
   (user's chosen "option 3", to do after MLOps).
4. Later: cross-validation, then more MLOps (experiment tracking, monitoring).

## 📌 Environment notes

- Repo: https://github.com/karthikcode26/ml-learning-journey
- Laptop on Python 3.9; run scripts from INSIDE the project folder, e.g.
  `cd 01-churn-prediction && python lessons/lesson_09_random_forest.py`
  (activate `.venv` from repo root first: `source .venv/bin/activate`).
- Reactivate venv each new terminal: `cd ~/workspace/ml-learning-journey && source .venv/bin/activate`.
- Churn dataset: 01-churn-prediction/data/telco_churn.csv (Kaggle, not in git).
- Data files aren't in git — after moving/cloning code, data stays put (MLOps gotcha).
