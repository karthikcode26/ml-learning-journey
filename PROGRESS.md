# Progress Log

A running checkpoint of the ML learning journey, so we can pick up easily.

## 📁 Repo structure (reorganized by project)

```
01-churn-prediction/   ← supervised learning (was "lessons/")
02-anomaly-detection/  ← unsupervised learning (was "anomaly-detection/")
```

## ✅ Project 01 — Churn Prediction (Supervised) — COMPLETE

- **L1** What ML is: rules-from-data; features vs. label vs. ignore.
- **L2** Loaded the real Telco Churn dataset (7,043 rows, 21 cols).
- **L3** Explored data: found hidden `TotalCharges` dirt (11 blanks); churners
  differ (shorter tenure, higher charges) => signal exists.
- **L4** Prepared data: cleaned, one-hot encoded, split X/y (telco_X/telco_y.csv).
- **L5** Trained first scikit-learn model (~80% acc vs 73.5% baseline).
- **L5b** Inspected the model; hand-computed a prediction (matched sklearn).
- **L5c** Logistic regression FROM SCRATCH (sigmoid + gradient descent), ~80%.
- **L5d** Saw what `.fit()` builds = 31 numbers (30 weights + 1 bias).
- **L6** Feature scaling + Pipeline (fixed convergence/overflow warnings).
- **L7** Save/load model with joblib (train once, predict many) — first MLOps step.
- **L8** Decision Trees — an intuitive alternative model (readable flowchart).

## ✅ Project 02 — Anomaly Detection (Unsupervised) — COMPLETE

- **L1** Intuition: anomalies sit alone; mean/std/z-score on tiny data.
- **L2** Generated realistic retail dataset (500 txns, ~4% anomalies).
- **L3** z-score method + why it FAILS on relationship anomalies ($500 for 1 item).
- **L4a** Isolation from scratch (random cuts, isolation depth); "forest" = ensemble.
- **L4b** Real scikit-learn IsolationForest; `contamination` dial; `-1`=anomaly.
- **L5** Tuned contamination; saw precision/recall trade-off in a table.

## 🧠 Big concepts understood

- **Two types of ML**: supervised (has labels, e.g. churn) vs. unsupervised
  (no labels, e.g. anomaly detection).
- Trained model = learned numbers; training = gradient descent loop.
- Train/test split, scaling, pipelines, save/load.
- Precision/recall trade-off (churn threshold ≈ anomaly contamination dial).
- Notebooks (incl. SageMaker) vs. scripts: explore vs. production.

## ⏭️ Next up — Step 1: more supervised models

- **L8 Decision Tree** (done) — next: try `max_depth=None` to SEE overfitting.
- **L9 Random Forest** — many trees voting (fixes overfitting).
- **L10 Compare** Logistic Regression vs. Tree vs. Random Forest on churn.
- Then: overfitting/cross-validation, feature engineering, then MLOps
  (experiment tracking → FastAPI serving → monitoring).

## 📌 Environment notes

- Repo: https://github.com/karthikcode26/ml-learning-journey
- Laptop on Python 3.9; **run scripts from INSIDE the project folder** now, e.g.
  `cd 01-churn-prediction && python lessons/lesson_05_train_model.py`
  (with `.venv` activated from the repo root).
- Churn dataset: `01-churn-prediction/data/telco_churn.csv` (Kaggle, not in git).
