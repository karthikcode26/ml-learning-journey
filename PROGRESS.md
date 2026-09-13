# Progress Log

A running checkpoint of the ML learning journey, so we can pick up easily.

## ✅ Completed so far

- **Setup** — Python venv + pandas/scikit-learn installed on laptop (Mac).
- **Lesson 1** — What ML is: rules-from-data; features vs. label vs. ignore.
- **Lesson 2** — Loaded the real Telco Churn dataset (7,043 rows, 21 cols).
- **Lesson 3** — Explored data: found the hidden `TotalCharges` dirt (11 blank
  rows), saw churners differ (shorter tenure, higher charges) => signal exists.
- **Lesson 4** — Prepared data: cleaned, one-hot encoded, split X/y, saved
  `data/telco_X.csv` and `data/telco_y.csv`.
- **Lesson 5** — Trained first scikit-learn model (~80% acc vs 73.5% baseline).
- **Lesson 5b** — Inspected the model; hand-computed one prediction (matched sklearn).
- **Lesson 5c** — Logistic regression FROM SCRATCH (sigmoid + gradient descent),
  matched sklearn (~80%).
- **Lesson 5d** — Saw what `.fit()` builds = 31 numbers (30 weights + 1 bias).
- **Lesson 6** — Feature scaling + Pipeline (fixed convergence/overflow warnings).
- **Lesson 7** — Save/load the trained model with joblib (train once, predict
  many). First real MLOps concept: separate training from prediction.

## 🧠 Concepts understood

- Trained model = weighted sum -> sigmoid -> threshold; weights are learned.
- Training loop = predict -> error -> nudge weights (gradient descent), repeat.
- Why split train/test; why scale features; why use pipelines.
- Notebooks (incl. SageMaker) vs. scripts: explore vs. production.

## ⏭️ Next up (pick one to start with tomorrow)

- **(a)** Exploratory Jupyter notebook with inline charts (experience notebooks).
- **(b)** Serve the model via a FastAPI web service (Stage 3 — serving).
- **(c)** Precision & recall metrics (the metrics that matter for churn).

## 📌 Environment notes

- Repo: https://github.com/karthikcode26/ml-learning-journey
- Laptop is on Python 3.9; run scripts from the project root as
  `python lessons/lesson_XX_....py` with the `.venv` activated.
- Dataset lives at `data/telco_churn.csv` (downloaded from Kaggle, not in git).
