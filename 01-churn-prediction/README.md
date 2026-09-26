# Project 01 — Customer Churn Prediction (Supervised Learning)

Predict **which customers will leave** (churn) using the classic Telco dataset.
This project teaches **supervised learning** — learning from labeled examples
where the answer (churned: Yes/No) is already known.

## What you learn

- The full ML workflow: load → explore → clean → encode → train → evaluate → save
- **Logistic Regression** — built from scratch (sigmoid + gradient descent) AND
  with scikit-learn
- What a trained model actually *is* (weights + bias)
- Feature scaling & scikit-learn Pipelines
- Saving/loading a model (the first MLOps step)
- Decision Trees (an intuitive alternative model)

## Setup (one time)

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

Download the **Telco Customer Churn** dataset (see `data/README.md`) and save it as:

```
01-churn-prediction/data/telco_churn.csv
```

## Run the lessons (in order)

```bash
cd 01-churn-prediction

python lessons/lesson_02_look_at_data.py        # peek at the data
python lessons/lesson_03_explore_features.py    # profile it
python lessons/lesson_04_prepare_data.py        # clean + encode (creates telco_X/telco_y)
python lessons/lesson_05_train_model.py         # train logistic regression
python lessons/lesson_05b_inspect_model.py      # see the weights
python lessons/lesson_05c_logistic_from_scratch.py  # the algorithm, by hand
python lessons/lesson_05d_what_fit_built.py     # what .fit() produced
python lessons/lesson_06_scaling_pipeline.py    # scaling + pipeline
python lessons/lesson_07_train_and_save.py      # save the model
python lessons/lesson_07_load_and_predict.py    # load + predict (no retraining)
python lessons/lesson_08_decision_tree.py       # a different model
```

`lesson_01_what_is_ml.md` is reading, not code.

> ML type: **Supervised** (we have the churn labels to learn from).
