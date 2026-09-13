"""
Lesson 5 — Train your first real model
=======================================
We finally TRAIN a model on the prepared data, using scikit-learn (the
standard ML library). Then we evaluate it honestly on data it never saw.

The full ML loop:
  1. LOAD the prepared X (features) and y (label)
  2. SPLIT into train + test (never evaluate on data you trained on!)
  3. TRAIN a logistic regression model
  4. PREDICT on the test set
  5. EVALUATE — accuracy vs. the baseline

Run from the project root (with .venv activated):

    python lessons/lesson_05_train_model.py

Prerequisite: run lesson_04 first (it creates data/telco_X.csv & telco_y.csv).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. LOAD the prepared data from Lesson 4.
X = pd.read_csv("data/telco_X.csv")
y = pd.read_csv("data/telco_y.csv").squeeze()  # squeeze: 1-column frame -> series
print("Loaded X:", X.shape, "| y:", y.shape)

# 2. SPLIT: 80% to train on, 20% held out to test on.
#    random_state=42 makes the split reproducible (same split every run).
#    stratify=y keeps the churn ratio (~26%) the same in train and test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

# 3. TRAIN. Two lines — this is what scikit-learn does for you.
#    max_iter is raised so the training math has room to converge.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)   # <-- this is "learning": finding the weights
print("\nModel trained! It learned a weight for each of the",
      X.shape[1], "features.")

# 4. PREDICT on the unseen test set.
y_pred = model.predict(X_test)

# 5. EVALUATE.
accuracy = accuracy_score(y_test, y_pred)

# Baseline: always guess the majority class ("No churn"). What accuracy is that?
baseline = max(y_test.mean(), 1 - y_test.mean())

print("\n=== Results ===")
print(f"  Model accuracy : {accuracy:.1%}")
print(f"  Baseline       : {baseline:.1%}  (always guessing 'no churn')")
verdict = "BEATS the baseline — it learned real patterns!" if accuracy > baseline + 0.01 \
    else "barely beats baseline — needs work."
print(f"  Verdict        : {verdict}")

# Bonus: what did the model think matters most? (largest weights)
print("\n=== Top signals the model learned ===")
weights = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
for name, w in weights.head(6).items():
    direction = "-> churn" if w > 0 else "-> stay"
    print(f"  {name:28s} {w:+.2f}  {direction}")
