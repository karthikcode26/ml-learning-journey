"""
Lesson 9 — Random Forest (many trees voting)
=============================================
A single Decision Tree is easy to read but tends to OVERFIT (memorize the
training data). The fix: build MANY trees, each slightly different, and let
them VOTE. That's a RANDOM FOREST.

You already met this idea! Isolation Forest (anomaly project) averaged many
random trees. Same principle here: many weak/varied trees combine into one
strong, stable model. This combining is called an ENSEMBLE.

Why the trees differ from each other:
  - each tree trains on a random sample of the rows (bootstrap), and
  - at each split it considers only a random subset of the features.
So no two trees are identical -> their mistakes cancel out when they vote.

Run from the 01-churn-prediction folder (with .venv activated):

    cd 01-churn-prediction
    python lessons/lesson_09_random_forest.py

Prerequisite: run lesson_04_prepare_data.py first (creates telco_X/telco_y.csv).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = pd.read_csv("data/telco_X.csv")
y = pd.read_csv("data/telco_y.csv").squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train a Random Forest. Same .fit()/.score() as always! ---
# n_estimators = how many trees vote (100 is a common default).
# max_depth kept moderate so each tree stays reasonable.
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
baseline = max(y_test.mean(), 1 - y_test.mean())

print("=== Random Forest results ===")
print(f"  Trees in the forest : {model.n_estimators}")
print(f"  Train accuracy      : {train_acc:.1%}")
print(f"  Test accuracy       : {test_acc:.1%}")
print(f"  Baseline            : {baseline:.1%}")
print(f"  Train-test gap      : {train_acc - test_acc:.1%}  (smaller = less overfitting)")
print()

# --- Feature importance: which features did the forest rely on most? ---
# (A forest can't print one flowchart, but it CAN rank feature usefulness.)
importances = pd.Series(model.feature_importances_, index=X.columns)
print("=== Top 8 most important features (across all trees) ===")
for name, imp in importances.sort_values(ascending=False).head(8).items():
    bar = "#" * int(imp * 100)
    print(f"  {name:32s} {imp:.3f}  {bar}")
print()
print("Note: unlike one tree, a forest gives IMPORTANCE (how useful each")
print("feature was) rather than a single readable flowchart. You trade some")
print("interpretability for better, more stable accuracy.")
