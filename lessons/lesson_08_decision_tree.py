"""
Lesson 8 — Decision Trees (a new, more intuitive model)
=======================================================
So far you know ONE supervised model: Logistic Regression (a weighted formula).
Now meet a DECISION TREE — the most intuitive model of all. It's literally a
flowchart of yes/no questions:

        Is Contract "month-to-month"?
            /yes                 \no
     tenure < 6 months?        predict: STAY
       /yes      \no
   CHURN      predict: STAY

The tree LEARNS which questions to ask (and in what order) from the data, by
repeatedly picking the split that best separates churners from non-churners.

Great news: swapping models in scikit-learn is trivial — same .fit()/.predict()
as Logistic Regression. Only the import and one line change.

Run (with .venv activated):

    python lessons/lesson_08_decision_tree.py

Prerequisite: run lesson_04_prepare_data.py first (creates telco_X/telco_y.csv).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

X = pd.read_csv("data/telco_X.csv")
y = pd.read_csv("data/telco_y.csv").squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train a Decision Tree. Same 2 lines as Logistic Regression! ---
# max_depth=4 keeps the tree small enough to READ and understand.
# (Without a limit, trees grow huge and memorize the training data.)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
baseline = max(y_test.mean(), 1 - y_test.mean())

print("=== Decision Tree results ===")
print(f"  Train accuracy : {train_acc:.1%}")
print(f"  Test accuracy  : {test_acc:.1%}")
print(f"  Baseline       : {baseline:.1%}")
print()

# --- The BEST part: we can PRINT the actual flowchart it learned ---
print("=== The tree's learned decision rules (top levels) ===")
rules = export_text(model, feature_names=list(X.columns), max_depth=3)
print(rules)

print("How to read it: each line is a yes/no question. Follow the branches down")
print("to a leaf, where 'class' is the prediction (0 = stay, 1 = churn).")
print("The tree LEARNED these questions from the data — you didn't write them.")
