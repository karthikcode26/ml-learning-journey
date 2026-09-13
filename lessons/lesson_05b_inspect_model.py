"""
Lesson 5b — What IS a trained model? Look inside it.
====================================================
A "trained model" is just a formula with LEARNED NUMBERS (weights) in it.
This script trains the model, then PRINTS those numbers so you can see that a
model is not magic — it's weighted arithmetic.

The prediction formula for logistic regression is:

    score = bias + (w1 * feature1) + (w2 * feature2) + ... + (w30 * feature30)
    churn_probability = sigmoid(score)     # squashes score into 0..1

Run from the project root (with .venv activated):

    python lessons/lesson_05b_inspect_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = pd.read_csv("data/telco_X.csv")
y = pd.read_csv("data/telco_y.csv").squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---- THE TRAINED MODEL, laid bare ----
print("A trained model = a bias + one weight per feature. Here they are:\n")
print(f"  bias (baseline lean) = {model.intercept_[0]:+.3f}\n")

weights = pd.Series(model.coef_[0], index=X.columns)
print("  Every learned weight (positive -> pushes toward CHURN):")
for name, w in weights.items():
    print(f"    {name:28s} {w:+.3f}")

# ---- Use those numbers by hand on ONE customer ----
print("\n--- Let's predict ONE customer by plugging numbers into the formula ---")
customer = X_test.iloc[0]
# score = bias + sum(weight_i * feature_i)  -- exactly what the model computes
score = model.intercept_[0] + (model.coef_[0] * customer.values).sum()
import math
prob = 1 / (1 + math.exp(-score))   # sigmoid: turn score into a probability
print(f"  Hand-computed churn score       : {score:+.3f}")
print(f"  Hand-computed churn probability : {prob:.1%}")
print(f"  scikit-learn's own probability  : {model.predict_proba([customer.values])[0][1]:.1%}")
print("  (They match — proving the model is just this formula!)")
