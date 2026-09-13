"""
Lesson 5d — What did model.fit() actually BUILD?
================================================
After `model.fit(X_train, y_train)`, the trained model lives inside the `model`
object. It is nothing more than:
    - model.coef_       -> the learned WEIGHTS (one number per feature)
    - model.intercept_  -> the learned BIAS  (one baseline number)

This script trains the model, then PRINTS everything it built, so you can see
that "the trained model" is just a handful of numbers.

Run from the project root (with .venv activated):

    python lessons/lesson_05d_what_fit_built.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load + split (same as before)
X = pd.read_csv("data/telco_X.csv")
y = pd.read_csv("data/telco_y.csv").squeeze()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# The line you asked about:
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)   # <-- AFTER this line, the model is "built"

# ---------------------------------------------------------------
# So... what did it build?  Let's look inside the `model` object.
# ---------------------------------------------------------------

print("=" * 60)
print("1. THE BIAS (one baseline number the model learned)")
print("=" * 60)
print("   model.intercept_ =", model.intercept_)
print()

print("=" * 60)
print("2. THE WEIGHTS (one number per feature)")
print("=" * 60)
print("   How many weights did it build? ->", len(model.coef_[0]))
print("   (One per feature, matching the", X.shape[1], "columns in X)")
print()
print("   The raw list of learned weights:")
print("  ", [round(w, 3) for w in model.coef_[0]])
print()

print("=" * 60)
print("3. THE WEIGHTS, paired with their feature names (readable)")
print("=" * 60)
weights = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("   (Positive weight pushes toward CHURN, negative toward STAY)\n")
for name, w in weights.items():
    direction = "-> churn" if w > 0 else "-> stay"
    print(f"     {name:32s} {w:+.3f}  {direction}")
print()

print("=" * 60)
print("4. THAT'S THE WHOLE MODEL.")
print("=" * 60)
print(f"   The trained model = {len(model.coef_[0])} weights + 1 bias = "
      f"{len(model.coef_[0]) + 1} numbers.")
print("   To predict a new customer:  sigmoid(bias + sum(weight_i * feature_i))")
print("   Nothing else is stored. No copy of the training data. Just numbers.")
