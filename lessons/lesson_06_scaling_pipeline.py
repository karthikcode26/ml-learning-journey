"""
Lesson 6 — Feature scaling & Pipelines (fixing the convergence warning)
=======================================================================
The convergence / overflow warnings you saw happen because features are on very
different scales (TotalCharges: 0-8000 vs. one-hot columns: 0 or 1). Gradient
descent struggles with that mix.

THE FIX: "scale" every feature to a comparable range first. We use a scikit-
learn PIPELINE, which chains steps (scale -> train) into one clean object.

    StandardScaler  -> rescales each feature to mean 0, std 1
    Pipeline        -> bundles [scale, model] so they always run together

Run from the project root (with .venv activated):

    python lessons/lesson_06_scaling_pipeline.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

X = pd.read_csv("data/telco_X.csv")
y = pd.read_csv("data/telco_y.csv").squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# A Pipeline = a sequence of steps applied in order.
#   Step 1: StandardScaler puts every feature on the same scale.
#   Step 2: LogisticRegression trains on the scaled features.
# The pipeline handles scaling for train AND test automatically & correctly.
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)

model.fit(X_train, y_train)   # scales, then trains — NO warnings now

acc = model.score(X_test, y_test)
baseline = max(y_test.mean(), 1 - y_test.mean())

print("=== Results (with scaling) ===")
print(f"  Model accuracy : {acc:.1%}")
print(f"  Baseline       : {baseline:.1%}")
print("  (No convergence warnings — the math is now well-behaved.)")

# The trained weights now live inside the pipeline's final step.
lr = model.named_steps["logisticregression"]
weights = pd.Series(lr.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("\n=== Top signals (weights are now comparable thanks to scaling) ===")
for name, w in weights.head(8).items():
    direction = "-> churn" if w > 0 else "-> stay"
    print(f"  {name:32s} {w:+.3f}  {direction}")
