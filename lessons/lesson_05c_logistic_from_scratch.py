"""
Lesson 5c — Logistic Regression FROM SCRATCH (no scikit-learn)
==============================================================
This is what scikit-learn's LogisticRegression does internally, written out in
plain Python so nothing is hidden. Two ideas power it:

  1. SIGMOID   — turns any number into a probability between 0 and 1.
  2. GRADIENT DESCENT — the "learning" loop: repeatedly nudge the weights a
     little in the direction that reduces prediction error.

We train it on the SAME Telco data and compare its accuracy to scikit-learn's,
to prove they do the same thing.

Run from the project root (with .venv activated):

    python lessons/lesson_05c_logistic_from_scratch.py
"""

import math
import pandas as pd
from sklearn.model_selection import train_test_split


# ---- THE CORE: sigmoid + gradient descent (the ~40 lines that matter) ----

def sigmoid(z):
    """Squash any real number into (0, 1) -> interpret as a probability."""
    if z < -60:
        return 0.0          # avoid math overflow on huge negatives
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def train(X, y, learning_rate=0.5, epochs=300):
    """Learn one weight per feature (+ a bias) via gradient descent."""
    n_samples = len(X)
    n_features = len(X[0])
    weights = [0.0] * n_features   # start at zero — the model knows nothing yet
    bias = 0.0

    for epoch in range(epochs):
        # Gradients: how much to change each weight to reduce error.
        dw = [0.0] * n_features
        db = 0.0
        for i in range(n_samples):
            # 1. Predict this row: score = w·x + bias, then sigmoid -> prob.
            score = sum(weights[j] * X[i][j] for j in range(n_features)) + bias
            pred = sigmoid(score)
            # 2. Error = how far the prediction is from the true label (0/1).
            error = pred - y[i]
            # 3. Accumulate the gradient contribution of this row.
            for j in range(n_features):
                dw[j] += error * X[i][j]
            db += error
        # 4. Step the weights downhill (reduce error) by learning_rate.
        for j in range(n_features):
            weights[j] -= learning_rate * dw[j] / n_samples
        bias -= learning_rate * db / n_samples

    return weights, bias


def predict(X, weights, bias, threshold=0.5):
    """Turn learned weights into 0/1 churn predictions."""
    out = []
    for row in X:
        score = sum(weights[j] * row[j] for j in range(len(row))) + bias
        out.append(1 if sigmoid(score) >= threshold else 0)
    return out


# ---- Load data, SCALE it, train, and compare to scikit-learn ----

def main():
    X_df = pd.read_csv("data/telco_X.csv")
    y = pd.read_csv("data/telco_y.csv").squeeze().tolist()

    # Feature scaling (standardization): put every feature on a comparable
    # range so gradient descent trains stably. scikit-learn handles this kind
    # of numerical conditioning internally; here we do it explicitly.
    means = X_df.mean()
    stds = X_df.std().replace(0, 1)
    X_scaled = ((X_df - means) / stds).values.tolist()

    Xtr, Xte, ytr, yte = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training from-scratch logistic regression...")
    weights, bias = train(Xtr, ytr, learning_rate=0.5, epochs=300)

    preds = predict(Xte, weights, bias)
    acc = sum(1 for p, t in zip(preds, yte) if p == t) / len(yte)
    print(f"  From-scratch accuracy : {acc:.1%}")

    # Compare with scikit-learn on the same scaled data.
    from sklearn.linear_model import LogisticRegression
    sk = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    sk_acc = sk.score(Xte, yte)
    print(f"  scikit-learn accuracy : {sk_acc:.1%}")
    print("\n  Same idea, same ballpark result — the library is just optimized.")


if __name__ == "__main__":
    main()
