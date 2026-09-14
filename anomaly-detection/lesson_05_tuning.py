"""
Anomaly Detection — Lesson 5: Tuning the dial & measuring the trade-off
=======================================================================
You understand the miss vs. false-alarm trade-off. Now SEE it in numbers.
We run IsolationForest at several `contamination` settings and watch how
catches, misses, and false alarms change.

We also introduce the two metrics that name this trade-off (same as churn!):
    PRECISION = of the points we FLAGGED, how many were truly anomalies?
                (high precision = few false alarms)
    RECALL    = of the REAL anomalies, how many did we catch?
                (high recall = few misses)

Run (with .venv activated):

    python anomaly-detection/lesson_05_tuning.py

Prerequisite: run lesson_02 first (creates transactions.csv).
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("anomaly-detection/transactions.csv")
X = df[["amount", "items"]]
truth = df["is_anomaly"]

print(f"Dataset: {len(df)} transactions, {truth.sum()} real anomalies "
      f"({truth.mean():.1%})\n")

print(f"{'contamination':>13} | {'caught':>6} | {'missed':>6} | "
      f"{'false_alarm':>11} | {'precision':>9} | {'recall':>6}")
print("-" * 70)

for c in [0.01, 0.02, 0.04, 0.08, 0.15]:
    model = IsolationForest(contamination=c, n_estimators=100, random_state=42)
    pred = (model.fit_predict(X) == -1).astype(int)   # 1 = anomaly

    caught = int(((pred == 1) & (truth == 1)).sum())
    missed = int(((pred == 0) & (truth == 1)).sum())
    false_alarm = int(((pred == 1) & (truth == 0)).sum())
    flagged = caught + false_alarm

    precision = caught / flagged if flagged else 0.0
    recall = caught / truth.sum() if truth.sum() else 0.0

    print(f"{c:>13} | {caught:>6} | {missed:>6} | {false_alarm:>11} | "
          f"{precision:>8.0%} | {recall:>5.0%}")

print()
print("Read the trade-off DOWN the table:")
print("  - Low contamination  -> flag few  -> more MISSES, fewer false alarms.")
print("  - High contamination -> flag many -> fewer misses, more FALSE ALARMS.")
print("  The 'best' setting depends on the business cost of each mistake.")
