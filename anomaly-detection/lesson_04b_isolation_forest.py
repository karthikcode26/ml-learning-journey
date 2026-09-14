"""
Anomaly Detection — Lesson 4b: The REAL IsolationForest (scikit-learn)
======================================================================
Your from-scratch version (Lesson 4a) did the "random cuts, count depth" idea
in ~60 lines. scikit-learn's IsolationForest does exactly that, optimized, in
a few lines. Here we use it and compare — they should agree.

Key parameter:
    contamination = the fraction of the data we EXPECT to be anomalies.
                    We planted ~4%, so we tell it 0.04. This sets how many
                    points it flags (the "sensitivity dial").

Run (with .venv activated):

    python anomaly-detection/lesson_04b_isolation_forest.py

Prerequisite: run lesson_02 first (creates transactions.csv).
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("anomaly-detection/transactions.csv")

# The detector only sees the FEATURES — never the is_anomaly label.
# (Unsupervised: it must find anomalies without being told the answers.)
X = df[["amount", "items"]]

# --- Create and train the model (2 lines) ---
model = IsolationForest(
    contamination=0.04,   # expect ~4% anomalies (we planted 20/500)
    n_estimators=100,     # 100 trees in the forest (remember: more = steadier)
    random_state=42,      # reproducible
)
model.fit(X)              # learns what "normal" looks like from the data

# --- Predict: -1 means anomaly, +1 means normal (scikit-learn's convention) ---
raw = model.predict(X)
df["flagged"] = (raw == -1).astype(int)   # convert to 1 = anomaly, 0 = normal

# --- Compare our flags to the planted truth (for grading only) ---
truth = df["is_anomaly"]
pred = df["flagged"]
caught = ((pred == 1) & (truth == 1)).sum()
missed = ((pred == 0) & (truth == 1)).sum()
false_alarm = ((pred == 1) & (truth == 0)).sum()

print("=== scikit-learn IsolationForest results ===")
print(f"  Real anomalies : {truth.sum()}")
print(f"  Caught         : {caught}")
print(f"  Missed         : {missed}")
print(f"  False alarms   : {false_alarm}")
print()
print("  (Lesson 3 z-score caught 17 and missed the relationship anomalies.")
print("   IsolationForest sees both features together, like your 4a version.)")
print()

# --- Show the anomaly SCORES: lower = more anomalous (like low 'depth') ---
df["score"] = model.score_samples(X)   # higher = more normal, lower = more anomalous
print("Most anomalous transactions (lowest scores):")
print(f"  {'amount':>9}  {'items':>5}  {'score':>7}  {'was_anomaly':>11}")
for _, r in df.nsmallest(8, "score").iterrows():
    print(f"  {r['amount']:>9}  {int(r['items']):>5}  {r['score']:>7.3f}  {int(r['is_anomaly']):>11}")
