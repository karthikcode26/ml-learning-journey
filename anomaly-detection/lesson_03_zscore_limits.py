"""
Anomaly Detection — Lesson 3: The z-score method, and where it FAILS
====================================================================
We apply Lesson 1's z-score idea to the full 500-transaction dataset, checking
each feature (amount, items) on its own. It will catch the obvious anomalies...
but MISS the sneaky "relationship" ones (e.g. $1117 for 1 item), because each
feature ALONE looks normal there.

Seeing this failure is the whole point — it motivates a smarter algorithm next.

Run (with .venv activated):

    python anomaly-detection/lesson_03_zscore_limits.py

Prerequisite: run lesson_02 first (creates transactions.csv).
"""

import pandas as pd

df = pd.read_csv("anomaly-detection/transactions.csv")

# --- z-score for a single column: (value - mean) / std ---
def zscores(series):
    return (series - series.mean()) / series.std()

# Flag a row if EITHER feature is more than 3 std from its own mean.
z_amount = zscores(df["amount"])
z_items = zscores(df["items"])
df["flagged"] = ((z_amount.abs() > 3) | (z_items.abs() > 3)).astype(int)

# --- How did we do? Compare our flags to the planted truth (is_anomaly). ---
truth = df["is_anomaly"]
pred = df["flagged"]

caught = ((pred == 1) & (truth == 1)).sum()   # real anomalies we flagged
missed = ((pred == 0) & (truth == 1)).sum()   # real anomalies we MISSED
false_alarm = ((pred == 1) & (truth == 0)).sum()  # normal wrongly flagged
total_anom = truth.sum()

print("=== Z-score method results (per-feature, threshold |z|>3) ===")
print(f"  Real anomalies in data : {total_anom}")
print(f"  Caught                 : {caught}")
print(f"  MISSED                 : {missed}   <-- the problem")
print(f"  False alarms           : {false_alarm}")
print()

# --- Show some anomalies we MISSED, to understand WHY ---
missed_rows = df[(pred == 0) & (truth == 1)]
print("Examples of MISSED anomalies (and why each feature looked 'normal'):")
print(f"  {'amount':>9}  {'items':>5}  {'z_amount':>9}  {'z_items':>8}")
for idx in missed_rows.index[:6]:
    print(f"  {df.loc[idx,'amount']:>9}  {df.loc[idx,'items']:>5}  "
          f"{z_amount[idx]:>9.2f}  {z_items[idx]:>8.2f}")
print()
print("Notice: for these, BOTH z-scores are small (each feature looks normal)")
print("individually — the weirdness is in the RELATIONSHIP between them, which")
print("a per-feature method simply cannot see. That's why we need a smarter one.")
