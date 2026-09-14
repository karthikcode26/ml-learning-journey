"""
Anomaly Detection — Lesson 2: Generate & look at realistic retail data
======================================================================
Lesson 1 used 7 hand-picked numbers. Real data has MANY transactions where
anomalies are RARE and HIDDEN. This script creates such a dataset so later
lessons have something realistic to work on.

Each transaction has TWO features (so it's more than one number):
    - amount        : dollars spent
    - items         : number of items in the order

NORMAL orders: small-ish amounts, a few items, and amount roughly tracks items.
ANOMALIES    : weird combos (e.g. huge amount, or many items for tiny amount).

We DO secretly know which are anomalies here (we plant them), ONLY so we can
check our detector later. In real life you usually don't have these labels.

Run it (pure Python, no libraries needed):

    python anomaly-detection/lesson_02_generate_data.py

Output: anomaly-detection/transactions.csv
"""

import csv
import os
import random

random.seed(7)   # reproducible: same dataset every run

N_NORMAL = 480
N_ANOMALY = 20   # ~4% anomalies — rare, like real life
OUT = "anomaly-detection/transactions.csv"


def make_normal():
    """A normal order: 1-8 items, ~ $12 per item, with a little noise."""
    items = random.randint(1, 8)
    amount = round(items * random.uniform(8, 16) + random.uniform(-3, 3), 2)
    return {"amount": max(amount, 1.0), "items": items, "is_anomaly": 0}


def make_anomaly():
    """A weird order — one of a few unusual patterns."""
    kind = random.choice(["huge_amount", "many_items_tiny_amount", "pricey_single"])
    if kind == "huge_amount":
        return {"amount": round(random.uniform(800, 3000), 2),
                "items": random.randint(1, 5), "is_anomaly": 1}
    if kind == "many_items_tiny_amount":
        return {"amount": round(random.uniform(1, 8), 2),
                "items": random.randint(25, 60), "is_anomaly": 1}
    # pricey_single: one item, absurd price
    return {"amount": round(random.uniform(400, 1200), 2),
            "items": 1, "is_anomaly": 1}


def main():
    rows = [make_normal() for _ in range(N_NORMAL)]
    rows += [make_anomaly() for _ in range(N_ANOMALY)]
    random.shuffle(rows)   # mix the anomalies in among the normal ones

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["amount", "items", "is_anomaly"])
        w.writeheader()
        w.writerows(rows)

    # --- Look at what we made ---
    total = len(rows)
    n_anom = sum(r["is_anomaly"] for r in rows)
    amounts = [r["amount"] for r in rows]
    print(f"Wrote {total} transactions to {OUT}")
    print(f"  Anomalies planted: {n_anom} ({n_anom/total:.1%}) — rare, as expected")
    print()
    print("Amount column overview:")
    print(f"  min    ${min(amounts):.2f}")
    print(f"  max    ${max(amounts):.2f}   <-- pulled up by anomalies")
    print(f"  mean   ${sum(amounts)/total:.2f}")
    print()
    print("First 8 transactions (peek):")
    print(f"  {'amount':>9}  {'items':>5}  {'is_anomaly':>10}")
    for r in rows[:8]:
        print(f"  {r['amount']:>9}  {r['items']:>5}  {r['is_anomaly']:>10}")


if __name__ == "__main__":
    main()
