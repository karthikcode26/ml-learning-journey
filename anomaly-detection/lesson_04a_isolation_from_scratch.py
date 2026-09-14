"""
Anomaly Detection — Lesson 4a: Isolation, from scratch
======================================================
THE BIG IDEA: anomalies are EASY TO ISOLATE.

Imagine playing "20 questions" by making random cuts through the data:
  "Is amount < 37?"  "Is items < 4?"  ...
Each cut splits the group in two. We keep cutting the group that still contains
our target point, until that point is ALONE (isolated).

  - A WEIRD point (out on its own) gets isolated in just a FEW cuts.
  - A NORMAL point (buried in the crowd) needs MANY cuts to isolate.

So: fewer cuts to isolate  ->  more likely an anomaly.

This script does ONE such "isolation game" per point, averaged over many random
trees, using pure Python so nothing is hidden. (scikit-learn's IsolationForest
does exactly this idea, optimized — next lesson.)

Run (pure Python, no libraries needed):

    python anomaly-detection/lesson_04a_isolation_from_scratch.py

Prerequisite: run lesson_02 first (creates transactions.csv).
"""

import csv
import random

random.seed(1)


def load():
    rows = list(csv.DictReader(open("anomaly-detection/transactions.csv")))
    # Each point = [amount, items]; keep the planted label only for checking.
    points = [[float(r["amount"]), float(r["items"])] for r in rows]
    truth = [int(r["is_anomaly"]) for r in rows]
    return points, truth


def isolation_depth(point, data, max_depth=20):
    """Count how many random cuts it takes to isolate `point` from `data`.
    We repeatedly pick a random feature and a random split value, and keep only
    the side of the split that contains our point. Depth = number of cuts."""
    group = data
    for depth in range(1, max_depth + 1):
        if len(group) <= 1:
            return depth                      # isolated!
        # pick a random feature (0 = amount, 1 = items)
        f = random.randint(0, len(point) - 1)
        vals = [row[f] for row in group]
        lo, hi = min(vals), max(vals)
        if lo == hi:
            return depth                      # can't split further -> isolated
        split = random.uniform(lo, hi)        # a random cut on that feature
        # keep the side of the cut where our point lives
        if point[f] < split:
            group = [row for row in group if row[f] < split]
        else:
            group = [row for row in group if row[f] >= split]
    return max_depth


def avg_depth(point, data, trees=50):
    """Average isolation depth over several random trees (more stable).
    More trees = steadier, more accurate scores (that's why it's a FOREST).
    Try changing this number: 15 catches ~17/20, 50 ~18/20, 100 ~20/20."""
    return sum(isolation_depth(point, data) for _ in range(trees)) / trees


def main():
    points, truth = load()

    # Compute an average isolation depth for every point.
    # (Sampling a subset as the "data" each time keeps it fast + is how the
    #  real algorithm works too.)
    depths = []
    for p in points:
        sample = random.sample(points, min(128, len(points)))
        depths.append(avg_depth(p, sample, trees=50))

    # LOW average depth = isolated quickly = likely anomaly.
    # Rank points by depth; the lowest are our anomaly candidates.
    ranked = sorted(range(len(points)), key=lambda i: depths[i])

    n_flag = truth.count(1)   # we'll flag as many as there are real anomalies
    flagged = set(ranked[:n_flag])
    caught = sum(1 for i in flagged if truth[i] == 1)

    print("=== Isolation-from-scratch results ===")
    print(f"  Real anomalies      : {truth.count(1)}")
    print(f"  Points we flagged   : {n_flag} (the lowest-depth = easiest to isolate)")
    print(f"  Correctly caught    : {caught}")
    print()
    print("  Compare: the z-score method (Lesson 3) missed the '$500 for 1 item'")
    print("  ones. Because isolation cuts on BOTH features, those get isolated")
    print("  fast and are caught here.")
    print()
    print("  A few flagged points (amount, items, avg_depth, was_anomaly):")
    for i in ranked[:8]:
        print(f"    ${points[i][0]:>8}  items={int(points[i][1]):>3}  "
              f"depth={depths[i]:.2f}  anomaly={truth[i]}")


if __name__ == "__main__":
    main()
