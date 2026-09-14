"""
Anomaly Detection — Lesson 1: The Intuition
============================================
An ANOMALY is a data point that is FAR from where most of the data lives.
That's it. Your brain already does this: given
    $20, $25, $22, $19, $24, $2000, $21
you instantly saw $2000 as the odd one out — because it's far from the cluster
around $20.

This script turns that instinct into simple math, step by step, so you can SEE
exactly how a computer measures "far from normal". No ML library yet — just the
idea. (Later lessons use a real algorithm.)

Run it (no special setup needed — pure Python):

    python anomaly-detection/lesson_01_intuition.py
"""

# Our tiny retail dataset: order amounts in dollars.
amounts = [20, 25, 22, 19, 24, 2000, 21]

print("Order amounts:", amounts)
print()

# --- Step 1: where does "most of the data" live? Use the AVERAGE (mean). ---
mean = sum(amounts) / len(amounts)
print(f"Step 1 — Average (mean) amount: ${mean:.2f}")
print("        This is the 'center' — roughly where normal orders sit.")
print("        (Notice the $2000 drags the average up — a hint it's unusual.)")
print()

# --- Step 2: how spread out is the data? Use STANDARD DEVIATION. ---
# Standard deviation = the typical distance of points from the mean.
# Small std = points are tightly clustered; large std = spread out.
variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
std = variance ** 0.5
print(f"Step 2 — Standard deviation (typical spread): ${std:.2f}")
print()

# --- Step 3: for each point, how many 'typical spreads' is it from the mean? ---
# This distance-in-std-units is called a Z-SCORE. Big |z| = far from normal.
print("Step 3 — Z-score for each order (distance from mean, in std units):")
for x in amounts:
    z = (x - mean) / std
    flag = "  <-- ANOMALY (far from normal)" if abs(z) > 2 else ""
    print(f"        ${x:<6} z = {z:+.2f}{flag}")
print()

print("Takeaway: we flagged points more than 2 standard deviations away.")
print("The $2000 order stands out with a huge z-score — exactly what your")
print("brain did, now expressed as math.")
