"""
Lesson 3 — Explore the features before modeling
================================================
"Know your data before you model it." A good engineer always inspects the data
first. Still NO model — we're building the habit of understanding data.

Run from the project root (with .venv activated):

    python lessons/lesson_03_explore_features.py
"""

import pandas as pd

df = pd.read_csv("data/telco_churn.csv")

print("=" * 60)
print("1. MISSING VALUES — are any cells empty?")
print("=" * 60)
# .isnull() marks empty cells True; .sum() counts them per column.
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "No obvious missing values.")
print()

print("=" * 60)
print("2. COLUMN TYPES — how does pandas see each column?")
print("=" * 60)
# 'object' usually means text/categorical; int64/float64 mean numbers.
print(df.dtypes)
print()

print("=" * 60)
print("3. THE SNEAKY ONE — TotalCharges")
print("=" * 60)
# TotalCharges LOOKS numeric but pandas read it as text ('object').
# That's a red flag: some rows contain a blank/space instead of a number.
print("Reported dtype:", df["TotalCharges"].dtype)
# Try converting to a number; anything that can't convert becomes NaN (missing).
converted = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Rows that FAILED to convert to a number:", converted.isnull().sum())
print()

print("=" * 60)
print("4. NUMERIC FEATURES — min / max / average")
print("=" * 60)
print(df[["tenure", "MonthlyCharges"]].describe().round(2))
print()

print("=" * 60)
print("5. DO CHURNERS DIFFER? — average tenure & charges by churn")
print("=" * 60)
# groupby: split rows by Churn value, then average the numeric columns.
print(df.groupby("Churn")[["tenure", "MonthlyCharges"]].mean().round(2))
