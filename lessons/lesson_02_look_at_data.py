"""
Lesson 2 — Load and look at real data
=====================================
Your FIRST piece of ML code. We don't build a model yet — we just LOAD the
Telco Churn dataset and look at it, so we understand what we're working with.

Run it from the project root (with your .venv activated):

    python lessons/lesson_02_look_at_data.py

Prerequisite: data/telco_churn.csv must exist (Lesson 1 download step).
"""

import pandas as pd  # pandas = tables for Python. A DataFrame is like a SQL table.

# 1. LOAD the data. read_csv turns the file into a "DataFrame" (df) — a table.
df = pd.read_csv("data/telco_churn.csv")

# 2. HOW BIG is it?  (rows = customers, columns = things we know about them)
print("Shape (rows, columns):", df.shape)
print()

# 3. WHAT COLUMNS do we have?  These are our features + the label.
print("Columns:")
for col in df.columns:
    print("   -", col)
print()

# 4. PEEK at the first 5 customers, so we see real values.
print("First 5 rows:")
print(df.head())
print()

# 5. LOOK AT THE LABEL: how many customers churned vs. stayed?
#    'Churn' is the answer we'll eventually predict.
print("Churn counts (our label):")
print(df["Churn"].value_counts())
print()
print("Churn rate:")
print(df["Churn"].value_counts(normalize=True).round(3))
