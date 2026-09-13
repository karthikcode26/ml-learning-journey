"""
Lesson 4 — Prepare the data for modeling
=========================================
Models eat CLEAN NUMBERS only. This script does the 3 prep jobs:
  1. CLEAN   — fix the 11 broken TotalCharges rows
  2. ENCODE  — turn text/categorical columns into numbers
  3. SPLIT   — separate features (X) from label (y)

Run from the project root (with .venv activated):

    python lessons/lesson_04_prepare_data.py
"""

import pandas as pd

df = pd.read_csv("data/telco_churn.csv")
print("Starting shape:", df.shape)

# ---------------------------------------------------------------
# JOB 0: Drop the column with no predictive value.
# ---------------------------------------------------------------
# customerID is just an identifier — noise for a model.
df = df.drop(columns=["customerID"])

# ---------------------------------------------------------------
# JOB 1: CLEAN — fix TotalCharges (text with 11 blank cells).
# ---------------------------------------------------------------
# Convert to numbers; blanks become NaN (true "missing").
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Missing TotalCharges after conversion:", df["TotalCharges"].isnull().sum())

# Those 11 are brand-new customers (tenure 0), so total billed = 0. Fill with 0.
df["TotalCharges"] = df["TotalCharges"].fillna(0)
print("Missing after fill:", df["TotalCharges"].isnull().sum())

# ---------------------------------------------------------------
# JOB 2: ENCODE — convert text columns into numbers.
# ---------------------------------------------------------------
# First, separate the LABEL (y) from the FEATURES (X).
# The label 'Churn' is Yes/No -> convert to 1/0.
y = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])

# For the features, use ONE-HOT ENCODING: each category becomes its own 0/1
# column. e.g. Contract -> Contract_Month-to-month, Contract_One year, ...
# pd.get_dummies does this automatically for every text column.
X = pd.get_dummies(X, drop_first=True)

# Convert any True/False columns to 1/0 integers (cleaner for models).
X = X.astype(int) if X.select_dtypes(include="bool").shape[1] else X

print("\nAfter encoding:")
print("  X shape (rows, features):", X.shape)
print("  Number of features grew because each category became its own column.")
print("\nFirst 8 feature columns now look like:")
print(list(X.columns[:8]))

# ---------------------------------------------------------------
# JOB 3: RESULT — X (numbers only) and y (0/1 label).
# ---------------------------------------------------------------
print("\nLabel y — churn counts:")
print(y.value_counts())
print("\nAll feature columns are numeric now?",
      all(str(t) != "object" for t in X.dtypes))

# Save the prepared data so the NEXT lesson (training) can reuse it.
X.to_csv("data/telco_X.csv", index=False)
y.to_csv("data/telco_y.csv", index=False)
print("\nSaved prepared data -> data/telco_X.csv and data/telco_y.csv")
