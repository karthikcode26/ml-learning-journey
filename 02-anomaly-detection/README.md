# Project 02 — Anomaly Detection (Unsupervised Learning)

Find **rare, unusual retail transactions** that don't fit the normal pattern
(e.g. $500 for a single item). This project teaches **unsupervised learning** —
the model discovers what "normal" looks like on its own, with **no labels**.

## What you learn

- The intuition: anomalies sit *alone*, far from the crowd
- The **z-score** method — and *why it fails* on relationship anomalies
- **Isolation Forest** — built from scratch (random cuts + isolation depth) AND
  with scikit-learn
- Tuning the `contamination` dial and the **precision/recall trade-off**

## Setup

Uses the same `.venv` and `requirements.txt` as the repo (pandas + scikit-learn).
Lessons 1, 2, and 4a are pure Python and need no libraries.

## Run the lessons (in order)

```bash
cd 02-anomaly-detection

python lessons/lesson_01_intuition.py                 # what is an anomaly? (mean/std/z-score)
python lessons/lesson_02_generate_data.py             # make transactions.csv (500 rows, ~4% anomalies)
python lessons/lesson_03_zscore_limits.py             # z-score method + where it fails
python lessons/lesson_04a_isolation_from_scratch.py   # the isolation idea, by hand
python lessons/lesson_04b_isolation_forest.py         # real scikit-learn IsolationForest
python lessons/lesson_05_tuning.py                    # tune contamination; precision/recall
```

> Note: run `lesson_02_generate_data.py` first — it creates `transactions.csv`
> (in this folder) that the later lessons read.

> ML type: **Unsupervised** (no labels; the `is_anomaly` column exists only to
> grade the detector afterward).
