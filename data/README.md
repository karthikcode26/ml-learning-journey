# Datasets

We keep dataset *instructions* here rather than the raw files (data can be large,
and `.gitignore` skips `.csv`/`.xlsx`/`.zip` in this folder).

## Lesson 1–4 dataset: Telco Customer Churn (recommended start)

The classic beginner churn dataset — ~7,000 telecom customers, each labeled as
churned or not. Clean and gentle, perfect for your first model.

**How to get it:**
1. Download from Kaggle:
   [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   (free Kaggle account required — click **Download**).
2. Unzip it and place the file here as:
   ```
   data/telco_churn.csv
   ```
   (Kaggle names it `WA_Fn-UseC_-Telco-Customer-Churn.csv` — feel free to rename it.)

## Other open datasets to explore later

| Dataset | Domain | Why it's good | Source |
|---------|--------|---------------|--------|
| Online Retail II | Retail | Real e-commerce transactions; needs cleaning (good DE practice) | [UCI](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| Credit Card Fraud | Finance | Teaches imbalanced data (fraud is rare) | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Bank Marketing | Finance | Clean classification: will a client subscribe? | [UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing) |

> Tip: start with **Telco Churn**. Once your first model works end-to-end,
> swapping in another dataset is a great self-directed exercise.
