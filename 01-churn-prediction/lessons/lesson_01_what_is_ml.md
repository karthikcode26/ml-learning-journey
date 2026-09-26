# Lesson 1 — What Machine Learning Actually Is

> Goal: understand the core idea of ML and look at real data. **No model yet.**

## The core idea

**Traditional programming:** you write the rules.
```
RULES + DATA  ->  ANSWERS
```
e.g. (SQL-style): `WHERE support_tickets > 5 AND tenure < 3 THEN 'will churn'`
You had to already know the rule.

**Machine Learning:** you provide data + answers; the machine finds the rules.
```
DATA + ANSWERS  ->  RULES  (the "rules" it produces are called the MODEL)
```
You show it thousands of labeled past customers, and it discovers the pattern.

## Key vocabulary

- **Features** — the inputs / clues (columns like tenure, monthly charges).
- **Label / target** — the answer we want to predict (did they churn: Yes/No).
- **Supervised learning** — learning from examples where the label is known.
- **Model** — the "learned rules" the algorithm produces from the data.

## Hands-on task

1. Download the **Telco Customer Churn** dataset (see `../data/README.md`).
2. Save it as `data/telco_churn.csv`.
3. Next: load it with pandas and identify the features vs. the label.
