# ML Learning Journey 🧠

Learning **Machine Learning** and **MLOps** from the basics — slowly, completely,
and hands-on. Built by a Data Engineer, one small step at a time.

Each project introduces a real ML task with bite-sized lessons: a concept, a
tiny runnable script, then a checkpoint before moving on.

---

## Projects

| # | Project | ML type | What it teaches |
|---|---------|---------|-----------------|
| **01** | [Churn Prediction](01-churn-prediction/) | **Supervised** | Learn from labeled answers: logistic regression, the full ML workflow, saving a model |
| **02** | [Anomaly Detection](02-anomaly-detection/) | **Unsupervised** | Find patterns without labels: z-score, Isolation Forest, precision/recall tuning |

Each project folder has its own `README.md` with setup and run instructions.

---

## The big picture: two types of ML

- **Supervised** (Project 01) — you *have* the answers (labels) and learn to
  predict them. Example: "will this customer churn?"
- **Unsupervised** (Project 02) — you *don't* have answers; the model finds
  structure or oddities on its own. Example: "is this transaction weird?"

---

## One-time setup

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then `cd` into a project folder and follow its README.

---

## Repo layout

```
ml-learning-journey/
├── README.md                  ← you are here
├── PROGRESS.md                ← learning log / checkpoint
├── requirements.txt           ← shared Python dependencies
├── 01-churn-prediction/       ← Project 01 (supervised)
│   ├── README.md
│   ├── data/                  ← dataset download instructions
│   └── lessons/
└── 02-anomaly-detection/      ← Project 02 (unsupervised)
    ├── README.md
    └── lessons/
```

## Progress & roadmap

See [PROGRESS.md](PROGRESS.md) for what's done and what's next
(more models, evaluation, feature engineering, then MLOps: tracking → serving → monitoring).
