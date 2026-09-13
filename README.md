# ML Learning Journey 🧠

Learning **Machine Learning** and **MLOps** from the basics — slowly, completely,
and hands-on. Built by a Data Engineer, one small step at a time.

Each lesson introduces **one concept**, with a tiny runnable example, before
moving on. No rushing, no black boxes.

---

## Lesson 0 — Set up your machine

Before any ML, let's get your laptop ready. This is a one-time setup.

### 1. Check if you have Python 3

Open your terminal and run:

```bash
python3 --version
```

- **Mac / Linux:** Python 3 is usually already installed. If not, install from
  [python.org](https://www.python.org/downloads/) or via Homebrew (`brew install python`).
- **Windows:** if it's missing, install from [python.org](https://www.python.org/downloads/)
  and **check "Add Python to PATH"** during install. Then use `python` instead of `python3`.

You want **Python 3.10 or newer**.

### 2. Clone this repository

```bash
git clone https://github.com/karthikcode26/ml-learning-journey.git
cd ml-learning-journey
```

### 3. Create a virtual environment

A "virtual environment" (venv) is an isolated sandbox for this project's
libraries, so they don't clash with other Python projects. This is a standard
professional practice.

```bash
# Create it (do this once)
python3 -m venv .venv

# Activate it (do this every time you work on the project)
# Mac / Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

When active, your prompt shows `(.venv)`. To leave later: `deactivate`.

### 4. Install the libraries

```bash
pip install -r requirements.txt
```

This installs the real ML toolkit (pandas, scikit-learn, etc.) that we'll use
throughout the lessons.

### 5. Verify it worked

```bash
python -c "import pandas, sklearn; print('Setup OK — pandas', pandas.__version__, '| scikit-learn', sklearn.__version__)"
```

If you see a version printed, **you're ready.** 🎉

---

## What's next

Once setup works on your laptop, we'll start **Lesson 1: What Machine Learning
actually is** — and download our first real open dataset (see `data/README.md`).

## Roadmap

- [x] **Lesson 0** — Machine setup (this file)
- [ ] **Lesson 1** — What ML is: rules vs. examples
- [ ] **Lesson 2** — Loading & exploring a real dataset (pandas)
- [ ] **Lesson 3** — Your first model (train / test / evaluate)
- [ ] **Lesson 4** — Metrics that matter (precision, recall, confusion matrix)
- [ ] Later — MLOps: experiment tracking, serving, monitoring
