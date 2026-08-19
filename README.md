# JITLine Fine Adept

A small learning project for pull request risk prediction. It starts with a tiny
CSV dataset, compares a simple heuristic with tree-based models, and prints the
classification metrics for each approach.

## Project Structure

```text
data/
  tiny_pr_data.csv              # Example pull request dataset
scripts/
  01_view_data.py               # Inspect the dataset
  02_heuristic.py               # Rule-based risk score baseline
  03_decision_tree.py           # Scikit-learn decision tree model
  04_xgboost.py                 # XGBoost model
jitline-study/
  paper/jitline.pdf             # Reference paper
```

## Setup

Conda is the recommended setup for this project:

```bash
conda env create -f environment.yml
conda activate jitline-fine-adept
```

If the environment already exists, update it with:

```bash
conda env update -f environment.yml --prune
```

You can also use a standard virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Run the scripts from the project root:

```bash
python scripts/01_view_data.py
python scripts/02_heuristic.py
python scripts/03_decision_tree.py
python scripts/04_xgboost.py
```

The dataset is intentionally tiny, so the metrics are useful for learning the
workflow rather than judging production model quality.
