# PolarScope

**Repository:** [https://github.com/anandms101/polarScope](https://github.com/anandms101/polarScope)

PolarScope is a small research prototype that identifies **polarizing movies** from IMDb-style reviews and exposes them through a simple **Streamlit web app**. It combines a TF–IDF + logistic regression sentiment classifier with several polarization metrics to distinguish consensus films (universally liked) from divisive ones (strong love / hate split).

## Features (first milestone)

- Train a binary sentiment classifier on the 50K IMDb movie reviews dataset.
- Aggregate review sentiments per movie and compute multiple polarization metrics (bimodality, entropy, confidence‑adjusted disagreement, variance baseline).
- Rank movies for two recommendation modes:
  - **Safe pick** – low polarization, high average sentiment.
  - **Debate night** – high polarization with sufficient review volume.
- Explore movies in a **Streamlit** UI with basic visualizations.

## Getting started

### 1. Create and activate a virtual environment

```bash
cd polarScope
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Prepare data

1. Download the IMDb 50K reviews dataset from Kaggle and save it as `data/raw/imdb_reviews_50k.csv`.
2. Download or create a per‑movie review dataset (a CSV with at least `movie_title`, `review_text` columns) and save it as `data/raw/movie_reviews_by_title.csv`.

The exact filenames and expected columns are documented in `src/data_loader.py`. If these CSV files are **not** present, the project will still run by falling back to a small built‑in demo dataset, which is useful for quick testing but not for final evaluation.

### 4. Train the sentiment model and precompute metrics

```bash
python train.py
```

This will:

- Train the TF–IDF + logistic regression sentiment model.
- Save the trained pipeline into `data/processed/sentiment_model.pkl`.
- Compute per‑movie polarization metrics and store them in `data/processed/movie_polarization.parquet`.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the printed local URL in your browser.

## Development notes

- Code is organized under `src/` for reusable modules and `app.py` / `train.py` for entry points.
- Heavy artifacts (raw data, processed data, models) are ignored by git via `.gitignore`.
- Keep experiments in `notebooks/` and move any production‑ready logic into `src/`.

