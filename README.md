# PolarScope

**Repository:** [https://github.com/anandms101/polarScope](https://github.com/anandms101/polarScope)

PolarScope is an AI-powered research prototype that identifies **polarizing movies** from real IMDb reviews and explains *why* audiences disagree. It pairs a TF-IDF + logistic-regression sentiment classifier with three complementary polarization metrics (bimodality coefficient, Shannon entropy, confidence-adjusted disagreement) and NMF topic extraction, all surfaced through an interactive **Streamlit** dashboard.

## Quick start (4 commands)

```bash
# 1. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download dataset (see "Dataset setup" below)

# 3. Train model + compute metrics
python train.py

# 4. Launch the app
streamlit run app.py
```

## Dataset setup

PolarScope uses the Kaggle dataset [IMDb Movie Reviews Grouped by Ratings](https://www.kaggle.com/datasets/mlopssss/imdb-movie-reviews-grouped-by-ratings). You only need to do this once.

### Option A — Manual download (simplest, no Kaggle CLI needed)

1. Go to **https://www.kaggle.com/datasets/mlopssss/imdb-movie-reviews-grouped-by-ratings**
2. Click **Download** (you'll need a free Kaggle account).
3. Unzip the downloaded file and place the CSV files into:

```
data/raw/kaggle/mlopssss__imdb-movie-reviews-grouped-by-ratings/
├── reviews_rating_1.csv
├── reviews_rating_2.csv
├── ...
└── reviews_rating_10.csv
```

That's it — `python train.py` will find them automatically.

### Option B — Kaggle CLI (one command)

If you have the [Kaggle CLI](https://www.kaggle.com/docs/api) configured (`~/.kaggle/kaggle.json`):

```bash
python scripts/download_kaggle_dataset.py
```

## Features

- **Sentiment analysis** — TF-IDF (unigram + bigram) pipeline with logistic regression trained on real IMDb data.
- **Multi-metric polarization scoring** — bimodality coefficient, Shannon entropy, confidence-adjusted disagreement, and a weighted composite score.
- **NMF topic extraction** — identifies discussion themes driving disagreement and highlights the most controversial topics per movie.
- **Two recommendation modes**:
  - **Safe pick** — crowd-pleasers with low polarization and high average sentiment.
  - **Debate night** — conversation-starters with high polarization and sufficient review volume.
- **Case studies** — side-by-side polarizing vs. consensus movies with topic explanations and review excerpts.
- **Evaluation** — Spearman baseline comparisons and bootstrap ranking stability (Kendall tau).
- **Title resolution** — IMDb `tt*` IDs are automatically resolved to human-readable movie titles.
- **Interactive Streamlit UI** — six tabs (Home, Movie Explorer, Recommendations, Metrics Comparison, Case Studies, Evaluation).

## Project structure

```
polarScope/
├── app.py                  # Streamlit web application
├── train.py                # CLI pipeline: data → model → metrics → evaluation
├── requirements.txt
├── setup.sh                # One-command environment bootstrap
├── src/
│   ├── data_loader.py      # Dataset loading, cleaning, weak-label generation
│   ├── sentiment.py        # TF-IDF + LR sentiment pipeline
│   ├── polarization.py     # Polarization metrics (bimodality, entropy, etc.)
│   ├── recommender.py      # Safe-pick and debate-night ranking
│   ├── topics.py           # NMF topic extraction and controversy scoring
│   ├── evaluation.py       # Bootstrap stability, baseline correlations, case studies
│   └── title_resolver.py   # IMDb ID → movie title resolution + caching
├── tests/                  # pytest unit tests for every module
├── scripts/
│   └── download_kaggle_dataset.py
└── docs/
    └── progress_report_1.md / .pdf
```

## What `python train.py` produces

| Artifact | Path |
|---|---|
| Trained sentiment model | `data/processed/sentiment_model.pkl` |
| Per-movie polarization metrics | `data/processed/movie_polarization.parquet` |
| IMDb title cache | `data/processed/title_cache.json` |
| Case studies with topics | `data/processed/movie_case_studies.json` |
| Evaluation results | `data/processed/evaluation_results.json` |

## Running tests

```bash
pytest -q
```

## Development notes

- Reusable logic lives in `src/`; entry points are `app.py` and `train.py`.
- Heavy artifacts (raw data, processed data, models) are excluded from version control via `.gitignore`.
- Experimental notebooks can go in `notebooks/`; move production-ready code into `src/`.
