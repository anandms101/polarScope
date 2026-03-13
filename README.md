# PolarScope

**Repository:** [https://github.com/anandms101/polarScope](https://github.com/anandms101/polarScope)

PolarScope is an AI-powered research prototype that identifies **polarizing movies** from real IMDb reviews and explains *why* audiences disagree. It pairs a TF-IDF + logistic-regression sentiment classifier with three complementary polarization metrics (bimodality coefficient, Shannon entropy, confidence-adjusted disagreement) and NMF topic extraction, all surfaced through an interactive **Streamlit** dashboard.

## Features

- **Sentiment analysis** — TF-IDF (unigram + bigram) pipeline with logistic regression, trained on real IMDb review data.
- **Multi-metric polarization scoring** — bimodality coefficient, Shannon entropy, confidence-adjusted disagreement, and a weighted composite score.
- **NMF topic extraction** — identifies the discussion themes driving disagreement and highlights the most controversial topics per movie.
- **Two recommendation modes**:
  - **Safe pick** — crowd-pleasers with low polarization and high average sentiment.
  - **Debate night** — conversation-starters with high polarization and sufficient review volume.
- **Case studies** — side-by-side polarizing vs. consensus movies with topic explanations and representative review excerpts.
- **Evaluation** — Spearman baseline comparisons and bootstrap ranking stability (Kendall tau).
- **Title resolution** — IMDb `tt*` IDs are automatically resolved to human-readable movie titles.
- **Interactive Streamlit UI** — six tabs (Home, Movie Explorer, Recommendations, Metrics Comparison, Case Studies, Evaluation) with charts, search, and live controls.

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

## Getting started

### Kaggle setup (for real datasets)

1. Create a Kaggle account and generate an API token (`kaggle.json`) from your Kaggle profile settings.
2. Place it at `~/.kaggle/kaggle.json` and restrict permissions:

```bash
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

3. Download the per-movie IMDb reviews dataset used for PolarScope:

```bash
python scripts/download_kaggle_dataset.py --dataset mlopssss/imdb-movie-reviews-grouped-by-ratings
```

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

PolarScope runs end-to-end on the Kaggle dataset `mlopssss/imdb-movie-reviews-grouped-by-ratings` which contains `MovieID`, `Rating`, and `Review` columns (split across multiple CSV files). After running the download script above, no additional manual data steps are required.

Optionally, you can also provide the labeled IMDb 50K sentiment dataset as `data/raw/imdb_reviews_50k.csv` to train on explicit sentiment labels instead of weak labels from ratings.

### 4. Train the sentiment model and precompute metrics

```bash
python train.py
```

This will:

- Train the TF-IDF + logistic regression sentiment model and save it to `data/processed/sentiment_model.pkl`.
- Compute per-movie polarization metrics → `data/processed/movie_polarization.parquet`.
- Resolve IMDb IDs to human-readable titles → `data/processed/title_cache.json`.
- Generate case studies with topic-level explanations → `data/processed/movie_case_studies.json`.
- Run bootstrap stability and baseline evaluations → `data/processed/evaluation_results.json`.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the printed local URL (typically `http://localhost:8501`) in your browser.

## Running tests

```bash
pytest -q
```

## Development notes

- Reusable logic lives in `src/`; entry points are `app.py` and `train.py`.
- Heavy artifacts (raw data, processed data, models) are excluded from version control via `.gitignore`.
- Experimental notebooks can go in `notebooks/`; move production-ready code into `src/`.
