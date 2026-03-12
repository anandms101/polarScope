from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import (
    DATA_RAW_DIR,
    demo_movie_reviews,
    demo_sentiment_training_data,
    ensure_dirs,
    load_movie_reviews,
    load_sentiment_data,
)
from src.polarization import compute_all_metrics
from src.sentiment import MODEL_PATH, evaluate_model, save_model, train_sentiment_model


PROCESSED_DIR = Path("data/processed")
METRICS_PATH = PROCESSED_DIR / "movie_polarization.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sentiment model and compute polarization metrics.")
    parser.add_argument(
        "--sentiment_csv",
        default=str(DATA_RAW_DIR / "imdb_reviews_50k.csv"),
        help="Path to Kaggle IMDb 50K sentiment CSV (review,sentiment).",
    )
    parser.add_argument(
        "--movie_csv",
        default=str(DATA_RAW_DIR / "movie_reviews_by_title.csv"),
        help="Path to per-movie reviews CSV (movie_title,review_text).",
    )
    args = parser.parse_args()

    ensure_dirs()

    sentiment_csv = Path(args.sentiment_csv)
    if sentiment_csv.exists():
        ds = load_sentiment_data(sentiment_csv)
        model = train_sentiment_model(ds.X_train, ds.y_train)
        report, cm = evaluate_model(model, ds.X_test, ds.y_test)
        print("Sentiment model evaluation:")
        print(report)
        print("Confusion matrix:")
        print(cm)
    else:
        print(
            f"[WARN] Sentiment CSV not found at {sentiment_csv}. "
            "Training a tiny demo model instead."
        )
        X, y = demo_sentiment_training_data()
        model = train_sentiment_model(X, y)

    save_model(model, MODEL_PATH)
    print(f"Saved sentiment model to {MODEL_PATH}")

    movie_csv = Path(args.movie_csv)
    if movie_csv.exists():
        movie_reviews = load_movie_reviews(movie_csv, min_reviews_per_movie=1)
    else:
        print(
            f"[WARN] Per-movie CSV not found at {movie_csv}. "
            "Using a tiny built-in demo dataset."
        )
        movie_reviews = demo_movie_reviews()

    df = compute_all_metrics(movie_reviews, sentiment_model=model, min_reviews=3)
    if df.empty:
        raise RuntimeError("No movies available for metrics computation (too few reviews?).")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(METRICS_PATH, index=False)
    print(f"Saved movie metrics to {METRICS_PATH} ({len(df)} movies).")


if __name__ == "__main__":
    main()

