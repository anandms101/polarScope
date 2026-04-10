from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_loader import (
    DATA_RAW_DIR,
    KAGGLE_RAW_DIR,
    build_weak_sentiment_labels_from_ratings,
    ensure_dirs,
    load_kaggle_grouped_by_ratings,
    load_sentiment_data,
)
from src.evaluation import bootstrap_ranking_stability, compare_baselines, generate_case_studies
from src.polarization import compute_all_metrics
from src.sentiment import MODEL_PATH, evaluate_model, save_model, train_sentiment_model
from src.title_resolver import build_title_map


PROCESSED_DIR = Path("data/processed")
METRICS_PATH = PROCESSED_DIR / "movie_polarization.parquet"
TOPICS_PATH = PROCESSED_DIR / "movie_case_studies.json"
EVAL_PATH = PROCESSED_DIR / "evaluation_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sentiment model and compute polarization metrics.")
    parser.add_argument(
        "--sentiment_csv",
        default=str(DATA_RAW_DIR / "imdb_reviews_50k.csv"),
        help="Path to Kaggle IMDb 50K sentiment CSV (review,sentiment).",
    )
    parser.add_argument(
        "--kaggle_dir",
        default=str(KAGGLE_RAW_DIR / "mlopssss__imdb-movie-reviews-grouped-by-ratings"),
        help="Path to unzipped Kaggle per-movie reviews dataset directory.",
    )
    parser.add_argument(
        "--max_movies",
        type=int,
        default=800,
        help="Max movies to include (subsample for medium-scale runs).",
    )
    parser.add_argument(
        "--max_reviews_per_movie",
        type=int,
        default=200,
        help="Cap reviews per movie to control runtime/memory.",
    )
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=120,
        help="Bootstrap iterations for ranking stability evaluation.",
    )
    parser.add_argument(
        "--case_studies_n",
        type=int,
        default=5,
        help="Number of polarizing and consensus case studies to generate.",
    )
    args = parser.parse_args()

    ensure_dirs()

    sentiment_csv = Path(args.sentiment_csv)
    if sentiment_csv.exists():
        ds = load_sentiment_data(sentiment_csv)
    else:
        # Use weak labels from Kaggle ratings dataset (real data) if the labeled 50K CSV isn't present.
        kaggle_df = load_kaggle_grouped_by_ratings(
            args.kaggle_dir,
            min_reviews_per_movie=5,
            max_movies=args.max_movies,
            max_reviews_per_movie=args.max_reviews_per_movie,
        )
        ds = build_weak_sentiment_labels_from_ratings(kaggle_df)

    model = train_sentiment_model(ds.X_train, ds.y_train)
    report, cm = evaluate_model(model, ds.X_test, ds.y_test)
    print("Sentiment model evaluation:")
    print(report)
    print("Confusion matrix:")
    print(cm)

    save_model(model, MODEL_PATH)
    print(f"Saved sentiment model to {MODEL_PATH}")

    kaggle_df = load_kaggle_grouped_by_ratings(
        args.kaggle_dir,
        min_reviews_per_movie=5,
        max_movies=args.max_movies,
        max_reviews_per_movie=args.max_reviews_per_movie,
    )
    # Group reviews by movie_id for polarization computation.
    movie_reviews = (
        kaggle_df.groupby("movie_id")["review"].apply(list).to_dict()  # type: ignore[assignment]
    )

    # Resolve IMDb IDs to human-readable titles.
    all_ids = set(movie_reviews.keys())
    title_map = build_title_map(all_ids)

    df = compute_all_metrics(movie_reviews, sentiment_model=model, min_reviews=5)
    if df.empty:
        raise RuntimeError("No movies available for metrics computation (too few reviews?).")

    df["title"] = df["movie"].map(title_map).fillna(df["movie"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(METRICS_PATH, index=False)
    print(f"Saved movie metrics to {METRICS_PATH} ({len(df)} movies).")

    # Save case studies with topic explanations (used by the UI).
    case = generate_case_studies(df, movie_reviews, sentiment_model=model, n=args.case_studies_n, n_topics=5)
    TOPICS_PATH.write_text(json.dumps(case, indent=2))
    print(f"Saved case studies (with topics) to {TOPICS_PATH}")

    # Evaluation summary (baseline comparisons + bootstrap stability).
    eval_summary = {
        "baseline_correlations": compare_baselines(df),
    }
    try:
        # Keep bootstrap fast by evaluating only movies with the most reviews.
        top_movies = df.sort_values("num_reviews", ascending=False).head(120)["movie"].tolist()
        eval_reviews = {m: movie_reviews[m] for m in top_movies if m in movie_reviews}
        eval_summary["bootstrap"] = bootstrap_ranking_stability(
            eval_reviews,
            sentiment_model=model,
            n_iterations=args.bootstrap_iterations,
            min_reviews=5,
        )
    except Exception as e:
        eval_summary["bootstrap_error"] = str(e)

    EVAL_PATH.write_text(json.dumps(eval_summary, indent=2))
    print(f"Saved evaluation summary to {EVAL_PATH}")


if __name__ == "__main__":
    main()

