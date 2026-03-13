from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_RAW_DIR = Path("data/raw")
KAGGLE_RAW_DIR = DATA_RAW_DIR / "kaggle"


def _strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _basic_normalize(text: str) -> str:
    text = _strip_html(text)
    return text.lower()


@dataclass(frozen=True)
class SentimentDataset:
    X_train: List[str]
    X_test: List[str]
    y_train: List[int]
    y_test: List[int]


def load_sentiment_data(
    csv_path: str | Path = DATA_RAW_DIR / "imdb_reviews_50k.csv",
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SentimentDataset:
    """
    Loads a labeled sentiment dataset (Kaggle IMDb 50K reviews).

    Expected columns:
    - review: text
    - sentiment: 'positive' | 'negative' (or already 1/0)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Sentiment dataset not found at {csv_path}. "
            f"Place Kaggle CSV as data/raw/imdb_reviews_50k.csv."
        )

    df = pd.read_csv(csv_path)
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(
            f"Expected columns ['review','sentiment'] in {csv_path}, got {list(df.columns)}"
        )

    X = df["review"].astype(str).map(_basic_normalize).tolist()
    y_raw = df["sentiment"]
    if y_raw.dtype.kind in {"i", "u", "b"}:
        y = y_raw.astype(int).tolist()
    else:
        y = y_raw.astype(str).str.lower().map({"negative": 0, "positive": 1}).tolist()
        if any(v is None for v in y):
            raise ValueError("Sentiment column contains values other than positive/negative.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return SentimentDataset(
        X_train=list(X_train),
        X_test=list(X_test),
        y_train=list(y_train),
        y_test=list(y_test),
    )


def load_movie_reviews(
    csv_path: str | Path = DATA_RAW_DIR / "movie_reviews_by_title.csv",
    *,
    movie_col_candidates: Tuple[str, ...] = ("movie_title", "movie", "title", "film"),
    review_col_candidates: Tuple[str, ...] = ("review_text", "review", "text", "content"),
    min_reviews_per_movie: int = 1,
) -> Dict[str, List[str]]:
    """
    Loads reviews grouped by movie title from a CSV.

    Expected columns (at least):
    - movie_title (or movie/title)
    - review_text (or review/text/content)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Per-movie review dataset not found at {csv_path}. "
            f"Place CSV as data/raw/movie_reviews_by_title.csv."
        )

    df = pd.read_csv(csv_path)
    movie_col = next((c for c in movie_col_candidates if c in df.columns), None)
    review_col = next((c for c in review_col_candidates if c in df.columns), None)
    if movie_col is None or review_col is None:
        raise ValueError(
            f"Expected movie column in {movie_col_candidates} and review column in {review_col_candidates}. "
            f"Got {list(df.columns)}"
        )

    df = df[[movie_col, review_col]].dropna()
    df[movie_col] = df[movie_col].astype(str).str.strip()
    df[review_col] = df[review_col].astype(str).map(_basic_normalize)

    grouped: Dict[str, List[str]] = (
        df.groupby(movie_col)[review_col].apply(list).to_dict()  # type: ignore[assignment]
    )
    grouped = {k: v for k, v in grouped.items() if len(v) >= min_reviews_per_movie}
    return grouped


def load_kaggle_grouped_by_ratings(
    dataset_dir: str | Path = KAGGLE_RAW_DIR / "mlopssss__imdb-movie-reviews-grouped-by-ratings",
    *,
    min_reviews_per_movie: int = 5,
    max_movies: int | None = None,
    max_reviews_per_movie: int | None = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Loads the Kaggle dataset `mlopssss/imdb-movie-reviews-grouped-by-ratings`.

    The dataset ships multiple CSV files like `reviews_rating_7.csv` with columns:
    - MovieID (IMDb title id, e.g. tt1234567)
    - Rating  (1-10 integer)
    - Review  (text)

    Returns a DataFrame with columns: movie_id, rating, review.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Kaggle dataset not found at {dataset_dir}. "
            "Run: python scripts/download_kaggle_dataset.py --dataset mlopssss/imdb-movie-reviews-grouped-by-ratings"
        )

    files = sorted(dataset_dir.glob("reviews_rating_*.csv"))
    if not files:
        raise FileNotFoundError(f"No reviews_rating_*.csv files found in {dataset_dir}")

    frames: List[pd.DataFrame] = []
    for fp in files:
        df = pd.read_csv(fp)
        expected = {"MovieID", "Rating", "Review"}
        if not expected.issubset(set(df.columns)):
            raise ValueError(f"Unexpected columns in {fp.name}: {list(df.columns)}")
        frames.append(df[list(expected)])

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna()
    all_df["MovieID"] = all_df["MovieID"].astype(str).str.strip()
    all_df["Rating"] = pd.to_numeric(all_df["Rating"], errors="coerce").astype("Int64")
    all_df["Review"] = all_df["Review"].astype(str).map(_basic_normalize)
    all_df = all_df.dropna(subset=["MovieID", "Rating", "Review"])

    # Filter by minimum review count per movie.
    counts = all_df["MovieID"].value_counts()
    keep_ids = counts[counts >= min_reviews_per_movie].index
    all_df = all_df[all_df["MovieID"].isin(keep_ids)].copy()

    # Optionally subsample movies and/or reviews per movie for a medium-scale run.
    if max_movies is not None:
        movie_ids = all_df["MovieID"].drop_duplicates().sample(
            n=min(max_movies, all_df["MovieID"].nunique()),
            random_state=random_state,
        )
        all_df = all_df[all_df["MovieID"].isin(movie_ids)].copy()

    if max_reviews_per_movie is not None:
        # Avoid groupby.apply FutureWarning by sampling within each group explicitly.
        sampled_frames: List[pd.DataFrame] = []
        for _, g in all_df.groupby("MovieID", sort=False):
            sampled_frames.append(
                g.sample(n=min(len(g), max_reviews_per_movie), random_state=random_state)
            )
        all_df = pd.concat(sampled_frames, ignore_index=True)

    out = all_df.rename(columns={"MovieID": "movie_id", "Rating": "rating", "Review": "review"})
    out["rating"] = out["rating"].astype(int)
    return out[["movie_id", "rating", "review"]].reset_index(drop=True)


def build_weak_sentiment_labels_from_ratings(
    df: pd.DataFrame,
    *,
    neg_max: int = 4,
    pos_min: int = 7,
) -> SentimentDataset:
    """
    Creates a labeled sentiment dataset using rating thresholds:
    - rating <= neg_max => negative (0)
    - rating >= pos_min => positive (1)
    - otherwise dropped
    """
    if not {"rating", "review"}.issubset(df.columns):
        raise ValueError("Expected columns: rating, review")

    sub = df[["rating", "review"]].dropna().copy()
    sub["rating"] = pd.to_numeric(sub["rating"], errors="coerce")
    sub = sub.dropna()
    sub["rating"] = sub["rating"].astype(int)

    neg = sub[sub["rating"] <= neg_max].copy()
    pos = sub[sub["rating"] >= pos_min].copy()
    neg["y"] = 0
    pos["y"] = 1
    lab = pd.concat([neg, pos], ignore_index=True)
    if lab.empty:
        raise RuntimeError("No weak labels produced from rating thresholds.")

    X = lab["review"].astype(str).tolist()
    y = lab["y"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return SentimentDataset(
        X_train=list(X_train),
        X_test=list(X_test),
        y_train=list(y_train),
        y_test=list(y_test),
    )


def demo_movie_reviews() -> Dict[str, List[str]]:
    """
    Tiny built-in dataset so the pipeline/UI can run without external files.
    """
    return {
        "Paddington_2": [
            "Charming, heartfelt, and genuinely funny. A delightful film.",
            "Absolutely wonderful. Warm story and great performances.",
            "A feel-good masterpiece; loved every minute.",
            "Sweet, clever, and uplifting. Highly recommended.",
        ],
        "The_Last_Jedi": [
            "Visually stunning but the story choices were frustrating and inconsistent.",
            "I loved the bold direction and character arcs. Best Star Wars in years.",
            "Terrible pacing, weird jokes, and it ruined beloved characters.",
            "Ambitious and emotional; I enjoyed it far more than the critics claim.",
        ],
        "Mother!": [
            "Anxiety-inducing, pretentious, and exhausting. Not for me.",
            "Brilliant allegory with intense acting. Unforgettable experience.",
            "I hated it. It felt like shock value without payoff.",
            "Loved the symbolism and the escalating chaos. Very effective.",
        ],
    }


def demo_sentiment_training_data() -> Tuple[List[str], List[int]]:
    """
    Small labeled sentiment set for quick demo training (not for final evaluation).
    """
    pos = [
        "I loved this movie. Great acting and a wonderful story.",
        "Fantastic film. Highly enjoyable and well made.",
        "A beautiful and uplifting experience.",
        "Brilliant performances and excellent direction.",
    ]
    neg = [
        "I hated this movie. Boring and poorly written.",
        "Terrible film. A complete waste of time.",
        "Bad acting and an awful story.",
        "Disappointing and frustrating throughout.",
    ]
    X = [_basic_normalize(t) for t in (pos + neg)]
    y = [1] * len(pos) + [0] * len(neg)
    return X, y


def ensure_dirs() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

