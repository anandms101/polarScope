from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_RAW_DIR = Path("data/raw")


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

