from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


MODEL_PATH = Path("data/processed/sentiment_model.pkl")


@dataclass(frozen=True)
class SentimentPredictions:
    labels: np.ndarray  # shape (n,)
    proba_pos: np.ndarray  # shape (n,)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    n_jobs=None,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_sentiment_model(X_train: List[str], y_train: List[int]) -> Pipeline:
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(pipeline: Pipeline, path: str | Path = MODEL_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_model(path: str | Path = MODEL_PATH) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sentiment model not found at {path}. Run `python train.py` first."
        )
    return joblib.load(path)


def predict_sentiment(pipeline: Pipeline, texts: Iterable[str]) -> SentimentPredictions:
    texts_list = list(texts)
    proba = pipeline.predict_proba(texts_list)
    proba_pos = proba[:, 1]
    labels = (proba_pos >= 0.5).astype(int)
    return SentimentPredictions(labels=labels, proba_pos=proba_pos)


def evaluate_model(pipeline: Pipeline, X_test: List[str], y_test: List[int]) -> Tuple[str, np.ndarray]:
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    return report, cm

