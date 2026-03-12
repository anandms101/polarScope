from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .sentiment import SentimentPredictions, predict_sentiment


@dataclass(frozen=True)
class MovieMetrics:
    movie: str
    num_reviews: int
    mean_sentiment: float
    variance_baseline: float
    bimodality: float
    entropy: float
    confidence_adj: float
    composite: float


def _minmax01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def bimodality_coefficient(values: np.ndarray) -> float:
    """
    Bimodality coefficient (BC) based on skewness and kurtosis.
    Returns a value typically in [0, 1+], where higher suggests bimodality.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 4:
        return 0.0

    g1 = skew(v, bias=False)
    g2 = kurtosis(v, fisher=False, bias=False)  # normal => 3
    denom = g2 + (3.0 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    if denom <= 0:
        return 0.0
    return float((g1 * g1 + 1.0) / denom)


def entropy_score(values: np.ndarray, *, bins: int = 10) -> float:
    """
    Normalized Shannon entropy over a histogram of values.
    Output is in [0, 1].
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    hist, _ = np.histogram(v, bins=bins, range=(0.0, 1.0), density=False)
    p = hist.astype(float)
    p_sum = p.sum()
    if p_sum <= 0:
        return 0.0
    p = p / p_sum
    p = p[p > 0]
    ent = -np.sum(p * np.log(p))
    return float(ent / np.log(bins))


def confidence_adjusted_disagreement(proba_pos: np.ndarray) -> float:
    """
    Confidence-adjusted disagreement: variance of signed sentiments weighted by confidence.

    sentiment in [-1, +1] derived from proba; confidence in [0,1] as |p-0.5|*2.
    """
    p = np.asarray(proba_pos, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 0.0
    signed = (p - 0.5) * 2.0  # [-1, 1]
    conf = np.abs(p - 0.5) * 2.0  # [0, 1]
    weighted = signed * conf
    return float(np.var(weighted))


def compute_movie_metrics(
    movie: str,
    reviews: List[str],
    *,
    sentiment_model,
    entropy_bins: int = 10,
    composite_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> MovieMetrics:
    preds: SentimentPredictions = predict_sentiment(sentiment_model, reviews)
    p = preds.proba_pos
    n = int(p.size)
    mean_sent = float(np.mean(p)) if n else 0.0
    var_baseline = float(np.var(p)) if n else 0.0

    bc_raw = bimodality_coefficient(p)
    ent = entropy_score(p, bins=entropy_bins)
    cad_raw = confidence_adjusted_disagreement(p)

    # Simple normalizations for UI-friendly comparisons.
    # BC: map [0.3, 0.75] into [0,1] (0.555 is common bimodality heuristic)
    bc = _minmax01(bc_raw, 0.30, 0.75)
    # CAD: map [0, 0.25] into [0,1] (empirical; values tend to be small)
    cad = _minmax01(cad_raw, 0.00, 0.25)

    w_bc, w_ent, w_cad = composite_weights
    composite = float(w_bc * bc + w_ent * ent + w_cad * cad)

    return MovieMetrics(
        movie=movie,
        num_reviews=n,
        mean_sentiment=mean_sent,
        variance_baseline=var_baseline,
        bimodality=bc,
        entropy=ent,
        confidence_adj=cad,
        composite=composite,
    )


def compute_all_metrics(
    movie_reviews: Dict[str, List[str]],
    *,
    sentiment_model,
    min_reviews: int = 3,
) -> pd.DataFrame:
    rows: List[dict] = []
    for movie, reviews in movie_reviews.items():
        if len(reviews) < min_reviews:
            continue
        mm = compute_movie_metrics(movie, reviews, sentiment_model=sentiment_model)
        rows.append(mm.__dict__)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    return df

