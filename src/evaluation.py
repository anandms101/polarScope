from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .polarization import compute_all_metrics
from .sentiment import predict_sentiment
from .topics import extract_topics, get_controversial_topics


@dataclass(frozen=True)
class BootstrapResult:
    kendall_tau: float


def compare_baselines(metrics_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compare composite score with baselines using Spearman correlation.
    Returns a dict of correlations.
    """
    if metrics_df.empty:
        return {}

    out: Dict[str, float] = {}
    for a, b in [
        ("composite", "variance_baseline"),
        ("composite", "mean_sentiment"),
        ("variance_baseline", "mean_sentiment"),
    ]:
        if a in metrics_df.columns and b in metrics_df.columns:
            r, _ = spearmanr(metrics_df[a], metrics_df[b])
            out[f"spearman_{a}_vs_{b}"] = float(r)
    return out


def bootstrap_ranking_stability(
    movie_reviews: Dict[str, List[str]],
    *,
    sentiment_model,
    n_iterations: int = 200,
    min_reviews: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Bootstrap stability of movie rankings.

    Procedure:
    - Compute a baseline ranking by composite score.
    - For each iteration: resample reviews within each movie with replacement, recompute scores,
      then compute Kendall tau between baseline order and resampled order.
    """
    rng = np.random.default_rng(random_state)
    base_df = compute_all_metrics(movie_reviews, sentiment_model=sentiment_model, min_reviews=min_reviews)
    if base_df.empty or base_df.shape[0] < 3:
        raise RuntimeError("Not enough movies for bootstrap stability analysis.")

    base_order = base_df["movie"].tolist()
    base_rank = {m: i for i, m in enumerate(base_order)}

    taus: List[float] = []
    movies = base_order

    for _ in range(n_iterations):
        boot_reviews: Dict[str, List[str]] = {}
        for m in movies:
            rs = movie_reviews[m]
            if len(rs) < min_reviews:
                continue
            idx = rng.integers(0, len(rs), size=len(rs))
            boot_reviews[m] = [rs[i] for i in idx]

        boot_df = compute_all_metrics(boot_reviews, sentiment_model=sentiment_model, min_reviews=min_reviews)
        boot_order = boot_df["movie"].tolist()
        if len(boot_order) != len(movies):
            # If some movie drops due to min_reviews, skip iteration.
            continue

        boot_rank = {m: i for i, m in enumerate(boot_order)}
        x = [base_rank[m] for m in movies]
        y = [boot_rank[m] for m in movies]
        tau, _ = kendalltau(x, y)
        if np.isfinite(tau):
            taus.append(float(tau))

    if not taus:
        raise RuntimeError("Bootstrap produced no valid tau samples.")

    taus_arr = np.asarray(taus)
    return {
        "n_iterations_requested": n_iterations,
        "n_iterations_used": int(len(taus)),
        "kendall_tau_mean": float(np.mean(taus_arr)),
        "kendall_tau_ci95": [float(np.quantile(taus_arr, 0.025)), float(np.quantile(taus_arr, 0.975))],
        "kendall_tau_samples": taus,  # keep for plotting
    }


def generate_case_studies(
    metrics_df: pd.DataFrame,
    movie_reviews: Dict[str, List[str]],
    *,
    sentiment_model,
    n: int = 5,
    n_topics: int = 5,
) -> Dict[str, List[dict]]:
    """
    Generates structured case studies for top-N polarizing and top-N consensus movies.
    """
    if metrics_df.empty:
        return {"polarizing": [], "consensus": []}

    df = metrics_df.copy()
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    polar = df.head(n)["movie"].tolist()
    cons = df.sort_values("composite", ascending=True).head(n)["movie"].tolist()

    def build(movie_id: str) -> dict:
        reviews = movie_reviews.get(movie_id, [])
        preds = predict_sentiment(sentiment_model, reviews)
        topics = extract_topics(reviews, preds.labels.tolist(), n_topics=n_topics)
        controversial = get_controversial_topics(topics, top_n=min(3, n_topics))
        row = df[df["movie"] == movie_id].iloc[0].to_dict()
        return {
            "movie": movie_id,
            "metrics": row,
            "controversial_topics": [asdict(t) for t in controversial],
            "example_pos_review": reviews[int(np.argmax(preds.proba_pos))] if reviews else "",
            "example_neg_review": reviews[int(np.argmin(preds.proba_pos))] if reviews else "",
        }

    return {
        "polarizing": [build(m) for m in polar],
        "consensus": [build(m) for m in cons],
    }

