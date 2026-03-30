from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class TopicSummary:
    topic_idx: int
    keywords: List[str]
    pos_ratio: float  # fraction of positive reviews among top-loading reviews
    neg_ratio: float
    rep_pos_review: str
    rep_neg_review: str


def extract_topics(
    reviews: Sequence[str],
    sentiment_labels: Sequence[int],
    *,
    n_topics: int = 5,
    top_keywords: int = 6,
    top_docs: int = 30,
    random_state: int = 42,
) -> List[TopicSummary]:
    """
    Extracts NMF topics from a set of reviews and summarizes polarization per topic.

    - Topics are learned with NMF over TF-IDF features.
    - For each topic we take the top-loading documents (reviews) and compute pos/neg split.
    """
    if len(reviews) != len(sentiment_labels):
        raise ValueError("reviews and sentiment_labels must have the same length")
    if len(reviews) < max(10, n_topics * 2):
        return []

    y = np.asarray(sentiment_labels, dtype=int)
    vec = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english",
    )
    X = vec.fit_transform(reviews)

    nmf = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=random_state,
        max_iter=400,
    )
    W = nmf.fit_transform(X)  # (n_docs, n_topics)
    H = nmf.components_  # (n_topics, n_terms)

    terms = np.array(vec.get_feature_names_out())
    topics: List[TopicSummary] = []

    for k in range(n_topics):
        top_term_idx = np.argsort(H[k])[::-1][:top_keywords]
        keywords = terms[top_term_idx].tolist()

        doc_scores = W[:, k]
        top_doc_idx = np.argsort(doc_scores)[::-1][: min(top_docs, len(reviews))]
        top_y = y[top_doc_idx]
        pos_ratio = float(np.mean(top_y == 1)) if top_y.size else 0.0
        neg_ratio = 1.0 - pos_ratio

        # Representative excerpts: pick the highest-loading positive and negative review.
        pos_candidates = [i for i in top_doc_idx.tolist() if y[i] == 1]
        neg_candidates = [i for i in top_doc_idx.tolist() if y[i] == 0]
        rep_pos = reviews[pos_candidates[0]] if pos_candidates else ""
        rep_neg = reviews[neg_candidates[0]] if neg_candidates else ""

        topics.append(
            TopicSummary(
                topic_idx=k,
                keywords=keywords,
                pos_ratio=pos_ratio,
                neg_ratio=neg_ratio,
                rep_pos_review=rep_pos,
                rep_neg_review=rep_neg,
            )
        )

    return topics


def get_controversial_topics(topics: Sequence[TopicSummary], *, top_n: int = 3) -> List[TopicSummary]:
    """
    Returns topics closest to a 50/50 split (most controversial).
    """
    if not topics:
        return []
    scored = sorted(topics, key=lambda t: abs(t.pos_ratio - 0.5))
    return scored[: min(top_n, len(scored))]

