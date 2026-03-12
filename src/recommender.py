from __future__ import annotations

import pandas as pd


def safe_pick(df: pd.DataFrame, *, top_n: int = 10, min_reviews: int = 5) -> pd.DataFrame:
    """
    Consensus recommendations: low polarization, high mean sentiment.
    """
    if df.empty:
        return df
    d = df.copy()
    d = d[d["num_reviews"] >= min_reviews]
    d = d.sort_values(["composite", "mean_sentiment"], ascending=[True, False])
    return d.head(top_n).reset_index(drop=True)


def debate_night(df: pd.DataFrame, *, top_n: int = 10, min_reviews: int = 5) -> pd.DataFrame:
    """
    Polarizing recommendations: high polarization with sufficient review volume.
    """
    if df.empty:
        return df
    d = df.copy()
    d = d[d["num_reviews"] >= min_reviews]
    d = d.sort_values(["composite", "num_reviews"], ascending=[False, False])
    return d.head(top_n).reset_index(drop=True)

