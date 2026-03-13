import pandas as pd

from src.recommender import debate_night, safe_pick


def test_safe_pick_orders_low_polarization_high_sentiment():
    df = pd.DataFrame(
        [
            {"movie": "a", "composite": 0.1, "mean_sentiment": 0.9, "num_reviews": 10},
            {"movie": "b", "composite": 0.2, "mean_sentiment": 0.95, "num_reviews": 10},
        ]
    )
    out = safe_pick(df, top_n=2, min_reviews=5)
    assert out.iloc[0]["movie"] == "a"


def test_debate_night_orders_high_polarization():
    df = pd.DataFrame(
        [
            {"movie": "a", "composite": 0.1, "num_reviews": 10, "mean_sentiment": 0.5},
            {"movie": "b", "composite": 0.9, "num_reviews": 10, "mean_sentiment": 0.5},
        ]
    )
    out = debate_night(df, top_n=2, min_reviews=5)
    assert out.iloc[0]["movie"] == "b"

