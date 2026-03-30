import numpy as np

from src.evaluation import compare_baselines


def test_compare_baselines_returns_keys():
    import pandas as pd

    df = pd.DataFrame(
        {
            "composite": [0.1, 0.2, 0.3],
            "variance_baseline": [0.2, 0.1, 0.4],
            "mean_sentiment": [0.7, 0.6, 0.5],
        }
    )
    out = compare_baselines(df)
    assert "spearman_composite_vs_variance_baseline" in out

