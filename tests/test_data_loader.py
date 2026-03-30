import pandas as pd

from src.data_loader import _basic_normalize, _strip_html, build_weak_sentiment_labels_from_ratings


def test_strip_html_removes_tags():
    s = "hello<br />world <b>bold</b>"
    out = _strip_html(s)
    assert "br" not in out.lower()
    assert "bold" in out.lower()


def test_basic_normalize_lowercases():
    assert _basic_normalize("HeLLo") == "hello"


def test_build_weak_labels_from_ratings():
    df = pd.DataFrame(
        {
            "rating": [1, 2, 4, 7, 8, 10, 5, 6],
            "review": ["a", "b", "c", "d", "e", "f", "g", "h"],
        }
    )
    ds = build_weak_sentiment_labels_from_ratings(df)
    assert len(ds.X_train) > 0
    assert set(ds.y_train).issubset({0, 1})

