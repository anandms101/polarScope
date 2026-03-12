from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import demo_movie_reviews, ensure_dirs, load_movie_reviews
from src.polarization import compute_all_metrics
from src.recommender import debate_night, safe_pick
from src.sentiment import load_model


PROCESSED_DIR = Path("data/processed")
METRICS_PATH = PROCESSED_DIR / "movie_polarization.parquet"
MOVIE_CSV_DEFAULT = Path("data/raw/movie_reviews_by_title.csv")


def _load_or_compute_metrics(movie_reviews: Dict[str, List[str]]) -> pd.DataFrame:
    ensure_dirs()
    model = load_model()
    df = compute_all_metrics(movie_reviews, sentiment_model=model, min_reviews=3)
    return df


def main() -> None:
    st.set_page_config(page_title="PolarScope", layout="wide")
    st.title("PolarScope")
    st.caption(
        "Detect polarizing movies from review disagreement using sentiment + multi-metric scoring."
    )

    with st.sidebar:
        st.header("Data")
        st.write("Provide a per-movie review CSV, or use the built-in demo dataset.")
        use_demo = st.toggle("Use built-in demo dataset", value=not MOVIE_CSV_DEFAULT.exists())
        min_reviews = st.slider("Min reviews per movie", min_value=3, max_value=50, value=3, step=1)
        st.divider()
        st.header("Actions")
        st.write("If you changed data, rerun `python train.py` for faster startup.")

    if use_demo:
        movie_reviews = demo_movie_reviews()
        st.info("Using built-in demo dataset (3 movies).")
    else:
        try:
            movie_reviews = load_movie_reviews(MOVIE_CSV_DEFAULT, min_reviews_per_movie=1)
            st.success(f"Loaded per-movie dataset from `{MOVIE_CSV_DEFAULT}`.")
        except Exception as e:
            st.error(str(e))
            st.stop()

    tab_home, tab_explorer, tab_recs, tab_compare = st.tabs(
        ["Home", "Movie Explorer", "Recommendations", "Metrics Comparison"]
    )

    # Try to use precomputed parquet if available; otherwise compute live.
    df = None
    if METRICS_PATH.exists():
        try:
            df = pd.read_parquet(METRICS_PATH)
        except Exception:
            df = None
    if df is None:
        try:
            model = load_model()
        except Exception as e:
            st.warning(
                "Sentiment model not found yet. Run `python train.py` once. "
                "For now, this page cannot compute metrics."
            )
            st.code("python train.py")
            st.stop()
        df = compute_all_metrics(movie_reviews, sentiment_model=model, min_reviews=min_reviews)

    if df.empty:
        st.warning("No movies passed the minimum review threshold.")
        st.stop()

    with tab_home:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Movies analyzed", int(df.shape[0]))
        c2.metric("Total reviews", int(df["num_reviews"].sum()))
        c3.metric("Most polarizing score", f"{df['composite'].max():.3f}")
        c4.metric("Least polarizing score", f"{df['composite'].min():.3f}")

        st.subheader("Top polarizing vs consensus")
        left, right = st.columns(2)
        with left:
            st.markdown("**Debate night (top)**")
            st.dataframe(
                debate_night(df, top_n=5, min_reviews=min_reviews)[
                    ["movie", "composite", "bimodality", "entropy", "confidence_adj", "num_reviews"]
                ],
                use_container_width=True,
            )
        with right:
            st.markdown("**Safe pick (top)**")
            st.dataframe(
                safe_pick(df, top_n=5, min_reviews=min_reviews)[
                    ["movie", "mean_sentiment", "composite", "num_reviews"]
                ],
                use_container_width=True,
            )

    with tab_explorer:
        st.subheader("Movie Explorer")
        movie = st.selectbox("Select a movie", options=df["movie"].tolist(), index=0)
        row = df[df["movie"] == movie].iloc[0]

        a, b, c = st.columns(3)
        a.metric("Composite polarization", f"{row['composite']:.3f}")
        b.metric("Mean sentiment (P[pos])", f"{row['mean_sentiment']:.3f}")
        c.metric("Reviews", int(row["num_reviews"]))

        radar_df = pd.DataFrame(
            {
                "metric": ["bimodality", "entropy", "confidence_adj"],
                "value": [row["bimodality"], row["entropy"], row["confidence_adj"]],
            }
        )
        fig_radar = px.line_polar(
            radar_df,
            r="value",
            theta="metric",
            line_close=True,
            range_r=[0, 1],
            title="Polarization profile",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        reviews = movie_reviews.get(movie, [])
        if reviews:
            st.subheader("Sample reviews")
            st.write("\n\n".join([f"- {r}" for r in reviews[:8]]))
        else:
            st.caption("No raw reviews available for this movie in the loaded dataset.")

    with tab_recs:
        st.subheader("Recommendations")
        mode = st.radio("Mode", options=["Safe pick", "Debate night"], horizontal=True)
        top_n = st.slider("Top N", min_value=5, max_value=30, value=10, step=1)
        if mode == "Safe pick":
            out = safe_pick(df, top_n=top_n, min_reviews=min_reviews)
        else:
            out = debate_night(df, top_n=top_n, min_reviews=min_reviews)
        st.dataframe(out, use_container_width=True)

    with tab_compare:
        st.subheader("Multi-metric vs variance baseline")
        fig = px.scatter(
            df,
            x="variance_baseline",
            y="composite",
            hover_name="movie",
            size="num_reviews",
            title="Composite polarization vs variance baseline",
            labels={
                "variance_baseline": "Variance baseline",
                "composite": "Composite score",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Variance alone can miss bimodal splits; the composite combines bimodality, entropy, and confidence-adjusted disagreement."
        )


if __name__ == "__main__":
    main()

