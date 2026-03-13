from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import KAGGLE_RAW_DIR, ensure_dirs, load_kaggle_grouped_by_ratings
from src.evaluation import bootstrap_ranking_stability, compare_baselines, generate_case_studies
from src.polarization import compute_all_metrics
from src.recommender import debate_night, safe_pick
from src.sentiment import load_model, predict_sentiment
from src.title_resolver import load_title_cache
from src.topics import extract_topics, get_controversial_topics


PROCESSED_DIR = Path("data/processed")
METRICS_PATH = PROCESSED_DIR / "movie_polarization.parquet"
KAGGLE_DIR_DEFAULT = KAGGLE_RAW_DIR / "mlopssss__imdb-movie-reviews-grouped-by-ratings"


def _polarization_badge(score: float) -> str:
    if score >= 0.50:
        return "Highly polarizing"
    if score >= 0.30:
        return "Moderately divisive"
    if score >= 0.15:
        return "Mild disagreement"
    return "Strong consensus"


def main() -> None:
    st.set_page_config(page_title="PolarScope", page_icon="🎬", layout="wide")

    st.markdown(
        "<h1 style='margin-bottom:0'>PolarScope</h1>"
        "<p style='margin-top:0; color:gray;'>Identify movies that polarize audiences and discover <em>why</em> they're divisive.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Powered by **TF-IDF sentiment analysis**, three complementary **polarization metrics**, "
        "and **NMF topic extraction** on real IMDb reviews."
    )

    with st.sidebar:
        st.image("https://img.shields.io/badge/PolarScope-v1.0-blue?style=for-the-badge", width=180)
        st.markdown("[GitHub](https://github.com/anandms101/polarScope) · [Kaggle dataset](https://www.kaggle.com/datasets/mlopssss/imdb-movie-reviews-grouped-by-ratings)")
        st.divider()
        st.header("Settings")
        min_reviews = st.slider("Min reviews per movie", 3, 50, 5, 1)
        max_movies = st.slider("Max movies (subsample)", 50, 2000, 800, 50)
        max_reviews_per_movie = st.slider("Max reviews per movie", 20, 400, 200, 20)
        st.divider()
        st.caption("Data: IMDb reviews grouped by ratings (Kaggle). Run `python train.py` after changing settings.")

    try:
        kaggle_df = load_kaggle_grouped_by_ratings(
            KAGGLE_DIR_DEFAULT,
            min_reviews_per_movie=min_reviews,
            max_movies=max_movies,
            max_reviews_per_movie=max_reviews_per_movie,
        )
    except Exception as e:
        st.error(str(e))
        st.stop()

    movie_reviews: Dict[str, List[str]] = (
        kaggle_df.groupby("movie_id")["review"].apply(list).to_dict()  # type: ignore[assignment]
    )

    tab_home, tab_explorer, tab_recs, tab_compare, tab_case, tab_eval = st.tabs(
        ["Home", "Movie Explorer", "Recommendations", "Metrics Comparison", "Case Studies", "Evaluation"]
    )

    try:
        model = load_model()
    except Exception:
        st.warning("Sentiment model not found. Run `python train.py` first.")
        st.code("python train.py")
        st.stop()

    df = None
    if METRICS_PATH.exists():
        try:
            df = pd.read_parquet(METRICS_PATH)
        except Exception:
            df = None
    if df is None:
        df = compute_all_metrics(movie_reviews, sentiment_model=model, min_reviews=min_reviews)

    if df.empty:
        st.warning("No movies passed the minimum review threshold.")
        st.stop()

    title_cache = load_title_cache()
    if "title" not in df.columns:
        df["title"] = df["movie"].map(title_cache).fillna(df["movie"])

    # ── Home ─────────────────────────────────────────────────────────────────
    with tab_home:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Movies analyzed", int(df.shape[0]))
        c2.metric("Total reviews", f"{int(df['num_reviews'].sum()):,}")
        c3.metric("Most polarizing", f"{df['composite'].max():.3f}")
        c4.metric("Least polarizing", f"{df['composite'].min():.3f}")

        st.subheader("Score distribution")
        fig_dist = px.histogram(
            df,
            x="composite",
            nbins=30,
            title="Distribution of composite polarization scores across all movies",
            labels={"composite": "Composite polarization score"},
            color_discrete_sequence=["#636EFA"],
        )
        st.plotly_chart(fig_dist, width="stretch")

        st.subheader("Top polarizing vs consensus")
        left, right = st.columns(2)
        with left:
            st.markdown("**Debate night (most divisive)**")
            dn = debate_night(df, top_n=5, min_reviews=min_reviews)
            st.dataframe(
                dn[["title", "composite", "bimodality", "entropy", "confidence_adj", "num_reviews"]],
                width="stretch",
                hide_index=True,
            )
        with right:
            st.markdown("**Safe pick (strongest consensus)**")
            sp = safe_pick(df, top_n=5, min_reviews=min_reviews)
            st.dataframe(
                sp[["title", "mean_sentiment", "composite", "num_reviews"]],
                width="stretch",
                hide_index=True,
            )

    # ── Movie Explorer ───────────────────────────────────────────────────────
    with tab_explorer:
        sorted_titles = sorted(df["title"].tolist())
        selected_title = st.selectbox("Search for a movie", options=sorted_titles, index=0)
        row = df[df["title"] == selected_title].iloc[0]
        movie = row["movie"]

        badge = _polarization_badge(row["composite"])
        st.markdown(f"### {selected_title}")
        st.caption(f"IMDb ID: {movie} · Verdict: **{badge}**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Composite", f"{row['composite']:.3f}")
        m2.metric("Bimodality", f"{row['bimodality']:.3f}")
        m3.metric("Entropy", f"{row['entropy']:.3f}")
        m4.metric("Confidence-adj", f"{row['confidence_adj']:.3f}")

        col_chart, col_radar = st.columns(2)
        with col_radar:
            radar_df = pd.DataFrame(
                {
                    "metric": ["Bimodality", "Entropy", "Confidence-adj"],
                    "value": [row["bimodality"], row["entropy"], row["confidence_adj"]],
                }
            )
            fig_radar = px.line_polar(
                radar_df, r="value", theta="metric", line_close=True,
                range_r=[0, 1], title="Polarization profile",
            )
            fig_radar.update_traces(fill="toself", fillcolor="rgba(99,110,250,0.25)")
            st.plotly_chart(fig_radar, width="stretch")

        reviews = movie_reviews.get(movie, [])
        if not reviews:
            with col_chart:
                st.caption("No raw reviews available for this movie.")
        else:
            preds = predict_sentiment(model, reviews)
            with col_chart:
                hist_fig = px.histogram(
                    x=preds.proba_pos, nbins=20,
                    title="Sentiment distribution P(positive)",
                    labels={"x": "P(positive)"},
                    color_discrete_sequence=["#EF553B"],
                )
                st.plotly_chart(hist_fig, width="stretch")

            topics = extract_topics(reviews, preds.labels.tolist(), n_topics=5)
            controversial = get_controversial_topics(topics, top_n=3)

            st.subheader("Topic explanations (NMF)")
            if not topics:
                st.caption("Not enough reviews to extract stable topics for this movie.")
            else:
                tdf = pd.DataFrame(
                    [
                        {
                            "Topic": f"T{t.topic_idx}",
                            "Keywords": ", ".join(t.keywords),
                            "% Positive": f"{t.pos_ratio * 100:.0f}%",
                            "% Negative": f"{t.neg_ratio * 100:.0f}%",
                        }
                        for t in topics
                    ]
                )
                st.dataframe(tdf, width="stretch", hide_index=True)

                if controversial:
                    st.markdown("**Most controversial topics (closest to 50/50 split)**")
                    for t in controversial:
                        with st.expander(f"T{t.topic_idx}: {', '.join(t.keywords[:4])} — pos {t.pos_ratio:.0%} / neg {t.neg_ratio:.0%}"):
                            if t.rep_pos_review:
                                st.success(t.rep_pos_review[:500])
                            if t.rep_neg_review:
                                st.error(t.rep_neg_review[:500])

            st.subheader("Sample reviews")
            for i, r in enumerate(reviews[:6]):
                label = "positive" if preds.labels[i] == 1 else "negative"
                conf = preds.proba_pos[i]
                st.markdown(f"**[{label} — {conf:.2f}]** {r[:400]}")

    # ── Recommendations ──────────────────────────────────────────────────────
    with tab_recs:
        st.subheader("Recommendations")
        st.markdown(
            "**Safe pick** finds crowd-pleasers (low polarization, high sentiment). "
            "**Debate night** finds divisive conversation-starters."
        )
        mode = st.radio("Mode", options=["Safe pick", "Debate night"], horizontal=True)
        top_n = st.slider("Top N", 5, 30, 10, 1)
        out = safe_pick(df, top_n=top_n, min_reviews=min_reviews) if mode == "Safe pick" else debate_night(df, top_n=top_n, min_reviews=min_reviews)
        display_cols = ["title", "composite", "mean_sentiment", "bimodality", "entropy", "confidence_adj", "num_reviews"]
        st.dataframe(out[[c for c in display_cols if c in out.columns]], width="stretch", hide_index=True)

    # ── Metrics Comparison ───────────────────────────────────────────────────
    with tab_compare:
        st.subheader("Multi-metric vs variance baseline")
        st.markdown(
            "Each dot is a movie. The X axis is a simple variance-of-sentiment baseline; "
            "the Y axis is our composite score. Movies above the diagonal are ones our "
            "multi-metric approach rates as *more* polarizing than variance alone would suggest."
        )
        fig = px.scatter(
            df, x="variance_baseline", y="composite", hover_name="title",
            size="num_reviews",
            title="Composite polarization vs variance baseline",
            labels={"variance_baseline": "Variance baseline", "composite": "Composite score"},
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=df["variance_baseline"].max(), y1=df["variance_baseline"].max(), line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig, width="stretch")

    # ── Case Studies ─────────────────────────────────────────────────────────
    with tab_case:
        st.subheader("Case Studies")
        st.markdown("Side-by-side comparison of the most polarizing and most consensus movies, "
                     "with topic-level explanations and representative review excerpts.")
        cs_n = st.slider("Case studies per group", 2, 8, 4, 1)
        case = generate_case_studies(df, movie_reviews, sentiment_model=model, n=cs_n, n_topics=5)

        def _render_case(item: dict) -> None:
            title = title_cache.get(item["movie"], item["movie"])
            badge = _polarization_badge(item["metrics"]["composite"])
            st.markdown(f"#### {title}")
            st.caption(f"Composite: {item['metrics']['composite']:.3f} · {badge}")
            if item["controversial_topics"]:
                for t in item["controversial_topics"]:
                    st.write(f"- **{', '.join(t['keywords'][:4])}** (pos={t['pos_ratio']:.0%}, neg={t['neg_ratio']:.0%})")
            if item["example_pos_review"]:
                st.success(f"**Positive:** {item['example_pos_review'][:350]}")
            if item["example_neg_review"]:
                st.error(f"**Negative:** {item['example_neg_review'][:350]}")
            st.divider()

        left, right = st.columns(2)
        with left:
            st.markdown("**Most polarizing**")
            for item in case["polarizing"]:
                _render_case(item)
        with right:
            st.markdown("**Most consensus**")
            for item in case["consensus"]:
                _render_case(item)

    # ── Evaluation ───────────────────────────────────────────────────────────
    with tab_eval:
        st.subheader("Evaluation")
        st.markdown(
            "We evaluate our composite metric against two baselines (variance-only and mean-sentiment-only) "
            "using Spearman correlation, and test ranking stability via bootstrap resampling."
        )

        corr = compare_baselines(df)
        if corr:
            st.markdown("**Baseline correlations (Spearman)**")
            corr_df = pd.DataFrame([corr]).T.reset_index()
            corr_df.columns = ["Pair", "Spearman r"]
            corr_df["Pair"] = corr_df["Pair"].str.replace("spearman_", "").str.replace("_vs_", " vs ")
            st.dataframe(corr_df, width="stretch", hide_index=True)

            composite_vs_var = corr.get("spearman_composite_vs_variance_baseline", None)
            if composite_vs_var is not None:
                if composite_vs_var < 0.85:
                    st.info(
                        f"The composite score has a Spearman r = {composite_vs_var:.3f} with the variance baseline, "
                        "indicating the multi-metric approach captures polarization signal beyond simple variance."
                    )
                else:
                    st.caption(
                        f"The composite score is highly correlated (r = {composite_vs_var:.3f}) with variance on this sample."
                    )

        st.markdown("**Bootstrap ranking stability (Kendall tau)**")
        n_boot = st.slider("Bootstrap iterations", 20, 300, 80, 20)
        top_eval_movies = df.sort_values("num_reviews", ascending=False).head(120)["movie"].tolist()
        eval_reviews = {m: movie_reviews[m] for m in top_eval_movies if m in movie_reviews}
        try:
            boot = bootstrap_ranking_stability(
                eval_reviews, sentiment_model=model, n_iterations=n_boot, min_reviews=min_reviews
            )
            tau_mean = boot["kendall_tau_mean"]
            ci = boot["kendall_tau_ci95"]
            st.metric("Mean Kendall tau", f"{tau_mean:.3f}", delta=f"CI95: [{ci[0]:.3f}, {ci[1]:.3f}]")
            fig_tau = px.histogram(
                x=boot["kendall_tau_samples"], nbins=25,
                title="Bootstrap Kendall tau distribution",
                labels={"x": "Kendall tau"},
                color_discrete_sequence=["#00CC96"],
            )
            st.plotly_chart(fig_tau, width="stretch")
            if tau_mean > 0.85:
                st.success("Rankings are highly stable under bootstrap resampling.")
            elif tau_mean > 0.65:
                st.info("Rankings are moderately stable; some movies may shift a few positions between samples.")
            else:
                st.warning("Rankings show notable instability. Consider requiring more reviews per movie.")
        except Exception as e:
            st.warning(str(e))


if __name__ == "__main__":
    main()
