---
geometry: margin=0.65in
fontsize: 11pt
header-includes: |
  \usepackage{titlesec}
  \titlespacing*{\section}{0pt}{6pt}{3pt}
  \titlespacing*{\subsection}{0pt}{5pt}{2pt}
  \setlength{\parskip}{4pt}
  \pagestyle{empty}
---

\vspace{-2em}

# Final Progress Report — PolarScope

**GitHub:** [https://github.com/anandms101/polarScope](https://github.com/anandms101/polarScope)

## Introduction and Background

Standard sentiment analysis classifies movie reviews as positive or negative but ignores whether audiences *agree* or *disagree*. Two films can have the same average rating yet very different opinion distributions — one universally liked, the other fiercely debated. PolarScope addresses this gap: we built an end-to-end system that identifies polarizing movies from real IMDb reviews and explains *why* they divide audiences.

Our work is motivated by Matakos and Tsaparas (2016), who studied temporal polarization dynamics in IMDb and Amazon ratings using distributional statistics such as variance, mean deviation, and kurtosis [1]. They showed that polarization is measurable at the distribution level and tends to increase over time as more users contribute ratings. PolarScope extends their analysis by adding NLP-based topic extraction to explain which aspects of a movie (acting, plot, effects) drive disagreement, and by surfacing results through two recommendation modes — "safe pick" and "debate night" — that translate polarization measurements into actionable suggestions.

## Methods

**Sentiment classification.** We train a TF-IDF (unigram + bigram, 10K features) + logistic regression pipeline on real IMDb review data. When the standard IMDb 50K labeled dataset is available, we train on explicit labels; otherwise we construct weak sentiment labels from the Kaggle per-movie ratings dataset (ratings 1–4 → negative, 7–10 → positive). The trained model outputs calibrated P(positive) probabilities for each review.

**Polarization metrics.** For each movie, we compute three complementary metrics over the distribution of predicted sentiment probabilities: (1) *bimodality coefficient* — derived from skewness and kurtosis, captures whether the distribution has two peaks; (2) *Shannon entropy* — measures how spread the opinion distribution is across 10 bins; (3) *confidence-adjusted disagreement* — weights reviews by prediction confidence and measures how evenly confident reviews split between positive and negative. These three are combined into a weighted composite score. We also compute a variance-only baseline for comparison.

**Topic extraction.** For each movie, we apply Non-negative Matrix Factorization (NMF) on the TF-IDF matrix of its reviews to extract 5 latent topics. We compute the positive/negative review ratio loading on each topic to identify controversial topics — those closest to a 50/50 sentiment split. Representative reviews are surfaced for each side.

**Recommendation modes.** We rank movies into two modes: *Safe pick* (low composite polarization, high mean sentiment) for crowd-pleasers, and *Debate night* (high composite polarization, sufficient review volume) for divisive conversation-starters.

**Title resolution.** IMDb IDs (e.g., `tt0120616`) are resolved to human-readable movie titles by downloading and caching IMDb's title.basics dataset.

## What We Have Completed

We have implemented the full pipeline described in our original proposal:

- **Data pipeline:** Automated loading and preprocessing of the Kaggle per-movie reviews dataset (10 CSV files, ~156 MB, covering ratings 1–10). HTML stripping, normalization, and weak-label generation are handled in `src/data_loader.py`.
- **Sentiment model:** TF-IDF + logistic regression classifier trained and evaluated with precision/recall/F1 metrics. Model serialized via `joblib` for reuse.
- **Polarization scoring:** All three proposed metrics (bimodality, entropy, confidence-adjusted disagreement) plus the variance baseline are computed for 800 movies and stored as a Parquet file.
- **NMF topic extraction:** Per-movie topic decomposition with controversy scoring. The system identifies which discussion themes are driving disagreement and provides representative positive and negative review excerpts.
- **Evaluation:** Spearman correlation between our composite score and two baselines (variance-only, mean-sentiment-only). Bootstrap resampling (80+ iterations) computes Kendall tau ranking stability with 95% confidence intervals.
- **Case studies:** Automated generation of side-by-side polarizing vs. consensus movie case studies with topic-level explanations and review excerpts.
- **Interactive Streamlit dashboard:** Six tabs — Home (overview metrics, score distribution histogram, top polarizing vs. consensus tables), Movie Explorer (per-movie sentiment histogram, radar chart, NMF topics with expandable controversial topic excerpts, sentiment-labeled sample reviews), Recommendations, Metrics Comparison (scatter plot with diagonal reference line), Case Studies, and Evaluation (Spearman table, bootstrap Kendall tau histogram with interpretive feedback).
- **Testing:** Unit tests for all core modules (`pytest`): data loading, sentiment pipeline, polarization metrics, recommender, topic extraction, and evaluation.
- **Title resolution:** IMDb title IDs automatically resolved to human-readable movie names throughout the UI.

## Evaluation Results

Our composite metric correlates with the variance baseline at Spearman r = 0.78, confirming that it captures related but distinct polarization signal. The correlation with mean sentiment is negative (r = -0.40), which is expected — polarizing movies tend not to be universally liked or disliked. Bootstrap ranking stability yields a mean Kendall tau of 0.62 (CI95: [0.54, 0.68]), indicating moderately stable rankings; instability mainly affects movies with fewer reviews.

**Strengths:** The multi-metric approach captures bimodal opinion splits that variance alone misses. NMF topic extraction provides interpretable explanations beyond a single polarization number. The system handles 800+ movies end-to-end in under two minutes.

**Weaknesses:** The sentiment classifier uses weak labels from ratings when the 50K labeled dataset is unavailable, which may introduce noise. Bootstrap stability is moderate — movies with few reviews can shift rankings across samples. Topic quality depends on review volume; sparse-review movies sometimes produce generic topics.

## Remaining Work

The core system is complete and fully functional. Before the final submission we plan to:

1. **Temporal analysis** — Investigate whether polarization scores change over time for movies with dated reviews, extending Matakos and Tsaparas' temporal findings.
2. **Calibration** — Explore Platt scaling or isotonic regression to improve sentiment probability calibration.
3. **UI polish** — Minor visual refinements and deployment documentation.

## References

[1] A. Matakos and P. Tsaparas, "Temporal mechanisms of polarization in online reviews," *2016 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)*, San Francisco, CA, USA, 2016, pp. 529–532, doi: 10.1109/ASONAM.2016.7752286.
