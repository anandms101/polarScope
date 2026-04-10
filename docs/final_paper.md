---
title: "PolarScope: Identifying and Explaining Polarization in Movie Reviews"
author:
  - Anand Mohan Singh
  - Trisha Ambati
  - Sampath Pranay Beela
date: "Northeastern University, CS 5100 Foundations of AI \\newline April 2026"
abstract: |
  Standard sentiment analysis classifies movie reviews as positive or negative but ignores whether
  audiences agree or disagree. Two films can share the same average rating yet exhibit very different
  opinion distributions; one may be universally liked while the other is fiercely debated. PolarScope
  addresses this gap with an end-to-end system that identifies polarizing movies from real IMDb
  reviews and explains why they divide audiences. We combine a TF-IDF and logistic regression
  sentiment classifier (90.8\% accuracy) with three complementary polarization metrics (bimodality
  coefficient, Shannon entropy, and confidence-adjusted disagreement) and Non-negative Matrix
  Factorization (NMF) topic extraction, all surfaced through an interactive Streamlit dashboard.
  Evaluation on 800 movies shows that our composite metric captures polarization signal beyond
  simple variance (Spearman $r = 0.78$), and bootstrap resampling confirms moderately stable
  rankings (mean Kendall $\tau = 0.62$, 95\% CI $[0.54, 0.68]$).
geometry: "left=1in, right=1in, top=1in, bottom=1in"
fontsize: 12pt
linestretch: 2
bibliography: references.bib
csl: ieee.csl
header-includes: |
  \usepackage{float}
  \usepackage{booktabs}
  \floatplacement{figure}{H}
  \pagestyle{plain}
---

# Introduction

Online review platforms such as IMDb collect millions of user opinions about movies, creating a rich but noisy signal about audience reception. Traditional sentiment analysis reduces each review to a binary positive or negative label, and aggregation typically produces a single average score per movie. However, this approach obscures a crucial dimension of audience behavior: disagreement. Two films can both average a 6.5 out of 10, yet one may be rated consistently around that midpoint while the other divides audiences into passionate admirers and vocal detractors. Recognizing this distinction is important for recommendation systems, content curation, and cultural analysis.

Some platforms already acknowledge this problem partially. Rotten Tomatoes separates critic and audience scores, and Metacritic highlights score distributions, but neither system attempts to quantify the degree of audience polarization or explain which aspects of a film drive disagreement. PolarScope addresses this gap by combining natural language processing with distributional statistics to measure, decompose, and explain opinion polarization at the review level.

Opinion polarization has been studied in political science and sociology for decades. DiMaggio et al. [@dimaggio2013polarization] provided an early framework for measuring attitudinal polarization in survey data, distinguishing between dispersion (the spread of opinions) and bimodality (clustering at extremes). Their statistical approach, which examines distributional properties rather than averages, directly informs our methodology. In the online review domain, Matakos and Tsaparas [@matakos2016temporal] studied temporal polarization dynamics in IMDb and Amazon ratings using distributional statistics including variance, mean deviation, and kurtosis. They showed that polarization is measurable at the distribution level and tends to increase over time as more users contribute ratings. Their work, however, operates solely on numerical ratings and does not analyze the textual content of reviews to explain what drives disagreement.

PolarScope extends this line of research in three ways. First, we move beyond numerical ratings to analyze the full text of reviews using a supervised sentiment classification pipeline, producing continuous probability estimates rather than discrete star ratings. Second, we combine three complementary polarization metrics (bimodality coefficient, Shannon entropy, and confidence-adjusted disagreement) into a weighted composite score that captures different facets of opinion disagreement. Third, we apply Non-negative Matrix Factorization (NMF) topic extraction to identify which specific discussion themes, such as acting, plot, or visual effects, drive the disagreement, making the polarization signal interpretable. The system provides two recommendation modes ("safe pick" for crowd-pleasers and "debate night" for conversation-starters) alongside an interactive Streamlit dashboard.

Sentiment analysis itself is a well-established area of AI and NLP research. Liu [@liu2012sentiment] offers a comprehensive survey of techniques ranging from lexicon-based methods to machine learning classifiers. We adopt the supervised machine learning approach, a core technique covered in foundational AI courses, specifically a TF-IDF vectorizer paired with logistic regression. This combination remains competitive for document-level sentiment classification and provides calibrated probability outputs that are essential for our downstream polarization metrics.

# Methods

## Data

We use the Kaggle dataset "IMDb Movie Reviews Grouped by Ratings" [@kaggle_imdb_grouped], which contains user reviews organized into 10 CSV files corresponding to ratings 1 through 10. Each file provides a MovieID (an IMDb title identifier such as `tt0120616`), a Rating (an integer from 1 to 10), and the full text of the user's review. The combined dataset spans approximately 156 MB and covers thousands of movies with varying review volumes.

During data loading, we strip HTML markup artifacts (e.g., `<br/>` tags) and normalize text to lowercase. We retain only movies with at least 5 reviews, subsample to 800 movies for a medium-scale run, and cap reviews at 200 per movie to control runtime and memory usage. All of these thresholds are configurable through command-line arguments in the training pipeline and through sidebar sliders in the dashboard.

## Sentiment Classification

We train a TF-IDF plus logistic regression pipeline using scikit-learn [@pedregosa2011sklearn]. The TF-IDF vectorizer extracts unigram and bigram features with a vocabulary capped at 10,000 terms, a minimum document frequency of 2, and a maximum document frequency of 95\%. The logistic regression classifier uses the L-BFGS solver with a maximum of 1,000 iterations.

When the standard IMDb 50K labeled sentiment dataset is available, we train on explicit positive and negative labels with a standard 80/20 train-test split. When that dataset is absent, we construct weak sentiment labels from the Kaggle per-movie ratings: reviews with ratings 1 through 4 are labeled negative, ratings 7 through 10 are labeled positive, and ratings 5 and 6 are discarded as ambiguous. This weak labeling strategy is grounded in the observation that extreme ratings correlate strongly with sentiment polarity.

The trained model outputs calibrated $P(\text{positive})$ probabilities for each review via `predict_proba`. These continuous outputs are essential for the downstream polarization metrics, which operate on probability distributions rather than binary labels.

## Polarization Metrics

For each movie, we compute the sentiment probability distribution over all of its reviews and derive three complementary metrics.

**Bimodality coefficient.** The bimodality coefficient (BC) is derived from the skewness ($g_1$) and excess kurtosis ($g_2$) of the sentiment probability distribution [@dimaggio2013polarization]:

$$BC = \frac{g_1^2 + 1}{g_2 + \frac{3(n-1)^2}{(n-2)(n-3)}}$$

A value above approximately 0.555 suggests a bimodal distribution, indicating that reviews cluster at both positive and negative extremes rather than around the center. We normalize the raw BC value by mapping the range $[0.30, 0.75]$ onto $[0, 1]$ for comparability.

**Shannon entropy.** We bin the sentiment probabilities into 10 equal-width bins over $[0, 1]$ and compute the normalized Shannon entropy:

$$H = -\frac{1}{\ln(10)} \sum_{i=1}^{10} p_i \ln(p_i)$$

where $p_i$ is the proportion of reviews falling in bin $i$. A uniform distribution (maximum disagreement) yields $H = 1$, while a single-bin distribution (complete consensus) yields $H = 0$. Entropy captures the overall spread of opinions regardless of whether they cluster at the extremes.

**Confidence-adjusted disagreement.** This metric weights each review by the model's prediction confidence. For a review with predicted probability $p$, we compute a signed sentiment $s = 2(p - 0.5)$ in $[-1, 1]$ and a confidence weight $c = 2|p - 0.5|$ in $[0, 1]$. The metric is the variance of the confidence-weighted signed sentiments:

$$CAD = \text{Var}(s \cdot c)$$

By down-weighting ambiguous predictions near 0.5, CAD isolates disagreement among reviews where the model is most confident. We normalize by mapping $[0, 0.25]$ onto $[0, 1]$.

**Composite score.** The three normalized metrics are combined into a single weighted composite:

$$\text{composite} = 0.4 \cdot BC_{\text{norm}} + 0.3 \cdot H + 0.3 \cdot CAD_{\text{norm}}$$

The bimodality coefficient receives the highest weight because bimodality is the strongest indicator of genuine audience polarization, as opposed to broad but unimodal disagreement. We also compute a variance-only baseline ($\text{Var}(p)$) for comparison.

## Topic Extraction

To explain what drives polarization, we apply NMF [@lee2011nmf] to the TF-IDF matrix of each movie's reviews. NMF decomposes the term-document matrix into non-negative factors, producing 5 latent topics per movie. Each topic is characterized by its top 6 keywords, defined as the highest-loading terms in the topic's component vector.

For each topic, we identify the 30 highest-loading reviews and compute the fraction that were classified as positive versus negative by the sentiment model. Topics whose positive/negative ratio falls closest to 50/50 are flagged as "controversial," since these represent the discussion themes where the audience is most divided. We also surface the most representative positive and negative review for each controversial topic as interpretive excerpts.

## Recommendation Modes

PolarScope provides two recommendation modes that translate polarization measurements into actionable suggestions. The "safe pick" mode ranks movies by ascending composite polarization score, breaking ties by descending mean sentiment, and surfaces crowd-pleasers that audiences broadly agree are enjoyable. The "debate night" mode ranks movies by descending composite score, breaking ties by descending review volume, and surfaces divisive conversation-starters with sufficient reviews to ensure the polarization signal is robust. Both modes enforce a minimum review threshold (default 5) to prevent recommendations based on unreliable metrics.

## Interactive Dashboard

We built an interactive Streamlit [@streamlit2024] dashboard organized into six tabs: Home (overview statistics and score distribution), Movie Explorer (searchable per-movie deep dives with sentiment histograms, radar charts, and NMF topics), Recommendations (safe pick and debate night modes), Metrics Comparison (composite vs. variance scatter plot), Case Studies (side-by-side polarizing vs. consensus comparisons), and Evaluation (baseline correlations and bootstrap stability). All visualizations are rendered with Plotly [@plotly2024], and data processing relies on pandas [@pandas2024], NumPy [@numpy2024], and SciPy [@scipy2024].

## Evaluation Methodology

We evaluate our composite metric along two dimensions: correlation with simpler baselines and ranking stability under perturbation. For baseline comparison, we compute Spearman rank correlations between the composite score and two baselines: (1) a variance-only score computed as $\text{Var}(p)$ over predicted sentiment probabilities, and (2) the mean sentiment score.

For ranking stability, we employ bootstrap resampling. In each iteration, we resample reviews within each movie with replacement, recompute all polarization metrics and the composite score, and re-rank movies accordingly. We then compute Kendall's $\tau$ between the original ranking and the bootstrap ranking. Over many iterations, the distribution of $\tau$ values and its 95\% confidence interval quantify how robust the rankings are to sampling variability in the underlying reviews.

# Results

We ran the full pipeline on 800 movies from the Kaggle dataset, processing up to 200 reviews per movie. The sentiment model was trained on weak labels derived from ratings (the standard IMDb 50K labeled dataset was not used in our primary evaluation). Training and metric computation completed in under two minutes on a standard laptop.

## Sentiment Model Performance

Table 1 reports the classification performance of the TF-IDF plus logistic regression sentiment model evaluated on a held-out test set of 2,291 reviews (from a total of 11,454 weakly labeled reviews, 80/20 split).

| Metric    | Precision | Recall | F1-score | Support |
|:----------|:---------:|:------:|:--------:|:-------:|
| Negative  | 0.933     | 0.825  | 0.876    | 898     |
| Positive  | 0.895     | 0.962  | 0.927    | 1,393   |
| **Overall accuracy** | | | **0.908** | **2,291** |

: Sentiment classifier performance on the held-out test set (weak labels from ratings).

The model achieves 90.8\% accuracy with strong F1 scores for both classes. Precision is higher for the negative class (0.933), while recall is higher for the positive class (0.962). This asymmetry reflects the nature of the weak labels: extreme-rated reviews (ratings 1 through 4 and 7 through 10) tend to carry clear sentiment signals, but negative reviews are somewhat more varied in expression. Figure 1 shows the dashboard Home tab with the composite score distribution across all 800 movies.

![Home tab: overview statistics, composite score distribution histogram, and top polarizing vs. consensus tables.](figures_homepage.png){width=90%}

## Baseline Correlations

Table 2 summarizes the Spearman rank correlations between our composite score and the two baselines.

| Metric pair                      | Spearman $r$ |
|:---------------------------------|:------------:|
| Composite vs. variance baseline  | 0.782        |
| Composite vs. mean sentiment     | $-$0.397     |
| Variance baseline vs. mean sent. | $-$0.670     |

: Spearman rank correlations between the composite polarization score and baseline metrics.

The composite score correlates positively with the variance baseline at $r = 0.782$, confirming that both metrics capture related polarization signal. The correlation is well below 1.0, however, which indicates that the bimodality and confidence-adjusted disagreement components contribute information not captured by variance alone. The negative correlation with mean sentiment ($r = -0.397$) is consistent with our expectation: polarizing movies tend to be neither universally liked nor universally disliked. Notably, the variance baseline has a stronger negative correlation with mean sentiment ($r = -0.670$) than our composite does, suggesting that our multi-metric approach partially decouples polarization measurement from average sentiment. Figure 2 visualizes this relationship.

![Metrics Comparison: composite polarization score vs. variance baseline. Points above the diagonal indicate movies our multi-metric approach rates as more polarizing than variance alone.](figures_scatter.png){width=90%}

## Bootstrap Ranking Stability

We performed 80 bootstrap iterations on the top 120 movies by review count and computed Kendall's $\tau$ between the original and each resampled ranking. The mean $\tau$ was 0.615 with a 95\% confidence interval of $[0.542, 0.678]$. The top and bottom of the polarization spectrum remain relatively consistent across bootstrap samples, while movies in the middle range may shift several positions. This level of stability is reasonable given the noise inherent in sentiment prediction and the relatively small review counts for some movies.

## Case Studies: Polarizing vs. Consensus Movies

To illustrate the system's output concretely, we present case studies from the top polarizing and top consensus movies identified by our pipeline.

**The Mummy (1999)** scored the highest composite polarization (0.686) among all 800 movies analyzed. Its bimodality coefficient was particularly high (0.890), indicating a strongly bimodal sentiment distribution. The NMF topics for this film included "film, fun, mummy, like" and "movie, mind, acting, just." The most positive review praised the adventure and humor ("Rick O'Connell leads the beautiful Evelyn Carnahan to Hamunaptra..."), while the most negative review dismissed the film bluntly ("Man, what a turkey. Let me see... \$150M on special effects and \$20 on plot and acting."). The sharp contrast in language illustrates what our bimodality metric captures quantitatively.

**Betrayed (2018)** ranked second in composite polarization (0.625). Unlike The Mummy, its high score was driven primarily by the confidence-adjusted disagreement metric (1.000, the maximum), with a lower bimodality coefficient (0.258). The NMF topics here ("good, topic, terrible, movie" and "great, thriller, suspense, acting terrible") showed 82\% negative loading, reflecting overall dissatisfaction but with a vocal minority of confident positive reviewers. This case demonstrates how different components of our composite score can dominate for different movies.

By contrast, **The Curious Case of Benjamin Button** (composite 0.100, mean sentiment 0.880) and **Digital Estate Planning** (composite 0.087, mean sentiment 0.940) exemplify consensus movies where audiences broadly agree. These films show nearly unimodal sentiment distributions and topic ratios heavily skewed toward positive sentiment. The mean composite score across all 800 movies was 0.373, with a range from 0.013 to 0.686. Figure 3 shows the Movie Explorer deep dive for a selected film, and Figure 4 shows the Evaluation tab output.

![Movie Explorer: per-movie sentiment probability histogram and polarization radar chart.](figures_explorer.png){width=90%}

![Evaluation tab: Spearman baseline correlations between composite, variance, and mean sentiment.](figures_eval_spearman.png){width=90%}

# Conclusions

PolarScope shows that opinion polarization in movie reviews can be measured, decomposed, and explained through a combination of sentiment classification, distributional metrics, and topic modeling. Our multi-metric composite score captures polarization signal that a single variance measure misses, particularly bimodal opinion distributions where audiences cluster at opposing extremes. The NMF topic extraction component adds an interpretive layer that moves beyond aggregate scores to pinpoint which discussion themes drive disagreement. Across the 800 movies we analyzed, the system identified films like *The Mummy* and *Betrayed* as the most divisive, while films like *The Curious Case of Benjamin Button* represented strong consensus, and the underlying topics and review excerpts made these classifications interpretable.

Compared to the approach of Matakos and Tsaparas [@matakos2016temporal], who analyzed polarization purely through rating distributions using variance, mean deviation, and kurtosis, PolarScope offers two key advantages. First, by operating on review text rather than numerical ratings alone, we can detect polarization even when ratings are absent or unreliable. Second, the NMF topic extraction provides explanations for *why* a movie is polarizing, not just *whether* it is. On the other hand, their temporal analysis captures how polarization evolves over time, a dimension PolarScope does not currently address. A simple variance-based approach also has the advantage of being easier to compute and interpret, though our evaluation shows it misses polarization patterns that our composite score captures (Spearman $r = 0.782$, not 1.0).

The system has several notable strengths. The pipeline is fully automated: given a set of movie reviews, it trains a sentiment model, computes polarization metrics for hundreds of movies, extracts explanatory topics, and presents everything through an interactive dashboard, all in under two minutes. The three-metric approach yields a more nuanced picture of polarization than any individual statistic. The sentiment model achieves 90.8\% accuracy on the weak-label test set, which is strong enough to produce meaningful probability distributions for downstream analysis. From an AI perspective, the project demonstrates how foundational techniques (supervised classification, feature engineering with TF-IDF, unsupervised topic modeling with NMF, and statistical evaluation via bootstrap resampling) can be composed into a system that solves a real analytical problem.

At the same time, the approach has clear limitations. The sentiment classifier relies on a TF-IDF plus logistic regression pipeline; more expressive models such as fine-tuned transformers could improve prediction quality, especially for nuanced or sarcastic reviews. When the labeled 50K dataset is unavailable, weak labels derived from numerical ratings introduce noise, because a user's star rating does not always match the sentiment expressed in their text. Bootstrap stability is moderate ($\tau = 0.615$) rather than high, meaning that rankings in the middle of the spectrum are sensitive to sampling, which is an inherent limitation when computing metrics over finite, noisy data.

Looking ahead, there are several promising directions for future work. Temporal analysis could investigate whether polarization scores shift over time for movies with dated reviews, building on the findings of Matakos and Tsaparas [@matakos2016temporal]. Probability calibration via Platt scaling or isotonic regression could improve the quality of the sentiment probabilities that feed into the polarization metrics. Aspect-based sentiment analysis could also replace the current document-level classification, enabling finer-grained measurement of disagreement about specific movie attributes such as acting, cinematography, or narrative structure.

# Works Cited
