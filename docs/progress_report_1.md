# Progress Report 1 — PolarScope

**GitHub:** [https://github.com/anandms101/polarScope](https://github.com/anandms101/polarScope)

## What have you already achieved?

So far we have implemented a minimal but working end‑to‑end version of PolarScope that goes from raw text reviews to a web‑based visualization of movie polarization. On the modeling side, we built a TF–IDF + logistic regression sentiment classifier and wrapped it as a reusable pipeline that can be trained on the Kaggle IMDb 50K dataset. The pipeline currently supports training, evaluation, and serialization via `joblib`, so we can easily retrain on larger or cleaner data later.

Building on this classifier, we implemented several polarization metrics that operate over the distribution of predicted sentiment scores for each movie. Concretely, the system computes a bimodality coefficient (using skewness and kurtosis of the sentiment probabilities), a normalized entropy score over binned probabilities, a confidence‑adjusted disagreement score that emphasizes high‑confidence extreme reviews, and a simple variance baseline. These metrics are combined into a composite polarization score that ranks movies from “consensus” to “highly divisive.”

To make the results accessible, we created a Streamlit UI that reads precomputed metrics and exposes them in three ways: (1) a home dashboard comparing the most and least polarizing movies, (2) a movie explorer with a radar plot of the three core metrics plus summary statistics, and (3) recommendation views for “Safe pick” (low composite score, high mean sentiment) and “Debate night” (high composite score). At this stage the app uses a small built‑in demo dataset grouped by movie so that the interface is fully interactive even before the Kaggle data is wired in.

## What are your immediate next steps?

Our first priority is to replace the demo data with real datasets. Concretely, we plan to (1) download the labeled IMDb 50K reviews into `data/raw/imdb_reviews_50k.csv` for proper sentiment training and (2) obtain or construct a per‑movie review file with at least `movie_title` and `review_text` columns. Once those are in place, we will retrain the sentiment model, recompute metrics, and check whether the learned polarization scores align with known polarizing films such as *The Last Jedi* and *Mother!*

The next major step is to add text‑based explanations. We intend to experiment with topic models (e.g., NMF or LDA on TF–IDF features) to decompose each movie’s reviews into interpretable aspects like acting, pacing, and plot. The goal is to show not just that a movie is polarizing, but which topics are driving disagreement and to surface representative reviews for each side. After that, we will extend the evaluation by comparing the composite metric against the variance baseline on a larger set of movies and, if time permits, use simple bootstrap resampling to check how stable the rankings are when we resample reviews.

## Are there any challenges or adjustments?

One practical challenge is that the standard IMDb 50K dataset is organized as independent reviews with labels but without reliable movie identifiers, whereas the project needs reviews grouped by film. For the progress demo we addressed this by hard‑coding a small synthetic dataset that mimics polarizing and consensus movies, but for the final project we will either need a separate Kaggle dataset with titles or will have to do additional preprocessing to align reviews to movies. A second challenge is designing polarization metrics that are robust to noisy model probabilities; the confidence‑adjusted disagreement attempt is a first pass, but we may need to tune normalization ranges or incorporate calibration if predictions are over‑ or under‑confident.

## What is your overall plan for the next month?

Over the next month we plan to iterate in roughly three phases. In the first 1–2 weeks we will focus on integrating the real datasets, retraining the sentiment model, and adding topic‑level explanations into the Streamlit UI. In week 3 we will prioritize evaluation and analysis: running the metrics on a larger set of movies, comparing against baselines, and writing up a few detailed case studies of consensus vs. polarizing films. In the final week we will refine the user interface, clean up the code and documentation, and prepare the final report and in‑class demo.
