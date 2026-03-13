from src.sentiment import build_pipeline, predict_sentiment, train_sentiment_model


def test_pipeline_builds():
    p = build_pipeline()
    assert hasattr(p, "fit")


def test_train_and_predict_shapes():
    # Need enough repeated terms due to min_df=2 in vectorizer.
    X = [
        "good movie good",
        "good film good",
        "great acting good",
        "bad movie bad",
        "terrible film bad",
        "awful plot bad",
    ]
    y = [1, 1, 1, 0, 0, 0]
    model = train_sentiment_model(X, y)
    preds = predict_sentiment(model, ["good", "bad"])
    assert preds.proba_pos.shape == (2,)
    assert ((preds.proba_pos >= 0.0) & (preds.proba_pos <= 1.0)).all()

