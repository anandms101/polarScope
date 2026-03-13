from src.topics import extract_topics, get_controversial_topics


def test_extract_topics_returns_list():
    reviews = [
        "great acting and pacing",
        "terrible acting and boring pacing",
        "amazing ending and visuals",
        "bad ending and confusing visuals",
        "loved the message and themes",
        "hated the message and themes",
        "wonderful soundtrack and direction",
        "awful soundtrack and direction",
        "strong performances",
        "weak performances",
        "great plot",
        "bad plot",
    ]
    labels = [1, 0] * 6
    topics = extract_topics(reviews, labels, n_topics=3)
    assert isinstance(topics, list)
    controversial = get_controversial_topics(topics, top_n=2)
    assert len(controversial) <= 2

