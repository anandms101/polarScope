import numpy as np

from src.polarization import bimodality_coefficient, confidence_adjusted_disagreement, entropy_score


def test_entropy_bounds():
    v = np.linspace(0, 1, 1000)
    e = entropy_score(v, bins=10)
    assert 0.0 <= e <= 1.0


def test_confidence_adjusted_non_negative():
    p = np.array([0.1, 0.9, 0.2, 0.8])
    cad = confidence_adjusted_disagreement(p)
    assert cad >= 0.0


def test_bimodality_higher_for_bimodal():
    unimodal = np.random.default_rng(0).normal(0.5, 0.05, size=500)
    unimodal = np.clip(unimodal, 0, 1)
    bimodal = np.concatenate(
        [
            np.random.default_rng(1).normal(0.2, 0.05, size=250),
            np.random.default_rng(2).normal(0.8, 0.05, size=250),
        ]
    )
    bimodal = np.clip(bimodal, 0, 1)
    assert bimodality_coefficient(bimodal) >= bimodality_coefficient(unimodal)

