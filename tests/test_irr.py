import math

import pytest

from inspect_degradation.validation.irr import (
    accuracy,
    cohens_kappa,
    krippendorff_alpha_nominal,
    pearson_r,
    weighted_cohens_kappa,
)


def test_kappa_perfect_agreement():
    assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_kappa_chance_disagreement_negative():
    a = ["pass", "fail", "pass", "fail"]
    b = ["fail", "pass", "fail", "pass"]
    assert cohens_kappa(a, b) < 0.0


def test_kappa_single_label_collapses_to_perfect_or_zero():
    # Both raters used "x" everywhere; pe == 1, undefined limit, we return 1.0.
    assert cohens_kappa(["x", "x", "x"], ["x", "x", "x"]) == 1.0


def test_kappa_rejects_length_mismatch():
    with pytest.raises(ValueError):
        cohens_kappa(["a"], ["a", "b"])


def test_kappa_rejects_empty():
    with pytest.raises(ValueError):
        cohens_kappa([], [])


def test_alpha_perfect_two_raters():
    raters = [["a", "b", "a"], ["a", "b", "a"]]
    assert krippendorff_alpha_nominal(raters) == 1.0


def test_alpha_handles_missing_values():
    raters = [["a", "b", None, "c"], ["a", "b", "c", "c"]]
    # Item 2 has only one rater present and is dropped.
    val = krippendorff_alpha_nominal(raters)
    assert 0.0 < val <= 1.0


def test_alpha_requires_two_raters():
    with pytest.raises(ValueError):
        krippendorff_alpha_nominal([["a", "b"]])


def test_pearson_r_identity():
    assert pearson_r([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_pearson_r_anti_correlated():
    assert pearson_r([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_pearson_r_zero_variance_returns_nan():
    assert math.isnan(pearson_r([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_accuracy_basic():
    assert accuracy([True, False, True], [True, True, True]) == pytest.approx(2 / 3)


# ---- weighted Cohen's kappa ----------------------------------------------


_RANK = {"low": 1, "medium": 2, "high": 3}


def _rank(label):
    return _RANK[label]


def test_weighted_kappa_perfect_agreement_is_one():
    a = ["low", "medium", "high", "medium"]
    assert weighted_cohens_kappa(a, a, rank=_rank) == 1.0


def test_weighted_kappa_penalizes_distant_disagreement_more_than_near():
    # Same number of disagreements; b1 confuses adjacent levels (low<->medium),
    # b2 confuses far levels (low<->high). Weighted kappa must rank b2 worse.
    a = ["low", "low", "high", "high"]
    b_near = ["medium", "medium", "high", "high"]
    b_far = ["high", "high", "high", "high"]
    near = weighted_cohens_kappa(a, b_near, rank=_rank)
    far = weighted_cohens_kappa(a, b_far, rank=_rank)
    assert near > far


def test_weighted_kappa_quadratic_weights_supported():
    a = ["low", "medium", "high"]
    b = ["medium", "medium", "high"]
    val = weighted_cohens_kappa(a, b, rank=_rank, weights="quadratic")
    assert -1.0 <= val <= 1.0


def test_weighted_kappa_rejects_unknown_weighting():
    with pytest.raises(ValueError):
        weighted_cohens_kappa(["low"], ["low"], rank=_rank, weights="bogus")  # type: ignore[arg-type]
