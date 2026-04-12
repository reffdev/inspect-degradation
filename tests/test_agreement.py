"""Tests for per-dimension agreement reports with bootstrap CIs."""

from __future__ import annotations

import math

from conftest import make_graded_step, make_graded_trace
from numpy.random import default_rng

from inspect_degradation.schema import (
    ComplexityLevel,
    Dependency,
    SeverityLevel,
    Validity,
)
from inspect_degradation.validation.agreement import (
    DimensionAgreement,
    pair_grades,
    score_agreement,
)


def _trace(trace_id, steps):
    return make_graded_trace(trace_id=trace_id, steps=steps)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_pairing_drops_unmatched_steps():
    pred = _trace("a", [make_graded_step(0), make_graded_step(1)])
    ref = _trace("a", [make_graded_step(0)])
    pairs = pair_grades([pred], [ref])
    assert [p.step_index for p in pairs] == [0]


def test_pairing_drops_unmatched_traces():
    pred = _trace("a", [make_graded_step(0)])
    ref = _trace("b", [make_graded_step(0)])
    assert pair_grades([pred], [ref]) == []


# ---------------------------------------------------------------------------
# Scoring — point values
# ---------------------------------------------------------------------------


def test_score_agreement_perfect_on_replicated_traces():
    """Perfect agreement on a multi-trace corpus.

    We replicate a matched pair across several traces so the
    cluster-bootstrap has enough units (> 1 trace) to try to compute
    a CI. With a single trace the bootstrap returns an
    ``insufficient_data`` estimate — correct, but we want to test
    the happy path here.
    """
    def _both(trace_id):
        return _trace(
            trace_id,
            [
                make_graded_step(
                    0, validity=Validity.pass_, complexity=ComplexityLevel.low
                ),
                make_graded_step(
                    1,
                    validity=Validity.fail,
                    severity=SeverityLevel.high,
                    dependency=Dependency.independent,
                    complexity=ComplexityLevel.medium,
                ),
            ],
        )

    preds = [_both(f"t{i}") for i in range(5)]
    refs = [_both(f"t{i}") for i in range(5)]
    pairs = pair_grades(preds, refs)
    report = score_agreement("g", pairs, rng=default_rng(0), n_resamples=300)

    assert report.n_pairs == 10
    # Perfect agreement → every κ is 1.0.
    assert math.isclose(report.per_dimension["validity"].value, 1.0)
    assert math.isclose(report.per_dimension["dependency"].value, 1.0)
    assert math.isclose(report.per_dimension["complexity"].value, 1.0)
    # Severity is only labeled on the failing step per trace.
    assert report.per_dimension["severity"].n_pairs == 5


def test_dimension_metrics_pin_the_mapping():
    pred = _trace("a", [make_graded_step(0)])
    ref = _trace("a", [make_graded_step(0)])
    report = score_agreement(
        "g", pair_grades([pred], [ref]), rng=default_rng(0), n_resamples=100
    )
    assert report.per_dimension["severity"].metric == "weighted_cohens_kappa"
    assert report.per_dimension["complexity"].metric == "weighted_cohens_kappa"
    assert report.per_dimension["validity"].metric == "cohens_kappa"
    assert report.per_dimension["dependency"].metric == "cohens_kappa"
    # Removed dimensions must not appear.
    assert "recovery" not in report.per_dimension
    assert "confidence" not in report.per_dimension


def test_score_agreement_severity_excludes_pass_steps():
    pred = _trace(
        "a",
        [
            make_graded_step(0, validity=Validity.pass_),
            make_graded_step(
                1,
                validity=Validity.fail,
                severity=SeverityLevel.low,
                dependency=Dependency.independent,
            ),
        ],
    )
    ref = _trace(
        "a",
        [
            make_graded_step(0, validity=Validity.pass_),
            make_graded_step(
                1,
                validity=Validity.fail,
                severity=SeverityLevel.medium,
                dependency=Dependency.independent,
            ),
        ],
    )
    pairs = pair_grades([pred], [ref])
    report = score_agreement("g", pairs, rng=default_rng(0), n_resamples=100)
    assert report.per_dimension["severity"].n_pairs == 1


# ---------------------------------------------------------------------------
# Scoring — bootstrap CI behavior
# ---------------------------------------------------------------------------


def test_single_trace_yields_insufficient_estimate():
    # One trace → one cluster → insufficient for cluster-bootstrap.
    pred = _trace("a", [make_graded_step(0), make_graded_step(1)])
    ref = _trace("a", [make_graded_step(0), make_graded_step(1)])
    report = score_agreement(
        "g", pair_grades([pred], [ref]), rng=default_rng(0), n_resamples=100
    )
    assert report.per_dimension["validity"].estimate.method == "insufficient_data"


def test_estimate_has_real_ci_on_multi_trace_mixed_corpus():
    # Half of the traces have a matching label, half disagree.
    matching = _trace(
        "m", [make_graded_step(0, validity=Validity.pass_)]
    )
    def _disagreeing(i):
        return _trace(
            f"d{i}",
            [make_graded_step(0, validity=Validity.fail, severity=SeverityLevel.medium, dependency=Dependency.independent)],
        )

    preds = [matching] * 5 + [_disagreeing(i) for i in range(5)]
    refs = [matching] * 5 + [
        _trace(
            f"d{i}",
            [make_graded_step(0, validity=Validity.pass_)],
        )
        for i in range(5)
    ]
    report = score_agreement(
        "g", pair_grades(preds, refs), rng=default_rng(0), n_resamples=300
    )
    est = report.per_dimension["validity"].estimate
    assert est.method.startswith("bootstrap")
    assert est.ci_low <= est.value <= est.ci_high


def test_empty_dimension_returns_empty_estimate():
    # No severity labels at all → severity estimate should be empty.
    pred = _trace(
        "a",
        [make_graded_step(0, validity=Validity.pass_)],
    )
    ref = _trace(
        "a",
        [make_graded_step(0, validity=Validity.pass_)],
    )
    report = score_agreement(
        "g", pair_grades([pred], [ref]), rng=default_rng(0), n_resamples=100
    )
    sev = report.per_dimension["severity"]
    assert sev.estimate.method == "empty"
    assert sev.n_pairs == 0


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_report_to_dict_is_json_safe():
    import json

    pred = _trace("a", [make_graded_step(0)])
    ref = _trace("a", [make_graded_step(0)])
    report = score_agreement(
        "g", pair_grades([pred], [ref]), rng=default_rng(0), n_resamples=50
    )
    out = report.to_dict()
    json.dumps(out)  # must not raise (degenerate nans are stringified)
    assert out["grader"] == "g"
    assert "validity" in out["per_dimension"]
    assert out["per_dimension"]["validity"]["metric"] == "cohens_kappa"
    assert "estimate" in out["per_dimension"]["validity"]
