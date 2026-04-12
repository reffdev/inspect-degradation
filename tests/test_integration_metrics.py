"""Tests for the validity-decomposition metrics in the integration layer.

These exercise the metrics directly on synthetic ``GradedTrace`` payloads
wrapped in fake ``Score`` objects -- no Inspect AI runtime needed. The
goal is to pin the semantics: ``error_rate``, ``neutral_rate``, and
``productive_rate`` partition the steps, and the slope metrics correctly
distinguish "more mistakes late" from "more flailing late".

After the @metric decorator migration, each metric is a factory:
``productive_rate()`` returns a callable that takes ``list[Score]``.
"""

from dataclasses import dataclass, field
from typing import Any

from conftest import make_graded_step, make_graded_trace

from inspect_degradation.integration.metrics import (
    error_rate_slope,
    loop_chain_length_mean,
    loop_rate,
    loop_rate_slope,
    neutral_rate,
    neutral_rate_slope,
    productive_rate,
)
from inspect_degradation.integration.scorer import GRADED_TRACE_METADATA_KEY
from inspect_degradation.schema import GradedTrace, Validity


@dataclass
class _FakeScore:
    metadata: dict[str, Any] = field(default_factory=dict)


def _score_for(trace: GradedTrace) -> _FakeScore:
    return _FakeScore(metadata={GRADED_TRACE_METADATA_KEY: trace.model_dump(mode="json")})


def _trace(validities: list[Validity]) -> GradedTrace:
    return make_graded_trace(
        steps=[make_graded_step(i, validity=v) for i, v in enumerate(validities)]
    )


def test_productive_and_neutral_rates_sum_with_error_rate():
    trace = _trace(
        [
            Validity.pass_,
            Validity.pass_,
            Validity.neutral,
            Validity.fail,
        ]
    )
    scores = [_score_for(trace)]
    p = productive_rate()(scores)
    n = neutral_rate()(scores)
    # Error rate isn't a top-level metric (it's a slope), but we can
    # reconstruct the count and check the partition explicitly.
    assert p == 0.5
    assert n == 0.25
    # The remaining 0.25 must be the error fraction.
    assert 1.0 - p - n == 0.25


def test_neutral_slope_distinguishes_flailing_from_failing():
    # "Flailing late": passes early, neutrals late, no failures.
    flailing = _trace(
        [Validity.pass_, Validity.pass_, Validity.pass_, Validity.neutral, Validity.neutral]
    )
    # "Failing late": passes early, failures late, no neutrals.
    failing = _trace(
        [Validity.pass_, Validity.pass_, Validity.pass_, Validity.fail, Validity.fail]
    )

    flail_neutral_slope = neutral_rate_slope()([_score_for(flailing)])
    flail_error_slope = error_rate_slope()([_score_for(flailing)])
    fail_neutral_slope = neutral_rate_slope()([_score_for(failing)])
    fail_error_slope = error_rate_slope()([_score_for(failing)])

    # Flailing trace: positive neutral slope, ~zero error slope.
    assert flail_neutral_slope > 0
    assert flail_error_slope == 0.0
    # Failing trace: positive error slope, ~zero neutral slope.
    assert fail_error_slope > 0
    assert fail_neutral_slope == 0.0


def test_metrics_skip_scores_without_graded_trace_metadata():
    # Coexistence with other scorers: a Score without our metadata key
    # must not contribute to the metric (and must not crash).
    blank = _FakeScore(metadata={})
    real = _score_for(_trace([Validity.pass_, Validity.pass_]))
    assert productive_rate()([blank, real]) == 1.0


def test_metrics_handle_empty_input():
    import math

    assert math.isnan(productive_rate()([]))
    assert math.isnan(neutral_rate()([]))
    assert math.isnan(error_rate_slope()([]))
    assert math.isnan(neutral_rate_slope()([]))
    assert math.isnan(loop_rate()([]))
    assert math.isnan(loop_rate_slope()([]))
    assert math.isnan(loop_chain_length_mean()([]))


# ---- loop metrics --------------------------------------------------------


def _loop_trace(specs):
    """Build a trace from a list of (validity, is_looping) tuples."""
    steps = []
    for i, (v, looping) in enumerate(specs):
        steps.append(make_graded_step(i, validity=v, is_looping=looping))
    return make_graded_trace(steps=steps)


def test_loop_rate_counts_labeled_steps_only():
    trace = _loop_trace(
        [
            (Validity.pass_, False),
            (Validity.neutral, True),
            (Validity.neutral, True),
            (Validity.fail, False),
        ]
    )
    # 2 of 4 labeled steps are loops.
    assert loop_rate()([_score_for(trace)]) == 0.5


def test_loop_rate_slope_detects_late_looping():
    # Early clean steps, late consecutive loops -- classic late-trace
    # looping signature; loop_rate_slope must be positive.
    late_loops = _loop_trace(
        [
            (Validity.pass_, False),
            (Validity.pass_, False),
            (Validity.pass_, False),
            (Validity.neutral, True),
            (Validity.neutral, True),
        ]
    )
    # Early loops, late clean -- slope must be negative.
    early_loops = _loop_trace(
        [
            (Validity.neutral, True),
            (Validity.neutral, True),
            (Validity.pass_, False),
            (Validity.pass_, False),
            (Validity.pass_, False),
        ]
    )
    assert loop_rate_slope()([_score_for(late_loops)]) > 0
    assert loop_rate_slope()([_score_for(early_loops)]) < 0


def test_loop_chain_length_mean_aggregates_across_traces():
    # Trace 1: one chain of length 2. Trace 2: one chain of length 4.
    # Expected mean: 3.0.
    t1 = _loop_trace(
        [
            (Validity.pass_, False),
            (Validity.neutral, True),
            (Validity.neutral, True),
            (Validity.pass_, False),
        ]
    )
    t2 = _loop_trace(
        [
            (Validity.neutral, True),
            (Validity.neutral, True),
            (Validity.neutral, True),
            (Validity.neutral, True),
        ]
    )
    assert loop_chain_length_mean()([_score_for(t1), _score_for(t2)]) == 3.0


def test_loop_metrics_skip_scores_without_metadata():
    blank = _FakeScore(metadata={})
    real = _score_for(
        _loop_trace([(Validity.pass_, False), (Validity.neutral, True)])
    )
    assert loop_rate()([blank, real]) == 0.5
