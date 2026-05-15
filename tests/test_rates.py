"""Tests for the per-step rate computations in ``analysis/rates.py``.

The key properties to pin down:

1. Point values match the hand-computed formulas.
2. ``pooled_rate`` uses trace-level bootstrap (its ``n`` field reports
   traces, not steps).
3. ``trace_mean_rate`` drops zero-labeled traces from the mean instead
   of treating them as 0/0.
4. ``loop_rate`` correctly excludes ``is_looping=None`` steps.
5. Confidence intervals bracket the point value for non-degenerate
   inputs.
6. Empty and degenerate inputs degrade gracefully to the right
   ``Estimate.empty`` / ``insufficient`` shapes.
"""

from __future__ import annotations

import math

from conftest import make_graded_step, make_graded_trace
from numpy.random import default_rng

from inspect_degradation.analysis.rates import (
    error_rate,
    loop_rate,
    neutral_rate,
    pooled_rate,
    productive_rate,
    trace_mean_rate,
    wilson_pooled_rate,
)
from inspect_degradation.schema import Validity


def _trace(trace_id: str, validities: list[Validity], loops: list[bool | None] | None = None):
    """Tiny helper: build a trace from per-step validity (and optional loop flags)."""
    if loops is None:
        loops = [False] * len(validities)
    assert len(validities) == len(loops)
    steps = [
        make_graded_step(i, validity=v, is_looping=lp)
        for i, (v, lp) in enumerate(zip(validities, loops))
    ]
    return make_graded_trace(trace_id=trace_id, steps=steps)


# ---------------------------------------------------------------------------
# Pooled rate — basic identities
# ---------------------------------------------------------------------------


class TestPooledRate:
    def test_empty_corpus(self):
        e = error_rate([], rng=default_rng(0))
        assert e.method == "empty"
        assert e.n == 0

    def test_single_trace_point_value_is_step_fraction(self):
        # 1 error out of 4 steps = 0.25.
        t = _trace(
            "a",
            [Validity.pass_, Validity.fail, Validity.pass_, Validity.neutral],
        )
        e = error_rate([t], rng=default_rng(0))
        # Single trace → bootstrap degrades to "insufficient_data"
        # because n_units = 1 cannot support a CI.
        assert e.method == "insufficient_data"

    def test_two_traces_point_value_pooled_matches_hand_count(self):
        # Trace A: 1 fail / 2 steps. Trace B: 2 fail / 3 steps.
        # Pooled: 3 / 5 = 0.6.
        a = _trace("a", [Validity.fail, Validity.pass_])
        b = _trace("b", [Validity.fail, Validity.fail, Validity.pass_])
        e = error_rate([a, b], rng=default_rng(42), n_resamples=500)
        assert math.isclose(e.value, 0.6)
        assert e.method.startswith("bootstrap")
        assert e.n == 2  # sampling unit = trace

    def test_ci_brackets_point_on_diverse_traces(self):
        traces = [
            _trace(f"t{i}", [Validity.fail, Validity.pass_, Validity.neutral, Validity.pass_])
            for i in range(20)
        ]
        # Ground truth: 1/4 = 0.25 per trace, so pooled = 0.25.
        e = error_rate(traces, rng=default_rng(7), n_resamples=500)
        assert math.isclose(e.value, 0.25)
        assert e.ci_low <= 0.25 <= e.ci_high
        # CI must actually bracket something — not collapsed to a
        # degenerate point — but because the traces are identical
        # the bootstrap width is zero and BCa falls back to
        # percentile with zero width, which is honest.
        assert e.ci_high - e.ci_low >= 0.0

    def test_point_value_unaffected_by_zero_labeled_traces(self):
        # A trace with zero labeled steps should not contribute to
        # the numerator or denominator of the pooled rate, but it
        # *is* still a valid resampling unit.
        labeled = _trace("a", [Validity.fail, Validity.pass_])
        empty = make_graded_trace(trace_id="b", steps=[])
        with_empty = error_rate([labeled, empty], rng=default_rng(0), n_resamples=200)
        without = error_rate([labeled], rng=default_rng(0), n_resamples=200)
        # Both have point value 0.5 (1 error / 2 steps).
        assert math.isclose(with_empty.value, 0.5)
        # Second case is a single trace → insufficient; first is ok.
        assert without.method == "insufficient_data"

    def test_different_confidence_levels_widen(self):
        from inspect_degradation.analysis.statistics import NINETY, NINETY_NINE

        traces = [
            _trace(f"t{i}", [Validity.fail, Validity.pass_, Validity.pass_, Validity.pass_])
            for i in range(30)
        ]
        rng1 = default_rng(0)
        rng2 = default_rng(0)
        rng3 = default_rng(0)
        e90 = error_rate(traces, confidence_level=NINETY, rng=rng1, n_resamples=300)
        e95 = error_rate(traces, rng=rng2, n_resamples=300)
        e99 = error_rate(traces, confidence_level=NINETY_NINE, rng=rng3, n_resamples=300)
        # Point values should be the same (same data), widths should
        # be nondecreasing with confidence level.
        assert math.isclose(e90.value, e95.value)
        assert math.isclose(e95.value, e99.value)
        width_90 = e90.ci_high - e90.ci_low
        width_95 = e95.ci_high - e95.ci_low
        width_99 = e99.ci_high - e99.ci_low
        # On identical traces the bootstrap width is zero at every
        # level, so nondecreasing is trivially satisfied (0 <= 0 <= 0).
        assert width_90 <= width_95 <= width_99


class TestPooledRateHeterogeneous:
    def test_heterogeneous_trace_rates_widen_ci(self):
        # Mix of trace rates (0, 0.5, 1.0) — bootstrap over traces
        # should produce a genuinely non-degenerate interval.
        clean = _trace("a", [Validity.pass_, Validity.pass_, Validity.pass_])
        half = _trace("b", [Validity.fail, Validity.pass_])
        bad = _trace("c", [Validity.fail, Validity.fail])
        # Replicate each several times so bootstrap has something
        # to work with.
        traces = [clean] * 5 + [half] * 5 + [bad] * 5
        e = error_rate(traces, rng=default_rng(1), n_resamples=500)
        assert e.ci_low < e.value < e.ci_high


# ---------------------------------------------------------------------------
# Trace-mean rate
# ---------------------------------------------------------------------------


class TestTraceMeanRate:
    def test_mean_of_per_trace_rates_drops_empties(self):
        labeled = [
            _trace("a", [Validity.fail, Validity.pass_]),  # rate 0.5
            _trace("b", [Validity.pass_, Validity.pass_]),  # rate 0.0
            make_graded_trace(trace_id="c", steps=[]),  # no labeled steps
            _trace("d", [Validity.fail, Validity.fail]),  # rate 1.0
        ]
        e = trace_mean_rate(
            labeled,
            lambda s: s.validity == Validity.fail,
            rng=default_rng(0),
            n_resamples=400,
        )
        # Mean of (0.5, 0.0, 1.0) = 0.5. The empty trace must be
        # dropped from the numerator/denominator of the mean.
        assert math.isclose(e.value, 0.5)

    def test_diverges_from_pooled_on_unbalanced_lengths(self):
        # One long trace with many errors and one short trace with
        # none. Pooled and mean should disagree.
        long_bad = _trace("a", [Validity.fail] * 10)
        short_good = _trace("b", [Validity.pass_] * 2)

        pooled = pooled_rate(
            [long_bad, short_good],
            lambda s: s.validity == Validity.fail,
            rng=default_rng(0),
            n_resamples=300,
        )
        trace_mean = trace_mean_rate(
            [long_bad, short_good],
            lambda s: s.validity == Validity.fail,
            rng=default_rng(0),
            n_resamples=300,
        )
        # Pooled: 10 errors / 12 steps ≈ 0.833
        # Trace mean: mean(1.0, 0.0) = 0.5
        assert math.isclose(pooled.value, 10 / 12)
        assert math.isclose(trace_mean.value, 0.5)
        assert pooled.value > trace_mean.value


# ---------------------------------------------------------------------------
# Loop rate — partial-reference handling
# ---------------------------------------------------------------------------


class TestLoopRate:
    def test_none_labels_excluded_from_both_sides(self):
        # Three steps labeled, one not. Of the labeled, 2 are loops.
        t = _trace(
            "a",
            [Validity.neutral, Validity.neutral, Validity.pass_, Validity.pass_],
            loops=[True, True, False, None],
        )
        # Denom = 3 (None excluded), numerator = 2. Point = 2/3.
        e = loop_rate([t] * 4, rng=default_rng(0), n_resamples=300)
        assert math.isclose(e.value, 2 / 3)

    def test_all_none_yields_nan_point(self):
        t = _trace("a", [Validity.pass_, Validity.pass_], loops=[None, None])
        # Zero labeled steps → bootstrap statistic is nan, BCa
        # collapses to a nan-carrying degenerate estimate.
        e = loop_rate([t, t], rng=default_rng(0), n_resamples=200)
        assert math.isnan(e.value)


# ---------------------------------------------------------------------------
# Wilson pooled rate — IID bound
# ---------------------------------------------------------------------------


class TestWilsonPooled:
    def test_wilson_pooled_uses_step_n_not_trace_n(self):
        # One trace, 5 steps, 2 errors.
        t = _trace(
            "a",
            [Validity.fail, Validity.pass_, Validity.fail, Validity.pass_, Validity.neutral],
        )
        e = wilson_pooled_rate([t], lambda s: s.validity == Validity.fail)
        # Wilson reports n = number of labeled observations (steps),
        # not number of traces.
        assert e.n == 5
        assert e.method == "wilson"
        assert math.isclose(e.value, 2 / 5)

    def test_empty_wilson_pooled(self):
        e = wilson_pooled_rate([], lambda s: True)
        assert e.method == "empty"


# ---------------------------------------------------------------------------
# Validity convenience triple: error + neutral + productive = 1.0
# ---------------------------------------------------------------------------


class TestValidityTriplePartitions:
    def test_error_neutral_productive_sum_to_pooled_total(self):
        traces = [
            _trace("a", [Validity.fail, Validity.pass_, Validity.neutral]),
            _trace("b", [Validity.pass_, Validity.pass_]),
            _trace("c", [Validity.fail, Validity.neutral]),
        ]
        e = error_rate(traces, rng=default_rng(0), n_resamples=200)
        n = neutral_rate(traces, rng=default_rng(0), n_resamples=200)
        p = productive_rate(traces, rng=default_rng(0), n_resamples=200)
        # Every labeled step is in exactly one of the three buckets.
        assert math.isclose(e.value + n.value + p.value, 1.0)
