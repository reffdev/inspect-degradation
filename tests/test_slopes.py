"""Tests for degradation-slope estimation in ``analysis/slopes.py``.

What we pin here:

1. Per-trace slope correctness on synthetic increasing / flat /
   decreasing traces.
2. Degenerate traces (too short, zero-variance x) are dropped with
   the right reason.
3. ``per_trace_mean_slope`` reports the average of surviving per-trace
   slopes, not the pooled OLS.
4. Pooled OLS differs from per-trace mean when trace lengths are
   imbalanced (documents the confound the architecture was designed
   around).
5. Bootstrap CI bracketing and reproducibility.
6. Loop-rate-slope correctly handles ``None`` loop labels.
7. Sign sanity: a monotonically increasing error pattern yields a
   positive slope; a flat trace yields zero.
"""

from __future__ import annotations

import math

import pytest
from conftest import make_graded_step, make_graded_trace
from numpy.random import default_rng

from inspect_degradation.analysis.slopes import (
    SlopeResult,
    error_rate_slope,
    loop_rate_slope,
    neutral_rate_slope,
    per_trace_mean_slope,
    pooled_slope,
)
from inspect_degradation.schema import Validity


def _trace(trace_id: str, validities: list[Validity], loops=None):
    if loops is None:
        loops = [False] * len(validities)
    steps = [
        make_graded_step(i, validity=v, is_looping=lp)
        for i, (v, lp) in enumerate(zip(validities, loops))
    ]
    return make_graded_trace(trace_id=trace_id, steps=steps)


def _increasing_error_trace(trace_id: str, n: int):
    """All pass until halfway, then all fail."""
    half = n // 2
    return _trace(
        trace_id,
        [Validity.pass_] * half + [Validity.fail] * (n - half),
    )


# ---------------------------------------------------------------------------
# Per-trace slope correctness
# ---------------------------------------------------------------------------


class TestPerTraceSlopePointValues:
    def test_flat_no_errors_yields_zero_slope(self):
        traces = [_trace(f"t{i}", [Validity.pass_] * 10) for i in range(5)]
        r = error_rate_slope(traces, rng=default_rng(0), n_resamples=300)
        # Every per-trace slope is exactly 0 (constant y).
        assert r.n_traces_used == 5
        assert math.isclose(r.value, 0.0)

    def test_flat_all_errors_yields_zero_slope(self):
        traces = [_trace(f"t{i}", [Validity.fail] * 10) for i in range(5)]
        r = error_rate_slope(traces, rng=default_rng(0), n_resamples=300)
        assert r.n_traces_used == 5
        assert math.isclose(r.value, 0.0)

    def test_late_errors_yield_positive_slope(self):
        traces = [_increasing_error_trace(f"t{i}", 10) for i in range(5)]
        r = error_rate_slope(traces, rng=default_rng(0), n_resamples=300)
        assert r.value > 0, f"expected positive slope, got {r.value}"
        assert r.n_traces_used == 5

    def test_early_errors_yield_negative_slope(self):
        traces = [
            _trace(f"t{i}", [Validity.fail] * 5 + [Validity.pass_] * 5)
            for i in range(5)
        ]
        r = error_rate_slope(traces, rng=default_rng(0), n_resamples=300)
        assert r.value < 0

    def test_known_slope_value_exact(self):
        # One trace with binary y=[0,0,0,1,1,1] at x=[0..5].
        # Slope: cov(x,y)/var(x).
        # mean_x = 2.5, mean_y = 0.5
        # cov = sum((x-2.5)(y-0.5)) / 6
        # x-2.5 = -2.5,-1.5,-0.5,0.5,1.5,2.5
        # y-0.5 = -0.5,-0.5,-0.5,0.5,0.5,0.5
        # products: 1.25, 0.75, 0.25, 0.25, 0.75, 1.25 → sum = 4.5
        # bias cov = 4.5/6 = 0.75
        # var_x (biased) = sum((x-2.5)^2)/6 = 17.5/6 ≈ 2.9167
        # slope = 0.75 / 2.9167 ≈ 0.25714
        t = _trace(
            "a",
            [
                Validity.pass_,
                Validity.pass_,
                Validity.pass_,
                Validity.fail,
                Validity.fail,
                Validity.fail,
            ],
        )
        r = error_rate_slope([t, t], rng=default_rng(0), n_resamples=300)
        # Two identical traces → per-trace mean is the same slope.
        assert math.isclose(r.value, 4.5 / 17.5, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# Drop handling
# ---------------------------------------------------------------------------


class TestSlopeDropReasons:
    def test_too_short_traces_dropped(self):
        short = _trace("a", [Validity.pass_, Validity.fail])  # 2 steps
        ok = _trace("b", [Validity.pass_, Validity.pass_, Validity.fail])
        r = error_rate_slope([short, ok], rng=default_rng(0), n_resamples=100)
        assert r.n_traces_used == 1
        assert r.drop_reasons == {"too_short": 1}
        # Only one valid trace → insufficient CI.
        assert r.estimate.method == "insufficient_data"

    def test_all_traces_dropped_yields_empty_estimate(self):
        too_shorts = [_trace(f"t{i}", [Validity.pass_] * 2) for i in range(5)]
        r = error_rate_slope(too_shorts, rng=default_rng(0), n_resamples=100)
        assert r.n_traces_used == 0
        assert r.drop_reasons == {"too_short": 5}
        assert r.estimate.method == "empty"

    def test_loop_slope_drops_traces_with_no_loop_labels(self):
        unlabeled = _trace(
            "a", [Validity.pass_] * 5, loops=[None] * 5
        )
        labeled = _trace(
            "b", [Validity.pass_, Validity.neutral, Validity.neutral],
            loops=[False, True, True],
        )
        r = loop_rate_slope([unlabeled, labeled], rng=default_rng(0), n_resamples=100)
        # unlabeled has 0 labeled steps → too_short
        assert r.drop_reasons.get("too_short", 0) == 1
        assert r.n_traces_used == 1


# ---------------------------------------------------------------------------
# Per-trace vs pooled divergence
# ---------------------------------------------------------------------------


class TestPerTraceVsPooled:
    def test_divergence_on_imbalanced_lengths(self):
        # Long trace: strong positive slope.
        long_bad = _increasing_error_trace("long", 20)
        # Short traces: flat no-error, contribute zero.
        shorts = [_trace(f"s{i}", [Validity.pass_] * 3) for i in range(10)]

        per_trace = error_rate_slope(
            [long_bad] + shorts, rng=default_rng(0), n_resamples=200
        )
        pooled = pooled_slope(
            [long_bad] + shorts, lambda s: s.validity == Validity.fail
        )

        # Pooled is dragged up by the long trace's many steps; per-trace
        # mean is a simple average where the long trace is 1/11 of the
        # weight. They should disagree in magnitude and the per-trace
        # value should be smaller.
        assert per_trace.value < pooled.value


# ---------------------------------------------------------------------------
# Bootstrap CI coverage
# ---------------------------------------------------------------------------


class TestSlopeBootstrapCI:
    def test_ci_brackets_point_value(self):
        traces = [_increasing_error_trace(f"t{i}", 10) for i in range(20)]
        r = error_rate_slope(traces, rng=default_rng(7), n_resamples=500)
        assert r.estimate.has_ci
        assert r.estimate.ci_low <= r.value <= r.estimate.ci_high

    def test_reproducible_with_seed(self):
        traces = [_increasing_error_trace(f"t{i}", 10) for i in range(10)]
        rng1 = default_rng(123)
        rng2 = default_rng(123)
        r1 = error_rate_slope(traces, rng=rng1, n_resamples=200)
        r2 = error_rate_slope(traces, rng=rng2, n_resamples=200)
        assert math.isclose(r1.value, r2.value)
        assert math.isclose(r1.estimate.ci_low, r2.estimate.ci_low)
        assert math.isclose(r1.estimate.ci_high, r2.estimate.ci_high)


# ---------------------------------------------------------------------------
# Neutral slope — distinct from error slope
# ---------------------------------------------------------------------------


class TestNeutralSlope:
    def test_flat_error_positive_neutral_slope(self):
        # Agent doesn't error; just neutral rate climbs over time.
        traces = [
            _trace(
                f"t{i}",
                [Validity.pass_, Validity.pass_, Validity.pass_, Validity.neutral, Validity.neutral],
            )
            for i in range(10)
        ]
        e_slope = error_rate_slope(traces, rng=default_rng(0), n_resamples=200)
        n_slope = neutral_rate_slope(traces, rng=default_rng(0), n_resamples=200)
        # Zero errors throughout → error slope is 0.
        assert math.isclose(e_slope.value, 0.0)
        # Neutrals late → neutral slope positive.
        assert n_slope.value > 0


# ---------------------------------------------------------------------------
# Return type contract
# ---------------------------------------------------------------------------


class TestSlopeResultType:
    def test_to_dict_is_json_safe(self):
        import json

        traces = [_increasing_error_trace(f"t{i}", 10) for i in range(5)]
        r = error_rate_slope(traces, rng=default_rng(0), n_resamples=100)
        payload = r.to_dict()
        json.dumps(payload)  # must not raise
        assert "estimate" in payload
        assert "n_traces_total" in payload
        assert "n_traces_used" in payload
        assert "drop_reasons" in payload

    def test_value_property_matches_estimate(self):
        traces = [_increasing_error_trace(f"t{i}", 10) for i in range(5)]
        r = error_rate_slope(traces, rng=default_rng(0), n_resamples=100)
        assert r.value == r.estimate.value
