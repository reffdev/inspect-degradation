"""Tests for the statistics primitives: :class:`Estimate`, Wilson,
bootstrap (BCa + percentile), OLS-with-CI, and the normal PPF helper.

These tests are deliberately thorough because every higher-level
analysis function in the project builds on this module. Bugs here
would silently contaminate every downstream statistical claim.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.random import default_rng

from inspect_degradation.analysis.statistics import (
    NINETY,
    NINETY_FIVE,
    NINETY_NINE,
    ConfidenceLevel,
    Estimate,
    _normal_cdf,
    _normal_ppf,
    bootstrap_estimate,
    ols_slope_with_interval,
    wilson_proportion_interval,
)


# ---------------------------------------------------------------------------
# ConfidenceLevel
# ---------------------------------------------------------------------------


class TestConfidenceLevel:
    def test_default_presets_match_standard_alphas(self):
        assert math.isclose(NINETY_FIVE.level, 0.95)
        assert math.isclose(NINETY_FIVE.alpha, 0.05)
        assert math.isclose(NINETY_NINE.alpha, 0.01)
        assert math.isclose(NINETY.alpha, 0.10)

    def test_percentiles_split_alpha_two_tailed(self):
        assert math.isclose(NINETY_FIVE.lower_percentile, 2.5)
        assert math.isclose(NINETY_FIVE.upper_percentile, 97.5)
        assert math.isclose(NINETY_NINE.lower_percentile, 0.5)
        assert math.isclose(NINETY_NINE.upper_percentile, 99.5)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1, float("nan")])
    def test_out_of_range_raises(self, bad):
        with pytest.raises(ValueError):
            ConfidenceLevel(bad)


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


class TestEstimate:
    def test_ordered_interval_accepted(self):
        e = Estimate(
            value=0.5,
            ci_low=0.3,
            ci_high=0.7,
            n=10,
            method="test",
            confidence_level=NINETY_FIVE,
        )
        assert e.has_ci
        assert e.value == 0.5

    def test_reversed_interval_rejected(self):
        with pytest.raises(ValueError, match="interval must be ordered"):
            Estimate(
                value=0.5,
                ci_low=0.7,
                ci_high=0.3,
                n=10,
                method="test",
                confidence_level=NINETY_FIVE,
            )

    def test_nan_endpoints_allowed(self):
        e = Estimate(
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=0,
            method="empty",
            confidence_level=NINETY_FIVE,
        )
        assert not e.has_ci
        assert math.isnan(e.value)

    def test_half_nan_interval_not_ordered_check(self):
        # A finite-on-one-side interval is legal — we don't
        # arbitrarily reject it. The callers that produce this (e.g.
        # one-sided degenerate bootstrap) flag themselves via the
        # method string.
        Estimate(
            value=0.5,
            ci_low=float("nan"),
            ci_high=0.9,
            n=5,
            method="partial",
            confidence_level=NINETY_FIVE,
        )

    def test_negative_n_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Estimate(
                value=0.0,
                ci_low=0.0,
                ci_high=0.0,
                n=-1,
                method="test",
                confidence_level=NINETY_FIVE,
            )

    def test_empty_factory(self):
        e = Estimate.empty()
        assert e.n == 0
        assert e.method == "empty"
        assert math.isnan(e.value)
        assert not e.has_ci

    def test_insufficient_factory(self):
        e = Estimate.insufficient(n=1)
        assert e.n == 1
        assert e.method == "insufficient_data"

    def test_to_dict_stringifies_nans(self):
        e = Estimate.empty()
        d = e.to_dict()
        assert d["value"] == "nan"
        assert d["ci_low"] == "nan"
        assert d["ci_high"] == "nan"
        assert d["method"] == "empty"
        assert d["n"] == 0
        assert d["confidence_level"] == 0.95

    def test_to_dict_preserves_finite_values(self):
        e = Estimate(
            value=0.42,
            ci_low=0.3,
            ci_high=0.55,
            n=100,
            method="wilson",
            confidence_level=NINETY_FIVE,
            se=0.05,
        )
        d = e.to_dict()
        assert d["value"] == 0.42
        assert d["ci_low"] == 0.3
        assert d["ci_high"] == 0.55
        assert d["se"] == 0.05


# ---------------------------------------------------------------------------
# Normal CDF / PPF
# ---------------------------------------------------------------------------


class TestNormalHelpers:
    def test_cdf_at_zero_is_half(self):
        assert math.isclose(_normal_cdf(0.0), 0.5, abs_tol=1e-12)

    def test_cdf_symmetry(self):
        for x in [0.5, 1.0, 1.96, 2.58, 3.0]:
            assert math.isclose(
                _normal_cdf(x) + _normal_cdf(-x), 1.0, abs_tol=1e-12
            )

    def test_ppf_inverse_of_cdf(self):
        # Round-trip at a dense grid of probabilities; the PPF must
        # invert the CDF within Moro-approximation accuracy.
        for p in np.linspace(0.01, 0.99, 99):
            assert math.isclose(_normal_cdf(_normal_ppf(p)), p, abs_tol=1e-8)

    def test_ppf_standard_quantiles(self):
        # These are the values every statistics textbook memorizes;
        # a mistake in the Moro coefficients would show up here first.
        assert math.isclose(_normal_ppf(0.975), 1.959963984540054, abs_tol=1e-8)
        assert math.isclose(_normal_ppf(0.995), 2.5758293035489004, abs_tol=1e-8)
        assert math.isclose(_normal_ppf(0.95), 1.6448536269514722, abs_tol=1e-8)

    def test_ppf_symmetric_about_half(self):
        for p in [0.1, 0.25, 0.4, 0.45]:
            assert math.isclose(
                _normal_ppf(p) + _normal_ppf(1.0 - p), 0.0, abs_tol=1e-8
            )

    def test_ppf_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            _normal_ppf(-0.1)
        with pytest.raises(ValueError):
            _normal_ppf(1.5)

    def test_ppf_edge_values_return_infinite(self):
        assert _normal_ppf(0.0) == float("-inf")
        assert _normal_ppf(1.0) == float("inf")


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


class TestWilsonInterval:
    def test_empty_returns_empty_estimate(self):
        e = wilson_proportion_interval(0, 0)
        assert e.method == "empty"
        assert e.n == 0

    def test_point_value_is_raw_sample_proportion(self):
        # The midpoint of the Wilson interval is shrunk; we report the
        # raw p_hat as the value so readers see the observed proportion.
        e = wilson_proportion_interval(30, 100)
        assert math.isclose(e.value, 0.30)

    def test_interval_contains_point(self):
        for s, n in [(1, 10), (50, 100), (99, 100), (0, 10), (10, 10)]:
            e = wilson_proportion_interval(s, n)
            if e.n == 0:
                continue
            assert e.ci_low <= e.value <= e.ci_high, f"violation at s={s}, n={n}"

    def test_interval_bounded_in_unit_interval(self):
        for s, n in [(0, 10), (10, 10), (1, 10000), (9999, 10000)]:
            e = wilson_proportion_interval(s, n)
            assert 0.0 <= e.ci_low <= 1.0
            assert 0.0 <= e.ci_high <= 1.0

    def test_zero_successes_still_has_upper_bound(self):
        # Clopper-Pearson and normal-approx both break at p=0; Wilson
        # returns a non-trivial upper bound even so.
        e = wilson_proportion_interval(0, 30)
        assert math.isclose(e.ci_low, 0.0, abs_tol=1e-12)
        assert 0.0 < e.ci_high < 1.0

    def test_all_successes_still_has_lower_bound(self):
        # At p=1 Wilson's upper bound lands very near — but not
        # necessarily exactly at — 1.0 due to FP arithmetic in the
        # closed-form, and the clamping ``min(1.0, ...)`` may not
        # trigger if the sum is already < 1.0 by an ULP. Either
        # outcome is acceptable; the test is that it's practically at
        # the boundary.
        e = wilson_proportion_interval(30, 30)
        assert math.isclose(e.ci_high, 1.0, abs_tol=1e-12)
        assert 0.0 < e.ci_low < 1.0

    def test_ninety_five_interval_widths_sanity(self):
        # At n=100, p=0.5, the 95% Wilson CI is well-known:
        # roughly (0.404, 0.596). Check against textbook values.
        e = wilson_proportion_interval(50, 100, confidence_level=NINETY_FIVE)
        assert math.isclose(e.ci_low, 0.4038, abs_tol=1e-3)
        assert math.isclose(e.ci_high, 0.5962, abs_tol=1e-3)

    def test_ninety_nine_wider_than_ninety_five(self):
        e95 = wilson_proportion_interval(50, 100, confidence_level=NINETY_FIVE)
        e99 = wilson_proportion_interval(50, 100, confidence_level=NINETY_NINE)
        width_95 = e95.ci_high - e95.ci_low
        width_99 = e99.ci_high - e99.ci_low
        assert width_99 > width_95

    def test_larger_n_shrinks_interval(self):
        e_small = wilson_proportion_interval(50, 100)
        e_large = wilson_proportion_interval(500, 1000)
        width_small = e_small.ci_high - e_small.ci_low
        width_large = e_large.ci_high - e_large.ci_low
        assert width_large < width_small

    def test_method_is_wilson(self):
        e = wilson_proportion_interval(10, 50)
        assert e.method == "wilson"

    def test_negative_n_rejected(self):
        with pytest.raises(ValueError):
            wilson_proportion_interval(0, -1)

    def test_out_of_range_successes_rejected(self):
        with pytest.raises(ValueError):
            wilson_proportion_interval(11, 10)
        with pytest.raises(ValueError):
            wilson_proportion_interval(-1, 10)


# ---------------------------------------------------------------------------
# OLS slope with normal-theory CI
# ---------------------------------------------------------------------------


class TestOLSSlope:
    def test_empty_data(self):
        e = ols_slope_with_interval([], [])
        assert e.method == "empty"

    def test_insufficient_data(self):
        e = ols_slope_with_interval([1.0, 2.0], [1.0, 2.0])
        assert e.method == "insufficient_data"

    def test_zero_variance_in_x(self):
        e = ols_slope_with_interval([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        assert e.method == "degenerate"

    def test_exact_slope_on_perfect_line(self):
        x = list(range(10))
        y = [2.0 * xi + 1.0 for xi in x]
        e = ols_slope_with_interval(x, y)
        assert math.isclose(e.value, 2.0, abs_tol=1e-10)
        # Perfect fit: residuals are zero, SE is zero, interval collapses.
        assert math.isclose(e.se, 0.0, abs_tol=1e-10)
        assert math.isclose(e.ci_low, 2.0, abs_tol=1e-10)
        assert math.isclose(e.ci_high, 2.0, abs_tol=1e-10)

    def test_slope_with_noise_has_nontrivial_interval(self):
        rng = default_rng(42)
        x = np.arange(100, dtype=float)
        y = 0.5 * x + rng.normal(0, 5.0, size=100)
        e = ols_slope_with_interval(x, y)
        assert e.method == "ols_normal"
        assert e.ci_low < e.value < e.ci_high
        # The true slope (0.5) should lie inside the 95% CI on this
        # well-behaved synthetic.
        assert e.ci_low <= 0.5 <= e.ci_high
        assert e.se is not None and e.se > 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            ols_slope_with_interval([1, 2, 3], [1, 2])

    def test_interval_ordering_for_negative_slope(self):
        x = np.arange(50, dtype=float)
        y = -x + 10.0
        e = ols_slope_with_interval(x, y)
        assert math.isclose(e.value, -1.0, abs_tol=1e-10)
        assert e.ci_low <= e.value <= e.ci_high


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


class TestBootstrap:
    def test_empty_units_returns_empty(self):
        e = bootstrap_estimate([], lambda u: 0.0, rng=default_rng(0))
        assert e.method == "empty"

    def test_single_unit_returns_insufficient_with_point_value(self):
        # A single unit cannot support a CI, but the point value is
        # still well-defined: the statistic of the single-unit
        # "corpus". We report the value with nan interval so callers
        # don't lose the number entirely.
        e = bootstrap_estimate([3.0], lambda u: float(sum(u)), rng=default_rng(0))
        assert e.method == "insufficient_data"
        assert math.isclose(e.value, 3.0)
        assert math.isnan(e.ci_low)
        assert math.isnan(e.ci_high)

    def test_bootstrap_mean_recovers_point_and_covers_true_mean(self):
        rng = default_rng(123)
        data = list(rng.normal(loc=5.0, scale=1.0, size=200))

        def mean(units):
            return float(np.mean(units))

        e = bootstrap_estimate(data, mean, rng=rng, n_resamples=500)
        assert e.method in ("bootstrap_bca", "bootstrap_percentile")
        assert math.isclose(e.value, np.mean(data), rel_tol=1e-10)
        # True mean 5.0 must fall inside the 95% CI on well-behaved data.
        assert e.ci_low <= 5.0 <= e.ci_high
        assert e.se is not None

    def test_bootstrap_on_skewed_statistic_bca_tag(self):
        rng = default_rng(7)
        # Exponential data is skewed — BCa should kick in, not fall
        # back to percentile.
        data = list(rng.exponential(scale=2.0, size=150))

        def mean(units):
            return float(np.mean(units))

        e = bootstrap_estimate(data, mean, rng=rng, n_resamples=500)
        assert e.method == "bootstrap_bca"

    def test_percentile_method_honored(self):
        rng = default_rng(0)
        data = list(range(20))

        def mean(units):
            return float(np.mean(units))

        e = bootstrap_estimate(
            data, mean, rng=rng, n_resamples=200, method="percentile"
        )
        assert e.method == "bootstrap_percentile"

    def test_invalid_method_rejected(self):
        with pytest.raises(ValueError):
            bootstrap_estimate([1.0, 2.0], lambda u: 0.0, method="other")

    def test_trace_level_bootstrap_resamples_whole_groups(self):
        # Verify the resampling unit is what the caller passes in, not
        # individual observations. We pass grouped data as the units;
        # the statistic function sees lists of groups, and the
        # aggregate it computes must treat them as atomic.
        groups = [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ]

        def grand_mean(unit_list):
            flat = [x for g in unit_list for x in g]
            return float(np.mean(flat))

        rng = default_rng(99)
        e = bootstrap_estimate(groups, grand_mean, rng=rng, n_resamples=400)
        # Point value is the grand mean of the original.
        assert math.isclose(e.value, 2.0, abs_tol=1e-10)
        # CI must include the point value and be non-trivial.
        assert e.ci_low <= e.value <= e.ci_high
        assert e.ci_high - e.ci_low > 0.0

    def test_nan_resamples_trigger_partial_tag(self):
        rng = default_rng(555)
        data = [1.0, 1.0, 1.0, 1.0]

        call_count = {"n": 0}

        def nan_on_some(units):
            call_count["n"] += 1
            # Every other call is non-finite.
            return float("nan") if call_count["n"] % 2 == 0 else 1.0

        e = bootstrap_estimate(
            data, nan_on_some, rng=rng, n_resamples=100, method="percentile"
        )
        assert "partial" in e.method

    def test_degenerate_all_constant_tag(self):
        rng = default_rng(1)
        data = [1.0] * 20

        def mean(units):
            return float(np.mean(units))

        e = bootstrap_estimate(data, mean, rng=rng, n_resamples=200)
        # Every resample gives the same value, BCa's proportion-below
        # is ill-defined, and we should have fallen cleanly to
        # percentile — or marked degenerate if even that collapses.
        assert e.method in (
            "bootstrap_percentile",
            "bootstrap_bca",
            "degenerate",
        )
        # The point value is correct regardless.
        assert math.isclose(e.value, 1.0)

    def test_reproducibility_with_seeded_rng(self):
        data = list(range(50))

        def mean(units):
            return float(np.mean(units))

        rng1 = default_rng(42)
        e1 = bootstrap_estimate(data, mean, rng=rng1, n_resamples=200)
        rng2 = default_rng(42)
        e2 = bootstrap_estimate(data, mean, rng=rng2, n_resamples=200)
        assert math.isclose(e1.ci_low, e2.ci_low)
        assert math.isclose(e1.ci_high, e2.ci_high)

    def test_wider_ci_for_higher_confidence(self):
        rng1 = default_rng(42)
        rng2 = default_rng(42)
        data = list(np.arange(100, dtype=float))

        def mean(units):
            return float(np.mean(units))

        e95 = bootstrap_estimate(data, mean, rng=rng1, confidence_level=NINETY_FIVE)
        e99 = bootstrap_estimate(data, mean, rng=rng2, confidence_level=NINETY_NINE)
        assert (e99.ci_high - e99.ci_low) > (e95.ci_high - e95.ci_low)
