"""Tests for the measurement-error correction module.

Priority ordering of the tests below matches the module's value
hierarchy:

1. **Ground-truth recovery for SIMEX.** Generate clean synthetic
   data with a known regression slope, apply label-flip noise at
   a known rate, verify that SIMEX recovers the *original* slope
   substantially better than the naive fit on the noisy data.
   Without this test the whole measurement-error module could be
   silently wrong — the mechanism is subtle and a math error
   would go undetected.

2. **Ground-truth recovery for deconfound_proportion.** Same idea
   on a simpler problem: a known true rate, a confusion matrix
   with known TPR/FPR, verify the deconfounded estimate matches
   the true rate.

3. **ConfusionMatrix construction and derived metrics.** TPR/FPR
   formulas, κ consistency check, from_grade_pairs, from_label_pairs,
   degenerate inputs.

4. **SIMEX schedule, serialization, and plumbing.** Tests that
   don't touch the correction logic but verify the surface-area
   contracts downstream reports will rely on.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng

pytest.importorskip("statsmodels")  # noqa: E402

from inspect_degradation.analysis.measurement_error import (  # noqa: E402
    ConfusionMatrix,
    SimexResult,
    deconfound_proportion,
    simex_correct,
)
from inspect_degradation.analysis.statistics import NINETY_FIVE  # noqa: E402


# ---------------------------------------------------------------------------
# SIMEX ground-truth recovery — the load-bearing test
# ---------------------------------------------------------------------------


class TestSimexRecovery:
    """SIMEX must recover a known slope from noise-contaminated data.

    Setup:
      * Generate 3000 rows of synthetic data where ``y = a + b*x``
        with small noise, ``x`` uniformly in [0, 10], and
        ``b = 0.05``.
      * Threshold ``y`` at its median to produce a clean binary
        outcome whose OLS slope is known.
      * Flip each binary label with probability ``p_flip`` to
        simulate grader noise.
      * Fit a simple OLS slope on the *noisy* binary outcome — this
        is the naive estimate.
      * Apply SIMEX with ``flip_probability=p_flip`` and verify the
        corrected slope is closer to the clean OLS slope.

    Note: the "ground truth" here is the clean-data OLS slope on
    the thresholded binary outcome, not the underlying linear
    coefficient (thresholding nonlinearly transforms the slope).
    SIMEX corrects for *additional* noise on top of whatever
    transformation the modeling chain has already applied.
    """

    def _make_contaminated_corpus(
        self,
        *,
        n: int = 3000,
        true_slope_on_y: float = 0.05,
        p_flip: float = 0.15,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, float, float]:
        rng = default_rng(seed)
        x = rng.uniform(0, 10, size=n)
        noise = rng.normal(0, 0.2, size=n)
        y_continuous = 1.0 + true_slope_on_y * x + noise
        threshold = float(np.median(y_continuous))
        y_binary = (y_continuous > threshold).astype(bool)

        # Clean-data OLS slope — our reference for what SIMEX
        # should try to recover.
        clean_slope = float(
            np.cov(x, y_binary.astype(float), bias=True)[0, 1] / np.var(x)
        )

        # Add independent label flips with probability p_flip.
        flips = rng.random(size=n) < p_flip
        y_noisy = np.where(flips, ~y_binary, y_binary)

        df = pd.DataFrame({"x": x, "y": y_noisy})
        return df, clean_slope, p_flip

    def _ols_slope_fit(self, df: pd.DataFrame) -> float:
        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        return float(np.dot(x_centered, y_centered) / np.dot(x_centered, x_centered))

    def test_simex_recovers_slope_better_than_naive(self):
        df, clean_slope, p_flip = self._make_contaminated_corpus(seed=42)

        result = simex_correct(
            df,
            outcome_col="y",
            flip_probability=p_flip,
            fit_fn=self._ols_slope_fit,
            lambdas=(0.0, 0.5, 1.0, 1.5, 2.0),
            n_repeats=30,
            rng=default_rng(1),
        )

        naive_bias = abs(result.naive.value - clean_slope)
        corrected_bias = abs(result.corrected.value - clean_slope)

        # The naive estimate should be biased toward zero by label
        # noise; the corrected estimate should be closer to the
        # clean slope. We allow a generous tolerance because SIMEX
        # is a consistent estimator, not an exact one, and the
        # Monte Carlo noise at n_repeats=30 is non-trivial.
        assert corrected_bias < naive_bias, (
            f"SIMEX failed to improve on naive: clean={clean_slope:.4f}, "
            f"naive={result.naive.value:.4f}, "
            f"corrected={result.corrected.value:.4f}"
        )

    def test_simex_points_sequence_matches_schedule(self):
        df, _, p_flip = self._make_contaminated_corpus(seed=7)
        result = simex_correct(
            df,
            outcome_col="y",
            flip_probability=p_flip,
            fit_fn=self._ols_slope_fit,
            lambdas=(0.0, 0.5, 1.0, 1.5, 2.0),
            n_repeats=20,
            rng=default_rng(2),
        )
        assert [p.lambda_ for p in result.points] == [0.0, 0.5, 1.0, 1.5, 2.0]
        # All coefficients should be finite on this well-behaved case.
        for p in result.points:
            assert math.isfinite(p.coefficient)

    def test_simex_corrected_has_bootstrap_ci(self):
        df, _, p_flip = self._make_contaminated_corpus(seed=11)
        result = simex_correct(
            df,
            outcome_col="y",
            flip_probability=p_flip,
            fit_fn=self._ols_slope_fit,
            n_repeats=20,
            rng=default_rng(3),
        )
        assert result.corrected.method == "simex_bootstrap"
        assert result.corrected.ci_low <= result.corrected.value <= result.corrected.ci_high
        assert result.corrected.se is not None

    def test_simex_result_to_dict_is_json_safe(self):
        import json

        df, _, p_flip = self._make_contaminated_corpus(seed=13)
        result = simex_correct(
            df,
            outcome_col="y",
            flip_probability=p_flip,
            fit_fn=self._ols_slope_fit,
            lambdas=(0.0, 1.0, 2.0),
            n_repeats=10,
            rng=default_rng(4),
        )
        json.dumps(result.to_dict())


# ---------------------------------------------------------------------------
# SIMEX input validation
# ---------------------------------------------------------------------------


class TestSimexValidation:
    def test_missing_outcome_column_raises(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [True, False, True]})
        with pytest.raises(ValueError, match="outcome_col"):
            simex_correct(
                df,
                outcome_col="nope",
                flip_probability=0.1,
                fit_fn=lambda d: 0.0,
            )

    def test_flip_probability_out_of_range(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [True, False, True]})
        with pytest.raises(ValueError, match="flip_probability"):
            simex_correct(
                df,
                outcome_col="y",
                flip_probability=0.5,
                fit_fn=lambda d: 0.0,
            )

    def test_lambdas_must_start_at_zero(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [True, False, True]})
        with pytest.raises(ValueError, match="start at 0"):
            simex_correct(
                df,
                outcome_col="y",
                flip_probability=0.1,
                fit_fn=lambda d: 0.0,
                lambdas=(0.5, 1.0, 1.5),
            )

    def test_lambdas_need_three_points(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [True, False, True]})
        with pytest.raises(ValueError, match="quadratic"):
            simex_correct(
                df,
                outcome_col="y",
                flip_probability=0.1,
                fit_fn=lambda d: 0.0,
                lambdas=(0.0, 1.0),
            )

    def test_non_binary_outcome_rejected(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0.3, 0.5, 0.9]})
        with pytest.raises(ValueError, match="binary"):
            simex_correct(
                df,
                outcome_col="y",
                flip_probability=0.1,
                fit_fn=lambda d: 0.0,
            )


# ---------------------------------------------------------------------------
# Deconfound proportion — ground-truth recovery
# ---------------------------------------------------------------------------


class TestDeconfoundProportion:
    def test_recovers_true_rate_with_known_confusion_matrix(self):
        # True rate: 0.4. Grader confusion: TPR = 0.8, FPR = 0.1.
        # Expected measured rate: 0.8 * 0.4 + 0.1 * 0.6 = 0.38.
        # Deconfounding: (0.38 - 0.1) / (0.8 - 0.1) = 0.4. ✓
        cm = ConfusionMatrix(tp=80, fn=20, fp=10, tn=90)  # 100 true+, 100 true-
        assert math.isclose(cm.tpr, 0.8, abs_tol=1e-9)
        assert math.isclose(cm.fpr, 0.1, abs_tol=1e-9)
        estimate = deconfound_proportion(
            measured_rate=0.38, confusion=cm, n_measured=1000
        )
        assert estimate.method == "deconfound_proportion"
        assert math.isclose(estimate.value, 0.4, abs_tol=0.01)
        # CI must bracket the point.
        assert estimate.ci_low <= estimate.value <= estimate.ci_high

    def test_perfect_grader_returns_measured_rate_unchanged(self):
        cm = ConfusionMatrix(tp=100, fn=0, fp=0, tn=100)  # perfect
        estimate = deconfound_proportion(measured_rate=0.42, confusion=cm)
        # TPR=1, FPR=0 → deconfounded rate = (0.42 - 0)/(1 - 0) = 0.42.
        assert math.isclose(estimate.value, 0.42, abs_tol=1e-9)

    def test_unusable_confusion_matrix_flagged(self):
        # TPR == FPR → no information, correction impossible.
        cm = ConfusionMatrix(tp=50, fn=50, fp=50, tn=50)
        estimate = deconfound_proportion(measured_rate=0.5, confusion=cm)
        assert estimate.method == "unusable_confusion_matrix"
        assert math.isnan(estimate.value)

    def test_out_of_range_measured_rate_rejected(self):
        cm = ConfusionMatrix(tp=10, fn=10, fp=5, tn=15)
        with pytest.raises(ValueError, match="measured_rate"):
            deconfound_proportion(measured_rate=1.5, confusion=cm)

    def test_ci_widens_at_higher_confidence(self):
        from inspect_degradation.analysis.statistics import NINETY_NINE

        cm = ConfusionMatrix(tp=80, fn=20, fp=10, tn=90)
        e95 = deconfound_proportion(
            measured_rate=0.38, confusion=cm, n_measured=1000
        )
        e99 = deconfound_proportion(
            measured_rate=0.38,
            confusion=cm,
            confidence_level=NINETY_NINE,
            n_measured=1000,
        )
        width_95 = e95.ci_high - e95.ci_low
        width_99 = e99.ci_high - e99.ci_low
        assert width_99 >= width_95


# ---------------------------------------------------------------------------
# ConfusionMatrix construction and metrics
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_rates_on_canonical_matrix(self):
        cm = ConfusionMatrix(tp=80, fn=20, fp=10, tn=90)
        assert math.isclose(cm.tpr, 0.8)
        assert math.isclose(cm.fpr, 0.1)
        assert math.isclose(cm.tnr, 0.9)
        assert math.isclose(cm.fnr, 0.2)
        assert math.isclose(cm.accuracy, 0.85)
        assert cm.total == 200
        assert cm.actual_positive == 100
        assert cm.actual_negative == 100

    def test_from_label_pairs_basic(self):
        predicted = ["fail", "fail", "pass", "pass", "fail"]
        reference = ["fail", "pass", "pass", "fail", "fail"]
        cm = ConfusionMatrix.from_label_pairs(
            predicted, reference, positive_label="fail"
        )
        # pred=fail, ref=fail: indices 0, 4 → tp=2
        # pred=fail, ref=pass: index 1 → fp=1
        # pred=pass, ref=pass: index 2 → tn=1
        # pred=pass, ref=fail: index 3 → fn=1
        assert cm.tp == 2
        assert cm.fp == 1
        assert cm.tn == 1
        assert cm.fn == 1

    def test_from_label_pairs_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            ConfusionMatrix.from_label_pairs(
                ["fail", "pass"], ["fail"], positive_label="fail"
            )

    def test_negative_counts_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            ConfusionMatrix(tp=-1, fn=0, fp=0, tn=10)

    def test_empty_matrix_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            ConfusionMatrix(tp=0, fn=0, fp=0, tn=0)

    def test_degenerate_rows_return_nan(self):
        # No actual positives → tpr and fnr undefined.
        cm = ConfusionMatrix(tp=0, fn=0, fp=5, tn=10)
        assert math.isnan(cm.tpr)
        assert math.isnan(cm.fnr)
        # The opposite row is still well-defined.
        assert math.isclose(cm.fpr, 5 / 15)

    def test_cohens_kappa_matches_irr_module(self):
        # Match against the existing cohens_kappa primitive on the
        # same data — they must agree to floating-point precision.
        from inspect_degradation.validation.irr import cohens_kappa

        predicted = ["fail", "fail", "pass", "pass", "fail", "pass", "fail"]
        reference = ["fail", "pass", "pass", "fail", "fail", "pass", "fail"]
        cm = ConfusionMatrix.from_label_pairs(
            predicted, reference, positive_label="fail"
        )
        kappa_from_cm = cm.cohens_kappa()
        kappa_from_irr = cohens_kappa(predicted, reference)
        assert math.isclose(kappa_from_cm, kappa_from_irr, abs_tol=1e-12)

    def test_tpr_estimate_has_wilson_ci(self):
        cm = ConfusionMatrix(tp=80, fn=20, fp=10, tn=90)
        est = cm.tpr_estimate()
        assert est.method == "wilson"
        assert math.isclose(est.value, 0.8)
        assert est.ci_low < est.value < est.ci_high
        assert est.n == 100  # actual positives

    def test_to_dict_is_json_safe(self):
        import json

        cm = ConfusionMatrix(tp=80, fn=20, fp=10, tn=90)
        json.dumps(cm.to_dict())

    def test_from_grade_pairs_with_extractor(self):
        # A lightweight stand-in for GradePair with a .predicted and
        # .reference attribute; the actual GradePair class works
        # identically.
        from dataclasses import dataclass

        @dataclass
        class _FakeStep:
            label: str | None

        @dataclass
        class _FakePair:
            predicted: _FakeStep
            reference: _FakeStep

        pairs = [
            _FakePair(_FakeStep("fail"), _FakeStep("fail")),
            _FakePair(_FakeStep("fail"), _FakeStep("pass")),
            _FakePair(_FakeStep("pass"), _FakeStep("pass")),
            _FakePair(_FakeStep(None), _FakeStep("fail")),  # dropped
            _FakePair(_FakeStep("pass"), _FakeStep("fail")),
        ]
        cm = ConfusionMatrix.from_grade_pairs(
            pairs,
            extractor=lambda s: s.label,
            positive_label="fail",
        )
        # 4 pairs contribute (one dropped for None).
        assert cm.total == 4
        assert cm.tp == 1
        assert cm.fp == 1
        assert cm.tn == 1
        assert cm.fn == 1
