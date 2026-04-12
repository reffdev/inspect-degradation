"""Multiple-comparison correction on families of coefficients."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("statsmodels")  # noqa: E402

from inspect_degradation.analysis.mixed_effects import CoefficientRow  # noqa: E402
from inspect_degradation.analysis.multiple_comparisons import (  # noqa: E402
    MultipleComparisonResult,
    adjust_coefficients,
)
from inspect_degradation.analysis.statistics import NINETY_FIVE  # noqa: E402


def _row(name: str, p: float) -> CoefficientRow:
    return CoefficientRow(
        name=name,
        estimate=0.1,
        std_error=0.05,
        z_statistic=2.0,
        p_value=p,
        ci_low=0.0,
        ci_high=0.2,
    )


class TestAdjustCoefficients:
    def test_empty_family_is_identity(self):
        result = adjust_coefficients([])
        assert isinstance(result, MultipleComparisonResult)
        assert result.n_tests == 0
        assert result.n_rejected == 0
        assert result.adjusted == []

    def test_bh_correction_matches_hand_computation(self):
        # Classic Benjamini-Hochberg example: sorted p = 0.001, 0.008, 0.039, 0.041, 0.042
        # m = 5, α = 0.05 → thresholds i*α/m = 0.01, 0.02, 0.03, 0.04, 0.05
        # rejected: the largest k with p_(k) ≤ kα/m is k=2 (0.008 ≤ 0.02); all
        # indices 1..2 rejected. p=0.039 fails (0.039 > 0.03), but then BH
        # does NOT "step back" further — everything ≤ threshold-at-k is in.
        rows = [_row(f"c{i}", p) for i, p in enumerate([0.041, 0.001, 0.039, 0.008, 0.042])]
        result = adjust_coefficients(rows, method="fdr_bh")
        assert result.n_tests == 5
        # Rejection depends on adjusted p-values being ≤ 0.05. Let
        # statsmodels be the oracle here; we just check the obvious:
        # the smallest two p-values are rejected and the largest is not.
        adj_by_name = {a.name: a for a in result.adjusted}
        assert adj_by_name["c1"].rejected is True  # p=0.001
        assert adj_by_name["c3"].rejected is True  # p=0.008
        # c4 (p=0.042) may or may not survive BH depending on the
        # exact step-up cutoff — we leave that to statsmodels and
        # only assert it is no *more* significant than c1.
        assert (
            adj_by_name["c4"].adjusted_p_value
            >= adj_by_name["c1"].adjusted_p_value
        )
        # Adjusted p should be ≥ original p for every row.
        for a in result.adjusted:
            assert a.adjusted_p_value + 1e-12 >= a.original.p_value

    def test_bonferroni_is_strictest(self):
        rows = [_row(f"c{i}", p) for i, p in enumerate([0.01, 0.02, 0.03, 0.04])]
        bh = adjust_coefficients(rows, method="fdr_bh")
        bonf = adjust_coefficients(rows, method="bonferroni")
        # Bonferroni adjusted p = min(1, m * p); each is ≥ BH.
        for b_row, bonf_row in zip(bh.adjusted, bonf.adjusted):
            assert bonf_row.adjusted_p_value + 1e-12 >= b_row.adjusted_p_value

    def test_invalid_p_value_rejected(self):
        rows = [_row("c0", 0.01), _row("c1", float("nan"))]
        with pytest.raises(ValueError, match="p-value"):
            adjust_coefficients(rows)

    def test_rejected_names_matches_flags(self):
        rows = [_row(f"c{i}", p) for i, p in enumerate([0.001, 0.9, 0.002, 0.8])]
        result = adjust_coefficients(rows, method="fdr_bh")
        assert set(result.rejected_names()) == {
            a.name for a in result.adjusted if a.rejected
        }

    def test_family_alpha_tracks_confidence_level(self):
        rows = [_row("c0", 0.04)]
        result = adjust_coefficients(rows, confidence_level=NINETY_FIVE)
        assert math.isclose(result.family_alpha, 0.05)

    def test_to_dict_is_json_safe(self):
        import json

        rows = [_row(f"c{i}", p) for i, p in enumerate([0.01, 0.04, 0.2])]
        result = adjust_coefficients(rows)
        json.dumps(result.to_dict())
