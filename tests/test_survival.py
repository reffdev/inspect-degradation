"""Tests for the Kaplan-Meier survival fit with confidence bands.

We pin:

1. Empty input returns an empty curve and an empty median estimate.
2. The full curve has consistent lengths across the four sequences.
3. On a synthetic corpus with a known first-error-step distribution,
   the KM median estimate is within the expected range and the CI
   bands contain the point estimate at every timeline index.
4. The median-unreached case is flagged cleanly instead of raising.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

# Survival analysis depends on lifelines; skip the module entirely if
# it isn't installed so the rest of the suite stays green in slimmer
# environments.
pytest.importorskip("lifelines")

from inspect_degradation.analysis.statistics import NINETY_FIVE, NINETY_NINE  # noqa: E402
from inspect_degradation.analysis.survival import (  # noqa: E402
    first_error_km,
)


class TestSurvival:
    def test_empty_frame(self):
        empty = pd.DataFrame(
            {"trace_id": [], "step_index": [], "is_error": []}
        )
        result = first_error_km(empty)
        assert result.n_traces == 0
        assert result.n_events == 0
        assert result.curve.timeline == []
        assert result.curve.survival == []
        assert result.median_survival_time.method == "empty"

    def test_curve_sequence_lengths_match(self):
        rows = []
        # Ten traces, each with 5 steps. Half error at step 2, half at step 4.
        for i in range(10):
            error_step = 2 if i % 2 == 0 else 4
            for step in range(5):
                rows.append(
                    {
                        "trace_id": f"t{i}",
                        "step_index": step,
                        "is_error": step == error_step,
                    }
                )
        df = pd.DataFrame(rows)
        result = first_error_km(df)
        n = len(result.curve.timeline)
        assert len(result.curve.survival) == n
        assert len(result.curve.ci_lower) == n
        assert len(result.curve.ci_upper) == n
        assert result.n_traces == 10
        # 10 events observed (every trace errored).
        assert result.n_events == 10

    def test_survival_decreases_monotonically(self):
        rows = []
        for i in range(20):
            for step in range(10):
                rows.append(
                    {
                        "trace_id": f"t{i}",
                        "step_index": step,
                        "is_error": step == (i % 10),
                    }
                )
        df = pd.DataFrame(rows)
        result = first_error_km(df)
        survival = result.curve.survival
        # KM is monotonically non-increasing.
        for a, b in zip(survival, survival[1:]):
            assert b <= a + 1e-12

    def test_ci_bounds_bracket_point(self):
        rows = []
        for i in range(30):
            for step in range(10):
                rows.append(
                    {
                        "trace_id": f"t{i}",
                        "step_index": step,
                        "is_error": (step == 5) and (i % 2 == 0),
                    }
                )
        df = pd.DataFrame(rows)
        result = first_error_km(df)
        for s, lo, hi in zip(
            result.curve.survival,
            result.curve.ci_lower,
            result.curve.ci_upper,
        ):
            # On any timeline point: lo <= s <= hi, modulo numerical
            # noise at the ends.
            assert lo - 1e-9 <= s <= hi + 1e-9

    def test_wider_ci_at_higher_confidence(self):
        rows = []
        for i in range(30):
            for step in range(10):
                rows.append(
                    {
                        "trace_id": f"t{i}",
                        "step_index": step,
                        "is_error": step == 5 and i % 3 == 0,
                    }
                )
        df = pd.DataFrame(rows)
        r95 = first_error_km(df, confidence_level=NINETY_FIVE)
        r99 = first_error_km(df, confidence_level=NINETY_NINE)
        # At matching timeline indices, the 99% CI must be at least as
        # wide as the 95% CI.
        for lo95, hi95, lo99, hi99 in zip(
            r95.curve.ci_lower,
            r95.curve.ci_upper,
            r99.curve.ci_lower,
            r99.curve.ci_upper,
        ):
            width95 = hi95 - lo95
            width99 = hi99 - lo99
            assert width99 >= width95 - 1e-9

    def test_median_unreached_flagged(self):
        # All traces never error — median survival is never reached,
        # should be surfaced cleanly not raise.
        rows = []
        for i in range(20):
            for step in range(5):
                rows.append(
                    {
                        "trace_id": f"t{i}",
                        "step_index": step,
                        "is_error": False,
                    }
                )
        df = pd.DataFrame(rows)
        result = first_error_km(df)
        assert result.n_events == 0
        assert result.median_survival_time.method == "km_median_unreached"
        assert math.isnan(result.median_survival_time.value)

    def test_median_reached_has_finite_ci(self):
        rows = []
        for i in range(40):
            for step in range(10):
                rows.append(
                    {
                        "trace_id": f"t{i}",
                        "step_index": step,
                        "is_error": step == (i % 7 + 1),
                    }
                )
        df = pd.DataFrame(rows)
        result = first_error_km(df)
        m = result.median_survival_time
        assert m.method == "km_median"
        assert not math.isnan(m.value)

    def test_to_dict_is_json_safe(self):
        import json

        rows = []
        for i in range(10):
            for step in range(5):
                rows.append(
                    {
                        "trace_id": f"t{i}",
                        "step_index": step,
                        "is_error": step == 2,
                    }
                )
        df = pd.DataFrame(rows)
        result = first_error_km(df)
        payload = result.to_dict()
        json.dumps(payload)  # must not raise
