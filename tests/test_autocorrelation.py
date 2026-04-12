"""Within-trace autocorrelation diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from inspect_degradation.analysis.autocorrelation import (
    AutocorrelationResult,
    LjungBoxResult,
    ljung_box_per_trace,
    per_trace_acf,
)


def _frame_from_series(series_by_trace: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for tid, vals in series_by_trace.items():
        for i, v in enumerate(vals):
            rows.append({"trace_id": tid, "step_index": i, "is_error": v})
    return pd.DataFrame(rows)


class TestPerTraceACF:
    def test_iid_series_acf_near_zero(self):
        rng = np.random.default_rng(0)
        series = {
            f"t{i}": list(rng.integers(0, 2, size=80).astype(float))
            for i in range(40)
        }
        df = _frame_from_series(series)
        result = per_trace_acf(df, max_lag=4)
        assert isinstance(result, AutocorrelationResult)
        assert result.n_traces_used == 40
        # Mean ACF at every lag should be close to zero.
        for m in result.mean_acf:
            assert abs(m) < 0.15

    def test_strongly_correlated_series_detected(self):
        # Construct an AR(1)-style binary series via persistence: each
        # step copies the prior with high probability.
        rng = np.random.default_rng(1)
        series: dict[str, list[float]] = {}
        for i in range(30):
            x = [float(rng.integers(0, 2))]
            for _ in range(79):
                x.append(x[-1] if rng.random() < 0.9 else 1.0 - x[-1])
            series[f"t{i}"] = x
        df = _frame_from_series(series)
        result = per_trace_acf(df, max_lag=3)
        assert result.mean_acf[0] > 0.4  # strong lag-1 correlation

    def test_empty_frame_returns_nan_padded(self):
        result = per_trace_acf(pd.DataFrame(), max_lag=3)
        assert result.n_traces_used == 0
        assert all(math.isnan(v) for v in result.mean_acf)

    def test_to_dict_is_json_safe(self):
        import json

        df = _frame_from_series({"t0": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]})
        result = per_trace_acf(df, max_lag=2)
        json.dumps(result.to_dict())


# Ljung-Box requires statsmodels.
pytest.importorskip("statsmodels")


class TestLjungBox:
    def test_iid_corpus_low_rejection_rate(self):
        rng = np.random.default_rng(2)
        series = {
            f"t{i}": list(rng.integers(0, 2, size=60).astype(float))
            for i in range(50)
        }
        df = _frame_from_series(series)
        result = ljung_box_per_trace(df, lags=4, alpha=0.05)
        assert isinstance(result, LjungBoxResult)
        # Under the i.i.d. null we expect ~5% rejection. Allow a wide
        # tolerance because the corpus is small.
        assert result.rejection_rate.value < 0.25

    def test_correlated_corpus_high_rejection_rate(self):
        rng = np.random.default_rng(3)
        series: dict[str, list[float]] = {}
        for i in range(40):
            x = [float(rng.integers(0, 2))]
            for _ in range(79):
                x.append(x[-1] if rng.random() < 0.9 else 1.0 - x[-1])
            series[f"t{i}"] = x
        df = _frame_from_series(series)
        result = ljung_box_per_trace(df, lags=4)
        assert result.rejection_rate.value > 0.5

    def test_skips_constant_traces(self):
        df = _frame_from_series(
            {
                "const": [0.0] * 20,
                "varied": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            }
        )
        result = ljung_box_per_trace(df, lags=2)
        assert result.n_traces_tested == 1

    def test_invalid_alpha_rejected(self):
        df = _frame_from_series({"t0": [0.0, 1.0, 0.0, 1.0, 1.0]})
        with pytest.raises(ValueError, match="alpha"):
            ljung_box_per_trace(df, alpha=1.5)

    def test_per_trace_payload_optional(self):
        df = _frame_from_series(
            {f"t{i}": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0] for i in range(5)}
        )
        without = ljung_box_per_trace(df, lags=2, include_per_trace=False)
        assert without.per_trace == []
        with_rows = ljung_box_per_trace(df, lags=2, include_per_trace=True)
        assert len(with_rows.per_trace) == with_rows.n_traces_tested

    def test_to_dict_is_json_safe(self):
        import json

        df = _frame_from_series(
            {f"t{i}": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0] for i in range(3)}
        )
        result = ljung_box_per_trace(df, lags=2, include_per_trace=True)
        json.dumps(result.to_dict())
