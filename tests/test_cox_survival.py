"""Cox proportional-hazards regression for first-error timing."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from numpy.random import default_rng

pytest.importorskip("lifelines")  # noqa: E402

from inspect_degradation.analysis.survival import (  # noqa: E402
    CoxResult,
    cox_first_error,
)


def _make_two_model_corpus(
    *,
    n_per_model: int = 60,
    steps: int = 20,
    fast_hazard: float = 0.15,
    slow_hazard: float = 0.03,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic corpus where 'fast' model errors sooner than 'slow'.

    Each step independently errors at the model-specific hazard.
    First-error distributions are therefore geometric, which gives
    Cox a clean, textbook scenario to recover the log-hazard-ratio.
    """
    rng = default_rng(seed)
    rows: list[dict] = []
    for model_name, hazard in [("fast", fast_hazard), ("slow", slow_hazard)]:
        for t in range(n_per_model):
            trace_id = f"{model_name}_{t}"
            for s in range(steps):
                is_error = bool(rng.random() < hazard)
                rows.append(
                    {
                        "trace_id": trace_id,
                        "step_index": s,
                        "is_error": is_error,
                        "model": model_name,
                    }
                )
                if is_error:
                    break
    return pd.DataFrame(rows)


class TestCoxFirstError:
    def test_recovers_expected_hazard_direction(self):
        df = _make_two_model_corpus(seed=1)
        result = cox_first_error(df, covariates=["model"])
        assert isinstance(result, CoxResult)
        # One dummy term: model_slow (fast is reference). Slow should
        # have *lower* hazard → negative log-hazard coefficient.
        row = result.coefficient("model_slow")
        assert row.estimate < 0
        assert row.ci_high < 0  # strictly significant
        hr, lo, hi = result.hazard_ratios["model_slow"]
        assert hr < 1.0
        assert lo < hr < hi
        assert result.n_events > 0

    def test_concordance_is_in_unit_interval(self):
        df = _make_two_model_corpus(seed=2)
        result = cox_first_error(df, covariates=["model"])
        assert 0.0 <= result.concordance <= 1.0
        # The synthetic effect is large; concordance should beat chance.
        assert result.concordance > 0.55

    def test_empty_frame_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            cox_first_error(pd.DataFrame(), covariates=["model"])

    def test_missing_covariate_rejected(self):
        df = _make_two_model_corpus(seed=3)
        with pytest.raises(ValueError, match="not in df"):
            cox_first_error(df, covariates=["nope"])

    def test_all_censored_rejected(self):
        df = pd.DataFrame(
            [
                {"trace_id": "a", "step_index": 0, "is_error": False, "model": "x"},
                {"trace_id": "a", "step_index": 1, "is_error": False, "model": "x"},
                {"trace_id": "b", "step_index": 0, "is_error": False, "model": "y"},
                {"trace_id": "b", "step_index": 1, "is_error": False, "model": "y"},
            ]
        )
        with pytest.raises(ValueError, match="observed event"):
            cox_first_error(df, covariates=["model"])

    def test_no_usable_covariates_rejected(self):
        df = _make_two_model_corpus(seed=4)
        # Collapse to a single model value → no usable covariate.
        df = df[df["model"] == "fast"].copy()
        with pytest.raises(ValueError, match="covariate"):
            cox_first_error(df, covariates=["model"])

    def test_to_dict_is_json_safe(self):
        import json

        df = _make_two_model_corpus(seed=5)
        result = cox_first_error(df, covariates=["model"])
        json.dumps(result.to_dict())

    def test_hazard_ratio_consistent_with_log_coef(self):
        df = _make_two_model_corpus(seed=6)
        result = cox_first_error(df, covariates=["model"])
        row = result.coefficient("model_slow")
        hr, _, _ = result.hazard_ratios["model_slow"]
        assert math.isclose(hr, math.exp(row.estimate), rel_tol=1e-9)
