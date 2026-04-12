"""Crossed random-effects mixed model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")  # noqa: E402

from inspect_degradation.analysis.mixed_effects import (  # noqa: E402
    fit_crossed_effects_model,
)


def _make_crossed_corpus(
    *,
    n_tasks: int = 8,
    n_models: int = 4,
    n_steps: int = 12,
    slope: float = 0.04,
    seed: int = 0,
) -> pd.DataFrame:
    """Tasks × models fully crossed; both random effects nonzero."""
    rng = np.random.default_rng(seed)
    task_offsets = rng.normal(0, 0.05, size=n_tasks)
    model_offsets = rng.normal(0, 0.08, size=n_models)
    rows = []
    for t in range(n_tasks):
        for m in range(n_models):
            trace_id = f"task{t}_model{m}"
            for s in range(n_steps):
                p = 0.1 + slope * s + task_offsets[t] + model_offsets[m]
                p = float(np.clip(p, 0.0, 1.0))
                rows.append(
                    {
                        "trace_id": trace_id,
                        "task_id": f"task{t}",
                        "model": f"model{m}",
                        "step_index": s,
                        "is_error": bool(rng.random() < p),
                    }
                )
    return pd.DataFrame(rows)


class TestFitCrossedEffectsModel:
    def test_recovers_positive_slope_with_two_grouping_variables(self):
        df = _make_crossed_corpus(seed=1)
        result = fit_crossed_effects_model(
            df,
            formula="is_error ~ step_index",
            primary_group="task_id",
            crossed_group="model",
        )
        assert result.method == "crossed_lmm"
        assert result.fit_usable
        slope_row = result.coefficient("step_index")
        assert slope_row.estimate > 0
        assert slope_row.ci_low > 0  # significant
        # Crossed-group variance was captured.
        assert result.extras["crossed_group"] == "model"

    def test_rejects_missing_columns(self):
        df = _make_crossed_corpus(seed=2)
        with pytest.raises(ValueError, match="not in dataframe"):
            fit_crossed_effects_model(
                df,
                formula="is_error ~ step_index",
                primary_group="task_id",
                crossed_group="nope",
            )

    def test_rejects_identical_groups(self):
        df = _make_crossed_corpus(seed=3)
        with pytest.raises(ValueError, match="distinct"):
            fit_crossed_effects_model(
                df,
                formula="is_error ~ step_index",
                primary_group="task_id",
                crossed_group="task_id",
            )

    def test_insufficient_groups_returns_unusable_result(self):
        df = _make_crossed_corpus(seed=4)
        df = df[df["model"] == "model0"].copy()  # collapse crossed dim
        result = fit_crossed_effects_model(
            df,
            formula="is_error ~ step_index",
            primary_group="task_id",
            crossed_group="model",
        )
        assert not result.fit_usable
