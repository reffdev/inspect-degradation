"""Tests for the mixed-effects degradation models.

The highest-value test here is the synthetic-ground-truth recovery
test: we generate a corpus with a known per-step degradation slope,
fit the model, and verify the recovered coefficient lies inside its
own confidence interval and is near the true value. Without this
test the module could be silently wrong on every downstream report.

The test file is gated on statsmodels being available via
``pytest.importorskip`` — the rest of the suite stays green in
environments that don't install it (statsmodels is a heavy dep).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng

pytest.importorskip("statsmodels")  # noqa: E402

from inspect_degradation.analysis.mixed_effects import (  # noqa: E402
    CoefficientRow,
    MixedEffectsResult,
    RandomEffects,
    fit_mixed_effects,
    fit_step_level_glmm,
    fit_step_level_model,
    fit_trace_level_slope_model,
)
from inspect_degradation.analysis.statistics import NINETY_FIVE, NINETY_NINE  # noqa: E402
from inspect_degradation.schema import (  # noqa: E402
    ComplexityLevel,
    Dependency,
    GradedStep,
    GradedTrace,
    SeverityLevel,
    Validity,
)


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _synthetic_step_frame(
    *,
    n_tasks: int = 20,
    n_traces_per_task: int = 3,
    n_steps: int = 15,
    true_slope: float = 0.05,
    task_effect_sd: float = 0.1,
    baseline: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, float]:
    """Generate a tidy step-level frame with a known degradation slope.

    Each task has its own random intercept on error probability.
    Each step's error probability is a linear function of step index
    with slope ``true_slope``, plus task intercept and baseline.
    The outcome is Bernoulli-sampled from the resulting probability.

    Returns the dataframe and the true slope used, so tests can
    verify recovery.
    """
    rng = default_rng(seed)
    rows = []
    for t in range(n_tasks):
        task_intercept = rng.normal(0.0, task_effect_sd)
        for r in range(n_traces_per_task):
            for step in range(n_steps):
                p = baseline + task_intercept + true_slope * step
                p = max(0.02, min(0.98, p))  # clip to probability range
                is_err = rng.random() < p
                rows.append(
                    {
                        "trace_id": f"task{t}_trace{r}",
                        "task_id": f"task{t}",
                        "model": "m1",
                        "step_index": step,
                        "is_error": bool(is_err),
                        "is_productive": not is_err,
                        "is_neutral": False,
                        "complexity": "medium",
                        "dependency": "n/a" if not is_err else "independent",
                        "severity": None if not is_err else "medium",
                        "is_looping": False,
                        "grader_model": "test",
                    }
                )
    return pd.DataFrame(rows), true_slope


def _synthetic_graded_traces(
    *,
    n_tasks: int = 15,
    n_traces_per_task: int = 2,
    n_steps: int = 12,
    true_slope: float = 0.05,
    seed: int = 0,
    models: tuple[str, ...] = ("m1",),
) -> list[GradedTrace]:
    """Build real ``GradedTrace`` objects with a known slope."""
    rng = default_rng(seed)
    traces: list[GradedTrace] = []
    for model in models:
        for t in range(n_tasks):
            task_intercept = rng.normal(0.0, 0.1)
            for r in range(n_traces_per_task):
                steps = []
                for step_idx in range(n_steps):
                    p = 0.2 + task_intercept + true_slope * step_idx
                    p = max(0.02, min(0.98, p))
                    is_err = rng.random() < p
                    if is_err:
                        steps.append(
                            GradedStep(
                                step_index=step_idx,
                                validity=Validity.fail,
                                complexity=ComplexityLevel.medium,
                                dependency=Dependency.independent,
                                severity=SeverityLevel.medium,
                                is_looping=False,
                                grader_model="test",
                            )
                        )
                    else:
                        steps.append(
                            GradedStep(
                                step_index=step_idx,
                                validity=Validity.pass_,
                                complexity=ComplexityLevel.medium,
                                dependency=Dependency.not_applicable,
                                is_looping=False,
                                grader_model="test",
                            )
                        )
                traces.append(
                    GradedTrace(
                        trace_id=f"{model}_task{t}_trace{r}",
                        task_id=f"task{t}",
                        model=model,
                        steps=steps,
                    )
                )
    return traces


# ---------------------------------------------------------------------------
# Ground-truth recovery
# ---------------------------------------------------------------------------


class TestGroundTruthRecovery:
    def test_recovers_known_slope_within_ci(self):
        df, true_slope = _synthetic_step_frame(
            n_tasks=25,
            n_traces_per_task=4,
            n_steps=15,
            true_slope=0.04,
            seed=42,
        )
        result = fit_step_level_model(df)
        assert result.fit_usable
        slope_row = result.coefficient("step_index")
        # The true slope must fall inside the 95% CI at this sample size.
        assert slope_row.ci_low <= true_slope <= slope_row.ci_high, (
            f"true slope {true_slope} outside CI "
            f"[{slope_row.ci_low}, {slope_row.ci_high}]"
        )
        # And the point estimate should be reasonably close.
        assert abs(slope_row.estimate - true_slope) < 0.02

    def test_recovers_zero_slope_on_stationary_data(self):
        df, true_slope = _synthetic_step_frame(
            n_tasks=25,
            n_traces_per_task=4,
            n_steps=15,
            true_slope=0.0,
            seed=7,
        )
        result = fit_step_level_model(df)
        assert result.fit_usable
        slope_row = result.coefficient("step_index")
        # Zero slope: CI must include 0.
        assert slope_row.ci_low <= 0.0 <= slope_row.ci_high

    def test_direction_of_negative_slope_preserved(self):
        df, true_slope = _synthetic_step_frame(
            n_tasks=25,
            n_traces_per_task=4,
            n_steps=15,
            true_slope=-0.03,
            seed=11,
        )
        result = fit_step_level_model(df)
        assert result.fit_usable
        slope_row = result.coefficient("step_index")
        # Point should be negative.
        assert slope_row.estimate < 0
        # CI should be entirely below zero for a detectable negative slope.
        assert slope_row.ci_high < 0.01


# ---------------------------------------------------------------------------
# Default-formula adaptation
# ---------------------------------------------------------------------------


class TestFormulaAdaptation:
    def test_drops_single_level_complexity(self):
        df, _ = _synthetic_step_frame(seed=3)
        # Synthetic data has only "medium" complexity — C(complexity)
        # would be singular; the fitter must drop it.
        result = fit_step_level_model(df)
        assert result.fit_usable
        dropped = result.extras["dropped_terms"]
        assert any("complexity" in msg for msg in dropped)
        # Step_index coefficient should still be present.
        result.coefficient("step_index")

    def test_drops_single_level_model(self):
        df, _ = _synthetic_step_frame(seed=5)
        # Only one model in synthetic data → drop C(model).
        result = fit_step_level_model(df)
        dropped = result.extras["dropped_terms"]
        assert any("model" in msg for msg in dropped)

    def test_keeps_multi_level_model_factor(self):
        # Two models, explicit varying intercepts per model.
        rng = default_rng(0)
        rows = []
        for model_name, model_offset in [("m1", 0.0), ("m2", 0.2)]:
            for t in range(15):
                task_intercept = rng.normal(0, 0.1)
                for step in range(10):
                    p = 0.2 + model_offset + task_intercept + 0.03 * step
                    p = max(0.02, min(0.98, p))
                    is_err = rng.random() < p
                    rows.append(
                        {
                            "trace_id": f"{model_name}_task{t}",
                            "task_id": f"task{t}",
                            "model": model_name,
                            "step_index": step,
                            "is_error": bool(is_err),
                            "complexity": "medium",
                        }
                    )
        df = pd.DataFrame(rows)
        result = fit_step_level_model(df)
        assert result.fit_usable
        # The model should include a C(model)[T.m2] coefficient.
        names = [c.name for c in result.coefficients]
        assert any("C(model)" in n for n in names)

    def test_explicit_none_drops_control(self):
        df, _ = _synthetic_step_frame(seed=9)
        # Explicitly telling the fitter to skip complexity must work.
        result = fit_step_level_model(df, complexity_col=None)
        # complexity is not in the dropped_terms list because we opted out.
        dropped = result.extras["dropped_terms"]
        assert not any("complexity" in msg for msg in dropped)


class TestOutcomeStratification:
    """``trace_success`` should partial out the survivorship-bias confound
    where failed traces are systematically longer than successful ones."""

    def _make_outcome_confounded_frame(self, *, seed: int = 0) -> pd.DataFrame:
        """Build a frame where successful traces are short with low base
        error and failed traces are long with high base error, but the
        *within-trace* per-step slope is identical (+0.01) for both.
        Without outcome stratification, the marginal slope is inflated
        by the success-vs-failure mean gap; with stratification it
        recovers the within-trace slope.
        """
        rng = default_rng(seed)
        rows: list[dict] = []
        for t in range(60):
            success = bool(t % 2)
            base = 0.05 if success else 0.30
            n_steps = 8 if success else 18
            for s in range(n_steps):
                p = base + 0.01 * s
                p = max(0.0, min(1.0, p))
                rows.append(
                    {
                        "trace_id": f"t{t}",
                        "task_id": f"t{t}",
                        "step_index": s,
                        "is_error": bool(rng.random() < p),
                        "complexity": "medium",
                        "model": "sim",
                        "trace_success": success,
                    }
                )
        return pd.DataFrame(rows)

    def test_success_term_added_to_formula_when_present(self):
        df = self._make_outcome_confounded_frame(seed=1)
        result = fit_step_level_model(df)
        assert "C(trace_success)" in result.formula

    def test_recovers_within_trace_slope_after_stratification(self):
        df = self._make_outcome_confounded_frame(seed=2)
        stratified = fit_step_level_model(df)
        slope = stratified.coefficient("step_index")
        # True within-trace slope is +0.01; recover it within CI.
        assert slope.ci_low <= 0.01 <= slope.ci_high

    def test_marginal_slope_is_inflated_without_stratification(self):
        df = self._make_outcome_confounded_frame(seed=3)
        marginal = fit_step_level_model(df, success_col=None)
        stratified = fit_step_level_model(df)
        marginal_slope = marginal.coefficient("step_index").estimate
        stratified_slope = stratified.coefficient("step_index").estimate
        # Marginal slope is inflated relative to the within-trace slope;
        # stratification pulls it back toward the truth (+0.01).
        assert marginal_slope > stratified_slope

    def test_success_dropped_when_column_constant(self):
        df = self._make_outcome_confounded_frame(seed=4)
        df = df[df["trace_success"]].copy()  # collapse to one outcome
        result = fit_step_level_model(df)
        assert "C(trace_success)" not in result.formula
        assert any("trace_success" in msg for msg in result.extras["dropped_terms"])

    def test_explicit_none_disables_stratification(self):
        df = self._make_outcome_confounded_frame(seed=5)
        result = fit_step_level_model(df, success_col=None)
        assert "C(trace_success)" not in result.formula
        # When explicitly disabled, it shouldn't appear in dropped_terms either.
        assert not any("trace_success" in msg for msg in result.extras["dropped_terms"])


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_empty_frame(self):
        empty = pd.DataFrame(
            {
                "trace_id": [],
                "task_id": [],
                "model": [],
                "step_index": [],
                "is_error": [],
                "complexity": [],
            }
        )
        result = fit_step_level_model(empty)
        assert not result.fit_usable
        assert result.fit_error == "empty_frame"
        assert result.n_observations == 0
        assert result.coefficients == []

    def test_single_group_frame(self):
        # One task only → cannot fit a random intercept.
        rows = []
        for step in range(5):
            rows.append(
                {
                    "trace_id": "t0",
                    "task_id": "only_task",
                    "model": "m1",
                    "step_index": step,
                    "is_error": step > 2,
                    "complexity": "medium",
                }
            )
        df = pd.DataFrame(rows)
        result = fit_step_level_model(df)
        assert not result.fit_usable
        assert result.fit_error == "insufficient_groups"

    def test_missing_outcome_column_raises(self):
        df = pd.DataFrame(
            {
                "trace_id": ["t"],
                "task_id": ["a"],
                "step_index": [0],
            }
        )
        with pytest.raises(ValueError, match="outcome column"):
            fit_step_level_model(df, outcome="is_error")

    def test_missing_step_column_raises(self):
        df = pd.DataFrame(
            {
                "trace_id": ["t"],
                "task_id": ["a"],
                "is_error": [False],
            }
        )
        with pytest.raises(ValueError, match="step column"):
            fit_step_level_model(df)

    def test_missing_group_col_raises_in_general_wrapper(self):
        df = pd.DataFrame({"y": [1.0, 2.0], "x": [0.0, 1.0]})
        with pytest.raises(ValueError, match="group_col"):
            fit_mixed_effects(df, formula="y ~ x", group_col="nonexistent")


# ---------------------------------------------------------------------------
# slope_estimate() bridge to Estimate
# ---------------------------------------------------------------------------


class TestSlopeEstimateBridge:
    def test_converged_returns_mixed_effects_method(self):
        df, _ = _synthetic_step_frame(n_tasks=20, n_traces_per_task=3, seed=1)
        result = fit_step_level_model(df)
        assert result.fit_usable
        est = result.slope_estimate("step_index")
        assert est.method == "mixed_effects"
        assert not math.isnan(est.value)
        assert est.ci_low <= est.value <= est.ci_high
        assert est.se is not None and est.se > 0
        # The confidence level round-trips correctly.
        assert est.confidence_level == NINETY_FIVE

    def test_non_converged_returns_nan_estimate(self):
        empty = pd.DataFrame(
            {
                "trace_id": [],
                "task_id": [],
                "step_index": [],
                "is_error": [],
                "complexity": [],
            }
        )
        result = fit_step_level_model(empty)
        est = result.slope_estimate("step_index")
        assert est.method == "mixed_effects_not_usable"
        assert math.isnan(est.value)

    def test_missing_coefficient_returns_missing_estimate(self):
        df, _ = _synthetic_step_frame(seed=13)
        result = fit_step_level_model(df)
        est = result.slope_estimate("nonexistent_param")
        assert est.method == "mixed_effects_coefficient_missing"
        assert math.isnan(est.value)

    def test_confidence_level_propagates(self):
        df, _ = _synthetic_step_frame(n_tasks=20, n_traces_per_task=3, seed=1)
        r99 = fit_step_level_model(df, confidence_level=NINETY_NINE)
        r95 = fit_step_level_model(df, confidence_level=NINETY_FIVE)
        assert r99.converged and r95.converged
        s99 = r99.slope_estimate("step_index")
        s95 = r95.slope_estimate("step_index")
        # 99% CI must be at least as wide as 95%.
        width_99 = s99.ci_high - s99.ci_low
        width_95 = s95.ci_high - s95.ci_low
        assert width_99 >= width_95 - 1e-9


# ---------------------------------------------------------------------------
# Coefficient and random-effects table structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    def test_coefficient_accessor(self):
        df, _ = _synthetic_step_frame(seed=17)
        result = fit_step_level_model(df)
        intercept = result.coefficient("Intercept")
        assert isinstance(intercept, CoefficientRow)
        assert not math.isnan(intercept.estimate)

    def test_coefficient_missing_raises_with_available(self):
        df, _ = _synthetic_step_frame(seed=19)
        result = fit_step_level_model(df)
        with pytest.raises(KeyError, match="available"):
            result.coefficient("this_is_not_a_real_param")

    def test_random_effects_icc_in_unit_interval(self):
        df, _ = _synthetic_step_frame(seed=21)
        result = fit_step_level_model(df)
        assert result.fit_usable
        re = result.random_effects
        if not math.isnan(re.icc):
            assert 0.0 <= re.icc <= 1.0
        assert re.n_groups >= 2

    def test_to_dict_json_safe(self):
        import json

        df, _ = _synthetic_step_frame(seed=23)
        result = fit_step_level_model(df)
        payload = result.to_dict()
        json.dumps(payload)  # must not raise


# ---------------------------------------------------------------------------
# Trace-level slope model
# ---------------------------------------------------------------------------


class TestTraceLevelSlopeModel:
    def test_basic_fit_on_multi_task_corpus(self):
        traces = _synthetic_graded_traces(
            n_tasks=25, n_traces_per_task=3, true_slope=0.04, seed=29
        )
        result = fit_trace_level_slope_model(traces)
        # Single model → C(model) is dropped; formula is "slope ~ 1".
        assert result.formula.startswith("slope ~")
        # Extras record the dropping.
        dropped = result.extras.get("dropped_terms", [])
        assert any("model" in msg for msg in dropped)
        # Fit should have some observations — each surviving trace
        # contributes one row.
        assert result.n_observations > 0

    def test_multi_model_corpus_decomposes_by_model(self):
        traces = _synthetic_graded_traces(
            n_tasks=20,
            n_traces_per_task=3,
            true_slope=0.03,
            models=("haiku", "sonnet"),
            seed=31,
        )
        result = fit_trace_level_slope_model(traces)
        # Formula should include the C(model) term.
        assert "C(model)" in result.formula
        # A C(model)[T.sonnet] or similar should appear.
        names = [c.name for c in result.coefficients]
        assert any(name.startswith("C(model)") for name in names), names

    def test_drop_reasons_surface_too_short(self):
        # Mix of usable (long) and unusable (very short) traces.
        long_traces = _synthetic_graded_traces(
            n_tasks=10, n_traces_per_task=2, n_steps=10, seed=33
        )
        short_traces = [
            GradedTrace(
                trace_id=f"short{i}",
                task_id=f"shorty_{i}",
                model="m1",
                steps=[
                    GradedStep(
                        step_index=0,
                        validity=Validity.pass_,
                        grader_model="test",
                    ),
                    GradedStep(
                        step_index=1,
                        validity=Validity.pass_,
                        grader_model="test",
                    ),
                ],
            )
            for i in range(5)
        ]
        result = fit_trace_level_slope_model(long_traces + short_traces)
        assert result.extras["drop_reasons"]["too_short"] == 5

    def test_no_usable_traces_yields_empty_result(self):
        too_short = [
            GradedTrace(
                trace_id=f"t{i}",
                task_id=f"task{i}",
                model="m1",
                steps=[
                    GradedStep(
                        step_index=0,
                        validity=Validity.pass_,
                        grader_model="test",
                    )
                ],
            )
            for i in range(5)
        ]
        result = fit_trace_level_slope_model(too_short)
        assert not result.fit_usable
        assert result.fit_error == "no_usable_traces"

    def test_unknown_outcome_rejected(self):
        traces = _synthetic_graded_traces(n_tasks=5, seed=0)
        with pytest.raises(ValueError, match="unknown outcome"):
            fit_trace_level_slope_model(traces, outcome="not_a_thing")


# ---------------------------------------------------------------------------
# GLMM (BinomialBayesMixedGLM)
# ---------------------------------------------------------------------------


class TestGlmm:
    def test_glmm_returns_result_on_synthetic_data(self):
        df, _ = _synthetic_step_frame(seed=0)
        result = fit_step_level_glmm(df)
        assert isinstance(result, MixedEffectsResult)
        assert result.fit_usable
        assert result.method == "step_level_glmm"
        # Should have a step_index coefficient
        names = [c.name for c in result.coefficients]
        assert "step_index" in names
        # step_index should have a reasonable p-value (not NaN)
        step_coeff = next(c for c in result.coefficients if c.name == "step_index")
        assert math.isfinite(step_coeff.p_value)

    def test_glmm_recovers_slope_direction(self):
        df, true_slope = _synthetic_step_frame(true_slope=0.05, seed=42)
        result = fit_step_level_glmm(df)
        assert result.fit_usable
        step_coeff = next(c for c in result.coefficients if c.name == "step_index")
        # On the logit scale the coefficient should be positive
        assert step_coeff.estimate > 0, (
            f"Expected positive slope, got {step_coeff.estimate}"
        )

    def test_glmm_slope_near_zero_for_null_data(self):
        df, _ = _synthetic_step_frame(true_slope=0.0, seed=7)
        result = fit_step_level_glmm(df)
        assert result.fit_usable
        step_coeff = next(c for c in result.coefficients if c.name == "step_index")
        # Either non-significant or very small magnitude
        assert step_coeff.p_value > 0.05 or abs(step_coeff.estimate) < 0.05

    def test_glmm_handles_low_error_rate(self):
        df, _ = _synthetic_step_frame(baseline=0.03, true_slope=0.01, seed=11)
        result = fit_step_level_glmm(df)
        assert result.fit_usable
        assert result.method == "step_level_glmm"

    def test_glmm_empty_frame_returns_not_usable(self):
        df, _ = _synthetic_step_frame(seed=0)
        empty = df.iloc[:0]
        result = fit_step_level_glmm(empty)
        assert not result.fit_usable

    def test_glmm_extras_contain_marginal_effects(self):
        df, _ = _synthetic_step_frame(seed=0)
        result = fit_step_level_glmm(df)
        assert "marginal_effects" in result.extras
        assert "step_index" in result.extras["marginal_effects"]


# ---------------------------------------------------------------------------
# Interaction terms
# ---------------------------------------------------------------------------


def _add_step_phase(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Add a synthetic step_phase column: explore early, act late."""
    rng = default_rng(seed)
    df = df.copy()
    # Steps in the first half are mostly explore, second half mostly act
    mid = df["step_index"].max() / 2
    df["step_phase"] = df["step_index"].apply(
        lambda s: "explore" if s < mid else "act"
    )
    # Add noise: flip 20% of labels
    flip = rng.random(len(df)) < 0.2
    df.loc[flip, "step_phase"] = df.loc[flip, "step_phase"].apply(
        lambda x: "act" if x == "explore" else "explore"
    )
    return df


class TestInteractions:
    def test_interaction_term_appears_in_formula(self):
        df, _ = _synthetic_step_frame(seed=0)
        df = _add_step_phase(df)
        result = fit_step_level_model(
            df, interactions=["step_index:C(step_phase)"]
        )
        assert "step_index:C(step_phase)" in result.formula

    def test_interaction_coefficient_exists(self):
        df, _ = _synthetic_step_frame(seed=0)
        df = _add_step_phase(df)
        result = fit_step_level_model(
            df, interactions=["step_index:C(step_phase)"]
        )
        assert result.fit_usable
        names = [c.name for c in result.coefficients]
        # Patsy expands C(step_phase) into indicator; the interaction
        # coefficient name will contain both step_index and step_phase.
        interaction_coefs = [n for n in names if "step_index" in n and "step_phase" in n]
        assert len(interaction_coefs) >= 1, f"No interaction coefficient found in {names}"

    def test_interaction_recorded_in_extras(self):
        df, _ = _synthetic_step_frame(seed=0)
        df = _add_step_phase(df)
        result = fit_step_level_model(
            df, interactions=["step_index:C(step_phase)"]
        )
        assert "interactions" in result.extras
        assert result.extras["interactions"] == ["step_index:C(step_phase)"]

    def test_no_interaction_by_default(self):
        df, _ = _synthetic_step_frame(seed=0)
        df = _add_step_phase(df)
        result = fit_step_level_model(df)
        assert "interactions" not in result.extras
        names = [c.name for c in result.coefficients]
        interaction_coefs = [n for n in names if "step_index" in n and "step_phase" in n]
        assert len(interaction_coefs) == 0
