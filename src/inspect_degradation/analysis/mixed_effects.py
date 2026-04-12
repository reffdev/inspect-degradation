"""Mixed-effects regression for within-run degradation.

This is the module that carries the project's central scientific
claim: that agents degrade within a run *controlling for* step
complexity, task identity, and model. Without the confound
decomposition every headline number is vulnerable to the obvious
reviewer objections (later steps are just harder; some tasks are
just worse; some models are just worse). This module exists to
answer those objections with one regression fit.

## Two flavors of mixed model

The module exposes two complementary model shapes, each answering
a slightly different question.

**Step-level model** (:func:`fit_step_level_model`) —
``is_error ~ step_index + C(complexity) + C(model) + (1 | task_id)``.
The fixed-effect coefficient on ``step_index`` is the degradation
slope on the probability scale (because ``is_error`` is 0/1), with
complexity, model, and task identity absorbed into the other
terms. A nonzero, significant ``step_index`` coefficient after
controlling for those is the formal version of "the agent gets
worse as the trace goes on, for reasons beyond the task structure."
This is the model PROJECT_PLAN.md specifies.

Note that ``is_error`` is binary, so the strictly correct model is
a generalized LMM with a logit link. ``statsmodels.MixedLM`` is a
linear model; we are approximating the logistic response with a
linear-probability model. For error rates in the 20–50% range the
LPM slope coefficient is nearly identical to the GLMM-logit
coefficient transformed to the probability scale (Angrist & Pischke,
*Mostly Harmless Econometrics*, §3.4.2); near 0 and 1 the
approximation degrades. A GLMM implementation is now available
via :func:`fit_step_level_glmm` using variational Bayes.

**Trace-level slope model** (:func:`fit_trace_level_slope_model`) —
``slope ~ C(model) + C(task_family) + (1 | task_id)``. Here the
outcome is one *per-trace* degradation slope computed via
:func:`inspect_degradation.analysis.slopes.per_trace_mean_slope`,
and the model decomposes those slopes across model identity and
task clustering. Slopes are continuous, so this is exactly linear
and doesn't lean on the linear-probability approximation. The
question it answers is different: "how does the per-trace slope
vary by model and by task family" — which for the writeup is
arguably the more interesting decomposition.

Both models return the same :class:`MixedEffectsResult` shape so
downstream code can compose either one uniformly.

## Result shape

Every fit returns :class:`MixedEffectsResult` carrying:

* :class:`CoefficientRow` objects for every fixed effect, each
  exposing point estimate, SE, z-statistic, p-value, and CI.
* Random-effects variance components.
* Model-fit diagnostics (log-likelihood, number of groups, number
  of observations, convergence status, any fit warnings).
* A :meth:`MixedEffectsResult.slope_estimate` bridge returning an
  :class:`~inspect_degradation.analysis.statistics.Estimate` for
  any named fixed effect — so the same point-and-interval type
  used everywhere else in the analysis layer travels out to
  reports.

## Convergence

``statsmodels.MixedLM`` emits ``ConvergenceWarning`` noisily during
normal use when random-effect variance hits the boundary or when
the optimizer needs to retry with a different method. The wrapper
here *captures* those warnings and surfaces them in
:attr:`MixedEffectsResult.fit_warnings` rather than letting them
escape to stderr. A caller running a batch fit (Phase 1's multi-
configuration validation script, for example) gets a clean log and
can inspect warnings per-fit in the result object.

If the underlying fit raises, the wrapper catches it and returns a
:class:`MixedEffectsResult` with :attr:`converged` set to False and
the exception message in :attr:`fit_error`. Callers can branch on
``result.converged`` rather than wrapping every call in try/except.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from inspect_degradation.analysis.statistics import (
    NINETY_FIVE,
    ConfidenceLevel,
    Estimate,
)


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoefficientRow:
    """One fixed-effect coefficient from a mixed-effects fit.

    Attributes:
        name: Parameter name as it appears in the formula
            (e.g. ``"step_index"``, ``"C(complexity)[T.medium]"``).
        estimate: Point estimate of the coefficient.
        std_error: Standard error of the estimate.
        z_statistic: Wald z-statistic (``estimate / std_error``).
        p_value: Two-sided Wald p-value.
        ci_low: Lower bound of the CI at the fit's confidence level.
        ci_high: Upper bound.
    """

    name: str
    estimate: float
    std_error: float
    z_statistic: float
    p_value: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict[str, Any]:
        def _f(x: float) -> float | str:
            return "nan" if math.isnan(x) else x

        return {
            "name": self.name,
            "estimate": _f(self.estimate),
            "std_error": _f(self.std_error),
            "z_statistic": _f(self.z_statistic),
            "p_value": _f(self.p_value),
            "ci_low": _f(self.ci_low),
            "ci_high": _f(self.ci_high),
        }


@dataclass(frozen=True)
class RandomEffects:
    """Random-effects variance components from a mixed-effects fit.

    Attributes:
        group_variance: Variance of the group-level random
            intercept (σ²_group).
        residual_variance: Residual variance (σ²_ε).
        n_groups: Number of levels in the grouping variable.
        icc: Intraclass correlation coefficient,
            σ²_group / (σ²_group + σ²_ε). Dimensionless in [0, 1].
            A large ICC (> 0.1) means a meaningful fraction of the
            outcome variance is between-group rather than
            within-group — which is exactly why the mixed-effects
            structure is needed.
    """

    group_variance: float
    residual_variance: float
    n_groups: int
    icc: float

    def to_dict(self) -> dict[str, Any]:
        def _f(x: float) -> float | str:
            return "nan" if math.isnan(x) else x

        return {
            "group_variance": _f(self.group_variance),
            "residual_variance": _f(self.residual_variance),
            "n_groups": self.n_groups,
            "icc": _f(self.icc),
        }


@dataclass(frozen=True)
class MixedEffectsResult:
    """Structured result from a mixed-effects fit.

    Attributes:
        formula: The statsmodels formula that produced this fit.
        group_col: The column used as the grouping (random-intercept)
            variable.
        n_observations: Number of rows in the input frame that were
            fit (after any formula-driven row dropping for missing
            values).
        coefficients: One :class:`CoefficientRow` per fixed-effect
            parameter, in the order statsmodels produced them.
        random_effects: :class:`RandomEffects` variance components.
        log_likelihood: Log-likelihood of the fit.
        confidence_level: The confidence level used for the CI
            columns on the coefficient rows.
        converged: Whether ``statsmodels``' underlying optimizer
            self-reported success. **Not** the right thing to gate
            reporting on — statsmodels routinely marks fits as
            non-converged when the random-effect variance hits a
            boundary, even though the fixed-effect coefficients are
            fine. Use :attr:`fit_usable` for "is it safe to cite".
        fit_usable: ``True`` when the fit produced finite fixed-
            effect coefficients and standard errors, regardless of
            whether the optimizer self-reported success. This is
            what :meth:`slope_estimate` gates on.
        fit_warnings: Any ``ConvergenceWarning`` messages caught
            during the fit, as a list of strings.
        fit_error: Exception message if the fit raised; empty string
            otherwise.
        method: Short tag identifying which wrapper produced this
            result (``"step_level_lmm"``, ``"trace_slope_lmm"``,
            ``"custom_lmm"``).
        extras: Free-form dict for method-specific metadata (e.g.
            per-trace slope counts for the trace-level model).
    """

    formula: str
    group_col: str
    n_observations: int
    coefficients: list[CoefficientRow]
    random_effects: RandomEffects
    log_likelihood: float
    confidence_level: ConfidenceLevel
    converged: bool
    fit_usable: bool
    fit_warnings: list[str]
    fit_error: str
    method: str
    extras: dict[str, Any] = field(default_factory=dict)

    def coefficient(self, name: str) -> CoefficientRow:
        """Return the :class:`CoefficientRow` for a named fixed effect.

        Raises :class:`KeyError` with the list of available names if
        the name is not present — this is deliberately louder than
        returning ``None`` because every caller of this method is
        asserting the effect's presence.
        """
        for row in self.coefficients:
            if row.name == name:
                return row
        available = [row.name for row in self.coefficients]
        raise KeyError(
            f"fixed effect {name!r} not present in fit; available: {available}"
        )

    def slope_estimate(self, name: str = "step_index") -> Estimate:
        """Return an :class:`Estimate` for a named fixed-effect coefficient.

        This is the bridge between the mixed-effects result and the
        analysis layer's uniform :class:`Estimate` contract. Gated on
        :attr:`fit_usable` rather than :attr:`converged` — a fit with
        finite coefficients and SEs is safe to cite even if
        statsmodels' optimizer flagged a boundary issue on the
        random-effect variance. Fits with empty coefficients or
        non-finite SEs return an :class:`Estimate` with
        ``method="mixed_effects_not_usable"`` and nan CI.
        """
        if not self.fit_usable:
            return Estimate(
                value=float("nan"),
                ci_low=float("nan"),
                ci_high=float("nan"),
                n=self.n_observations,
                method="mixed_effects_not_usable",
                confidence_level=self.confidence_level,
            )
        try:
            row = self.coefficient(name)
        except KeyError:
            return Estimate(
                value=float("nan"),
                ci_low=float("nan"),
                ci_high=float("nan"),
                n=self.n_observations,
                method="mixed_effects_coefficient_missing",
                confidence_level=self.confidence_level,
            )
        return Estimate(
            value=row.estimate,
            ci_low=row.ci_low,
            ci_high=row.ci_high,
            n=self.n_observations,
            method="mixed_effects",
            confidence_level=self.confidence_level,
            se=row.std_error,
        )

    def to_dict(self) -> dict[str, Any]:
        def _f(x: float) -> float | str:
            return "nan" if math.isnan(x) else x

        return {
            "formula": self.formula,
            "group_col": self.group_col,
            "n_observations": self.n_observations,
            "coefficients": [c.to_dict() for c in self.coefficients],
            "random_effects": self.random_effects.to_dict(),
            "log_likelihood": _f(self.log_likelihood),
            "confidence_level": self.confidence_level.level,
            "converged": self.converged,
            "fit_usable": self.fit_usable,
            "fit_warnings": list(self.fit_warnings),
            "fit_error": self.fit_error,
            "method": self.method,
            "extras": dict(self.extras),
        }


# ---------------------------------------------------------------------------
# General-purpose fit wrapper
# ---------------------------------------------------------------------------


def fit_mixed_effects(
    df: pd.DataFrame,
    *,
    formula: str,
    group_col: str,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    method: str = "custom_lmm",
    reml: bool = True,
    extras: dict[str, Any] | None = None,
    vc_formula: dict[str, str] | None = None,
) -> MixedEffectsResult:
    """Fit a linear mixed-effects model against a formula.

    Thin wrapper around ``statsmodels.formula.api.mixedlm``. Captures
    convergence warnings and raised exceptions into the returned
    :class:`MixedEffectsResult` instead of letting them escape.

    Args:
        df: Input dataframe. Must contain every column referenced
            by ``formula`` and the ``group_col``.
        formula: Statsmodels Patsy formula. The LHS is the outcome;
            the RHS lists fixed effects. Random intercepts are
            specified via ``group_col``, not in the formula.
        group_col: Name of the dataframe column used as the grouping
            variable for the random intercept.
        confidence_level: Two-sided confidence level applied to the
            coefficient CIs.
        method: Short tag propagated into the result's ``method``
            field, for traceability when callers wrap this with
            higher-level helpers.
        reml: Whether to fit via REML (Restricted Maximum
            Likelihood, the statistical default for variance
            estimation) or plain ML. REML is preferred for variance
            components; ML is preferred for likelihood-ratio tests
            across models with different fixed effects. Default REML.
        extras: Optional metadata passed through to
            :attr:`MixedEffectsResult.extras`. Used by higher-level
            wrappers to attach context (per-trace slope counts,
            drop reasons, etc.).

    Returns:
        A :class:`MixedEffectsResult`. A failing or non-converging
        fit produces a well-formed result with ``converged=False``
        and the failure details recorded; callers should check
        ``converged`` before citing coefficients.
    """
    # Import statsmodels here so modules that don't use mixed-effects
    # don't pay the import cost (statsmodels is heavy).
    import statsmodels.formula.api as smf

    # Validate inputs cheaply before paying the fit cost.
    if group_col not in df.columns:
        raise ValueError(
            f"group_col {group_col!r} not in dataframe columns: {list(df.columns)}"
        )

    extras_dict = dict(extras or {})

    if df.empty:
        return _empty_result(
            formula=formula,
            group_col=group_col,
            method=method,
            confidence_level=confidence_level,
            extras=extras_dict,
            reason="empty_frame",
        )
    if df[group_col].nunique() < 2:
        return _empty_result(
            formula=formula,
            group_col=group_col,
            method=method,
            confidence_level=confidence_level,
            extras=extras_dict,
            reason="insufficient_groups",
            n_observations=len(df),
        )

    # Patsy (the formula engine) interprets ``bool`` dtype columns as
    # two-level categoricals and expands them to a 2-column design
    # matrix, which statsmodels' MixedLM then rejects with a cryptic
    # "endog has multiple columns" ValueError. The tidy frame
    # produced by :func:`traces_to_frame` carries ``is_error`` etc. as
    # bool, so the wrapper has to coerce them to float before fitting.
    #
    # Similarly, columns containing Python enum objects (e.g.
    # ComplexityLevel from traces_to_frame) cause Patsy index errors.
    # Coerce them to their string values so Patsy treats them as
    # ordinary categoricals.
    needs_copy = False
    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    # Detect columns holding enum objects: dtype is 'object' and the
    # first non-null value has a .value attribute (Enum convention).
    enum_cols: list[str] = []
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().iloc[0] if len(df[c].dropna()) else None
            if sample is not None and hasattr(sample, "value"):
                enum_cols.append(c)

    if bool_cols or enum_cols:
        df = df.copy()
        for c in bool_cols:
            df[c] = df[c].astype(float)
        for c in enum_cols:
            df[c] = df[c].apply(lambda v: v.value if hasattr(v, "value") else v)

    warning_messages: list[str] = []
    fit_error = ""
    result_obj: Any = None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if vc_formula is not None:
                model = smf.mixedlm(
                    formula, df, groups=df[group_col], vc_formula=vc_formula
                )
            else:
                model = smf.mixedlm(formula, df, groups=df[group_col])
            result_obj = model.fit(reml=reml)
        except Exception as exc:  # pragma: no cover — exercised in failure tests
            fit_error = f"{type(exc).__name__}: {exc}"
        for w in caught:
            warning_messages.append(f"{w.category.__name__}: {w.message}")

    if result_obj is None:
        return MixedEffectsResult(
            formula=formula,
            group_col=group_col,
            n_observations=len(df),
            coefficients=[],
            random_effects=RandomEffects(
                group_variance=float("nan"),
                residual_variance=float("nan"),
                n_groups=int(df[group_col].nunique()),
                icc=float("nan"),
            ),
            log_likelihood=float("nan"),
            confidence_level=confidence_level,
            converged=False,
            fit_usable=False,
            fit_warnings=warning_messages,
            fit_error=fit_error,
            method=method,
            extras=extras_dict,
        )

    # Extract coefficient table.
    params: pd.Series = result_obj.fe_params
    bse: pd.Series = result_obj.bse_fe
    pvalues: pd.Series = result_obj.pvalues
    ci = result_obj.conf_int(alpha=confidence_level.alpha)

    coefficients: list[CoefficientRow] = []
    for name in params.index:
        est = float(params.loc[name])
        se = float(bse.loc[name])
        z = est / se if se > 0 else float("nan")
        p_raw = pvalues.loc[name] if name in pvalues.index else float("nan")
        p = float(p_raw)
        ci_low, ci_high = float(ci.loc[name, 0]), float(ci.loc[name, 1])
        coefficients.append(
            CoefficientRow(
                name=name,
                estimate=est,
                std_error=se,
                z_statistic=z,
                p_value=p,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )

    # Random-effects variance.
    try:
        group_variance = float(result_obj.cov_re.iloc[0, 0])
    except (AttributeError, IndexError):
        group_variance = float("nan")
    residual_variance = float(result_obj.scale)
    denom = group_variance + residual_variance
    icc = group_variance / denom if denom > 0 else float("nan")

    rand = RandomEffects(
        group_variance=group_variance,
        residual_variance=residual_variance,
        n_groups=int(result_obj.model.n_groups),
        icc=icc,
    )

    converged = bool(getattr(result_obj, "converged", True))

    # "Fit usable" is distinct from "optimizer reported converged":
    # the fit is usable if we extracted finite fixed-effect
    # coefficients and standard errors. statsmodels flags many
    # boundary-variance fits as non-converged even when the fixed
    # effects are completely sane, so gating citation on the raw
    # converged flag would throw away legitimate fits. We compute
    # fit_usable here from the coefficient table we actually built.
    fit_usable = bool(coefficients) and all(
        math.isfinite(c.estimate) and math.isfinite(c.std_error)
        for c in coefficients
    )

    return MixedEffectsResult(
        formula=formula,
        group_col=group_col,
        n_observations=int(result_obj.nobs),
        coefficients=coefficients,
        random_effects=rand,
        log_likelihood=float(result_obj.llf),
        confidence_level=confidence_level,
        converged=converged,
        fit_usable=fit_usable,
        fit_warnings=warning_messages,
        fit_error=fit_error,
        method=method,
        extras=extras_dict,
    )


def _empty_result(
    *,
    formula: str,
    group_col: str,
    method: str,
    confidence_level: ConfidenceLevel,
    extras: dict[str, Any],
    reason: str,
    n_observations: int = 0,
) -> MixedEffectsResult:
    """Construct a "cannot fit" result with a specific reason string."""
    return MixedEffectsResult(
        formula=formula,
        group_col=group_col,
        n_observations=n_observations,
        coefficients=[],
        random_effects=RandomEffects(
            group_variance=float("nan"),
            residual_variance=float("nan"),
            n_groups=0,
            icc=float("nan"),
        ),
        log_likelihood=float("nan"),
        confidence_level=confidence_level,
        converged=False,
        fit_usable=False,
        fit_warnings=[],
        fit_error=reason,
        method=method,
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Step-level model (the PROJECT_PLAN.md version)
# ---------------------------------------------------------------------------


def fit_step_level_model(
    df: pd.DataFrame,
    *,
    outcome: str = "is_error",
    step_col: str = "step_index",
    complexity_col: str | None = "complexity",
    model_col: str | None = "model",
    success_col: str | None = "trace_success",
    phase_col: str | None = "step_phase",
    interactions: list[str] | None = None,
    group_col: str = "task_id",
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    reml: bool = True,
) -> MixedEffectsResult:
    """Fit the step-level degradation model.

    Canonical formula:
    ``<outcome> ~ step_index + C(complexity) + C(model) + C(trace_success)``
    with a random intercept for ``task_id``. Adapts gracefully to
    the columns actually present in the dataframe:

    * If ``complexity_col`` is None or the column has fewer than two
      distinct values, the ``C(complexity)`` term is dropped.
    * If ``model_col`` is None or has fewer than two distinct
      values, the ``C(model)`` term is dropped.
    * If ``success_col`` is None or has fewer than two distinct
      values, the ``C(trace_success)`` term is dropped.

    This lets the same function run on a single-model corpus (where
    ``C(model)`` would be singular) or on a frame that doesn't
    carry complexity labels. Adaptation is logged into
    ``extras["dropped_terms"]`` on the result so downstream reports
    can note which controls were active.

    **Why ``trace_success`` is in the canonical formula.** Without it,
    the ``step_index`` slope conflates real within-trace degradation
    with a *survivorship-bias* artifact: failed traces are
    systematically longer than successful ones (failed agents keep
    trying, successful agents stop), so "errors increase with step
    index" partly reflects "long traces are the failed ones, and
    failed traces have a higher unconditional error rate." Including
    ``C(trace_success)`` partials out this between-trace selection
    effect so the slope can be interpreted as "per-step change in
    error probability conditional on eventual outcome." For corpora
    where every trace has the same outcome (or where ``success`` is
    not labeled), the term is dropped automatically.

    **Interpretation.** With a binary 0/1 outcome, this is a
    *linear probability model*: the coefficient on ``step_index``
    is the per-step change in the probability of error, holding
    complexity, model, task, and outcome constant. For error rates
    between about 15% and 85% the LPM slope is within a few percent
    of the logistic-GLMM slope transformed to the probability scale;
    at more extreme rates the approximation can be biased toward
    zero (the linear fit hits the [0, 1] floor/ceiling). See module
    docstring for the rationale and the GLMM fallback tracked for
    Phase 2.

    Args:
        df: Tidy graded-trace frame, typically produced by
            :func:`inspect_degradation.analysis.frame.traces_to_frame`.
        outcome: Name of the binary outcome column. Default
            ``"is_error"``; other candidates are ``"is_neutral"`` or
            ``"is_productive"`` for flailing and progress models.
        step_col: Name of the step-index column. Default
            ``"step_index"``.
        complexity_col: Optional complexity control column. Pass
            ``None`` to drop it explicitly. Must be a categorical.
        model_col: Optional model-identity control column. Pass
            ``None`` to drop it explicitly.
        success_col: Optional eventual-outcome control column.
            Default ``"trace_success"``. Pass ``None`` to disable
            outcome stratification entirely (e.g., when running a
            "what does the marginal slope look like ignoring
            outcome?" sensitivity check).
        interactions: Optional list of Patsy interaction terms to
            append to the formula. For example,
            ``["step_index:C(step_phase)"]`` adds a step_index x
            step_phase interaction that tests whether degradation
            exists *within* each phase. Terms are appended as-is;
            the caller is responsible for using valid Patsy syntax.
        group_col: Grouping column for the random intercept. Default
            ``"task_id"``.
        confidence_level: Two-sided confidence level for coefficient
            CIs.
        reml: Whether to fit via REML. Default True.

    Returns:
        A :class:`MixedEffectsResult` with ``method="step_level_lmm"``.
    """
    if outcome not in df.columns:
        raise ValueError(
            f"outcome column {outcome!r} not in dataframe columns: {list(df.columns)}"
        )
    if step_col not in df.columns:
        raise ValueError(
            f"step column {step_col!r} not in dataframe columns: {list(df.columns)}"
        )

    terms: list[str] = [step_col]
    dropped: list[str] = []

    # Complexity is ordinal (low < medium < high), so encode it as a
    # numeric rank rather than using C() categorical expansion. This
    # respects the ordering, uses one coefficient instead of k-1
    # dummies, and avoids a statsmodels indexing bug triggered by
    # C() on very unbalanced categoricals with grouped data.
    _COMPLEXITY_RANK = {"low": 0, "medium": 1, "high": 2}
    if complexity_col and complexity_col in df.columns:
        if df[complexity_col].nunique(dropna=True) >= 2:
            complexity_num_col = f"{complexity_col}_num"
            df = df.copy()
            df[complexity_num_col] = df[complexity_col].apply(
                lambda v: _COMPLEXITY_RANK.get(
                    v.value if hasattr(v, "value") else v, 0
                )
            )
            terms.append(complexity_num_col)
        else:
            dropped.append(f"{complexity_col}: fewer than 2 distinct values")
    elif complexity_col:
        dropped.append(f"{complexity_col}: not in dataframe")

    if model_col and model_col in df.columns:
        if df[model_col].nunique(dropna=True) >= 2:
            terms.append(f"C({model_col})")
        else:
            dropped.append(f"{model_col}: fewer than 2 distinct values")
    elif model_col:
        dropped.append(f"{model_col}: not in dataframe")

    if success_col and success_col in df.columns:
        if df[success_col].nunique(dropna=True) >= 2:
            terms.append(f"C({success_col})")
        else:
            dropped.append(f"{success_col}: fewer than 2 distinct values")
    elif success_col:
        dropped.append(f"{success_col}: not in dataframe")

    if phase_col and phase_col in df.columns:
        if df[phase_col].nunique(dropna=True) >= 2:
            terms.append(f"C({phase_col})")
        else:
            dropped.append(f"{phase_col}: fewer than 2 distinct values")
    elif phase_col:
        dropped.append(f"{phase_col}: not in dataframe")

    if interactions:
        terms.extend(interactions)

    formula = f"{outcome} ~ " + " + ".join(terms)
    extras: dict[str, Any] = {"dropped_terms": dropped}
    if interactions:
        extras["interactions"] = list(interactions)

    return fit_mixed_effects(
        df=df,
        formula=formula,
        group_col=group_col,
        confidence_level=confidence_level,
        method="step_level_lmm",
        reml=reml,
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Trace-level slope model
# ---------------------------------------------------------------------------


def fit_trace_level_slope_model(
    traces: Iterable[Any],
    *,
    outcome: str = "is_error",
    task_family_col: str | None = None,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    reml: bool = True,
) -> MixedEffectsResult:
    """Fit a mixed-effects model on per-trace degradation slopes.

    Workflow:

    1. Compute one OLS slope of ``<outcome> ~ step_index`` per trace
       using :func:`inspect_degradation.analysis.slopes.per_trace_mean_slope`
       machinery internally. Degenerate traces (too short, zero
       variance) are dropped with reasons recorded in
       ``extras["drop_reasons"]``.
    2. Build a flat frame with one row per surviving trace, with
       columns ``slope``, ``task_id``, ``model``, and optionally
       ``task_family`` (if ``task_family_col`` is set on each
       trace's metadata).
    3. Fit ``slope ~ C(model)`` (plus task-family if present) with
       a random intercept on ``task_id``.

    The advantage over :func:`fit_step_level_model` is that the
    outcome is continuous, so the linear assumption is exactly
    right and the confounds decompose cleanly. The question this
    answers is *"how does the per-trace degradation slope vary by
    model and task"* — distinct from *"how does error rate depend
    on step position controlling for everything else."*

    Args:
        traces: Iterable of :class:`~inspect_degradation.schema.GradedTrace`
            (or any object that :class:`~inspect_degradation.analysis.slopes.per_trace_mean_slope`
            accepts).
        outcome: Which binary attribute of a step to regress.
            Default ``"is_error"``.
        task_family_col: Optional metadata key to pull from each
            trace's ``.metadata`` dict as a task-family fixed
            effect. If None or absent on a trace, the term is
            dropped.
        confidence_level: Two-sided confidence level.
        reml: Whether to fit via REML.

    Returns:
        :class:`MixedEffectsResult` with ``method="trace_slope_lmm"``.
    """
    from inspect_degradation.analysis.slopes import _per_trace_slope
    from inspect_degradation.schema import Validity

    predicate_map = {
        "is_error": lambda s: s.validity == Validity.fail,
        "is_neutral": lambda s: s.validity == Validity.neutral,
        "is_productive": lambda s: s.validity == Validity.pass_,
    }
    if outcome not in predicate_map:
        raise ValueError(
            f"unknown outcome {outcome!r}; must be one of {sorted(predicate_map)}"
        )
    predicate = predicate_map[outcome]

    trace_list = list(traces)
    rows: list[dict[str, Any]] = []
    drop_reasons: dict[str, int] = {}
    for trace in trace_list:
        pt = _per_trace_slope(trace, predicate)
        if pt.dropped_reason is not None:
            drop_reasons[pt.dropped_reason] = drop_reasons.get(pt.dropped_reason, 0) + 1
            continue
        row: dict[str, Any] = {
            "slope": pt.slope,
            "task_id": trace.task_id or trace.trace_id,
            "model": trace.model or "unknown",
        }
        if task_family_col:
            row["task_family"] = trace.metadata.get(task_family_col, "unknown")
        rows.append(row)

    extras: dict[str, Any] = {
        "drop_reasons": drop_reasons,
        "n_traces_total": len(trace_list),
        "n_traces_used": len(rows),
        "outcome": outcome,
    }

    frame = pd.DataFrame(rows)
    if frame.empty:
        return _empty_result(
            formula=f"slope ~ ... (outcome={outcome})",
            group_col="task_id",
            method="trace_slope_lmm",
            confidence_level=confidence_level,
            extras=extras,
            reason="no_usable_traces",
        )

    terms: list[str] = []
    if frame["model"].nunique() >= 2:
        terms.append("C(model)")
    else:
        extras.setdefault("dropped_terms", []).append(
            "model: fewer than 2 distinct values"
        )
    if task_family_col:
        if frame["task_family"].nunique() >= 2:
            terms.append("C(task_family)")
        else:
            extras.setdefault("dropped_terms", []).append(
                "task_family: fewer than 2 distinct values"
            )

    formula = "slope ~ " + (" + ".join(terms) if terms else "1")

    return fit_mixed_effects(
        df=frame,
        formula=formula,
        group_col="task_id",
        confidence_level=confidence_level,
        method="trace_slope_lmm",
        reml=reml,
        extras=extras,
    )


# ---------------------------------------------------------------------------
# Crossed random effects
# ---------------------------------------------------------------------------


def fit_crossed_effects_model(
    df: pd.DataFrame,
    *,
    formula: str,
    primary_group: str,
    crossed_group: str,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    reml: bool = True,
) -> MixedEffectsResult:
    """Mixed model with two crossed (non-nested) random intercepts.

    Use this when both grouping variables vary independently — the
    canonical example is ``trace_id`` (one row per agent run) and
    ``model`` (the underlying LLM), which are crossed because every
    model is run on every task and vice versa, so neither nests
    inside the other. ``fit_step_level_model`` only carries one
    random intercept (typically ``task_id``) and so cannot
    distinguish between-trace and between-model variance; this
    estimator can.

    Implementation: statsmodels' :class:`MixedLM` only supports a
    single grouping column directly, but a constant outer group
    combined with two ``vc_formula`` variance components is the
    standard idiom for crossed random effects in this library.
    The first column is treated as the primary group and the second
    is added as a variance component.

    Args:
        df: Tidy step-level frame. Must contain every column
            referenced by ``formula`` plus both group columns.
        formula: Patsy formula for the fixed-effects part. Same
            shape as :func:`fit_mixed_effects`.
        primary_group: Column used as the principal grouping
            variable for the random intercept.
        crossed_group: Column used as the *additional* crossed
            random effect.
        confidence_level: CI level for coefficients.
        reml: Whether to fit by REML (default) or ML.

    Returns:
        :class:`MixedEffectsResult` with the crossed-group variance
        recorded in ``extras['crossed_variance']`` and the primary
        group variance on the standard ``random_effects`` field.
        ``method`` is set to ``"crossed_lmm"``.
    """
    for col in (primary_group, crossed_group):
        if col not in df.columns:
            raise ValueError(
                f"crossed-effects column {col!r} not in dataframe columns: "
                f"{list(df.columns)}"
            )
    if primary_group == crossed_group:
        raise ValueError(
            "primary_group and crossed_group must be distinct columns"
        )

    # Pre-check the crossed group: fit_mixed_effects only validates
    # the primary group, but a single-level crossed effect is just
    # noise added to the residual and should be flagged the same
    # way as an insufficient primary group.
    if not df.empty and df[crossed_group].nunique() < 2:
        return _empty_result(
            formula=formula,
            group_col=primary_group,
            method="crossed_lmm",
            confidence_level=confidence_level,
            extras={"crossed_group": crossed_group},
            reason="insufficient_crossed_groups",
            n_observations=len(df),
        )

    vc = {crossed_group: f"0 + C({crossed_group})"}
    return fit_mixed_effects(
        df=df,
        formula=formula,
        group_col=primary_group,
        confidence_level=confidence_level,
        method="crossed_lmm",
        reml=reml,
        extras={"crossed_group": crossed_group},
        vc_formula=vc,
    )


# ---------------------------------------------------------------------------
# GLMM (logit-link via variational Bayes)
# ---------------------------------------------------------------------------


def fit_step_level_glmm(
    df: pd.DataFrame,
    *,
    outcome: str = "is_error",
    step_col: str = "step_index",
    complexity_col: str | None = "complexity",
    model_col: str | None = "model",
    success_col: str | None = "trace_success",
    phase_col: str | None = "step_phase",
    interactions: list[str] | None = None,
    group_col: str = "task_id",
    confidence_level: ConfidenceLevel = NINETY_FIVE,
) -> MixedEffectsResult:
    """Logit-link GLMM for step-level error via variational Bayes.

    Uses :class:`statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM`
    to fit a proper logistic mixed model with a random intercept for
    ``task_id``. This is the strictly correct model for a binary
    outcome and avoids the linear-probability approximation bias that
    :func:`fit_step_level_model` exhibits when error rates are near 0
    or 1.

    The returned :class:`MixedEffectsResult` reports coefficients on
    the **logit scale** (standard for GLMM reporting). For
    probability-scale interpretation, ``extras["marginal_effects"]``
    provides the average marginal effect of each coefficient.

    Args:
        df: Tidy graded-trace frame.
        outcome: Binary outcome column. Default ``"is_error"``.
        step_col: Step-index column. Default ``"step_index"``.
        complexity_col: Optional complexity control column.
        model_col: Optional model-identity control column.
        success_col: Optional eventual-outcome control column.
        phase_col: Optional step-phase control column.
        group_col: Grouping column for random intercept.
        confidence_level: Confidence level for CIs.

    Returns:
        A :class:`MixedEffectsResult` with ``method="step_level_glmm"``.
    """
    from scipy.stats import norm as _norm
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    method = "step_level_glmm"
    extras: dict[str, Any] = {"dropped_terms": []}
    dropped: list[str] = extras["dropped_terms"]

    # --- Validate required columns ---
    if outcome not in df.columns:
        raise ValueError(
            f"outcome column {outcome!r} not in dataframe columns: {list(df.columns)}"
        )
    if step_col not in df.columns:
        raise ValueError(
            f"step column {step_col!r} not in dataframe columns: {list(df.columns)}"
        )

    # --- Edge cases ---
    formula_placeholder = f"{outcome} ~ {step_col}"
    if df.empty:
        return _empty_result(
            formula=formula_placeholder,
            group_col=group_col,
            method=method,
            confidence_level=confidence_level,
            extras=extras,
            reason="empty_frame",
        )
    if group_col not in df.columns or df[group_col].nunique() < 2:
        return _empty_result(
            formula=formula_placeholder,
            group_col=group_col,
            method=method,
            confidence_level=confidence_level,
            extras=extras,
            reason="insufficient_groups",
            n_observations=len(df),
        )

    # --- Coerce bool / enum columns ---
    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    enum_cols: list[str] = []
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().iloc[0] if len(df[c].dropna()) else None
            if sample is not None and hasattr(sample, "value"):
                enum_cols.append(c)
    if bool_cols or enum_cols:
        df = df.copy()
        for c in bool_cols:
            df[c] = df[c].astype(float)
        for c in enum_cols:
            df[c] = df[c].apply(lambda v: v.value if hasattr(v, "value") else v)
    else:
        df = df.copy()

    # --- Build formula (same adaptive logic as fit_step_level_model) ---
    terms: list[str] = [step_col]

    _COMPLEXITY_RANK = {"low": 0, "medium": 1, "high": 2}
    if complexity_col and complexity_col in df.columns:
        if df[complexity_col].nunique(dropna=True) >= 2:
            complexity_num_col = f"{complexity_col}_num"
            df[complexity_num_col] = df[complexity_col].apply(
                lambda v: _COMPLEXITY_RANK.get(
                    v.value if hasattr(v, "value") else v, 0
                )
            )
            terms.append(complexity_num_col)
        else:
            dropped.append(f"{complexity_col}: fewer than 2 distinct values")
    elif complexity_col:
        dropped.append(f"{complexity_col}: not in dataframe")

    if model_col and model_col in df.columns:
        if df[model_col].nunique(dropna=True) >= 2:
            terms.append(f"C({model_col})")
        else:
            dropped.append(f"{model_col}: fewer than 2 distinct values")
    elif model_col:
        dropped.append(f"{model_col}: not in dataframe")

    if success_col and success_col in df.columns:
        if df[success_col].nunique(dropna=True) >= 2:
            terms.append(f"C({success_col})")
        else:
            dropped.append(f"{success_col}: fewer than 2 distinct values")
    elif success_col:
        dropped.append(f"{success_col}: not in dataframe")

    if phase_col and phase_col in df.columns:
        if df[phase_col].nunique(dropna=True) >= 2:
            terms.append(f"C({phase_col})")
        else:
            dropped.append(f"{phase_col}: fewer than 2 distinct values")
    elif phase_col:
        dropped.append(f"{phase_col}: not in dataframe")

    if interactions:
        terms.extend(interactions)

    formula = f"{outcome} ~ " + " + ".join(terms)
    if interactions:
        extras["interactions"] = list(interactions)

    # --- Ensure outcome is float for statsmodels ---
    df[outcome] = df[outcome].astype(float)

    # --- Fit the GLMM ---
    warning_messages: list[str] = []
    result_obj: Any = None

    vc_formulas = {group_col: f"0 + C({group_col})"}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = BinomialBayesMixedGLM.from_formula(
                formula, vc_formulas=vc_formulas, data=df
            )
            result_obj = model.fit_vb()
        except Exception as exc:
            fit_error = f"{type(exc).__name__}: {exc}"
            warning_messages.extend(
                f"{w.category.__name__}: {w.message}" for w in caught
            )
            return MixedEffectsResult(
                formula=formula,
                group_col=group_col,
                n_observations=len(df),
                coefficients=[],
                random_effects=RandomEffects(
                    group_variance=float("nan"),
                    residual_variance=float("nan"),
                    n_groups=int(df[group_col].nunique()),
                    icc=float("nan"),
                ),
                log_likelihood=float("nan"),
                confidence_level=confidence_level,
                converged=False,
                fit_usable=False,
                fit_warnings=warning_messages,
                fit_error=fit_error,
                method=method,
                extras=extras,
            )
        for w in caught:
            warning_messages.append(f"{w.category.__name__}: {w.message}")

    # --- Extract fixed-effect coefficients ---
    fe_mean = np.asarray(result_obj.fe_mean)
    fe_sd = np.asarray(result_obj.fe_sd)
    param_names = list(result_obj.model.fep_names)

    coefficients: list[CoefficientRow] = []
    z_crit = _norm.ppf(1.0 - confidence_level.alpha / 2)

    for i, name in enumerate(param_names):
        est = float(fe_mean[i])
        se = float(fe_sd[i])
        z = est / se if se > 0 else float("nan")
        p = float(2.0 * (1.0 - _norm.cdf(abs(z)))) if math.isfinite(z) else float("nan")
        ci_low = est - z_crit * se
        ci_high = est + z_crit * se
        coefficients.append(
            CoefficientRow(
                name=name,
                estimate=est,
                std_error=se,
                z_statistic=z,
                p_value=p,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )

    # --- Random-effects variance (latent-variable ICC) ---
    vcp_mean = np.asarray(result_obj.vcp_mean)
    # vcp_mean is log-sd; variance = exp(log_sd)^2
    group_variance = float(np.sum(np.exp(vcp_mean) ** 2)) if len(vcp_mean) > 0 else 0.0
    # Logistic residual variance on latent scale: pi^2/3
    residual_variance = math.pi**2 / 3.0
    denom = group_variance + residual_variance
    icc = group_variance / denom if denom > 0 else float("nan")

    rand = RandomEffects(
        group_variance=group_variance,
        residual_variance=residual_variance,
        n_groups=int(df[group_col].nunique()),
        icc=icc,
    )

    # --- Marginal effects (average marginal effect on probability scale) ---
    try:
        import patsy

        dmat = patsy.dmatrix(
            " + ".join(terms), data=df, return_type="dataframe"
        )
        # Add intercept column if not present
        if "Intercept" not in dmat.columns:
            dmat.insert(0, "Intercept", 1.0)
        X = dmat.values
        linear_pred = X @ fe_mean
        # logistic pdf: exp(-x)/(1+exp(-x))^2 = pdf of logistic(0,1)
        from scipy.stats import logistic as _logistic

        pdf_vals = _logistic.pdf(linear_pred)
        marginal_effects: dict[str, float] = {}
        for i, name in enumerate(param_names):
            # AME = mean(pdf(X@beta) * beta_i)
            marginal_effects[name] = float(np.mean(pdf_vals * fe_mean[i]))
    except Exception:
        marginal_effects = {}

    extras["marginal_effects"] = marginal_effects

    # --- Log-likelihood / ELBO ---
    log_likelihood = float(getattr(result_obj, "elbo", float("nan")))
    if not isinstance(log_likelihood, float) or not math.isfinite(log_likelihood):
        # elbo may be an array; take the last value
        try:
            log_likelihood = float(np.asarray(log_likelihood).flat[-1])
        except Exception:
            log_likelihood = float("nan")

    # --- Convergence & fit_usable ---
    converged = True  # VB always "converges" (iterates to tolerance)
    fit_usable = bool(coefficients) and all(
        math.isfinite(c.estimate) and math.isfinite(c.std_error)
        for c in coefficients
    )

    return MixedEffectsResult(
        formula=formula,
        group_col=group_col,
        n_observations=len(df),
        coefficients=coefficients,
        random_effects=rand,
        log_likelihood=log_likelihood,
        confidence_level=confidence_level,
        converged=converged,
        fit_usable=fit_usable,
        fit_warnings=warning_messages,
        fit_error="",
        method=method,
        extras=extras,
    )


__all__ = [
    "CoefficientRow",
    "MixedEffectsResult",
    "RandomEffects",
    "fit_crossed_effects_model",
    "fit_mixed_effects",
    "fit_step_level_glmm",
    "fit_step_level_model",
    "fit_trace_level_slope_model",
]
