"""Survival / hazard analysis for first-error timing.

Exposes two complementary estimators:

* :func:`first_error_km` — Kaplan-Meier estimate of "probability the
  agent has not yet made its first error by step N" with CI bands.
  Good for overall curves and median survival time, no covariates.
* :func:`cox_first_error` — Cox proportional-hazards regression for
  first-error timing with covariates (complexity, model, task).
  Returns hazard-ratio style :class:`CoefficientRow` entries. Good
  for "does task complexity accelerate first error *after*
  controlling for model identity?" style questions.

Both fitters accept the tidy DataFrame produced by
:func:`inspect_degradation.analysis.frame.traces_to_frame`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import pandas as pd

from inspect_degradation.analysis.mixed_effects import CoefficientRow
from inspect_degradation.analysis.statistics import (
    NINETY_FIVE,
    ConfidenceLevel,
    Estimate,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class KMCurve:
    """A Kaplan-Meier survival curve with confidence bands.

    All three sequences (``timeline``, ``survival``, ``ci_lower``,
    ``ci_upper``) are the same length and are ordered by step index.
    Callers plotting a curve should zip them together.

    Attributes:
        timeline: Step indices at which the estimate is evaluated.
            Includes the origin (step 0) and every distinct event /
            censoring time observed in the data.
        survival: Estimated P(no error yet) at each timeline index.
        ci_lower: Lower bound of the CI on the survival estimate.
        ci_upper: Upper bound of the CI on the survival estimate.
        n_traces: Number of traces contributing to the fit.
        n_events: Number of observed "events" (first errors).
        confidence_level: Which CI level the bands correspond to.
    """

    timeline: list[int]
    survival: list[float]
    ci_lower: list[float]
    ci_upper: list[float]
    n_traces: int
    n_events: int
    confidence_level: ConfidenceLevel

    def __post_init__(self) -> None:
        n = len(self.timeline)
        if len(self.survival) != n or len(self.ci_lower) != n or len(self.ci_upper) != n:
            raise ValueError(
                f"KMCurve sequences must be equal length; got "
                f"timeline={n} survival={len(self.survival)} "
                f"ci_lower={len(self.ci_lower)} ci_upper={len(self.ci_upper)}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "timeline": list(self.timeline),
            "survival": list(self.survival),
            "ci_lower": list(self.ci_lower),
            "ci_upper": list(self.ci_upper),
            "n_traces": self.n_traces,
            "n_events": self.n_events,
            "confidence_level": self.confidence_level.level,
        }


@dataclass(frozen=True)
class KMResult:
    """Complete result of a Kaplan-Meier fit.

    Carries both the full curve (for plotting and curve comparisons)
    and a summary :class:`Estimate` for the median survival time — the
    scalar "by what step has half the corpus made its first error"
    that reports typically cite.
    """

    curve: KMCurve
    median_survival_time: Estimate
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def n_traces(self) -> int:
        return self.curve.n_traces

    @property
    def n_events(self) -> int:
        return self.curve.n_events

    def to_dict(self) -> dict[str, object]:
        return {
            "curve": self.curve.to_dict(),
            "median_survival_time": self.median_survival_time.to_dict(),
            "metadata": dict(self.metadata),
        }


def first_error_km(
    df: pd.DataFrame,
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
) -> KMResult:
    """Kaplan-Meier estimate of P(no error yet) by step index.

    Args:
        df: DataFrame with columns ``trace_id``, ``step_index``,
            ``is_error``. Typically produced by
            :func:`inspect_degradation.analysis.frame.traces_to_frame`.
        confidence_level: Two-sided confidence level for the survival
            bands. Default 95%.

    Returns:
        :class:`KMResult` containing the full curve with CI bands and
        a scalar median-survival-time estimate with its own CI from
        the Kaplan-Meier median-time computation.

    Traces with no errors are right-censored at their final observed
    step. Empty input returns a :class:`KMResult` with an empty curve
    and an :meth:`Estimate.empty`-shaped median.
    """
    if df.empty:
        return KMResult(
            curve=KMCurve(
                timeline=[],
                survival=[],
                ci_lower=[],
                ci_upper=[],
                n_traces=0,
                n_events=0,
                confidence_level=confidence_level,
            ),
            median_survival_time=Estimate.empty(confidence_level=confidence_level),
        )

    durations: list[int] = []
    events: list[int] = []
    for _, group in df.sort_values("step_index").groupby("trace_id"):
        errs = group[group["is_error"]]
        if len(errs):
            durations.append(int(errs["step_index"].iloc[0]))
            events.append(1)
        else:
            durations.append(int(group["step_index"].max()))
            events.append(0)

    try:
        from lifelines import KaplanMeierFitter
        from lifelines.utils import median_survival_times
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "lifelines is required for survival analysis; install via "
            "the package's main dependency group"
        ) from exc

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations,
        event_observed=events,
        alpha=confidence_level.alpha,
    )

    # confidence_interval_ is a DataFrame whose columns are
    # "KM_estimate_lower_<1-alpha>" and "KM_estimate_upper_<1-alpha>".
    ci_df = kmf.confidence_interval_
    ci_lower_col, ci_upper_col = ci_df.columns[0], ci_df.columns[1]

    curve = KMCurve(
        timeline=[int(t) for t in kmf.timeline],
        survival=[float(s) for s in kmf.survival_function_.iloc[:, 0]],
        ci_lower=[float(x) for x in ci_df[ci_lower_col]],
        ci_upper=[float(x) for x in ci_df[ci_upper_col]],
        n_traces=len(durations),
        n_events=int(sum(events)),
        confidence_level=confidence_level,
    )

    # Median survival time + its CI, also from lifelines.
    median_value = float(kmf.median_survival_time_)
    try:
        median_ci_df = median_survival_times(kmf.confidence_interval_)
        median_ci_low = float(median_ci_df.iloc[0, 0])
        median_ci_high = float(median_ci_df.iloc[0, 1])
    except Exception:
        # lifelines can return non-numeric placeholders when the
        # median is not reached (i.e., more than half the corpus
        # never errors). Surface this cleanly rather than crash.
        median_ci_low = float("nan")
        median_ci_high = float("nan")

    if math.isnan(median_value) or not math.isfinite(median_value):
        median_estimate = Estimate(
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=len(durations),
            method="km_median_unreached",
            confidence_level=confidence_level,
        )
    else:
        median_estimate = Estimate(
            value=median_value,
            ci_low=median_ci_low,
            ci_high=median_ci_high,
            n=len(durations),
            method="km_median",
            confidence_level=confidence_level,
        )

    return KMResult(
        curve=curve,
        median_survival_time=median_estimate,
        metadata={"fitter": "lifelines.KaplanMeierFitter"},
    )


@dataclass(frozen=True)
class CoxResult:
    """Result of a Cox proportional-hazards fit on first-error timing.

    Coefficients are reported on the log-hazard scale (``estimate``)
    with a matching hazard ratio (``hazard_ratio`` = exp(estimate))
    for readability. A positive coefficient means the covariate
    *accelerates* first error (higher hazard).

    Attributes:
        coefficients: One :class:`CoefficientRow` per model term, on
            the log-hazard scale. Reuses the mixed-effects row type
            so report code can treat both model families uniformly.
        hazard_ratios: ``{name: (hr, ci_low, ci_high)}`` on the
            hazard-ratio scale, for direct display.
        concordance: lifelines' concordance index (0.5 = chance,
            1.0 = perfect ordering). A rough analogue of AUC for
            survival data.
        log_likelihood: Fitted partial log-likelihood.
        n_traces: Number of traces (rows after aggregation) used.
        n_events: Number of observed first-error events.
        confidence_level: CI level for the coefficient intervals.
        formula: Patsy formula used to build the design matrix.
    """

    coefficients: list[CoefficientRow]
    hazard_ratios: dict[str, tuple[float, float, float]]
    concordance: float
    log_likelihood: float
    n_traces: int
    n_events: int
    confidence_level: ConfidenceLevel
    formula: str
    ph_test: dict[str, float] | None = None
    ph_violated: bool | None = None

    def coefficient(self, name: str) -> CoefficientRow:
        for row in self.coefficients:
            if row.name == name:
                return row
        raise KeyError(f"no coefficient named {name!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "coefficients": [c.to_dict() for c in self.coefficients],
            "hazard_ratios": {
                k: {"hr": v[0], "ci_low": v[1], "ci_high": v[2]}
                for k, v in self.hazard_ratios.items()
            },
            "concordance": self.concordance,
            "log_likelihood": self.log_likelihood,
            "n_traces": self.n_traces,
            "n_events": self.n_events,
            "confidence_level": self.confidence_level.level,
            "formula": self.formula,
            "ph_test": dict(self.ph_test) if self.ph_test is not None else None,
            "ph_violated": self.ph_violated,
        }


def cox_first_error(
    df: pd.DataFrame,
    *,
    covariates: list[str] | None = None,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    test_proportional_hazards: bool = False,
    ph_violation_alpha: float = 0.05,
) -> CoxResult:
    """Fit a Cox proportional-hazards model for first-error timing.

    One row per trace is constructed: ``duration`` = step index of
    the first observed error (or the last observed step for
    error-free traces), ``event`` = 1 iff an error was observed.
    Covariates are taken from the *first* step of each trace, which
    is the right convention for static, trace-level attributes like
    ``model`` and ``task_id``. For step-varying attributes like
    ``complexity`` we use the mean (or modal category after one-hot
    expansion) — this is a deliberate simplification; time-varying
    Cox models are out of scope for Phase 1.

    Args:
        df: Tidy step-level frame as produced by ``traces_to_frame``.
            Must contain ``trace_id``, ``step_index``, ``is_error``
            plus every column named in ``covariates``.
        covariates: Column names to include as fixed effects. Each
            must exist in ``df``. Categorical columns (dtype object
            or category) are one-hot encoded with a reference level.
            Numeric columns enter as-is. Defaults to ``["model"]``
            when the column exists and has ≥2 distinct values,
            otherwise an empty list (falls back to an intercept-only
            fit, which lifelines rejects, so at least one usable
            covariate is required).
        confidence_level: Confidence level for coefficient CIs.

    Returns:
        :class:`CoxResult` with per-covariate rows and hazard ratios.

    Raises:
        RuntimeError: lifelines not installed.
        ValueError: no usable covariates after filtering, or the
            event count is zero (cannot fit a Cox model with no
            observed events).
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "lifelines is required for Cox regression; install via the "
            "package's main dependency group"
        ) from exc

    if df.empty:
        raise ValueError("cannot fit Cox model on empty frame")

    for required in ("trace_id", "step_index", "is_error"):
        if required not in df.columns:
            raise ValueError(f"df missing required column {required!r}")

    if covariates is None:
        covariates = []
        if "model" in df.columns and df["model"].nunique() >= 2:
            covariates.append("model")
    if not covariates:
        raise ValueError(
            "cox_first_error requires at least one covariate with "
            ">=2 distinct values; none supplied or inferable"
        )
    for col in covariates:
        if col not in df.columns:
            raise ValueError(f"covariate {col!r} not in df")

    # Build one row per trace: duration + event + covariates taken
    # from the first step.
    rows: list[dict] = []
    for trace_id, group in df.sort_values("step_index").groupby("trace_id"):
        errs = group[group["is_error"]]
        if len(errs):
            duration = int(errs["step_index"].iloc[0])
            event = 1
        else:
            duration = int(group["step_index"].max())
            event = 0
        row: dict = {
            "trace_id": trace_id,
            "duration": max(duration, 0) + 1,  # lifelines requires >0
            "event": event,
        }
        first = group.iloc[0]
        for col in covariates:
            row[col] = first[col]
        rows.append(row)

    trace_df = pd.DataFrame(rows)
    if int(trace_df["event"].sum()) == 0:
        raise ValueError(
            "cox_first_error requires at least one observed event; "
            "all traces are right-censored"
        )

    # One-hot encode categorical covariates. Numeric columns pass
    # through. We drop the first dummy to avoid collinearity.
    cat_cols = [
        c for c in covariates
        if trace_df[c].dtype == object or str(trace_df[c].dtype) == "category"
    ]
    num_cols = [c for c in covariates if c not in cat_cols]

    design = trace_df[["duration", "event"]].copy()
    term_names: list[str] = []
    for c in num_cols:
        if trace_df[c].nunique() < 2:
            continue
        design[c] = trace_df[c].astype(float)
        term_names.append(c)
    for c in cat_cols:
        if trace_df[c].nunique() < 2:
            continue
        dummies = pd.get_dummies(
            trace_df[c], prefix=c, drop_first=True, dtype=float
        )
        for col in dummies.columns:
            design[col] = dummies[col].values
            term_names.append(col)

    if not term_names:
        raise ValueError(
            "cox_first_error: no covariates survived filtering "
            "(all constant within the corpus)"
        )

    cph = CoxPHFitter(alpha=confidence_level.alpha)
    cph.fit(design, duration_col="duration", event_col="event")

    # Coefficient table.
    summary = cph.summary
    coef_col = "coef"
    se_col = "se(coef)"
    p_col = "p"
    lower_col = next(c for c in summary.columns if c.startswith("coef lower"))
    upper_col = next(c for c in summary.columns if c.startswith("coef upper"))

    coefficients: list[CoefficientRow] = []
    hazard_ratios: dict[str, tuple[float, float, float]] = {}
    for name in term_names:
        if name not in summary.index:
            continue
        est = float(summary.loc[name, coef_col])
        se = float(summary.loc[name, se_col])
        z = est / se if se > 0 else float("nan")
        coefficients.append(
            CoefficientRow(
                name=name,
                estimate=est,
                std_error=se,
                z_statistic=z,
                p_value=float(summary.loc[name, p_col]),
                ci_low=float(summary.loc[name, lower_col]),
                ci_high=float(summary.loc[name, upper_col]),
            )
        )
        hazard_ratios[name] = (
            math.exp(est),
            math.exp(float(summary.loc[name, lower_col])),
            math.exp(float(summary.loc[name, upper_col])),
        )

    formula = "Surv(duration, event) ~ " + " + ".join(term_names)

    # Optional Schoenfeld-residuals proportional-hazards test. The test
    # asks: do the residuals correlate with time? If yes (small p), the
    # log-hazard ratio is *not* constant over time and the reported HR
    # is a weighted average rather than the constant the model claims.
    # We surface per-term p-values plus a corpus-wide "violated" flag
    # so reports can either drop to a stratified Cox or document the
    # caveat.
    ph_test_summary: dict[str, float] | None = None
    ph_violated: bool | None = None
    if test_proportional_hazards:
        try:
            from lifelines.statistics import proportional_hazard_test

            ph_result = proportional_hazard_test(cph, design, time_transform="rank")
            ph_test_summary = {
                str(name): float(ph_result.p_value[i])
                for i, name in enumerate(ph_result.summary.index)
            }
            ph_violated = any(p < ph_violation_alpha for p in ph_test_summary.values())
        except Exception as exc:  # pragma: no cover — diagnostic, never fatal
            ph_test_summary = {"_error": float("nan")}
            ph_violated = None
            log.warning("proportional_hazard_test failed: %s", exc)

    return CoxResult(
        coefficients=coefficients,
        hazard_ratios=hazard_ratios,
        concordance=float(cph.concordance_index_),
        log_likelihood=float(cph.log_likelihood_),
        n_traces=len(trace_df),
        n_events=int(trace_df["event"].sum()),
        confidence_level=confidence_level,
        formula=formula,
        ph_test=ph_test_summary,
        ph_violated=ph_violated,
    )


__all__ = ["CoxResult", "KMCurve", "KMResult", "cox_first_error", "first_error_km"]
