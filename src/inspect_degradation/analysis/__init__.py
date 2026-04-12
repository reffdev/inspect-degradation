"""Statistical analysis over graded traces.

Operates on :class:`inspect_degradation.schema.GradedTrace` objects
regardless of whether they came from the Inspect integration (live)
or the offline grader.

## Organization

* :mod:`.statistics` — foundation: the :class:`Estimate` type,
  Wilson proportion intervals, OLS slope + CI, bootstrap (BCa and
  percentile). Pure numerics, no project-type dependencies.
* :mod:`.rates` — per-step rates (error, neutral, productive,
  looping) with trace-level bootstrap CIs.
* :mod:`.slopes` — per-trace OLS degradation slopes aggregated with
  bootstrap CIs, plus a pooled-OLS variant for comparison.
* :mod:`.cascade_chains` — contiguous dependent-error chain lengths
  and the mean-failing-run metric, with bootstrap CIs.
* :mod:`.loops` — contiguous loop-run lengths and mean-loop-length,
  with bootstrap CIs.
* :mod:`.mixed_effects` — linear mixed-effects models for the
  central degradation-controlling-for-confounds analysis, in both
  step-level and per-trace-slope flavors.
* :mod:`.measurement_error` — confusion-matrix deconfounding and
  SIMEX correction for regression coefficients biased by grader
  measurement error.
* :mod:`.survival` — Kaplan-Meier first-error survival with
  confidence bands.
* :mod:`.frame` — flatten traces into a tidy pandas DataFrame for
  ad-hoc analysis.
* :mod:`.change_point` — placeholder; not yet production-ready.

Every statistical claim the analysis layer makes is returned as an
:class:`~inspect_degradation.analysis.statistics.Estimate` so
confidence intervals are a first-class contract, not an afterthought.
"""

from inspect_degradation.analysis.autocorrelation import (
    AutocorrelationResult,
    LjungBoxResult,
    ljung_box_per_trace,
    per_trace_acf,
)
from inspect_degradation.analysis.cascade_chains import (
    cascade_chain_length_mean_estimate,
    cascade_chain_lengths,
    mean_failing_run_length_estimate,
    mean_steps_to_non_failure,
)
from inspect_degradation.analysis.frame import traces_to_frame
from inspect_degradation.analysis.loops import (
    loop_chain_length_mean_estimate,
    loop_chain_lengths,
    raw_loop_rate,
)
from inspect_degradation.analysis.measurement_error import (
    ConfusionMatrix,
    SimexPoint,
    SimexResult,
    deconfound_proportion,
    simex_correct,
)
from inspect_degradation.analysis.change_point import (
    ChangePointResult,
    naive_change_point,
    pelt_change_points,
)
from inspect_degradation.analysis.mixed_effects import (
    CoefficientRow,
    MixedEffectsResult,
    RandomEffects,
    fit_crossed_effects_model,
    fit_mixed_effects,
    fit_step_level_model,
    fit_trace_level_slope_model,
)
from inspect_degradation.analysis.rates import (
    error_rate,
    loop_rate,
    neutral_rate,
    pooled_rate,
    productive_rate,
    trace_mean_rate,
    wilson_pooled_rate,
)
from inspect_degradation.analysis.slopes import (
    SlopeResult,
    error_rate_slope,
    loop_rate_slope,
    neutral_rate_slope,
    per_trace_mean_slope,
    pooled_slope,
)
from inspect_degradation.analysis.statistics import (
    NINETY,
    NINETY_FIVE,
    NINETY_NINE,
    ConfidenceLevel,
    Estimate,
    bootstrap_estimate,
    ols_slope_with_interval,
    wilson_proportion_interval,
)
from inspect_degradation.analysis.multiple_comparisons import (
    AdjustedCoefficient,
    MultipleComparisonResult,
    adjust_coefficients,
)
from inspect_degradation.analysis.power import (
    PowerResult,
    simulate_mixed_effects_power,
)
from inspect_degradation.analysis.survival import (
    CoxResult,
    cox_first_error,
    first_error_km,
)

__all__ = [
    "AdjustedCoefficient",
    "AutocorrelationResult",
    "ChangePointResult",
    "CoefficientRow",
    "LjungBoxResult",
    "fit_crossed_effects_model",
    "ljung_box_per_trace",
    "naive_change_point",
    "pelt_change_points",
    "per_trace_acf",
    "ConfidenceLevel",
    "ConfusionMatrix",
    "CoxResult",
    "MultipleComparisonResult",
    "PowerResult",
    "adjust_coefficients",
    "cox_first_error",
    "simulate_mixed_effects_power",
    "Estimate",
    "MixedEffectsResult",
    "NINETY",
    "NINETY_FIVE",
    "NINETY_NINE",
    "RandomEffects",
    "SimexPoint",
    "SimexResult",
    "SlopeResult",
    "bootstrap_estimate",
    "cascade_chain_length_mean_estimate",
    "cascade_chain_lengths",
    "deconfound_proportion",
    "error_rate",
    "error_rate_slope",
    "first_error_km",
    "fit_mixed_effects",
    "fit_step_level_model",
    "fit_trace_level_slope_model",
    "loop_chain_length_mean_estimate",
    "loop_chain_lengths",
    "loop_rate",
    "loop_rate_slope",
    "mean_failing_run_length_estimate",
    "mean_steps_to_non_failure",
    "neutral_rate",
    "neutral_rate_slope",
    "ols_slope_with_interval",
    "per_trace_mean_slope",
    "pooled_rate",
    "pooled_slope",
    "productive_rate",
    "raw_loop_rate",
    "simex_correct",
    "trace_mean_rate",
    "traces_to_frame",
    "wilson_pooled_rate",
    "wilson_proportion_interval",
]
