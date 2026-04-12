"""Degradation slopes with trace-aware uncertainty.

This module computes how per-step rates change with position in a
trace — the "is the agent getting worse as the trace goes on"
quantity that the whole project exists to measure.

## The architectural choice: per-trace slopes vs. pooled OLS

There are two common ways to compute a degradation slope from a
corpus of graded traces, and they answer different questions:

**Pooled OLS.** Concatenate all ``(step_index, is_error)`` pairs
across all traces into one flat dataset and fit a single linear
regression. This is the quick-and-dirty version and what the earlier
``integration/metrics.py`` did. It has two problems:

1. *Longer traces dominate the fit.* A 30-step trace contributes 30
   points; a 5-step trace contributes 5. If trace length is
   correlated with the error pattern (which it usually is — hard
   tasks produce long traces), the slope is confounded by selection.
2. *Within-trace correlation is ignored.* The standard OLS CI
   assumes independent observations. Steps within a trace are not
   independent; they come from one agent at one moment. The nominal
   CI understates uncertainty.

**Per-trace slopes, then meta-analysis.** For each trace
independently, fit a single-feature OLS slope of
``is_error ~ step_index``. Drop traces where the slope is
ill-defined (fewer than 3 steps, zero variance in y). Report the
mean of the surviving per-trace slopes, with a CI from
**trace-level bootstrap** over the slope set. This is the right
sampling unit: the trace is the replicate, the slope is the
summary statistic, and the CI comes from the empirical distribution
of slopes across the corpus.

This module exposes both — :func:`per_trace_mean_slope` is the
default and correct one for Phase 3, :func:`pooled_slope` is kept
for comparison experiments and for single-trace reports — and tags
the returned ``Estimate`` with which method was used.

## Degenerate traces

Any trace that cannot support a slope is dropped from the
per-trace-slope pool with a tag in the ``dropped_reason`` field of
the returned :class:`SlopeResult`. The caller sees how many traces
contributed and why the rest were excluded.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.random import Generator

from inspect_degradation.analysis.statistics import (
    NINETY_FIVE,
    ConfidenceLevel,
    Estimate,
    bootstrap_estimate,
    ols_slope_with_interval,
)
from inspect_degradation.schema import GradedStep, GradedTrace, Validity


#: Classify a single step as match / non-match / not-labeled. ``None``
#: excludes the step from the regression (used for ``is_looping`` on
#: partial references). Identical in signature to
#: :data:`inspect_degradation.analysis.rates.StepPredicate` and shares
#: the same contract.
StepPredicate: TypeAlias = Callable[[GradedStep], bool | None]


def _is_error(step: GradedStep) -> bool:
    return step.validity == Validity.fail


def _is_neutral(step: GradedStep) -> bool:
    return step.validity == Validity.neutral


def _is_looping(step: GradedStep) -> bool | None:
    return step.is_looping


# ---------------------------------------------------------------------------
# Per-trace slope computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PerTraceSlope:
    """One trace's slope result, plus metadata for auditing drops."""

    trace_id: str
    slope: float  # nan when the trace had to be dropped
    n_steps: int
    dropped_reason: str | None  # None when slope is valid


def _per_trace_slope(
    trace: GradedTrace, predicate: StepPredicate
) -> _PerTraceSlope:
    """Compute a single trace's OLS slope of predicate vs. step index.

    Drop reasons:

    * ``too_short``: fewer than 3 labeled steps.
    * ``no_x_variance``: all labeled steps have the same step index
      (pathological, but possible if a source loader does something
      strange).
    * ``constant_y``: every labeled step has the same predicate value;
      the slope is mathematically zero but the usual closed-form
      handles it fine, so we return 0.0 rather than dropping.
    """
    xs: list[int] = []
    ys: list[int] = []
    for step in trace.steps:
        result = predicate(step)
        if result is None:
            continue
        xs.append(step.step_index)
        ys.append(1 if result else 0)

    n_labeled = len(xs)
    if n_labeled < 3:
        return _PerTraceSlope(
            trace_id=trace.trace_id,
            slope=float("nan"),
            n_steps=n_labeled,
            dropped_reason="too_short",
        )
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    if np.isclose(x_arr.std(), 0.0):
        return _PerTraceSlope(
            trace_id=trace.trace_id,
            slope=float("nan"),
            n_steps=n_labeled,
            dropped_reason="no_x_variance",
        )
    x_centered = x_arr - x_arr.mean()
    y_centered = y_arr - y_arr.mean()
    ssx = float(np.dot(x_centered, x_centered))
    slope = float(np.dot(x_centered, y_centered) / ssx)
    return _PerTraceSlope(
        trace_id=trace.trace_id,
        slope=slope,
        n_steps=n_labeled,
        dropped_reason=None,
    )


# ---------------------------------------------------------------------------
# SlopeResult — the public return type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlopeResult:
    """A degradation-slope estimate with per-trace provenance.

    Attributes:
        estimate: The :class:`Estimate` carrying point value, CI,
            method tag, sample size, etc.
        n_traces_total: How many traces were in the input.
        n_traces_used: How many contributed a finite slope to the
            meta-analysis (i.e., were not dropped).
        drop_reasons: Counter-like dict mapping drop-reason strings
            (``"too_short"``, ``"no_x_variance"``, ...) to how many
            traces were dropped for that reason. Empty when nothing
            was dropped. Exposed so reports can flag "we dropped
            40% of traces as too_short" — an important caveat that
            pooled slopes completely hide.
    """

    estimate: Estimate
    n_traces_total: int
    n_traces_used: int
    drop_reasons: dict[str, int]

    @property
    def value(self) -> float:
        return self.estimate.value

    def to_dict(self) -> dict[str, object]:
        return {
            "estimate": self.estimate.to_dict(),
            "n_traces_total": self.n_traces_total,
            "n_traces_used": self.n_traces_used,
            "drop_reasons": dict(self.drop_reasons),
        }


# ---------------------------------------------------------------------------
# Per-trace mean slope
# ---------------------------------------------------------------------------


def per_trace_mean_slope(
    traces: Iterable[GradedTrace],
    predicate: StepPredicate,
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
) -> SlopeResult:
    """Mean of per-trace slopes with a trace-level bootstrap CI.

    Default choice for degradation slope measurement at corpus scale.
    Each trace contributes exactly one slope to the meta-analysis
    (regardless of trace length), degenerate traces are dropped with
    an auditable reason, and the CI is bootstrapped over the
    resulting per-trace slope sequence — so longer traces don't
    dominate the fit and within-trace correlation is handled by
    construction.

    Args:
        traces: Iterable of :class:`GradedTrace`.
        predicate: Per-step match predicate.
        confidence_level: Two-sided confidence level.
        n_resamples: Number of bootstrap resamples.
        rng: Optional seeded generator for reproducibility.

    Returns:
        A :class:`SlopeResult` with the bootstrap :class:`Estimate`,
        counts of contributing and dropped traces, and the drop
        reasons.
    """
    trace_list = list(traces)
    n_total = len(trace_list)

    per_trace = [_per_trace_slope(t, predicate) for t in trace_list]
    drop_reasons: dict[str, int] = {}
    valid_slopes: list[float] = []
    for pt in per_trace:
        if pt.dropped_reason is not None:
            drop_reasons[pt.dropped_reason] = drop_reasons.get(pt.dropped_reason, 0) + 1
            continue
        valid_slopes.append(pt.slope)
    n_used = len(valid_slopes)

    if n_used == 0:
        return SlopeResult(
            estimate=Estimate.empty(confidence_level=confidence_level),
            n_traces_total=n_total,
            n_traces_used=0,
            drop_reasons=drop_reasons,
        )
    if n_used == 1:
        # Point value is well-defined (it is the one surviving slope)
        # but CI is not computable from a single unit. Report the
        # point with nan interval rather than erasing the value.
        return SlopeResult(
            estimate=Estimate(
                value=float(valid_slopes[0]),
                ci_low=float("nan"),
                ci_high=float("nan"),
                n=1,
                method="insufficient_data",
                confidence_level=confidence_level,
            ),
            n_traces_total=n_total,
            n_traces_used=1,
            drop_reasons=drop_reasons,
        )

    def mean_of_slopes(units: Sequence[float]) -> float:
        arr = np.asarray(units, dtype=float)
        if arr.size == 0:
            return float("nan")
        return float(arr.mean())

    estimate = bootstrap_estimate(
        valid_slopes,
        mean_of_slopes,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        rng=rng,
    )
    return SlopeResult(
        estimate=estimate,
        n_traces_total=n_total,
        n_traces_used=n_used,
        drop_reasons=drop_reasons,
    )


# ---------------------------------------------------------------------------
# Pooled slope (legacy, with analytic CI)
# ---------------------------------------------------------------------------


def pooled_slope(
    traces: Iterable[GradedTrace],
    predicate: StepPredicate,
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
) -> SlopeResult:
    """OLS slope on pooled ``(step_index, predicate(step))`` pairs.

    This pools *all* labeled steps across all traces into one flat
    regression. The analytic CI assumes independent observations,
    which **is not true** for steps within a trace — the CI is
    conservative-when-the-data-is-iid and over-confident otherwise.
    Use :func:`per_trace_mean_slope` instead for the Phase 3 headline
    number; :func:`pooled_slope` is exposed for single-trace reports
    and for comparisons against the old pre-bootstrap behavior.
    """
    trace_list = list(traces)
    n_total = len(trace_list)

    xs: list[int] = []
    ys: list[int] = []
    for trace in trace_list:
        for step in trace.steps:
            result = predicate(step)
            if result is None:
                continue
            xs.append(step.step_index)
            ys.append(1 if result else 0)

    estimate = ols_slope_with_interval(xs, ys, confidence_level=confidence_level)
    # Pooled OLS treats each step as a unit, so n_traces_used is a bit
    # of a misnomer — we report the number of traces that contributed
    # at least one labeled step, which is the honest "how many groups
    # is this slope summarizing."
    n_used = sum(
        1
        for trace in trace_list
        if any(predicate(s) is not None for s in trace.steps)
    )
    return SlopeResult(
        estimate=estimate,
        n_traces_total=n_total,
        n_traces_used=n_used,
        drop_reasons={},
    )


# ---------------------------------------------------------------------------
# Convenience functions for canonical predicates
# ---------------------------------------------------------------------------


def error_rate_slope(
    traces: Iterable[GradedTrace],
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
) -> SlopeResult:
    """Per-trace-mean slope of P(error) regressed on step index."""
    return per_trace_mean_slope(
        traces,
        _is_error,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        rng=rng,
    )


def neutral_rate_slope(
    traces: Iterable[GradedTrace],
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
) -> SlopeResult:
    """Per-trace-mean slope of P(neutral) regressed on step index.

    Companion to :func:`error_rate_slope`. A trace where the agent
    isn't breaking but is spinning its wheels shows a flat error slope
    and a positive neutral slope — two different degradation patterns.
    """
    return per_trace_mean_slope(
        traces,
        _is_neutral,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        rng=rng,
    )


def loop_rate_slope(
    traces: Iterable[GradedTrace],
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
) -> SlopeResult:
    """Per-trace-mean slope of P(is_looping) regressed on step index.

    The third of the three degradation-shape metrics. Unlabeled steps
    (``is_looping is None``) are dropped from each trace's regression.
    Traces where loop labels are entirely absent become ``too_short``
    drops automatically.
    """
    return per_trace_mean_slope(
        traces,
        _is_looping,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        rng=rng,
    )


__all__ = [
    "SlopeResult",
    "StepPredicate",
    "error_rate_slope",
    "loop_rate_slope",
    "neutral_rate_slope",
    "per_trace_mean_slope",
    "pooled_slope",
]
