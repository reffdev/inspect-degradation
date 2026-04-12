"""Per-step rate computations with trace-level uncertainty.

This module provides the analysis-layer interface for validity rates
(``error_rate``, ``neutral_rate``, ``productive_rate``) and for the
``is_looping`` rate. Every function returns an :class:`Estimate` — no
bare floats escape to callers. The existing integration-layer scalar
wrappers in :mod:`inspect_degradation.integration.metrics` delegate
here and unwrap to ``.value`` where the Inspect AI metric API demands
a scalar.

## Pooled vs trace-level rates — why the distinction matters

A "rate" on trace data can mean two different things:

* **Pooled rate.** Total matching steps divided by total labeled
  steps, across the whole corpus. This is what an Inspect eval
  normally reports: one big bucket. It answers "across all the steps
  in the corpus, what fraction were errors?"

* **Trace-level mean rate.** Compute the rate per trace, then take
  the unweighted mean across traces. This answers "in a typical
  trace, what fraction of steps are errors?" — a subtly different
  question, and the one Phase 3's mixed-effects analysis cares about.

The two agree on balanced corpora but diverge when trace lengths
differ. Phase 3's Nebius corpus is length-heterogeneous (easy tasks
are short, hard tasks are long), so the difference is not academic.

## Why trace-level bootstrap for both

The uncertainty on either rate has the same problem: the sampling
unit is **the trace**, not the step. Two steps in the same trace
come from the same agent at the same moment and are correlated in
ways the Bernoulli model does not capture. Observation-level Wilson
intervals on pooled rates would understate the CI width.

Our approach:

* For **pooled rates**, we compute the point value as
  ``matches / total`` over all steps, and derive the CI from a
  **trace-level** bootstrap that recomputes the pooled rate on
  resampled traces. This gives a point estimate that matches what
  anyone would write by hand and a CI that honors the correlation
  structure.

* For **trace-level mean rates**, the statistic is already "mean of
  per-trace rates" and bootstraps over traces naturally. Traces with
  zero labeled steps are dropped from the denominator before the mean
  — including them as 0/0 would silently bias the mean downward.

Wilson intervals are exposed separately (see
:func:`wilson_pooled_rate`) for users who genuinely want the IID
interval — e.g., for comparing a single trace's rate against a known
baseline — but pooled rates at corpus level should default to the
trace-bootstrap path.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TypeAlias

from numpy.random import Generator

from inspect_degradation.analysis.statistics import (
    NINETY_FIVE,
    ConfidenceLevel,
    Estimate,
    bootstrap_estimate,
    wilson_proportion_interval,
)
from inspect_degradation.schema import GradedStep, GradedTrace, Validity


#: A predicate that classifies a single :class:`GradedStep` as a match
#: (``True``) or non-match (``False``) for the rate being computed.
#: ``None`` means "this step is not labeled for this dimension" and
#: excludes the step from both numerator and denominator — used for
#: partial references like TRAIL that don't label ``is_looping``.
StepPredicate: TypeAlias = Callable[[GradedStep], bool | None]


# ---------------------------------------------------------------------------
# Core predicates
# ---------------------------------------------------------------------------


def _is_error(step: GradedStep) -> bool:
    return step.validity == Validity.fail


def _is_neutral(step: GradedStep) -> bool:
    return step.validity == Validity.neutral


def _is_productive(step: GradedStep) -> bool:
    return step.validity == Validity.pass_


def _is_looping(step: GradedStep) -> bool | None:
    # None preserves the "unlabeled" semantics; the aggregator
    # excludes None from both numerator and denominator.
    return step.is_looping


# ---------------------------------------------------------------------------
# Per-trace raw counts
# ---------------------------------------------------------------------------


def _trace_counts(trace: GradedTrace, predicate: StepPredicate) -> tuple[int, int]:
    """Return ``(matches, labeled)`` for one trace under ``predicate``.

    Steps where the predicate returns ``None`` are excluded from both
    numerator and denominator.
    """
    matches = 0
    labeled = 0
    for step in trace.steps:
        result = predicate(step)
        if result is None:
            continue
        labeled += 1
        if result:
            matches += 1
    return matches, labeled


# ---------------------------------------------------------------------------
# Pooled rate with trace-level bootstrap CI
# ---------------------------------------------------------------------------


def pooled_rate(
    traces: Iterable[GradedTrace],
    predicate: StepPredicate,
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
) -> Estimate:
    """Pooled-over-steps rate with a trace-level bootstrap CI.

    The point estimate is ``sum(matches) / sum(labeled)`` across all
    traces. The CI is computed by resampling **whole traces** with
    replacement and recomputing the pooled rate on each resample —
    this is the right sampling unit because steps within a trace are
    correlated.

    Args:
        traces: Iterable of :class:`GradedTrace`.
        predicate: Per-step match predicate.
        confidence_level: Two-sided confidence level.
        n_resamples: Number of bootstrap resamples.
        rng: Optional seeded generator for reproducibility.

    Returns:
        An :class:`Estimate` whose ``n`` field is the number of traces
        (the resampling unit), not the number of steps. Traces with
        zero labeled steps still count in ``n`` — they carry zero
        weight in the rate but are a legitimate part of the resampling
        pool.
    """
    trace_list = list(traces)
    if not trace_list:
        return Estimate.empty(confidence_level=confidence_level)

    counts: list[tuple[int, int]] = [
        _trace_counts(trace, predicate) for trace in trace_list
    ]

    def pool(unit_counts: Sequence[tuple[int, int]]) -> float:
        matches = sum(c[0] for c in unit_counts)
        labeled = sum(c[1] for c in unit_counts)
        if labeled == 0:
            return float("nan")
        return matches / labeled

    return bootstrap_estimate(
        counts,
        pool,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Trace-mean rate
# ---------------------------------------------------------------------------


def trace_mean_rate(
    traces: Iterable[GradedTrace],
    predicate: StepPredicate,
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
) -> Estimate:
    """Mean of per-trace rates, with a trace-level bootstrap CI.

    Per-trace rate is ``matches / labeled`` within that trace; traces
    with no labeled steps are dropped before the mean, not included
    as ``0/0``. The statistic is well-defined because the resampling
    unit is explicitly the trace, so dropped traces are a function of
    the data, not of the resample index.

    Args:
        traces: Iterable of :class:`GradedTrace`.
        predicate: Per-step match predicate.
        confidence_level: Two-sided confidence level.
        n_resamples: Number of bootstrap resamples.
        rng: Optional seeded generator for reproducibility.

    Returns:
        An :class:`Estimate` whose ``n`` is the number of traces in
        the input (including dropped-due-to-no-labels traces, for
        resampling fidelity). Point value is the mean of labeled
        traces only.
    """
    trace_list = list(traces)
    if not trace_list:
        return Estimate.empty(confidence_level=confidence_level)

    counts: list[tuple[int, int]] = [
        _trace_counts(trace, predicate) for trace in trace_list
    ]

    def mean_of_rates(unit_counts: Sequence[tuple[int, int]]) -> float:
        per_trace: list[float] = []
        for matches, labeled in unit_counts:
            if labeled == 0:
                continue
            per_trace.append(matches / labeled)
        if not per_trace:
            return float("nan")
        return sum(per_trace) / len(per_trace)

    return bootstrap_estimate(
        counts,
        mean_of_rates,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Wilson interval on pooled data
# ---------------------------------------------------------------------------


def wilson_pooled_rate(
    traces: Iterable[GradedTrace],
    predicate: StepPredicate,
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
) -> Estimate:
    """Pooled-over-steps rate with a **Wilson** (IID-assumption) CI.

    Use when you have good reason to believe the steps are independent
    — e.g., you're reporting a single-trace rate, or your unit of
    analysis really is the step and you've accounted for within-trace
    correlation some other way. For corpus-level error rates on
    realistic traces, prefer :func:`pooled_rate` instead.

    The ``n`` on the returned estimate is the number of **labeled
    steps**, not traces — because Wilson is an IID bound and steps
    are its natural sampling unit.
    """
    trace_list = list(traces)
    matches = 0
    labeled = 0
    for trace in trace_list:
        m, lbl = _trace_counts(trace, predicate)
        matches += m
        labeled += lbl
    return wilson_proportion_interval(
        matches, labeled, confidence_level=confidence_level
    )


# ---------------------------------------------------------------------------
# Convenience functions for the canonical validity/loop predicates
# ---------------------------------------------------------------------------


def error_rate(
    traces: Iterable[GradedTrace],
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
) -> Estimate:
    """Pooled error rate across traces with trace-level bootstrap CI."""
    return pooled_rate(
        traces,
        _is_error,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        rng=rng,
    )


def neutral_rate(
    traces: Iterable[GradedTrace],
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
) -> Estimate:
    """Pooled neutral rate across traces with trace-level bootstrap CI."""
    return pooled_rate(
        traces,
        _is_neutral,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        rng=rng,
    )


def productive_rate(
    traces: Iterable[GradedTrace],
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
) -> Estimate:
    """Pooled productive rate across traces with trace-level bootstrap CI."""
    return pooled_rate(
        traces,
        _is_productive,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        rng=rng,
    )


def loop_rate(
    traces: Iterable[GradedTrace],
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
) -> Estimate:
    """Pooled loop rate across traces with trace-level bootstrap CI.

    Steps whose ``is_looping`` field is ``None`` (partial references)
    are excluded from both numerator and denominator of every trace's
    rate — the right behavior for TRAIL-style reference data that
    doesn't carry loop labels.
    """
    return pooled_rate(
        traces,
        _is_looping,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        rng=rng,
    )


__all__ = [
    "StepPredicate",
    "error_rate",
    "loop_rate",
    "neutral_rate",
    "pooled_rate",
    "productive_rate",
    "trace_mean_rate",
    "wilson_pooled_rate",
]
