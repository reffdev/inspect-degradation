"""Inspect AI metric aggregators over graded traces.

These read the serialized :class:`GradedTrace` that
:func:`inspect_degradation.integration.scorer.degradation_scorer` stashes
in ``Score.metadata`` and aggregate it across samples in an eval run.

Each metric here is a thin scalar wrapper -- the actual statistical
work lives in :mod:`inspect_degradation.analysis`. The Inspect AI
``@metric`` decorator expects scalar return values, which is why this
module exposes ``float``-returning callables rather than the
:class:`~inspect_degradation.analysis.statistics.Estimate` objects
the analysis layer produces. Callers doing their own reporting outside
Inspect should prefer the analysis-layer APIs directly so they keep
the confidence intervals the scalar API throws away.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.scorer import Score, metric  # type: ignore

from inspect_degradation.analysis import rates as _rates
from inspect_degradation.analysis import slopes as _slopes
from inspect_degradation.analysis.cascade_chains import (
    cascade_chain_length_mean_estimate,
    mean_failing_run_length_estimate,
)
from inspect_degradation.analysis.loops import raw_loop_rate
from inspect_degradation.integration.scorer import GRADED_TRACE_METADATA_KEY
from inspect_degradation.schema import GradedTrace, Validity


def _graded_traces_from_scores(scores: list[Any]) -> list[GradedTrace]:
    """Reconstitute :class:`GradedTrace` objects from Inspect scores.

    Accepts both ``list[SampleScore]`` (new API) and ``list[Score]``
    (deprecated API / tests). Scores without our metadata key are
    skipped -- this lets the metric coexist with other scorers in the
    same eval without exploding.
    """
    out: list[GradedTrace] = []
    for item in scores:
        # SampleScore wraps the Score in .score; plain Score has .metadata directly.
        score = getattr(item, "score", item)
        meta = getattr(score, "metadata", None) or {}
        payload = meta.get(GRADED_TRACE_METADATA_KEY)
        if payload is None:
            continue
        out.append(GradedTrace.model_validate(payload))
    return out


# ---------------------------------------------------------------------------
# Rate scalars
# ---------------------------------------------------------------------------


@metric
def error_rate():
    """Pooled fraction of steps marked ``fail``.

    Thin scalar wrapper around :func:`inspect_degradation.analysis.rates.error_rate`.
    Returns ``nan`` for empty input.
    """
    def metric(scores: list[Score]) -> float:
        traces = _graded_traces_from_scores(scores)
        return _rates.error_rate(traces).value
    return metric


@metric
def neutral_rate():
    """Pooled fraction of steps marked ``neutral``."""
    def metric(scores: list[Score]) -> float:
        traces = _graded_traces_from_scores(scores)
        return _rates.neutral_rate(traces).value
    return metric


@metric
def productive_rate():
    """Pooled fraction of steps marked ``pass``."""
    def metric(scores: list[Score]) -> float:
        traces = _graded_traces_from_scores(scores)
        return _rates.productive_rate(traces).value
    return metric


@metric
def loop_rate():
    """Pooled fraction of steps flagged ``is_looping=True``.

    Steps whose ``is_looping`` field is ``None`` (partial references)
    are excluded from both numerator and denominator. Returns ``nan``
    when no labeled data is present.
    """
    def metric(scores: list[Score]) -> float:
        traces = _graded_traces_from_scores(scores)
        value = raw_loop_rate(traces)
        return float("nan") if value is None else value
    return metric


# ---------------------------------------------------------------------------
# Slope scalars
# ---------------------------------------------------------------------------


@metric
def error_rate_slope():
    """Per-trace-mean slope of P(error) regressed on step index.

    Positive slope => error rate concentrates later in the trace.
    Together with :func:`neutral_rate_slope` and :func:`loop_rate_slope`
    this decomposes within-run degradation into three failure-mode
    signatures: more mistakes late, more flailing late, more looping
    late.
    """
    def metric(scores: list[Score]) -> float:
        traces = _graded_traces_from_scores(scores)
        return _slopes.error_rate_slope(traces).value
    return metric


@metric
def neutral_rate_slope():
    """Per-trace-mean slope of P(neutral) regressed on step index."""
    def metric(scores: list[Score]) -> float:
        traces = _graded_traces_from_scores(scores)
        return _slopes.neutral_rate_slope(traces).value
    return metric


@metric
def loop_rate_slope():
    """Per-trace-mean slope of P(is_looping) regressed on step index."""
    def metric(scores: list[Score]) -> float:
        traces = _graded_traces_from_scores(scores)
        return _slopes.loop_rate_slope(traces).value
    return metric


# ---------------------------------------------------------------------------
# First-error and cascade summaries
# ---------------------------------------------------------------------------


@metric
def first_error_step_median():
    """Median step index of the first error across traces.

    Uses the raw empirical median of first-error step indices across
    traces that errored at all.
    """
    def metric(scores: list[Score]) -> float:
        import numpy as np

        traces = _graded_traces_from_scores(scores)
        firsts: list[int] = []
        for trace in traces:
            for step in trace.steps:
                if step.validity == Validity.fail:
                    firsts.append(step.step_index)
                    break
        if not firsts:
            return float("nan")
        return float(np.median(firsts))
    return metric


@metric
def cascade_chain_length_mean():
    """Mean length of dependent-error chains across traces."""
    def metric(scores: list[Score]) -> float:
        traces = _graded_traces_from_scores(scores)
        return cascade_chain_length_mean_estimate(traces).value
    return metric


@metric
def mean_failure_run_length():
    """Mean length of contiguous failing-step runs across traces.

    Right-censors traces that end inside a failing run (an
    unrecovered failure should not shorten the average).
    """
    def metric(scores: list[Score]) -> float:
        traces = _graded_traces_from_scores(scores)
        return mean_failing_run_length_estimate(traces).value
    return metric


@metric
def loop_chain_length_mean():
    """Mean length of contiguous loop runs across traces."""
    def metric(scores: list[Score]) -> float:
        from inspect_degradation.analysis.loops import loop_chain_length_mean_estimate

        traces = _graded_traces_from_scores(scores)
        return loop_chain_length_mean_estimate(traces).value
    return metric


__all__ = [
    "cascade_chain_length_mean",
    "error_rate",
    "error_rate_slope",
    "first_error_step_median",
    "loop_chain_length_mean",
    "loop_rate",
    "loop_rate_slope",
    "mean_failure_run_length",
    "neutral_rate",
    "neutral_rate_slope",
    "productive_rate",
]
