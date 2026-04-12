"""Cascade-chain analysis: chains of dependent errors plus failing-run metrics.

Two primitives:

* :func:`cascade_chain_lengths` — raw list of contiguous dependent-
  error chain lengths. Pure combinatorial function; no uncertainty.
  Kept because downstream plotting and percentile work wants the raw
  distribution.

* :func:`mean_steps_to_non_failure` — mean length of contiguous
  failing-step runs, derived purely from ``validity`` transitions.
  Right-censors traces that end inside a failing run (an unrecovered
  failure should not shorten the average).

And the uncertainty-bearing variants:

* :func:`cascade_chain_length_mean_estimate` — bootstrap CI on the
  mean chain length, resampling **whole traces** so correlated
  within-trace chains are handled correctly.

* :func:`mean_failing_run_length_estimate` — same idea for the
  non-failure metric.

Both bootstrap variants keep the scalar-returning functions as inner
statistics, so the uncertainty and the point-value pipelines are
guaranteed to agree on the underlying computation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from numpy.random import Generator

from inspect_degradation.analysis.statistics import (
    NINETY_FIVE,
    ConfidenceLevel,
    Estimate,
    bootstrap_estimate,
)
from inspect_degradation.schema import Dependency, GradedTrace, Validity


# ---------------------------------------------------------------------------
# Chain-length list (no CI, pure combinatorial)
# ---------------------------------------------------------------------------


def cascade_chain_lengths(
    traces: Iterable[GradedTrace],
    *,
    allow_partial_dependency: bool = False,
) -> list[int]:
    """Return the length of every dependent-error chain across traces.

    A chain starts at an independent failing step (or at any failing
    step if the current chain-length counter is zero — this handles
    traces whose first failure has no declared dependency) and
    continues through subsequent *dependent* failing steps. Any
    non-failing step (or a new independent failure) closes the open
    chain.

    Args:
        traces: Iterable of graded traces.
        allow_partial_dependency: If False (default), the analysis
            raises :class:`ValueError` on the first failing step
            with no dependency label. If True, missing dependency
            labels are treated as :attr:`Dependency.independent` —
            the conservative choice (it produces *shorter* chains
            than treating them as dependent and so under-estimates
            cascade severity rather than over-estimating it). The
            relaxed mode is intended for corpora where the grader
            occasionally returns a fallback step (e.g., the parse-
            failure neutral fallback in
            :class:`~inspect_degradation.grader.llm.LLMGrader`),
            which would otherwise wedge the entire analysis on a
            single bad response.

    Raises:
        ValueError: a failing step is missing a dependency label
            and ``allow_partial_dependency`` is False.
    """
    lengths: list[int] = []
    for trace in traces:
        current = 0
        for step in trace.steps:
            if step.validity == Validity.fail:
                dep = step.dependency
                if dep is None:
                    if not allow_partial_dependency:
                        raise ValueError(
                            f"trace {trace.trace_id!r} step {step.step_index} has "
                            f"no dependency label; cascade analysis requires fully "
                            f"graded traces (use the LLM grader, not a partial human "
                            f"reference) — pass allow_partial_dependency=True to "
                            f"treat missing labels as independent (conservative)"
                        )
                    dep = Dependency.independent
                if dep == Dependency.independent or current == 0:
                    if current:
                        lengths.append(current)
                    current = 1
                else:
                    current += 1
            else:
                if current:
                    lengths.append(current)
                    current = 0
        if current:
            lengths.append(current)
    return lengths


# ---------------------------------------------------------------------------
# Mean failing-run length (no CI, backwards-compatible scalar API)
# ---------------------------------------------------------------------------


def mean_steps_to_non_failure(traces: Iterable[GradedTrace]) -> float | None:
    """Mean length of contiguous failing-step runs; ``None`` if none.

    Derived purely from ``validity`` transitions. Right-censors traces
    that end inside a failing run — an unrecovered failure should
    not artificially shorten the average. See the
    :mod:`inspect_degradation.analysis.loops` module for a symmetric
    treatment of loop runs (which are *not* right-censored, because
    a trace ending inside a loop is legitimately part of the loop
    distribution).
    """
    spans: list[int] = []
    for trace in traces:
        run_start: int | None = None
        for step in trace.steps:
            if step.validity == Validity.fail:
                if run_start is None:
                    run_start = step.step_index
            else:
                if run_start is not None:
                    spans.append(step.step_index - run_start)
                    run_start = None
        # Intentionally do NOT append an open run at trace end:
        # right-censored by design.
    if not spans:
        return None
    return sum(spans) / len(spans)


# ---------------------------------------------------------------------------
# Uncertainty-bearing variants
# ---------------------------------------------------------------------------


def _mean_chain_length_over_traces(
    units: Sequence[GradedTrace],
    *,
    allow_partial_dependency: bool = False,
) -> float:
    """Bootstrap-safe inner statistic for chain-length mean.

    Takes a sequence of :class:`GradedTrace` (not a pre-computed
    chain list), so the bootstrap correctly reflects the chain-from-
    trace combinatorics under resampling.
    """
    lengths = cascade_chain_lengths(
        units, allow_partial_dependency=allow_partial_dependency
    )
    if not lengths:
        return float("nan")
    return sum(lengths) / len(lengths)


def cascade_chain_length_mean_estimate(
    traces: Iterable[GradedTrace],
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
    allow_partial_dependency: bool = False,
) -> Estimate:
    """Mean cascade-chain length across traces, with trace-level bootstrap CI.

    Chain-length distributions are heavy-tailed — one trace stuck in
    a 15-step dependent-error chain can dominate the mean, and
    reporting the mean without a CI hides exactly that concern. The
    bootstrap resamples whole traces, so the CI correctly reflects
    the variance caused by which traces happen to be in the sample.

    See :func:`cascade_chain_lengths` for the
    ``allow_partial_dependency`` semantics.
    """
    trace_list = list(traces)

    def _stat(units: Sequence[GradedTrace]) -> float:
        return _mean_chain_length_over_traces(
            units, allow_partial_dependency=allow_partial_dependency
        )

    return bootstrap_estimate(
        trace_list,
        _stat,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        rng=rng,
    )


def _mean_failing_run_over_traces(units: Sequence[GradedTrace]) -> float:
    """Bootstrap-safe inner statistic for failing-run mean."""
    value = mean_steps_to_non_failure(units)
    return float("nan") if value is None else value


def mean_failing_run_length_estimate(
    traces: Iterable[GradedTrace],
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
) -> Estimate:
    """Mean contiguous failing-run length, with trace-level bootstrap CI."""
    trace_list = list(traces)
    return bootstrap_estimate(
        trace_list,
        _mean_failing_run_over_traces,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        rng=rng,
    )


__all__ = [
    "cascade_chain_length_mean_estimate",
    "cascade_chain_lengths",
    "mean_failing_run_length_estimate",
    "mean_steps_to_non_failure",
]
