"""Loop analysis: contiguous runs of ``is_looping`` steps.

This is a distinct concern from :mod:`cascade_chains`, which handles
chains of dependent *errors*. Loops are a different failure mode —
the agent is functioning but trapped in a cycle — and keeping the
two analyses in separate modules makes the metric naming and the
downstream interpretation explicit.

## What this module exposes

* :func:`loop_chain_lengths` — raw list of contiguous loop-run
  lengths. Pure combinatorial function; trace-end runs are counted
  (not right-censored, because a trace that ends inside a loop is
  unambiguously part of the loop distribution).

* :func:`raw_loop_rate` — scalar pooled loop rate, returning a float
  or ``None``. Used for backwards-compatible scalar integrations
  (the Inspect AI metric decorator needs scalars). New analysis code
  should use :func:`inspect_degradation.analysis.rates.loop_rate`
  instead, which returns an :class:`Estimate` with a trace-level
  bootstrap CI.

* :func:`loop_chain_length_mean_estimate` — mean loop-run length
  with a trace-level bootstrap CI.

Steps where ``is_looping`` is ``None`` are treated as *not looping*
for the chain-length calculation (a missing label is not evidence
of looping), and excluded from both numerator and denominator of
:func:`raw_loop_rate` (a missing label is not evidence of
*not* looping, either).
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
from inspect_degradation.schema import GradedTrace


# ---------------------------------------------------------------------------
# Chain-length list
# ---------------------------------------------------------------------------


def loop_chain_lengths(traces: Iterable[GradedTrace]) -> list[int]:
    """Return the length of every contiguous ``is_looping=True`` run.

    A "loop chain" is a maximal run of consecutive steps all flagged
    as ``is_looping``. A single isolated loop step contributes a
    chain of length 1; 12 steps stuck in the same cycle contribute
    one chain of length 12.

    Steps with ``is_looping is None`` are treated as *not* looping —
    they terminate any open chain. This is the safe default for
    partial references: a missing label is not evidence of looping.

    Unlike failing-step runs (which are right-censored), a trace
    ending inside a loop *does* contribute its open run to the list.
    The reason is asymmetric: an unrecovered failure has meaning on
    its own (the task ended in a bad state); an unfinished loop is
    just "still in the loop at trace end" and should count as loop
    time in full.
    """
    lengths: list[int] = []
    for trace in traces:
        current = 0
        for step in trace.steps:
            if step.is_looping is True:
                current += 1
            else:
                if current:
                    lengths.append(current)
                    current = 0
        if current:
            lengths.append(current)
    return lengths


# ---------------------------------------------------------------------------
# Raw (scalar) loop rate
# ---------------------------------------------------------------------------


def raw_loop_rate(traces: Iterable[GradedTrace]) -> float | None:
    """Scalar pooled loop rate — fraction of labeled steps flagged looping.

    Returns ``None`` when no step in any trace carries a non-null
    ``is_looping`` label; this lets callers distinguish "no loops
    observed" (returns 0.0) from "no data to compute a rate"
    (returns None).

    For a rate with uncertainty, call
    :func:`inspect_degradation.analysis.rates.loop_rate`, which
    wraps this logic in a trace-level bootstrap CI.
    """
    labeled = 0
    looping = 0
    for trace in traces:
        for step in trace.steps:
            if step.is_looping is None:
                continue
            labeled += 1
            if step.is_looping:
                looping += 1
    if labeled == 0:
        return None
    return looping / labeled


# ---------------------------------------------------------------------------
# Chain-length mean with bootstrap CI
# ---------------------------------------------------------------------------


def _mean_loop_chain_length_over_traces(units: Sequence[GradedTrace]) -> float:
    """Bootstrap-safe inner statistic for loop-chain-length mean."""
    lengths = loop_chain_lengths(units)
    if not lengths:
        return float("nan")
    return sum(lengths) / len(lengths)


def loop_chain_length_mean_estimate(
    traces: Iterable[GradedTrace],
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
) -> Estimate:
    """Mean loop-chain length across traces, with trace-level bootstrap CI.

    Distinguishes "one brief loop" from "stuck in the same cycle for
    12 consecutive steps" — two very different agent states with the
    same pooled ``loop_rate``. The bootstrap resamples whole traces
    so chain-length variance across the corpus is correctly
    reflected in the CI.
    """
    trace_list = list(traces)
    return bootstrap_estimate(
        trace_list,
        _mean_loop_chain_length_over_traces,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        rng=rng,
    )


__all__ = [
    "loop_chain_length_mean_estimate",
    "loop_chain_lengths",
    "raw_loop_rate",
]
