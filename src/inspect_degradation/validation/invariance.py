"""Falsification tests for grader invariances.

The degradation hypothesis — "agents get worse later in a trace" —
only means something if the grader's judgement of a single step
doesn't itself depend on where the step appears. If the grader is
systematically harsher on step 30 than it would be on the *same
content* at step 5, any measured "degradation" signal is partially
or entirely a grader artefact.

Two falsification tests live here:

* :func:`position_invariance_test` — regrade each step after
  telling the grader it occupies a different position in the
  trace, and measure how often the rubric judgement (validity)
  changes. Low disagreement = position-invariant grader.
* :func:`task_invariance_test` — regrade each step in the context
  of a *different* task prompt, and measure disagreement.

Both tests return a :class:`InvarianceReport` carrying a Wilson
proportion interval on the disagreement rate. Reports should cite
"grader disagreement under position relabeling: 4.1% [2.8, 5.9]"
and the upper bound of the CI becomes a load-bearing number for
the subsequent degradation claim. If position disagreement CI
overlaps the measured per-step degradation slope × trace length,
the degradation signal is not distinguishable from grader drift
and the finding should be flagged.

These tests are expensive — they require re-running the grader on
a sizeable sample of steps — so the default subsample size is
small and callers can widen it. Compute cost scales as
``n_steps * n_relabelings``.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from inspect_degradation.analysis.statistics import (
    NINETY_FIVE,
    ConfidenceLevel,
    Estimate,
    wilson_proportion_interval,
)
from inspect_degradation.schema import GradedStep, GradedTrace


@dataclass(frozen=True)
class InvarianceReport:
    """Result of an invariance falsification test.

    Attributes:
        name: Identifier for the test (e.g. ``"position"``).
        disagreement_rate: Wilson CI on the fraction of resampled
            steps whose validity judgement changed under the
            relabeling.
        n_steps_tested: How many steps were re-graded.
        n_disagreements: How many changed their validity label.
        details: Per-step disagreement diagnostics for audit.
    """

    name: str
    disagreement_rate: Estimate
    n_steps_tested: int
    n_disagreements: int
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "disagreement_rate": self.disagreement_rate.to_dict(),
            "n_steps_tested": self.n_steps_tested,
            "n_disagreements": self.n_disagreements,
            "details": list(self.details),
        }


# Type alias for the pluggable regrade callable. Takes
# ``(original_step, new_position, new_task_id)`` and returns a
# re-graded :class:`GradedStep`. Implementations are expected to
# call into whatever grader the surrounding pipeline uses.
RegradeFn = Callable[[GradedStep, int, str], GradedStep]


def _sample_steps(
    traces: Sequence[GradedTrace],
    *,
    sample_size: int,
    rng: random.Random,
) -> list[tuple[GradedTrace, GradedStep]]:
    pool: list[tuple[GradedTrace, GradedStep]] = []
    for t in traces:
        for s in t.steps:
            pool.append((t, s))
    if not pool:
        return []
    if sample_size >= len(pool):
        return pool
    return rng.sample(pool, sample_size)


def _require_task_id(trace: GradedTrace) -> str:
    """Resolve a usable task_id, falling back to trace_id.

    The schema allows ``task_id=None`` (some sources only carry a
    trace_id), but the invariance tests' :data:`RegradeFn` callback
    is typed as ``str``. Falling back to ``trace_id`` keeps the
    contract intact and is the right semantics for position-jitter
    probes anyway: a step's "task identity" is at worst the trace
    it lives in.
    """
    return trace.task_id if trace.task_id is not None else trace.trace_id


def position_invariance_test(
    traces: Sequence[GradedTrace],
    regrade_fn: RegradeFn,
    *,
    sample_size: int = 50,
    position_jitter: int = 10,
    seed: int | None = None,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
) -> InvarianceReport:
    """Does the grader judge the same step differently at a new position?

    Draws a random sample of ``sample_size`` steps from ``traces``,
    re-grades each one with its reported position offset by
    ``±position_jitter`` (never below 0), and counts how many
    re-grades change the step's :class:`Validity` label. The
    disagreement rate is reported as a Wilson CI — the right
    primitive for a binomial proportion of this shape.

    Args:
        traces: Corpus of already-graded traces to probe.
        regrade_fn: Callable the test uses to obtain a new grade
            at the jittered position. See :data:`RegradeFn`.
        sample_size: Number of steps to re-grade. Compute cost
            scales linearly with this.
        position_jitter: Magnitude of the position shift. The
            actual shift is uniformly drawn from
            ``[-jitter, +jitter]`` excluding 0 and clamped to ≥0.
        seed: Optional seed for the sampling and jitter RNG.
        confidence_level: CI level for the disagreement rate.

    Returns:
        :class:`InvarianceReport` named ``"position"``.
    """
    rng = random.Random(seed)
    sample = _sample_steps(traces, sample_size=sample_size, rng=rng)

    disagreements = 0
    details: list[dict[str, Any]] = []
    for trace, step in sample:
        shift = 0
        while shift == 0:
            shift = rng.randint(-position_jitter, position_jitter)
        new_pos = max(0, step.step_index + shift)
        regraded = regrade_fn(step, new_pos, _require_task_id(trace))
        disagreed = regraded.validity != step.validity
        if disagreed:
            disagreements += 1
        details.append(
            {
                "trace_id": trace.trace_id,
                "step_index": step.step_index,
                "new_position": new_pos,
                "original_validity": step.validity.value,
                "regraded_validity": regraded.validity.value,
                "disagreed": disagreed,
            }
        )

    rate = wilson_proportion_interval(
        disagreements, len(sample), confidence_level=confidence_level
    )
    return InvarianceReport(
        name="position",
        disagreement_rate=rate,
        n_steps_tested=len(sample),
        n_disagreements=disagreements,
        details=details,
    )


def task_invariance_test(
    traces: Sequence[GradedTrace],
    regrade_fn: RegradeFn,
    *,
    sample_size: int = 50,
    seed: int | None = None,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
) -> InvarianceReport:
    """Does the grader judge the same step differently under a new task?

    For each sampled step, swaps in a *different* trace's
    ``task_id`` and re-grades. Disagreement rate is computed the
    same way as :func:`position_invariance_test`.

    A well-calibrated grader should only change its verdict if the
    new task prompt legitimately makes the same action
    inappropriate — some non-zero rate is expected. The test's job
    is to bound that rate so downstream degradation claims can be
    interpreted relative to it.

    Args:
        traces: Corpus of already-graded traces to probe. Must
            contain at least two distinct ``task_id`` values; if
            not, a :class:`ValueError` is raised.
        regrade_fn: Callable the test uses to obtain a new grade
            under the swapped task id. See :data:`RegradeFn`.
        sample_size: Number of steps to re-grade.
        seed: Optional seed for the RNG.
        confidence_level: CI level for the disagreement rate.

    Returns:
        :class:`InvarianceReport` named ``"task"``.
    """
    distinct_tasks = {_require_task_id(t) for t in traces}
    if len(distinct_tasks) < 2:
        raise ValueError(
            "task_invariance_test requires >=2 distinct task_id values; "
            f"got {len(distinct_tasks)}"
        )

    rng = random.Random(seed)
    sample = _sample_steps(traces, sample_size=sample_size, rng=rng)

    disagreements = 0
    details: list[dict[str, Any]] = []
    for trace, step in sample:
        own_task = _require_task_id(trace)
        other_tasks = [t for t in distinct_tasks if t != own_task]
        new_task = rng.choice(other_tasks)
        regraded = regrade_fn(step, step.step_index, new_task)
        disagreed = regraded.validity != step.validity
        if disagreed:
            disagreements += 1
        details.append(
            {
                "trace_id": trace.trace_id,
                "step_index": step.step_index,
                "original_task_id": own_task,
                "new_task_id": new_task,
                "original_validity": step.validity.value,
                "regraded_validity": regraded.validity.value,
                "disagreed": disagreed,
            }
        )

    rate = wilson_proportion_interval(
        disagreements, len(sample), confidence_level=confidence_level
    )
    return InvarianceReport(
        name="task",
        disagreement_rate=rate,
        n_steps_tested=len(sample),
        n_disagreements=disagreements,
        details=details,
    )


__all__ = [
    "InvarianceReport",
    "RegradeFn",
    "position_invariance_test",
    "task_invariance_test",
]
