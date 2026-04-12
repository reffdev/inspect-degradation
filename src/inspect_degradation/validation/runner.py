"""Validation runner: grade a labeled corpus and emit an agreement report.

This is the top-level entry point for Phase 1 grader-validation experiments.
It is intentionally a thin function rather than a class — the heavy lifting
lives in :class:`Grader` (which handles concurrency) and :mod:`.agreement`
(which handles scoring). The runner exists only to wire them together,
to keep one canonical pairing convention, and to handle resumability via
an optional :class:`GradedTraceStore` cache.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass

from numpy.random import Generator

log = logging.getLogger(__name__)

from inspect_degradation.analysis.statistics import (
    NINETY_FIVE,
    ConfidenceLevel,
)
from inspect_degradation.grader.interface import Grader
from inspect_degradation.schema import GradedTrace
from inspect_degradation.store import GradedTraceStore
from inspect_degradation.trace import Trace
from inspect_degradation.validation.agreement import (
    AgreementReport,
    pair_grades,
    score_agreement,
)


@dataclass(frozen=True)
class ValidationResult:
    """The full output of a validation run.

    Carries both the per-dimension agreement report (the headline numbers)
    and the actual graded traces (so callers can reuse them — for cost
    accounting, error analysis, or re-scoring with a different metric —
    without re-paying the grading cost).
    """

    report: AgreementReport
    predicted: list[GradedTrace]
    n_from_cache: int
    n_freshly_graded: int


async def run_validation(
    *,
    grader: Grader,
    traces: Iterable[Trace],
    reference: Iterable[GradedTrace],
    cache: GradedTraceStore | None = None,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    bootstrap_resamples: int = 2000,
    rng: Generator | None = None,
) -> ValidationResult:
    """Grade ``traces`` with ``grader`` and score against ``reference``.

    The reference must already be in :class:`GradedTrace` form (use
    :class:`inspect_degradation.schema.HUMAN_GRADER` as the
    ``grader_model`` for human labels). Converting raw human annotations
    into :class:`GradedTrace` is the responsibility of the dataset
    adapter, not this runner, so the runner stays source-agnostic.

    If ``cache`` is provided, traces whose ``trace_id`` already appears in
    the cache are loaded from it instead of being re-graded. Newly graded
    traces are appended to the cache before scoring, so a crash mid-run
    leaves a partial-but-valid cache the next call can resume from.

    The cache is identity-based on ``trace_id`` only — it does *not* check
    that the cached grades came from the same grader configuration.
    Mixing graders into one cache file is a user error; use one cache
    file per ``ExperimentConfig`` to keep results comparable.
    """
    traces_list = list(traces)
    cached_ids: set[str] = cache.completed_trace_ids() if cache is not None else set()

    cached_by_id: dict[str, GradedTrace] = {}
    if cache is not None and cached_ids:
        for trace in cache:
            if trace.trace_id in cached_ids:
                cached_by_id[trace.trace_id] = trace

    # Separate cached traces from those that need fresh grading.
    cached_traces: list[tuple[int, GradedTrace]] = []
    pending_traces: list[tuple[int, Trace]] = []
    for i, trace in enumerate(traces_list):
        if trace.trace_id in cached_by_id:
            cached_traces.append((i, cached_by_id[trace.trace_id]))
        else:
            pending_traces.append((i, trace))

    n_from_cache = len(cached_traces)

    # Grade pending traces concurrently. The grader's instance-level
    # semaphore caps total in-flight API calls to max_concurrency,
    # so launching all traces at once is safe — they compete for the
    # same semaphore slots instead of running one at a time.
    #
    # We use return_exceptions=True so a single failing trace does
    # not cancel every other in-flight trace. Failed traces are
    # logged and skipped — the cache keeps every trace that
    # completed successfully, and the next run will retry the rest.
    _completed = 0
    _total_pending = len(pending_traces)

    async def _grade_one_trace(trace: Trace) -> GradedTrace:
        graded_steps = await grader.grade_trace(trace)
        nonlocal _completed
        _completed += 1
        log.info(
            "[%s] %d/%d traces graded (%d cached, %d steps)",
            grader.name,
            _completed,
            _total_pending,
            n_from_cache,
            len(graded_steps),
        )
        return GradedTrace(
            trace_id=trace.trace_id,
            task_id=trace.task_id,
            model=trace.model,
            source=trace.source,
            success=trace.success,
            steps=graded_steps,
            metadata={**trace.metadata, "validation_grader": grader.name},
        )

    freshly_graded: list[tuple[int, GradedTrace]] = []
    if pending_traces:
        log.info(
            "[%s] grading %d traces (max_concurrency=%d, %d cached)",
            grader.name,
            _total_pending,
            grader.max_concurrency,
            n_from_cache,
        )
        tasks = [_grade_one_trace(trace) for _, trace in pending_traces]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        n_errors = 0
        for (i, trace), outcome in zip(pending_traces, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                n_errors += 1
                log.error(
                    "trace %s failed: %s: %s",
                    trace.trace_id,
                    type(outcome).__name__,
                    outcome,
                )
                continue
            graded = outcome
            if cache is not None:
                cache.append(graded)
            freshly_graded.append((i, graded))
        if n_errors:
            log.warning(
                "%d/%d traces failed; they will be retried on the next run",
                n_errors,
                len(pending_traces),
            )

    n_freshly_graded = len(freshly_graded)

    # Reassemble in original trace order so pairing with references
    # stays correct regardless of completion order.
    all_traces = cached_traces + freshly_graded
    all_traces.sort(key=lambda pair: pair[0])
    predicted = [graded for _, graded in all_traces]

    pairs = pair_grades(predicted, reference)
    report = score_agreement(
        grader.name,
        pairs,
        confidence_level=confidence_level,
        n_resamples=bootstrap_resamples,
        rng=rng,
    )
    return ValidationResult(
        report=report,
        predicted=predicted,
        n_from_cache=n_from_cache,
        n_freshly_graded=n_freshly_graded,
    )
