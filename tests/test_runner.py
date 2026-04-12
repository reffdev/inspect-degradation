"""End-to-end tests for the validation runner.

The runner orchestrates grading + persistence + scoring. The grader here
is a deterministic fake — no LLM calls — so these tests pin the
runner's behavior independently of any model. Coverage focuses on the
two non-trivial properties:

1. The cache short-circuits already-graded traces and the count is
   reported accurately.
2. Pairing across (trace_id, step_index) wires the agreement scorer to
   the right reference grades.
"""

import pytest
from conftest import make_graded_step, make_graded_trace, make_trace

from inspect_degradation.grader.interface import Grader, StepContext
from inspect_degradation.schema import (
    ComplexityLevel,
    Dependency,
    GradedStep,
    Validity,
)
from inspect_degradation.store import GradedTraceStore
from inspect_degradation.trace import Trace
from inspect_degradation.validation.runner import run_validation


class _StubGrader(Grader):
    """Always returns ``pass``. Records how many times it was called."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "stub"

    async def grade_step(self, ctx: StepContext) -> GradedStep:
        self.calls += 1
        return GradedStep(
            step_index=ctx.step_index,
            validity=Validity.pass_,
            complexity=ComplexityLevel.low,
            dependency=Dependency.not_applicable,
            is_looping=False,
            grader_model=self.name,
        )


def _trace_with_id(trace_id: str, n_steps: int) -> Trace:
    base = make_trace(n_steps=n_steps)
    return Trace(
        trace_id=trace_id,
        task_goal=base.task_goal,
        steps=base.steps,
    )


def _matching_reference(trace_id: str, n_steps: int):
    return make_graded_trace(
        trace_id=trace_id,
        steps=[make_graded_step(i, validity=Validity.pass_) for i in range(n_steps)],
    )


@pytest.mark.asyncio
async def test_runner_grades_and_scores_without_cache():
    grader = _StubGrader()
    traces = [_trace_with_id("a", 2), _trace_with_id("b", 3)]
    references = [_matching_reference("a", 2), _matching_reference("b", 3)]

    result = await run_validation(grader=grader, traces=traces, reference=references)

    assert result.n_freshly_graded == 2
    assert result.n_from_cache == 0
    assert grader.calls == 5  # 2 + 3 steps
    assert result.report.n_pairs == 5
    # Stub always returns pass, references always pass: perfect agreement.
    assert result.report.per_dimension["validity"].value == 1.0


@pytest.mark.asyncio
async def test_cache_short_circuits_previously_graded_traces(tmp_path):
    cache_path = tmp_path / "c.jsonl"

    grader1 = _StubGrader()
    traces = [_trace_with_id("a", 2), _trace_with_id("b", 2)]
    references = [_matching_reference("a", 2), _matching_reference("b", 2)]
    cache = GradedTraceStore(cache_path)

    first = await run_validation(
        grader=grader1, traces=traces, reference=references, cache=cache
    )
    assert first.n_freshly_graded == 2
    assert first.n_from_cache == 0
    assert grader1.calls == 4

    # Reopen cache, run again with a fresh grader instance — nothing
    # should be re-graded, but the report must still be computed.
    grader2 = _StubGrader()
    cache2 = GradedTraceStore(cache_path)
    second = await run_validation(
        grader=grader2, traces=traces, reference=references, cache=cache2
    )
    assert second.n_freshly_graded == 0
    assert second.n_from_cache == 2
    assert grader2.calls == 0
    assert second.report.n_pairs == 4


@pytest.mark.asyncio
async def test_partial_cache_only_skips_matching_trace_ids(tmp_path):
    cache_path = tmp_path / "c.jsonl"
    grader = _StubGrader()
    traces = [_trace_with_id("a", 1), _trace_with_id("b", 1)]
    references = [_matching_reference("a", 1), _matching_reference("b", 1)]

    # Pre-populate the cache with only "a".
    pre = GradedTraceStore(cache_path)
    pre.append(
        make_graded_trace(
            trace_id="a",
            steps=[make_graded_step(0, validity=Validity.pass_)],
        )
    )

    cache = GradedTraceStore(cache_path)
    result = await run_validation(
        grader=grader, traces=traces, reference=references, cache=cache
    )
    assert result.n_from_cache == 1
    assert result.n_freshly_graded == 1
    assert grader.calls == 1
