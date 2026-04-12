"""Tests for the Grader ABC's orchestration behavior.

We use a fake grader so these tests stay fast and have no LLM dependency.
The fake exercises exactly the contract the interface promises: it must
receive correctly-sliced ``StepContext``s and the results must come back
in trace order even if the fake completes them out of order.
"""

import asyncio

import pytest
from conftest import make_trace

from inspect_degradation.grader.interface import Grader, StepContext
from inspect_degradation.schema import (
    ComplexityLevel,
    Dependency,
    GradedStep,
    Validity,
)


class _FakeGrader(Grader):
    def __init__(self, *, delays: list[float] | None = None):
        self.calls: list[StepContext] = []
        self._delays = delays or []

    @property
    def name(self) -> str:
        return "fake"

    async def grade_step(self, ctx: StepContext) -> GradedStep:
        self.calls.append(ctx)
        if self._delays:
            await asyncio.sleep(self._delays[ctx.step_index])
        return GradedStep(
            step_index=ctx.step_index,
            validity=Validity.pass_,
            complexity=ComplexityLevel.low,
            dependency=Dependency.not_applicable,
            is_looping=False,
            grader_model=self.name,
        )


@pytest.mark.asyncio
async def test_grade_trace_passes_correct_prior_steps():
    grader = _FakeGrader()
    trace = make_trace(n_steps=4)
    await grader.grade_trace(trace)
    assert [c.step_index for c in sorted(grader.calls, key=lambda c: c.step_index)] == [0, 1, 2, 3]
    by_idx = {c.step_index: c for c in grader.calls}
    assert by_idx[0].prior_steps == ()
    assert by_idx[3].prior_steps == trace.steps[:3]
    assert all(c.task_goal == trace.task_goal for c in grader.calls)
    assert all(c.trace_id == trace.trace_id for c in grader.calls)


@pytest.mark.asyncio
async def test_grade_trace_results_preserve_order_under_concurrency():
    # Step 0 sleeps longer than step 2; without order preservation the
    # output would come back as [2, 1, 0].
    grader = _FakeGrader(delays=[0.05, 0.02, 0.0])
    trace = make_trace(n_steps=3)
    grades = await grader.grade_trace(trace)
    assert [g.step_index for g in grades] == [0, 1, 2]


@pytest.mark.asyncio
async def test_grade_trace_handles_empty_trace():
    grader = _FakeGrader()
    trace = make_trace(n_steps=0)
    grades = await grader.grade_trace(trace)
    assert grades == []
