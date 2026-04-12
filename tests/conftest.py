"""Shared test fixtures and small builders.

Centralizing the constructors keeps individual tests focused on the
behavior they actually exercise rather than on schema boilerplate.
"""

from __future__ import annotations

from inspect_degradation.schema import (
    ComplexityLevel,
    Dependency,
    GradedStep,
    GradedTrace,
    SeverityLevel,
    Validity,
)
from inspect_degradation.trace import Trace, TraceStep


def make_step(
    index: int,
    *,
    action: str = "do thing",
    observation: str | None = "ok",
    thought: str | None = None,
) -> TraceStep:
    return TraceStep(index=index, action=action, observation=observation, thought=thought)


def make_trace(
    *,
    trace_id: str = "t1",
    task_goal: str = "complete the task",
    n_steps: int = 3,
) -> Trace:
    return Trace(
        trace_id=trace_id,
        task_goal=task_goal,
        steps=tuple(make_step(i) for i in range(n_steps)),
    )


def make_graded_step(
    index: int,
    *,
    validity: Validity = Validity.pass_,
    complexity: ComplexityLevel | None = ComplexityLevel.low,
    dependency: Dependency | None = Dependency.not_applicable,
    severity: SeverityLevel | None = None,
    is_looping: bool | None = False,
    grader_model: str = "test",
) -> GradedStep:
    return GradedStep(
        step_index=index,
        validity=validity,
        complexity=complexity,
        dependency=dependency,
        severity=severity,
        is_looping=is_looping,
        grader_model=grader_model,
    )


def make_graded_trace(
    *,
    trace_id: str = "t1",
    steps: list[GradedStep] | None = None,
) -> GradedTrace:
    return GradedTrace(
        trace_id=trace_id,
        steps=steps if steps is not None else [make_graded_step(0)],
    )
