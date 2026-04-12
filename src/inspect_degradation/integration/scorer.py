"""Inspect AI ``@scorer`` for within-run degradation grading.

The scorer is a thin shim: it converts the live ``TaskState`` into a
:class:`Trace` via :func:`task_state_to_trace`, dispatches to a
:class:`Grader` (defaulting to :class:`LLMGrader` with the packaged v1
rubric), and packs the resulting :class:`GradedTrace` into the
``Score.metadata`` so downstream :mod:`inspect_degradation.integration.metrics`
aggregators can read it without re-grading.

We deliberately do not compute degradation metrics inside the scorer --
that's the metric layer's job, and keeping the scorer write-only over the
graded trace lets multiple metric aggregations share one grading pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inspect_degradation.grader.interface import Grader
from inspect_degradation.grader.llm import LLMGrader, LLMGraderConfig
from inspect_degradation.grader.rubric import Rubric
from inspect_degradation.integration.trace_adapter import task_state_to_trace
from inspect_degradation.schema import GradedTrace

if TYPE_CHECKING:  # pragma: no cover
    from inspect_ai.scorer import Score
    from inspect_ai.solver import TaskState

#: Key under which the serialized :class:`GradedTrace` is stashed in
#: ``Score.metadata``. Metric aggregators read this key.
GRADED_TRACE_METADATA_KEY = "inspect_degradation.graded_trace"


def degradation_scorer(
    *,
    grader: Grader | None = None,
    grader_model: str = "openai/gpt-4o-mini",
    rubric_name: str = "step_grader_v1",
):
    """Return an Inspect AI scorer that grades each step of an agent trace.

    Args:
        grader: A pre-built grader. If omitted, an :class:`LLMGrader` is
            constructed with the default rubric and ``grader_model``.
            Passing a custom grader is the supported way to use a cascade.
        grader_model: Inspect model spec used when ``grader`` is not given.
        rubric_name: Stem of a rubric YAML in ``inspect_degradation/prompts/``.

    The scorer's ``value`` is the fraction of steps that scored ``pass``;
    callers wanting richer aggregates pull them from the metric layer.

    Default metrics (``error_rate``, ``productive_rate``,
    ``error_rate_slope``) are attached to the scorer and will appear
    in the Inspect eval viewer automatically. Additional metrics from
    :mod:`inspect_degradation.integration.metrics` can be added via the
    ``metrics`` parameter of :func:`inspect_ai.eval.eval`.
    """
    # Lazy import: this module is part of the integration layer, but we
    # still want a clean error message if inspect_ai is missing at runtime
    # (e.g. in a downstream offline-only install).
    try:
        from inspect_ai.scorer import Score, Target, scorer  # type: ignore
        from inspect_ai.solver import TaskState  # type: ignore  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "inspect_ai is required for the integration layer; "
            "install with `pip install inspect-ai`."
        ) from e

    from inspect_degradation.integration.metrics import (
        error_rate,
        error_rate_slope,
        productive_rate,
    )

    resolved_grader = grader or LLMGrader(
        config=LLMGraderConfig(model=grader_model),
        rubric=Rubric.from_package(rubric_name),
    )

    @scorer(metrics=[error_rate(), productive_rate(), error_rate_slope()])
    def _build_scorer():
        async def score(state: "TaskState", target: "Target") -> "Score":
            trace = task_state_to_trace(state)
            graded_steps = await resolved_grader.grade_trace(trace)
            graded_trace = GradedTrace(
                trace_id=trace.trace_id,
                task_id=trace.task_id,
                model=trace.model,
                source=trace.source,
                success=_infer_success(state),
                steps=graded_steps,
                metadata={"grader": resolved_grader.name},
            )
            value = _productive_rate(graded_trace)
            return Score(
                value=value,
                metadata={
                    GRADED_TRACE_METADATA_KEY: graded_trace.model_dump(mode="json"),
                },
            )

        return score

    return _build_scorer()


def _infer_success(state: "TaskState") -> bool | None:
    """Best-effort extraction of task success from prior scorer results.

    If another scorer has already run (e.g. the task's primary correctness
    scorer), its result is available in ``state.scores``. We interpret a
    numeric score > 0 as success, and the string ``"C"`` (Inspect's
    CORRECT constant) as success.

    Returns ``None`` if no prior scores are available or if the result
    is ambiguous.
    """
    prior_scores = getattr(state, "scores", None)
    if not prior_scores:
        return None
    # Take the first available scorer result.
    for _name, score in prior_scores.items():
        value = getattr(score, "value", None)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return value > 0
        if isinstance(value, str):
            return value.upper() in ("C", "CORRECT", "1", "TRUE")
    return None


def _productive_rate(graded: GradedTrace) -> float:
    """Top-line ``Score.value`` for the scorer: fraction of *productive* steps.

    "Productive" means ``validity == pass`` -- neutrals do not count, by
    design. The corresponding rate-decomposition lives in
    :mod:`inspect_degradation.integration.metrics`
    (``error_rate``, ``neutral_rate``, ``productive_rate``); the three
    sum to 1.0 across any complete trace.
    """
    if not graded.steps:
        return float("nan")
    from inspect_degradation.schema import Validity

    productive = sum(1 for s in graded.steps if s.validity == Validity.pass_)
    return productive / len(graded.steps)
