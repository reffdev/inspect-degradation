"""Grader interface (abstract base class).

This module defines *only* the interface — no model wiring, no rubric loading,
no I/O. Concrete graders live alongside it (``llm.py``, ``cascade.py``) and
implement :meth:`Grader.grade_step`. The base class supplies a correct,
bounded-concurrency :meth:`Grader.grade_trace` so subclasses never need to
re-implement orchestration.

Why an ABC and not a Protocol: graders compose (``CascadeGrader`` wraps two
others) and we want ``isinstance`` checks plus shared behavior to live in
one place. Protocols would force every concrete grader to re-implement
``grade_trace`` identically.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from inspect_degradation.schema import GradedStep
from inspect_degradation.trace import Trace, TraceStep


@dataclass(frozen=True)
class GraderSnapshot:
    """Serializable, configuration-only picture of a grader.

    Concrete graders return one from :meth:`Grader.snapshot`. The shape is
    intentionally a recursive dataclass so a :class:`CascadeGrader` snapshot
    can nest its children's snapshots without erasing their type.

    Snapshots are pure data: they round-trip through JSON and never hold
    references to live model handles, retry state, or HTTP clients. The
    :mod:`inspect_degradation.experiment` module wraps a snapshot in a
    larger reproducibility envelope but does not need to know any
    grader-specific details, which is the point of this type.
    """

    kind: str
    name: str
    fields: dict[str, object] = field(default_factory=dict)
    children: dict[str, "GraderSnapshot"] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "fields": dict(self.fields),
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }


@dataclass(frozen=True)
class StepContext:
    """Everything a grader needs to judge one step.

    Constructed by :meth:`Grader.grade_trace` so concrete graders never have
    to slice traces themselves. ``prior_steps`` is read-only context — graders
    must not assume it contains anything beyond what the source provided.
    """

    task_goal: str
    step_index: int
    step: TraceStep
    prior_steps: tuple[TraceStep, ...]
    trace_id: str


class Grader(ABC):
    """Abstract grader interface.

    Subclasses implement :meth:`grade_step`. The default :meth:`grade_trace`
    fans out across steps with bounded concurrency and preserves order.
    """

    #: Maximum number of step grades a single ``grade_trace`` call may have
    #: in flight at once. Subclasses can override; ``CascadeGrader`` inherits
    #: whatever its primary grader uses, since the cascade itself adds no
    #: independent rate limit.
    max_concurrency: int = 8

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier written into ``GradedStep.grader_model``.

        For ``LLMGrader`` this is the model spec; for ``CascadeGrader`` it's
        a composite name. The validation harness uses this to disambiguate
        which grader produced which prediction.
        """

    @abstractmethod
    async def grade_step(self, ctx: StepContext) -> GradedStep:
        """Grade a single step. Concrete graders implement this."""

    def snapshot(self) -> GraderSnapshot:
        """Return a serializable picture of this grader's configuration.

        The default returns the bare ``name`` so unknown subclasses still
        round-trip; concrete graders override this to surface the fields a
        reproducibility report needs.
        """
        return GraderSnapshot(kind=type(self).__name__, name=self.name)

    @property
    def _semaphore(self) -> asyncio.Semaphore:
        """Instance-level semaphore shared across all concurrent trace calls.

        Lazy-initialized on first access so subclasses that don't call
        ``super().__init__()`` still work. The semaphore is bound to the
        grader instance, not to a single ``grade_trace`` call, so when
        multiple traces are graded concurrently their step calls compete
        for the same pool of slots — total in-flight API calls is always
        ≤ ``max_concurrency`` regardless of how many traces are in flight.
        """
        # _sem is set on the instance, not the class, to avoid sharing
        # across grader instances (which may have different concurrency).
        if not hasattr(self, "_sem"):
            self._sem = asyncio.Semaphore(self.max_concurrency)
        return self._sem

    async def grade_trace(self, trace: Trace) -> list[GradedStep]:
        """Grade every step in ``trace`` sequentially, preserving order.

        Steps are graded in index order so that each successive API
        call extends the prior call's prompt prefix (system prompt +
        task goal + prior steps). Providers that support prompt
        caching (Anthropic, OpenAI) can then serve the growing
        prefix from cache, significantly reducing per-token cost on
        longer traces — step N's prompt shares all of step N-1's
        prefix plus the new step content.

        Multiple ``grade_trace`` calls can be in flight at once
        (the validation runner launches traces in parallel). The
        instance-level semaphore ensures total in-flight API calls
        never exceeds :attr:`max_concurrency`. With sequential
        steps, each trace has at most one call in flight at a time,
        so ``max_concurrency`` effectively caps the number of
        traces graded simultaneously.

        Step grading is independent — each grader call only sees
        ``prior_steps``, never future ones — so this ordering is
        purely a cost optimization, not a correctness requirement.
        """
        sem = self._semaphore
        results: list[GradedStep] = []
        for index, step in enumerate(trace.steps):
            async with sem:
                ctx = StepContext(
                    task_goal=trace.task_goal,
                    step_index=index,
                    step=step,
                    prior_steps=trace.prior(index),
                    trace_id=trace.trace_id,
                )
                results.append(await self.grade_step(ctx))
        return results
