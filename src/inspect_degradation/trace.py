"""Raw (ungraded) agent trace types.

These are the contract between trace *sources* (Inspect AI live runs,
TRAIL/Nebius/SWE-smith offline datasets) and *graders*. They are deliberately
minimal: every source we care about can be reduced to an ordered list of
``(thought?, action, observation?)`` triples plus a top-level task goal.

The grading layer never sees source-specific structure — only ``Trace`` and
``TraceStep`` — which is what makes the grader reusable across data sources
and what lets us run the same statistical analysis on either live or
offline-graded runs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, NonNegativeInt


class TraceStep(BaseModel):
    """A single step in an agent trace, before grading.

    A ``step`` is the smallest unit a grader judges. For tool-calling agents
    this is typically one ``(tool_call, tool_result)`` pair; for chat-only
    agents it can be one assistant message. Sources decide the granularity
    once at load time and the rest of the pipeline is agnostic to it.
    """

    model_config = {"frozen": True}

    index: NonNegativeInt
    thought: str | None = Field(
        default=None,
        description="Agent reasoning text, if separable from the action.",
    )
    action: str = Field(
        description="What the agent did at this step (tool call repr, message text, etc.).",
    )
    observation: str | None = Field(
        default=None,
        description="What came back from the environment after the action.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """An ordered sequence of agent steps plus the originating task goal.

    The ``task_goal`` is required because every grading dimension is defined
    *relative to the goal* — without it, validity and complexity have no
    referent. Sources that don't carry an explicit goal must synthesize one
    from the task spec at load time rather than leaving it blank.
    """

    model_config = {"frozen": True}

    trace_id: str
    task_goal: str
    task_id: str | None = None
    model: str | None = None
    source: str | None = Field(
        default=None,
        description="Origin of the trace; e.g. 'trail', 'nebius', 'inspect-live'.",
    )
    success: bool | None = None
    steps: tuple[TraceStep, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def prior(self, index: int) -> tuple[TraceStep, ...]:
        """Return all steps strictly before ``index`` (the grader's read-only context)."""
        if index < 0 or index > len(self.steps):
            raise IndexError(f"step index {index} out of range for trace of length {len(self.steps)}")
        return self.steps[:index]
