"""Pydantic schema for per-step grades and graded traces.

These types are the persistence/transport contract between the three layers
(grader, analysis, integration) *and* between LLM-graded data and
human-labeled reference data.

Every gradable dimension is either nominal categorical (``Validity``,
``Dependency``) or ordinal categorical with three levels
(``ComplexityLevel``, ``SeverityLevel``). The schema deliberately rejects
continuous floats: a grader can reliably distinguish "obvious / moderate
/ ambiguous" but cannot reliably distinguish 0.42 from 0.47, and
pretending otherwise invites spurious precision into downstream
statistics. Three-level ordinals also align natively with TRAIL's
LOW/MEDIUM/HIGH impact taxonomy and with the kind of rubric language a
human annotator can apply consistently.

Strictness for *LLM-produced* grades is enforced one layer up, in
:class:`inspect_degradation.grader.response.GradeResponse`, which the LLM
grader pipeline always goes through. The two-layer split is deliberate:

* the storage type is permissive enough to round-trip any labeling source
  (TRAIL labels validity and severity but not the other dimensions),
* the model-side response type is strict enough that an LLM grader cannot
  silently omit a dimension and have it land in the dataset as ``None``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, Field, NonNegativeInt, model_validator


class Validity(str, Enum):
    """Whether a step moved the task meaningfully forward.

    Three levels, deliberately not a binary:

    * ``fail`` — actively wrong, off-base, or harmful relative to the goal.
    * ``neutral`` — not wrong, but doesn't meaningfully advance the task
      even though doing so was possible. Wheel-spinning, restating prior
      context, hedging, and idle exploration that produces no new
      information all land here. Distinguished from ``fail`` because the
      agent isn't broken, and from ``pass`` because progress isn't
      happening.
    * ``pass`` — a correct, productive step toward the goal.

    The neutral level is what lets the analysis layer separate
    "the agent makes more mistakes late in traces" from "the agent
    flails late in traces" — two different failure modes with two
    different fixes. See :func:`inspect_degradation.integration.metrics.neutral_rate`
    and ``productive_rate``.
    """

    fail = "fail"
    neutral = "neutral"
    pass_ = "pass"


class Dependency(str, Enum):
    """Whether a failing step's error originated independently or cascaded.

    ``independent`` means the agent had a workable state and got it wrong
    on its own.  ``dependent`` means this failure is a downstream
    consequence of an earlier one.  ``not_applicable`` is used when the
    step is not a failure, so dependency is not meaningful.
    """

    independent = "independent"
    dependent = "dependent"
    not_applicable = "n/a"


# ---------------------------------------------------------------------------
# Ordinal level enums.
#
# Each ordinal dimension gets its own type so the type system catches
# accidental cross-dimension assignments. They share a value vocabulary
# (low / medium / high) so graders see consistent language across fields,
# but the *semantics* of each level differ per dimension and are pinned in
# the rubric YAML, not in code.
# ---------------------------------------------------------------------------


_LEVEL_RANKS: dict[str, int] = {"low": 1, "medium": 2, "high": 3}


class _OrdinalLevelMixin:
    """Comparison + rank mixin for the three ordinal level enums.

    Plain Python class — *not* an Enum subclass — so concrete enums can
    legally mix it in. (Python forbids subclassing an Enum that inherits
    from ``Enum``, even one with no members.) Each concrete enum is
    declared as ``class X(_OrdinalLevelMixin, str, Enum)``; the mixin
    supplies ``rank`` and the ordering operators, the concrete class
    supplies the members.

    All four comparison operators are defined explicitly rather than via
    ``functools.total_ordering`` because that decorator interacts badly
    with Enum's auto-generated ``__eq__``.
    """

    _ranks_table: ClassVar[dict[str, int]] = _LEVEL_RANKS

    @property
    def rank(self) -> int:
        return self._ranks_table[self.value]  # type: ignore[attr-defined]

    def __lt__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.rank < other.rank  # type: ignore[attr-defined]

    def __le__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.rank <= other.rank  # type: ignore[attr-defined]

    def __gt__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.rank > other.rank  # type: ignore[attr-defined]

    def __ge__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.rank >= other.rank  # type: ignore[attr-defined]


class ComplexityLevel(_OrdinalLevelMixin, str, Enum):
    """How clear the correct next action is at this step.

    * ``low`` — the correct next move is obvious from the prior context.
    * ``medium`` — multiple reasonable next moves; a competent agent
      could pick any of several without being wrong.
    * ``high`` — genuinely ambiguous; even a careful operator would have
      to guess.

    Used as a control variable in mixed-effects models to separate "agent
    is degrading" from "the task is just getting harder".
    """

    low = "low"
    medium = "medium"
    high = "high"


class SeverityLevel(_OrdinalLevelMixin, str, Enum):
    """How consequential a failing step is for the rest of the task.

    Aligned with TRAIL's LOW / MEDIUM / HIGH impact taxonomy so human
    references map cleanly:

    * ``low`` — minor inconvenience; task is still recoverable trivially.
    * ``medium`` — meaningful degradation; recovery requires extra steps.
    * ``high`` — task-critical; recovery from this state is unlikely.
    """

    low = "low"
    medium = "medium"
    high = "high"


#: Sentinel ``grader_model`` value used by reference (human-labeled) traces.
HUMAN_GRADER = "human"


class GradedStep(BaseModel):
    """A single agent step after grading.

    The required fields are the minimum any labeling source must produce:
    a step index, a validity verdict, and a grader identifier. Everything
    else is optional so partial human references can use the same type as
    fully-graded LLM output.
    """

    step_index: NonNegativeInt
    validity: Validity
    grader_model: str

    complexity: ComplexityLevel | None = None
    dependency: Dependency | None = None
    severity: SeverityLevel | None = None
    is_looping: bool | None = None
    raw: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_cross_field_invariants(self) -> "GradedStep":
        # Severity is meaningful only for failing steps.
        if self.validity != Validity.fail and self.severity is not None:
            raise ValueError(
                f"severity must be null when validity={self.validity.value}"
            )
        # Dependency only applies to failing steps; non-failures are
        # neither independent nor dependent failures, they're not failures.
        if self.validity != Validity.fail and self.dependency not in (
            None,
            Dependency.not_applicable,
        ):
            raise ValueError(
                f"dependency must be null or 'n/a' when validity={self.validity.value}"
            )
        # A step cannot be both productive progress and a loop. If real
        # progress is being made, it is definitionally not a repetition
        # of prior work — the two labels are mutually exclusive and
        # pairing them would silently corrupt the loop_rate metric.
        if self.validity == Validity.pass_ and self.is_looping is True:
            raise ValueError(
                "is_looping=True is incompatible with validity='pass'; "
                "a step that makes pointable progress is not a loop"
            )
        return self


class GradedTrace(BaseModel):
    """A full agent trace plus per-step grades."""

    trace_id: str
    task_id: str | None = None
    model: str | None = None
    source: str | None = Field(
        default=None, description="e.g. 'trail', 'nebius', 'inspect-live'"
    )
    success: bool | None = None
    steps: list[GradedStep]
    metadata: dict[str, Any] = Field(default_factory=dict)
