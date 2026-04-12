"""LLM-as-judge step grader and composition primitives.

Public surface:

* :class:`Grader` — abstract interface; bring your own backend.
* :class:`LLMGrader` / :class:`LLMGraderConfig` — concrete grader backed
  by Inspect AI's model API. Supports single-model self-consistency
  via ``sample_n > 1``.
* :class:`EnsembleGrader` — the primary composition primitive. Runs
  multiple independent :class:`Grader` instances per step and
  majority-votes on validity. Heterogeneous across model families is
  the intended shape; single-member ensembles are the identity.
* :class:`Rubric` — versioned YAML rubric loader.
* :class:`GradeResponse` / :func:`parse_grade_response` /
  :class:`GraderResponseError` — model-side response parsing.
* :class:`StepContext` — what a grader sees per step.
"""

from inspect_degradation.grader.ensemble import (
    ENSEMBLE_KEY,
    EnsembleGrader,
    EnsembleMemberGrade,
)
from inspect_degradation.grader.interface import Grader, StepContext
from inspect_degradation.grader.llm import (
    SELF_CONSISTENCY_KEY,
    LLMGrader,
    LLMGraderConfig,
)
from inspect_degradation.grader.response import (
    GradeResponse,
    GraderResponseError,
    parse_grade_response,
)
from inspect_degradation.grader.rubric import Rubric

__all__ = [
    "ENSEMBLE_KEY",
    "EnsembleGrader",
    "EnsembleMemberGrade",
    "GradeResponse",
    "GraderResponseError",
    "Grader",
    "LLMGrader",
    "LLMGraderConfig",
    "Rubric",
    "SELF_CONSISTENCY_KEY",
    "StepContext",
    "parse_grade_response",
]
