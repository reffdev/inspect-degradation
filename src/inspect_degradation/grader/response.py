"""Parsing and validation of LLM grader responses.

The grader prompt asks the model to return a single JSON object whose
fields exactly match the rubric dimensions. Real models routinely wrap
that JSON in prose, markdown code fences, or trailing commentary, so we
cannot just call ``json.loads`` on the raw completion. This module is the
*only* place that knows how to recover from those wrappers, and the only
place that maps loose response fields onto the strict
:class:`GradedStep` schema.

Keeping parsing isolated means:
  * concrete graders can be tested without an LLM by feeding canned strings,
  * the rubric can change shape (add fields, rename) by editing one Pydantic
    model rather than chasing call sites,
  * future structured-output backends (tool-call grading, JSON schema mode)
    can be added as alternative parsers without touching the grader loop.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from inspect_degradation.schema import (
    ComplexityLevel,
    Dependency,
    GradedStep,
    SeverityLevel,
    Validity,
)


class GraderResponseError(ValueError):
    """Raised when an LLM grader response cannot be parsed or validated."""

    def __init__(self, message: str, *, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class GradeResponse(BaseModel):
    """The model-side schema for a single step grade.

    Mirrors the YAML rubric's documented output keys exactly. Required
    fields here correspond to the dimensions an LLM grader is *contractually
    expected* to fill — keeping the model-side requirements stricter than
    the storage type ensures the LLM grader pipeline cannot silently emit
    a partially-filled grade.
    """

    model_config = {"extra": "ignore"}

    validity: Validity
    complexity: ComplexityLevel
    dependency: Dependency
    severity: SeverityLevel | None = None
    is_looping: bool

    def to_graded_step(
        self,
        *,
        step_index: int,
        grader_model: str,
        raw: dict[str, Any] | None = None,
    ) -> GradedStep:
        """Lift a parsed response into the canonical :class:`GradedStep`.

        Enforces cross-field invariants the per-field validators can't catch:
          * ``severity`` must be present iff ``validity == fail``,
          * ``dependency`` must be ``"n/a"`` for non-failing steps —
            independence/dependence is only meaningful for failures,
          * ``is_looping`` must be ``False`` when ``validity == pass`` —
            a step that makes pointable progress is not a loop.
        """
        if self.validity == Validity.fail:
            if self.severity is None:
                raise GraderResponseError(
                    "validity=fail requires severity, got null",
                    raw=json.dumps(raw or {}),
                )
        else:
            if self.severity is not None:
                raise GraderResponseError(
                    f"severity must be null when validity={self.validity.value}",
                    raw=json.dumps(raw or {}),
                )
            if self.dependency != Dependency.not_applicable:
                raise GraderResponseError(
                    f"dependency must be 'n/a' when validity={self.validity.value}, "
                    f"got {self.dependency.value!r}",
                    raw=json.dumps(raw or {}),
                )

        if self.validity == Validity.pass_ and self.is_looping:
            raise GraderResponseError(
                "is_looping cannot be true when validity='pass'; "
                "a step that makes pointable progress is not a loop",
                raw=json.dumps(raw or {}),
            )

        return GradedStep(
            step_index=step_index,
            validity=self.validity,
            complexity=self.complexity,
            dependency=self.dependency,
            severity=self.severity,
            is_looping=self.is_looping,
            grader_model=grader_model,
            raw=raw,
        )


# Matches a fenced ```json ... ``` or ``` ... ``` block, capturing the body.
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)


def parse_grade_response(text: str) -> GradeResponse:
    """Parse a raw LLM completion into a validated :class:`GradeResponse`.

    Strategy, in order:
      1. If the completion contains a fenced code block, parse the *first*
         such block as JSON.
      2. Otherwise, locate the first balanced ``{...}`` substring and parse
         that. This handles models that emit prose before/after the object.
      3. Validate via Pydantic. Validation errors are wrapped in
         :class:`GraderResponseError` carrying the raw input for diagnosis.
    """
    if not text or not text.strip():
        raise GraderResponseError("empty grader response", raw=text)

    candidate = _extract_json_blob(text)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise GraderResponseError(
            f"grader response is not valid JSON: {e.msg} at pos {e.pos}",
            raw=text,
        ) from e

    if not isinstance(data, dict):
        raise GraderResponseError(
            f"grader response must be a JSON object, got {type(data).__name__}",
            raw=text,
        )

    try:
        return GradeResponse.model_validate(data)
    except ValidationError as e:
        raise GraderResponseError(
            f"grader response failed schema validation: {e.errors(include_url=False)}",
            raw=text,
        ) from e


def _extract_json_blob(text: str) -> str:
    """Extract the JSON body from a possibly-fenced, possibly-prosed completion."""
    fenced = _FENCE_RE.search(text)
    if fenced is not None:
        return fenced.group(1).strip()

    # Find the first balanced top-level { ... } substring. We deliberately
    # do not regex this — JSON nesting requires a stack.
    start = text.find("{")
    if start == -1:
        raise GraderResponseError("no JSON object found in grader response", raw=text)

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise GraderResponseError("unbalanced JSON object in grader response", raw=text)
