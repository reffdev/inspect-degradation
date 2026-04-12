"""Rubric loading and rendering.

Rubrics are versioned YAML files (see ``inspect_degradation/prompts/``) so
the prompt text can be iterated on without code changes — one of the design
constraints in PROJECT_PLAN.md.

This module is responsible for:
  * loading and validating the YAML against a Pydantic schema,
  * verifying at load time that the user template's placeholders match the
    set the renderer supplies (so typos surface immediately, not at the
    first grader call hours into a run),
  * rendering the user prompt with explicit, named arguments.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from inspect_degradation.trace import TraceStep


@dataclass
class RenderDiagnostics:
    """Per-render diagnostics returned alongside the rendered prompt.

    Lets the calling grader detect prior-context truncation without
    re-parsing the rendered text. The grader can aggregate these
    across a run to surface "k% of grader prompts hit the prior-
    context cap" — the kind of fact a reviewer needs to assess
    whether a step-index slope is real or a context-truncation
    artifact.
    """

    prior_steps_total: int = 0
    prior_steps_kept: int = 0
    prior_steps_dropped: int = 0
    prior_steps_char_length: int = 0
    prior_steps_truncated: bool = False

#: Placeholders the renderer is contractually required to supply. Rubric YAML
#: files may use any subset of these and nothing else; unknown placeholders
#: are a load-time error.
_ALLOWED_PLACEHOLDERS: frozenset[str] = frozenset(
    {"task_goal", "prior_steps", "step_index", "step"}
)


class Rubric(BaseModel):
    """A versioned grading rubric loaded from YAML.

    Construct via :meth:`from_yaml` or :meth:`from_package` rather than
    directly — those run the placeholder validation that the bare
    constructor cannot.
    """

    model_config = {"frozen": True}

    version: int
    name: str
    description: str
    system: str
    user_template: str
    placeholders: frozenset[str] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def _check_placeholders(self) -> "Rubric":
        found = _extract_placeholders(self.user_template)
        unknown = found - _ALLOWED_PLACEHOLDERS
        if unknown:
            raise ValueError(
                f"rubric '{self.name}' v{self.version} references unknown "
                f"template placeholders: {sorted(unknown)}. "
                f"Allowed: {sorted(_ALLOWED_PLACEHOLDERS)}."
            )
        # Pydantic frozen models still allow attribute set during validation.
        object.__setattr__(self, "placeholders", frozenset(found))
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Rubric":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"rubric file {path} did not parse to a mapping")
        return cls.model_validate(data)

    @classmethod
    def from_package(cls, name: str) -> "Rubric":
        """Load a rubric shipped inside the package by file stem.

        Example: ``Rubric.from_package("step_grader_v1")``.
        """
        filename = f"{name}.yaml"
        with resources.files("inspect_degradation.prompts").joinpath(filename).open(
            "r", encoding="utf-8"
        ) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def render_user(
        self,
        *,
        task_goal: str,
        step_index: int,
        step: TraceStep,
        prior_steps: tuple[TraceStep, ...],
        prior_context_char_budget: int | None = None,
    ) -> tuple[str, "RenderDiagnostics"]:
        """Render the user-message body for one step.

        Only fields the rubric actually references are formatted, so callers
        don't pay serialization cost for unused placeholders.

        Args:
            task_goal: Task instruction string.
            step_index: 0-based index of the step under review.
            step: The current step being graded.
            prior_steps: Steps preceding the one under review, in
                index order.
            prior_context_char_budget: Optional cap on the rendered
                prior-steps block, in characters. When set, the
                renderer keeps the *most recent* prior steps that
                fit and drops older ones. Truncation count is
                returned in the :class:`RenderDiagnostics` so the
                grader can warn or surface a per-run summary; the
                limit is intentionally a soft heuristic (chars, not
                tokens) because the package is provider-agnostic
                and tokenization differs.

                Why this matters: without a cap, very long traces
                push the prior-steps block past the model's context
                window, the provider silently truncates, and later
                step indices end up graded with less context than
                earlier ones — a step-index-correlated grader
                artifact masquerading as degradation. Bounding the
                block here turns silent truncation into a measured,
                uniform truncation we can audit and report.

        Returns:
            ``(rendered_text, diagnostics)``. Diagnostics carries
            the count of dropped prior steps and the rendered char
            length of the prior-steps block.
        """
        values: dict[str, Any] = {}
        diagnostics = RenderDiagnostics()
        if "task_goal" in self.placeholders:
            values["task_goal"] = task_goal
        if "step_index" in self.placeholders:
            values["step_index"] = step_index
        if "step" in self.placeholders:
            values["step"] = _format_step(step)
        if "prior_steps" in self.placeholders:
            kept, n_dropped, rendered = _format_prior_steps_capped(
                prior_steps, char_budget=prior_context_char_budget
            )
            values["prior_steps"] = rendered
            diagnostics.prior_steps_total = len(prior_steps)
            diagnostics.prior_steps_kept = kept
            diagnostics.prior_steps_dropped = n_dropped
            diagnostics.prior_steps_char_length = len(rendered)
            diagnostics.prior_steps_truncated = n_dropped > 0
        return self.user_template.format(**values), diagnostics


def _extract_placeholders(template: str) -> set[str]:
    """Return the set of named ``{field}`` placeholders in a format string.

    Skips positional and empty placeholders (which we forbid by convention).
    """
    found: set[str] = set()
    for _literal, field_name, _spec, _conv in string.Formatter().parse(template):
        if field_name is None:
            continue
        if field_name == "":
            raise ValueError("rubric templates must use named placeholders, not positional ones")
        # Strip attribute/index access ("foo.bar" -> "foo").
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        found.add(root)
    return found


def _format_step(step: TraceStep) -> str:
    parts: list[str] = []
    if step.thought:
        parts.append(f"THOUGHT:\n{step.thought}")
    parts.append(f"ACTION:\n{step.action}")
    if step.observation is not None:
        parts.append(f"OBSERVATION:\n{step.observation}")
    return "\n\n".join(parts)


def _format_prior_steps(prior: tuple[TraceStep, ...]) -> str:
    if not prior:
        return "(none)"
    blocks: list[str] = []
    for s in prior:
        blocks.append(f"--- step {s.index} ---\n{_format_step(s)}")
    return "\n\n".join(blocks)


def _format_prior_steps_capped(
    prior: tuple[TraceStep, ...],
    *,
    char_budget: int | None,
) -> tuple[int, int, str]:
    """Render prior steps under an optional character budget.

    Strategy: keep the *most recent* prior steps (closest to the
    step under review) and drop older ones from the front. The
    most-recent steps carry the most causally-relevant context for
    the grader's per-step judgement, so this is the right end to
    preserve when context is scarce.

    When ``char_budget`` is None, behavior matches
    :func:`_format_prior_steps`. When set, returns ``(kept_count,
    dropped_count, rendered)``. If even a single step exceeds the
    budget, the most-recent one is kept anyway with a leading
    "[earlier steps omitted: budget exhausted]" marker — better to
    over-spend the budget on one step than to render an empty
    prior-steps block at every step in a long trace.
    """
    if not prior:
        return 0, 0, "(none)"

    if char_budget is None:
        return len(prior), 0, _format_prior_steps(prior)

    blocks_reverse: list[str] = []  # most-recent first
    total_chars = 0
    kept = 0
    for s in reversed(prior):
        block = f"--- step {s.index} ---\n{_format_step(s)}"
        # Account for the "\n\n" separator the join will add.
        delta = len(block) + (2 if blocks_reverse else 0)
        if total_chars + delta > char_budget and blocks_reverse:
            break
        blocks_reverse.append(block)
        total_chars += delta
        kept += 1

    n_dropped = len(prior) - kept
    blocks = list(reversed(blocks_reverse))
    if n_dropped > 0:
        marker = f"[earlier {n_dropped} step(s) omitted: prior-context budget reached]"
        blocks.insert(0, marker)
    return kept, n_dropped, "\n\n".join(blocks)
