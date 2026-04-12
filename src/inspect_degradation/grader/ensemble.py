"""Ensemble grader: run N independent graders per step, vote on the result.

The ensemble is the package's primary mechanism for handling grader
uncertainty. It runs each step through multiple independent grader
instances — typically across different model families (Haiku + Sonnet +
GPT-4o-mini, say) — and reduces them to a single verdict by majority
vote on the ``validity`` label.

Why this is the primary path instead of a cascade:

* **Direct uncertainty signal.** Inter-grader disagreement is a
  well-supported signal in the LLM-judge literature (Wang et al. 2022
  on self-consistency; Kadavath et al. 2022 on sampled-answer
  consistency). Cross-model disagreement is stronger still because the
  errors of different model families are less correlated than
  errors within a single family.
* **No unvalidated assumption.** A cascade assumes that the cheap
  grader's uncertainty signal reliably identifies the same cases the
  expensive grader would get wrong. That assumption is not validated
  in the literature for the cascade-with-LLM-judge setting; Kapoor et
  al. 2024 specifically tested it and found it unreliable. The
  ensemble does not require the assumption at all — every step gets
  every grader's verdict.
* **Provider diversity.** The architecture does not privilege any
  single provider. An Anthropic + OpenAI + Google ensemble is as
  legitimate a configuration as an all-Anthropic one.

The single-model self-consistency primitive on :class:`LLMGrader`
(``sample_n > 1``) is still useful as a primitive *inside* an ensemble
— ``EnsembleGrader([haiku@sc3, sonnet@sc3])`` runs six total calls per
step, three Haiku samples and three Sonnet samples, and votes on the
aggregate. Single-grader ensembles are also legal (and act as the
identity), which makes the ensemble the natural home for any
multi-sample composition.

Voting rule
-----------

We majority-vote on **validity** and return the grade from the first
ensemble member whose validity matches the modal label. Whole-sample
selection rather than per-field voting is deliberate: per-field voting
can produce incoherent cross-field states (majority validity is
``fail`` but majority dependency is ``n/a``), and the cross-field
invariants in :class:`~inspect_degradation.schema.GradedStep` would
reject that anyway. The full per-grader provenance is preserved under
``raw["ensemble"]`` so post-hoc analysis can re-vote, inspect
disagreements, or bucket results by agreement level.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Any

from inspect_degradation.grader.interface import Grader, GraderSnapshot, StepContext
from inspect_degradation.schema import GradedStep


#: Metadata key stashed on returned :class:`GradedStep` instances.
#: Contains per-grader detail so downstream analysis can re-vote,
#: bucket by agreement level, or compute grader-specific calibration
#: without re-running the ensemble.
ENSEMBLE_KEY = "ensemble"


@dataclass(frozen=True)
class EnsembleMemberGrade:
    """One member's grade within an ensemble result, for provenance."""

    grader_name: str
    validity: str
    grade: GradedStep


class EnsembleGrader(Grader):
    """Compose N graders by running each per step and majority-voting.

    Members run concurrently within each step (each step's grading pass
    is itself an asyncio gather), so the ensemble's per-step latency is
    dominated by its slowest member rather than the sum. Member-level
    concurrency is bounded by each member's own ``max_concurrency``.

    The ensemble's :attr:`max_concurrency` is the **minimum** of its
    members' — running N traces in flight at once means running N × M
    underlying calls, so the tightest member caps the whole group.
    """

    def __init__(
        self,
        members: list[Grader],
        *,
        name: str | None = None,
    ) -> None:
        if not members:
            raise ValueError("EnsembleGrader requires at least one member grader")
        # Duplicate-instance check: this almost certainly means the
        # caller forgot to construct a second grader and passed the
        # same one twice. Silent acceptance would skew the vote.
        if len(set(id(m) for m in members)) != len(members):
            raise ValueError(
                "EnsembleGrader members must be distinct instances; "
                "pass separate Grader objects even when they share a model"
            )
        self._members = list(members)
        self._explicit_name = name

    @property
    def name(self) -> str:
        if self._explicit_name is not None:
            return self._explicit_name
        member_names = "+".join(m.name for m in self._members)
        return f"ensemble({member_names})"

    @property
    def max_concurrency(self) -> int:  # type: ignore[override]
        return min(m.max_concurrency for m in self._members)

    def snapshot(self) -> GraderSnapshot:
        return GraderSnapshot(
            kind="ensemble",
            name=self.name,
            fields={"n_members": len(self._members)},
            children={
                f"member_{i}": m.snapshot() for i, m in enumerate(self._members)
            },
        )

    async def grade_step(self, ctx: StepContext) -> GradedStep:
        # Fan out: every member grades the step independently.
        member_grades = await asyncio.gather(
            *(m.grade_step(ctx) for m in self._members)
        )
        return self._vote(member_grades)

    # ------------------------------------------------------------------ internals

    def _vote(self, member_grades: list[GradedStep]) -> GradedStep:
        """Majority-vote on validity and return the winner with provenance.

        Tie-break (when multiple validity labels share the modal count):
        return the grade from the first member, in construction order,
        whose validity is one of the tied labels. This is deterministic
        and preferences stability of results across reruns over any
        particular "fairness" notion.
        """
        counts = Counter(g.validity.value for g in member_grades)
        modal_validity, modal_count = counts.most_common(1)[0]
        unanimous = len(counts) == 1

        # Find the first member whose validity equals the modal label.
        chosen: GradedStep | None = None
        for member_grade, member in zip(member_grades, self._members):
            if member_grade.validity.value == modal_validity:
                chosen = member_grade
                break
        if chosen is None:
            # Unreachable: modal label must appear in the grade list.
            raise RuntimeError("ensemble vote failed to find modal grade")  # pragma: no cover

        provenance: dict[str, Any] = {
            "n_members": len(self._members),
            "unanimous": unanimous,
            "modal_validity": modal_validity,
            "modal_count": modal_count,
            "validity_counts": dict(counts),
            "member_grades": [
                {
                    "grader_model": g.grader_model,
                    "validity": g.validity.value,
                    "complexity": g.complexity.value if g.complexity else None,
                    "dependency": g.dependency.value if g.dependency else None,
                    "severity": g.severity.value if g.severity else None,
                    "is_looping": g.is_looping,
                }
                for g in member_grades
            ],
        }

        merged_raw: dict[str, Any] = dict(chosen.raw or {})
        merged_raw[ENSEMBLE_KEY] = provenance
        return chosen.model_copy(
            update={
                "grader_model": self.name,
                "raw": merged_raw,
            }
        )


__all__ = ["ENSEMBLE_KEY", "EnsembleGrader", "EnsembleMemberGrade"]
