"""Tests for :class:`EnsembleGrader`.

Uses fake member graders so the tests run fast and have no LLM
dependency. The fakes let us pin voting behavior precisely: who wins
on unanimous, split, and tied inputs; what provenance is preserved;
that per-member concurrency is bounded; that single-member ensembles
act as the identity.
"""

import asyncio

import pytest
from conftest import make_trace

from inspect_degradation.grader.ensemble import ENSEMBLE_KEY, EnsembleGrader
from inspect_degradation.grader.interface import Grader, StepContext
from inspect_degradation.schema import (
    ComplexityLevel,
    Dependency,
    GradedStep,
    SeverityLevel,
    Validity,
)


class _StubGrader(Grader):
    """Always returns a fixed validity with a per-instance call counter."""

    def __init__(
        self,
        name: str,
        *,
        validity: Validity,
        severity: SeverityLevel | None = None,
        dependency: Dependency = Dependency.not_applicable,
        is_looping: bool = False,
        delay: float = 0.0,
    ):
        self._name = name
        self._validity = validity
        self._severity = severity
        self._dependency = dependency
        self._is_looping = is_looping
        self._delay = delay
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def grade_step(self, ctx: StepContext) -> GradedStep:
        self.call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        return GradedStep(
            step_index=ctx.step_index,
            validity=self._validity,
            complexity=ComplexityLevel.medium,
            dependency=(
                self._dependency
                if self._validity == Validity.fail
                else Dependency.not_applicable
            ),
            severity=self._severity if self._validity == Validity.fail else None,
            is_looping=self._is_looping,
            grader_model=self._name,
        )


# ---------------------------------------------------------------------------
# Construction invariants
# ---------------------------------------------------------------------------


def test_empty_members_rejected():
    with pytest.raises(ValueError, match="at least one"):
        EnsembleGrader([])


def test_duplicate_instance_rejected():
    g = _StubGrader("g", validity=Validity.pass_)
    with pytest.raises(ValueError, match="distinct"):
        EnsembleGrader([g, g])


def test_distinct_graders_with_same_name_accepted():
    # Two separate Grader objects that happen to share a name (e.g. two
    # configurations of the same model family) are legal — the ensemble
    # only forbids the same python object appearing twice.
    g1 = _StubGrader("m", validity=Validity.pass_)
    g2 = _StubGrader("m", validity=Validity.pass_)
    EnsembleGrader([g1, g2])  # no raise


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unanimous_ensemble_returns_unanimous_provenance():
    members = [
        _StubGrader("a", validity=Validity.pass_),
        _StubGrader("b", validity=Validity.pass_),
        _StubGrader("c", validity=Validity.pass_),
    ]
    ensemble = EnsembleGrader(members)
    grade = await ensemble.grade_step(
        StepContext(
            task_goal="g",
            step_index=0,
            step=make_trace(n_steps=1).steps[0],
            prior_steps=(),
            trace_id="t",
        )
    )
    assert grade.validity == Validity.pass_
    assert all(m.call_count == 1 for m in members)

    provenance = grade.raw[ENSEMBLE_KEY]
    assert provenance["n_members"] == 3
    assert provenance["unanimous"] is True
    assert provenance["modal_validity"] == "pass"
    assert provenance["validity_counts"] == {"pass": 3}
    assert len(provenance["member_grades"]) == 3


@pytest.mark.asyncio
async def test_majority_vote_picks_modal_validity():
    members = [
        _StubGrader("a", validity=Validity.pass_),
        _StubGrader("b", validity=Validity.neutral),
        _StubGrader("c", validity=Validity.pass_),
    ]
    ensemble = EnsembleGrader(members)
    grade = await ensemble.grade_step(
        StepContext(
            task_goal="g",
            step_index=0,
            step=make_trace(n_steps=1).steps[0],
            prior_steps=(),
            trace_id="t",
        )
    )
    assert grade.validity == Validity.pass_
    provenance = grade.raw[ENSEMBLE_KEY]
    assert provenance["unanimous"] is False
    assert provenance["validity_counts"] == {"pass": 2, "neutral": 1}


@pytest.mark.asyncio
async def test_tie_break_returns_first_member_in_construction_order():
    # Two-member tie: each member votes differently. The winner must
    # be the grade from the first member in construction order.
    members = [
        _StubGrader("first", validity=Validity.neutral),
        _StubGrader("second", validity=Validity.pass_),
    ]
    ensemble = EnsembleGrader(members)
    grade = await ensemble.grade_step(
        StepContext(
            task_goal="g",
            step_index=0,
            step=make_trace(n_steps=1).steps[0],
            prior_steps=(),
            trace_id="t",
        )
    )
    # Modal validity is "neutral" (first one Counter saw) — but the
    # important thing is that the result is deterministic.
    assert grade.validity == Validity.neutral


@pytest.mark.asyncio
async def test_ensemble_grader_model_reflects_ensemble_name():
    members = [
        _StubGrader("haiku", validity=Validity.pass_),
        _StubGrader("sonnet", validity=Validity.pass_),
    ]
    ensemble = EnsembleGrader(members)
    grade = await ensemble.grade_step(
        StepContext(
            task_goal="g",
            step_index=0,
            step=make_trace(n_steps=1).steps[0],
            prior_steps=(),
            trace_id="t",
        )
    )
    assert grade.grader_model == "ensemble(haiku+sonnet)"


@pytest.mark.asyncio
async def test_single_member_ensemble_acts_as_identity():
    member = _StubGrader("solo", validity=Validity.pass_)
    ensemble = EnsembleGrader([member])
    grade = await ensemble.grade_step(
        StepContext(
            task_goal="g",
            step_index=0,
            step=make_trace(n_steps=1).steps[0],
            prior_steps=(),
            trace_id="t",
        )
    )
    assert grade.validity == Validity.pass_
    prov = grade.raw[ENSEMBLE_KEY]
    assert prov["unanimous"] is True
    assert prov["n_members"] == 1


@pytest.mark.asyncio
async def test_ensemble_preserves_member_grade_details_in_provenance():
    members = [
        _StubGrader(
            "a",
            validity=Validity.fail,
            severity=SeverityLevel.high,
            dependency=Dependency.independent,
        ),
        _StubGrader("b", validity=Validity.neutral),
    ]
    ensemble = EnsembleGrader(members)
    grade = await ensemble.grade_step(
        StepContext(
            task_goal="g",
            step_index=0,
            step=make_trace(n_steps=1).steps[0],
            prior_steps=(),
            trace_id="t",
        )
    )
    prov = grade.raw[ENSEMBLE_KEY]
    by_name = {m["grader_model"]: m for m in prov["member_grades"]}
    assert by_name["a"]["validity"] == "fail"
    assert by_name["a"]["severity"] == "high"
    assert by_name["a"]["dependency"] == "independent"
    assert by_name["b"]["validity"] == "neutral"
    assert by_name["b"]["severity"] is None


# ---------------------------------------------------------------------------
# Concurrency and naming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_members_run_concurrently_not_serially():
    # If members ran serially the total time would be ~3 × 0.03 = 0.09s;
    # concurrent should be ~0.03s. Give a generous 0.06s budget.
    import time

    members = [
        _StubGrader("a", validity=Validity.pass_, delay=0.03),
        _StubGrader("b", validity=Validity.pass_, delay=0.03),
        _StubGrader("c", validity=Validity.pass_, delay=0.03),
    ]
    ensemble = EnsembleGrader(members)
    start = time.perf_counter()
    await ensemble.grade_step(
        StepContext(
            task_goal="g",
            step_index=0,
            step=make_trace(n_steps=1).steps[0],
            prior_steps=(),
            trace_id="t",
        )
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 0.06, f"members ran serially: {elapsed:.3f}s"


def test_max_concurrency_is_min_of_members():
    class _CapGrader(_StubGrader):
        def __init__(self, name: str, cap: int):
            super().__init__(name, validity=Validity.pass_)
            self._cap = cap

        @property
        def max_concurrency(self) -> int:  # type: ignore[override]
            return self._cap

    ensemble = EnsembleGrader([_CapGrader("a", 16), _CapGrader("b", 4), _CapGrader("c", 8)])
    assert ensemble.max_concurrency == 4


def test_explicit_name_overrides_auto_generated():
    members = [
        _StubGrader("a", validity=Validity.pass_),
        _StubGrader("b", validity=Validity.pass_),
    ]
    ensemble = EnsembleGrader(members, name="production")
    assert ensemble.name == "production"


def test_snapshot_nests_member_snapshots():
    members = [
        _StubGrader("a", validity=Validity.pass_),
        _StubGrader("b", validity=Validity.pass_),
    ]
    ensemble = EnsembleGrader(members)
    snap = ensemble.snapshot()
    assert snap.kind == "ensemble"
    assert snap.fields["n_members"] == 2
    assert set(snap.children) == {"member_0", "member_1"}


@pytest.mark.asyncio
async def test_grade_trace_preserves_order_with_ensemble():
    members = [
        _StubGrader("a", validity=Validity.pass_),
        _StubGrader("b", validity=Validity.pass_),
    ]
    ensemble = EnsembleGrader(members)
    trace = make_trace(n_steps=4)
    grades = await ensemble.grade_trace(trace)
    assert [g.step_index for g in grades] == [0, 1, 2, 3]
    # Each of the 2 members graded each of the 4 steps.
    assert all(m.call_count == 4 for m in members)
