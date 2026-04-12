"""Falsification tests for grader position and task invariance."""

from __future__ import annotations

import pytest

from inspect_degradation.schema import (
    ComplexityLevel,
    Dependency,
    GradedStep,
    GradedTrace,
    SeverityLevel,
    Validity,
)
from inspect_degradation.validation.invariance import (
    InvarianceReport,
    position_invariance_test,
    task_invariance_test,
)


def _step(index: int, validity: Validity = Validity.pass_) -> GradedStep:
    return GradedStep(
        step_index=index,
        validity=validity,
        complexity=ComplexityLevel.low,
        dependency=Dependency.not_applicable,
        severity=(
            SeverityLevel.low if validity == Validity.fail else None
        ),
        is_looping=False,
        grader_model="test",
    )


def _make_corpus(n_traces: int = 4, n_steps: int = 6) -> list[GradedTrace]:
    traces: list[GradedTrace] = []
    for t in range(n_traces):
        traces.append(
            GradedTrace(
                trace_id=f"t{t}",
                task_id=f"task{t}",
                model="m",
                steps=[_step(i) for i in range(n_steps)],
            )
        )
    return traces


class TestPositionInvariance:
    def test_stable_grader_reports_zero_disagreement(self):
        # Identity regrade → zero disagreement by construction.
        traces = _make_corpus()
        report = position_invariance_test(
            traces,
            regrade_fn=lambda step, new_pos, new_task: step,
            sample_size=12,
            seed=7,
        )
        assert isinstance(report, InvarianceReport)
        assert report.name == "position"
        assert report.n_disagreements == 0
        assert report.disagreement_rate.value == 0.0
        assert report.n_steps_tested == 12

    def test_flipping_grader_reports_full_disagreement(self):
        traces = _make_corpus()

        def flip(step: GradedStep, new_pos: int, new_task: str) -> GradedStep:
            return _step(new_pos, validity=Validity.fail)

        report = position_invariance_test(
            traces,
            regrade_fn=flip,
            sample_size=8,
            seed=2,
        )
        assert report.n_disagreements == 8
        assert report.disagreement_rate.value == 1.0

    def test_sample_size_capped_by_pool(self):
        traces = _make_corpus(n_traces=1, n_steps=3)
        report = position_invariance_test(
            traces,
            regrade_fn=lambda s, p, t: s,
            sample_size=100,
            seed=0,
        )
        assert report.n_steps_tested == 3

    def test_details_record_every_probe(self):
        traces = _make_corpus()
        report = position_invariance_test(
            traces,
            regrade_fn=lambda s, p, t: s,
            sample_size=5,
            seed=3,
        )
        assert len(report.details) == 5
        for row in report.details:
            assert {"trace_id", "step_index", "new_position", "disagreed"} <= set(row)


class TestTaskInvariance:
    def test_requires_at_least_two_tasks(self):
        trace = GradedTrace(
            trace_id="only",
            task_id="solo",
            steps=[_step(0)],
        )
        with pytest.raises(ValueError, match="task_id"):
            task_invariance_test(
                [trace],
                regrade_fn=lambda s, p, t: s,
                sample_size=1,
            )

    def test_identity_regrade_zero_disagreement(self):
        traces = _make_corpus()
        report = task_invariance_test(
            traces,
            regrade_fn=lambda s, p, t: s,
            sample_size=10,
            seed=1,
        )
        assert report.name == "task"
        assert report.n_disagreements == 0

    def test_detail_records_swapped_task(self):
        traces = _make_corpus()
        report = task_invariance_test(
            traces,
            regrade_fn=lambda s, p, t: s,
            sample_size=4,
            seed=9,
        )
        for row in report.details:
            assert row["new_task_id"] != row["original_task_id"]

    def test_to_dict_is_json_safe(self):
        import json

        traces = _make_corpus()
        report = task_invariance_test(
            traces,
            regrade_fn=lambda s, p, t: s,
            sample_size=3,
            seed=4,
        )
        json.dumps(report.to_dict())
