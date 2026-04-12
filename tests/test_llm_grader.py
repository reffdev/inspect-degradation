"""Tests for :class:`LLMGrader` — config validation and the
self-consistency sampling path.

We monkey-patch ``_generate_with_retry`` to return canned JSON strings
so these tests exercise the full parse → response → majority-vote
pipeline without calling any real model.
"""

from __future__ import annotations

import asyncio
import itertools
import json

import pytest
from conftest import make_trace

from inspect_degradation.grader.interface import StepContext
from inspect_degradation.grader.llm import (
    SELF_CONSISTENCY_KEY,
    LLMGrader,
    LLMGraderConfig,
)
from inspect_degradation.grader.rubric import Rubric
from inspect_degradation.schema import Validity


def _rubric() -> Rubric:
    return Rubric.from_package("step_grader_v1")


def _ctx(step_index: int = 0) -> StepContext:
    trace = make_trace(n_steps=step_index + 1)
    return StepContext(
        task_goal=trace.task_goal,
        step_index=step_index,
        step=trace.steps[step_index],
        prior_steps=trace.steps[:step_index],
        trace_id=trace.trace_id,
    )


def _valid_grade_json(
    *,
    validity: str = "pass",
    dependency: str = "n/a",
    severity: str | None = None,
    is_looping: bool = False,
) -> str:
    return json.dumps(
        {
            "validity": validity,
            "complexity": "low",
            "dependency": dependency,
            "severity": severity,
            "is_looping": is_looping,
        }
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_sample_n_must_be_positive():
    with pytest.raises(ValueError, match="sample_n"):
        LLMGraderConfig(model="m", sample_n=0)


def test_sample_n_greater_than_one_requires_positive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        LLMGraderConfig(model="m", sample_n=3, temperature=0.0)


def test_sample_n_one_allows_temperature_zero():
    # No exception — single-sample grading at temperature 0 is the
    # standard deterministic mode.
    cfg = LLMGraderConfig(model="m", sample_n=1, temperature=0.0)
    assert cfg.sample_n == 1
    assert cfg.temperature == 0.0


def test_sample_n_greater_than_one_with_positive_temperature_ok():
    cfg = LLMGraderConfig(model="m", sample_n=3, temperature=0.7)
    assert cfg.sample_n == 3


# ---------------------------------------------------------------------------
# Single-sample grading path
# ---------------------------------------------------------------------------


def _patch_completions(grader: LLMGrader, completions: list[str]) -> list[int]:
    """Monkey-patch grader._generate_and_parse_with_retry to yield canned strings.

    Returns a counter list that the test can inspect to see how many
    times generate was invoked.
    """
    from inspect_degradation.grader.response import parse_grade_response

    counter = [0]
    iterator = iter(completions)

    async def fake_generate(prompt: str):
        counter[0] += 1
        try:
            text = next(iterator)
        except StopIteration:
            raise AssertionError(
                f"fake grader ran out of canned completions at call {counter[0]}"
            )
        return text, parse_grade_response(text)

    grader._generate_and_parse_with_retry = fake_generate  # type: ignore[method-assign]
    return counter


@pytest.mark.asyncio
async def test_single_sample_grading_produces_step_from_completion():
    grader = LLMGrader(
        config=LLMGraderConfig(model="m", sample_n=1, temperature=0.0),
        rubric=_rubric(),
    )
    calls = _patch_completions(grader, [_valid_grade_json(validity="pass")])
    step = await grader.grade_step(_ctx(0))
    assert calls[0] == 1
    assert step.validity == Validity.pass_
    assert SELF_CONSISTENCY_KEY not in (step.raw or {})


@pytest.mark.asyncio
async def test_single_sample_parse_failure_returns_fallback_step():
    """When parsing irrecoverably fails, the grader returns a neutral
    step rather than crashing the trace. This is the bug the smoke
    test surfaced when a real provider returned malformed JSON."""
    from inspect_degradation.grader.response import GraderResponseError
    from inspect_degradation.schema import Validity

    grader = LLMGrader(
        config=LLMGraderConfig(model="m", sample_n=1, temperature=0.0),
        rubric=_rubric(),
    )

    async def always_fail(prompt: str):
        raise GraderResponseError("unbalanced JSON object", raw="{ broken")

    grader._generate_and_parse_with_retry = always_fail  # type: ignore[method-assign]

    step = await grader.grade_step(_ctx(0))
    assert step.validity == Validity.neutral
    assert step.grader_model == "m"
    assert step.raw is not None
    assert "parse_error" in step.raw
    assert "unbalanced JSON" in step.raw["parse_error"]


@pytest.mark.asyncio
async def test_self_consistency_all_failures_return_fallback_step():
    """If every self-consistency sample fails parsing, fall back to a
    single neutral step rather than crashing the trace."""
    from inspect_degradation.grader.response import GraderResponseError
    from inspect_degradation.schema import Validity

    grader = LLMGrader(
        config=LLMGraderConfig(model="m", sample_n=3, temperature=0.7),
        rubric=_rubric(),
    )

    async def always_fail(prompt: str):
        raise GraderResponseError("unbalanced JSON object", raw="{ broken")

    grader._generate_and_parse_with_retry = always_fail  # type: ignore[method-assign]

    step = await grader.grade_step(_ctx(0))
    assert step.validity == Validity.neutral
    assert step.raw is not None
    assert "parse_error" in step.raw


@pytest.mark.asyncio
async def test_single_sample_name_does_not_carry_sc_suffix():
    grader = LLMGrader(
        config=LLMGraderConfig(model="anthropic/claude-x", sample_n=1),
        rubric=_rubric(),
    )
    assert grader.name == "anthropic/claude-x"


# ---------------------------------------------------------------------------
# Self-consistency grading path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_consistency_unanimous_samples_produce_unanimous_metadata():
    grader = LLMGrader(
        config=LLMGraderConfig(model="m", sample_n=3, temperature=0.7),
        rubric=_rubric(),
    )
    _patch_completions(
        grader,
        [
            _valid_grade_json(validity="pass"),
            _valid_grade_json(validity="pass"),
            _valid_grade_json(validity="pass"),
        ],
    )
    step = await grader.grade_step(_ctx(0))
    sc = step.raw[SELF_CONSISTENCY_KEY]
    assert sc["n_samples"] == 3
    assert sc["unanimous"] is True
    assert sc["validity_counts"] == {"pass": 3}
    assert step.validity == Validity.pass_


@pytest.mark.asyncio
async def test_self_consistency_split_samples_return_majority_verdict():
    grader = LLMGrader(
        config=LLMGraderConfig(model="m", sample_n=3, temperature=0.7),
        rubric=_rubric(),
    )
    _patch_completions(
        grader,
        [
            _valid_grade_json(validity="pass"),
            _valid_grade_json(validity="neutral"),
            _valid_grade_json(validity="pass"),
        ],
    )
    step = await grader.grade_step(_ctx(0))
    sc = step.raw[SELF_CONSISTENCY_KEY]
    assert sc["unanimous"] is False
    assert sc["validity_counts"] == {"pass": 2, "neutral": 1}
    # Majority wins on validity.
    assert step.validity == Validity.pass_


@pytest.mark.asyncio
async def test_self_consistency_captures_all_sample_completions():
    grader = LLMGrader(
        config=LLMGraderConfig(model="m", sample_n=2, temperature=0.7),
        rubric=_rubric(),
    )
    completions = [
        _valid_grade_json(validity="pass"),
        _valid_grade_json(validity="neutral"),
    ]
    _patch_completions(grader, list(completions))
    step = await grader.grade_step(_ctx(0))
    sc = step.raw[SELF_CONSISTENCY_KEY]
    assert sc["sample_completions"] == completions
    assert sc["sample_validities"] == ["pass", "neutral"]


@pytest.mark.asyncio
async def test_self_consistency_makes_n_model_calls():
    grader = LLMGrader(
        config=LLMGraderConfig(model="m", sample_n=4, temperature=0.5),
        rubric=_rubric(),
    )
    calls = _patch_completions(
        grader,
        [_valid_grade_json(validity="pass")] * 4,
    )
    await grader.grade_step(_ctx(0))
    assert calls[0] == 4


@pytest.mark.asyncio
async def test_self_consistency_name_carries_sample_suffix():
    grader = LLMGrader(
        config=LLMGraderConfig(model="anthropic/claude-x", sample_n=3, temperature=0.7),
        rubric=_rubric(),
    )
    assert grader.name == "anthropic/claude-x@sc3"


@pytest.mark.asyncio
async def test_grade_trace_with_self_consistency_preserves_step_order():
    # Each step needs sample_n completions; 3 steps × 3 samples = 9 completions.
    grader = LLMGrader(
        config=LLMGraderConfig(model="m", sample_n=3, temperature=0.7),
        rubric=_rubric(),
    )
    _patch_completions(grader, [_valid_grade_json(validity="pass")] * 9)
    trace = make_trace(n_steps=3)
    grades = await grader.grade_trace(trace)
    assert [g.step_index for g in grades] == [0, 1, 2]
    for g in grades:
        assert g.raw[SELF_CONSISTENCY_KEY]["n_samples"] == 3
