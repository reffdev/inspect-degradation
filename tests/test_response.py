import json

import pytest

from inspect_degradation.grader.response import (
    GraderResponseError,
    parse_grade_response,
)
from inspect_degradation.schema import (
    ComplexityLevel,
    Dependency,
    SeverityLevel,
    Validity,
)


def _valid_payload(**overrides) -> dict:
    base = {
        "validity": "pass",
        "complexity": "low",
        "dependency": "n/a",
        "severity": None,
        "is_looping": False,
    }
    base.update(overrides)
    return base


def test_parses_bare_json_object():
    text = json.dumps(_valid_payload())
    resp = parse_grade_response(text)
    assert resp.validity == Validity.pass_
    assert resp.dependency == Dependency.not_applicable
    assert resp.complexity == ComplexityLevel.low
    assert resp.is_looping is False


def test_parses_fenced_code_block():
    inner = json.dumps(_valid_payload())
    text = f"Here is the grade:\n```json\n{inner}\n```\nThanks!"
    resp = parse_grade_response(text)
    assert resp.validity == Validity.pass_


def test_parses_object_embedded_in_prose():
    payload = json.dumps(_valid_payload(complexity="medium"))
    text = f"My answer is {payload} and that's that."
    resp = parse_grade_response(text)
    assert resp.complexity == ComplexityLevel.medium


def test_extractor_handles_braces_inside_strings():
    # Direct unit test of the brace-balanced extractor: a string literal
    # containing unmatched braces must not confuse the depth counter.
    # No response field carries arbitrary text anymore so we exercise the
    # extractor at its API boundary rather than through a payload.
    from inspect_degradation.grader.response import _extract_json_blob

    text = 'prose {"a": "weird } brace { inside"} trailing prose'
    blob = _extract_json_blob(text)
    assert json.loads(blob) == {"a": "weird } brace { inside"}


def test_extractor_handles_escaped_quotes_in_strings():
    from inspect_degradation.grader.response import _extract_json_blob

    text = r'{"a": "an \"escaped\" quote"}'
    blob = _extract_json_blob(text)
    assert json.loads(blob) == {"a": 'an "escaped" quote'}


def test_rejects_empty_response():
    with pytest.raises(GraderResponseError, match="empty"):
        parse_grade_response("   ")


def test_rejects_non_object_json():
    with pytest.raises(GraderResponseError, match="object"):
        parse_grade_response("[1, 2, 3]")


def test_rejects_unknown_complexity_label():
    with pytest.raises(GraderResponseError, match="schema"):
        parse_grade_response(json.dumps(_valid_payload(complexity="extreme")))


def test_to_graded_step_requires_severity_for_failure():
    resp = parse_grade_response(
        json.dumps(_valid_payload(validity="fail", dependency="independent"))
    )
    with pytest.raises(GraderResponseError, match="severity"):
        resp.to_graded_step(step_index=2, grader_model="m")


def test_to_graded_step_rejects_severity_on_pass():
    resp = parse_grade_response(json.dumps(_valid_payload(severity="medium")))
    with pytest.raises(GraderResponseError, match="severity"):
        resp.to_graded_step(step_index=2, grader_model="m")


def test_to_graded_step_rejects_dependency_on_non_failure():
    # Non-failures must always carry dependency='n/a'.
    resp = parse_grade_response(
        json.dumps(_valid_payload(dependency="independent"))
    )
    with pytest.raises(GraderResponseError, match="n/a"):
        resp.to_graded_step(step_index=2, grader_model="m")


def test_is_looping_is_required_on_response():
    # Dropping is_looping entirely should fail schema validation.
    import json as _json

    payload = _valid_payload()
    payload.pop("is_looping")
    with pytest.raises(GraderResponseError, match="schema"):
        parse_grade_response(_json.dumps(payload))


def test_is_looping_true_with_pass_is_rejected_at_parse_time():
    import json as _json

    with pytest.raises(GraderResponseError, match="is_looping"):
        resp = parse_grade_response(
            _json.dumps(_valid_payload(is_looping=True))
        )
        resp.to_graded_step(step_index=0, grader_model="m")


def test_is_looping_true_with_neutral_round_trips():
    import json as _json

    resp = parse_grade_response(
        _json.dumps(_valid_payload(validity="neutral", is_looping=True))
    )
    step = resp.to_graded_step(step_index=1, grader_model="m")
    assert step.is_looping is True
    assert step.validity.value == "neutral"


def test_to_graded_step_happy_path():
    resp = parse_grade_response(
        json.dumps(
            _valid_payload(
                validity="fail",
                severity="high",
                dependency="dependent",
            )
        )
    )
    step = resp.to_graded_step(step_index=3, grader_model="m")
    assert step.step_index == 3
    assert step.severity == SeverityLevel.high
    assert step.dependency == Dependency.dependent
    assert step.grader_model == "m"
