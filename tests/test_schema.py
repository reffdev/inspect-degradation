from conftest import make_graded_step, make_graded_trace

from inspect_degradation.schema import (
    ComplexityLevel,
    Dependency,
    GradedStep,
    SeverityLevel,
    Validity,
)


def test_graded_step_roundtrip():
    s = make_graded_step(0)
    assert GradedStep.model_validate_json(s.model_dump_json()) == s


def test_graded_trace_holds_steps():
    t = make_graded_trace(steps=[make_graded_step(i) for i in range(3)])
    assert len(t.steps) == 3
    assert all(isinstance(s, GradedStep) for s in t.steps)


def test_validity_enum_values_are_human_readable():
    # The rubric and the validation harness rely on the enum *values*, not
    # the member names — pin them so a rename can't silently break either.
    assert Validity.pass_.value == "pass"
    assert Validity.fail.value == "fail"
    assert Validity.neutral.value == "neutral"
    assert {v.value for v in Validity} == {"fail", "neutral", "pass"}
    assert Dependency.not_applicable.value == "n/a"


def test_ordinal_levels_have_consistent_ordering():
    assert ComplexityLevel.low < ComplexityLevel.medium < ComplexityLevel.high
    assert SeverityLevel.low < SeverityLevel.medium < SeverityLevel.high


def test_ordinal_levels_are_type_distinct():
    # We deliberately do not allow comparing across distinct ordinal types;
    # the type system should reject mixing severity and complexity even
    # though both happen to use {low, medium, high} values.
    result = ComplexityLevel.low.__lt__(SeverityLevel.high)  # type: ignore[arg-type]
    assert result is NotImplemented


def test_severity_required_only_when_failing():
    # Pass step with severity is rejected.
    import pytest
    with pytest.raises(Exception):
        GradedStep(
            step_index=0,
            validity=Validity.pass_,
            severity=SeverityLevel.low,
            grader_model="t",
        )
