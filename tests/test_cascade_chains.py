from inspect_degradation.analysis.cascade_chains import cascade_chain_lengths
from inspect_degradation.schema import (
    ComplexityLevel,
    Dependency,
    GradedStep,
    GradedTrace,
    SeverityLevel,
    Validity,
)


def _step(i, validity, dep=Dependency.not_applicable):
    return GradedStep(
        step_index=i,
        validity=validity,
        complexity=ComplexityLevel.medium,
        dependency=dep,
        severity=SeverityLevel.medium if validity == Validity.fail else None,
        grader_model="test",
    )


def test_chain_lengths_simple_cascade():
    trace = GradedTrace(
        trace_id="t",
        steps=[
            _step(0, Validity.pass_),
            _step(1, Validity.fail, Dependency.independent),
            _step(2, Validity.fail, Dependency.dependent),
            _step(3, Validity.fail, Dependency.dependent),
            _step(4, Validity.pass_),
        ],
    )
    assert cascade_chain_lengths([trace]) == [3]


def test_chain_lengths_multiple_chains():
    trace = GradedTrace(
        trace_id="t",
        steps=[
            _step(0, Validity.fail, Dependency.independent),
            _step(1, Validity.pass_),
            _step(2, Validity.fail, Dependency.independent),
            _step(3, Validity.fail, Dependency.dependent),
        ],
    )
    assert cascade_chain_lengths([trace]) == [1, 2]
