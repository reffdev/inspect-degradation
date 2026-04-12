import pytest
from conftest import make_step, make_trace

from inspect_degradation.trace import Trace, TraceStep


def test_prior_returns_strict_prefix():
    trace = make_trace(n_steps=4)
    assert trace.prior(0) == ()
    assert trace.prior(2) == tuple(make_step(i) for i in range(2))
    assert trace.prior(4) == trace.steps


def test_prior_rejects_out_of_range():
    trace = make_trace(n_steps=2)
    with pytest.raises(IndexError):
        trace.prior(-1)
    with pytest.raises(IndexError):
        trace.prior(3)


def test_trace_is_frozen():
    trace = make_trace()
    with pytest.raises(Exception):
        trace.trace_id = "other"  # type: ignore[misc]


def test_step_requires_action():
    with pytest.raises(Exception):
        TraceStep(index=0, action=None)  # type: ignore[arg-type]
