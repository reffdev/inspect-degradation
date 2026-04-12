"""Tests for the loop-analysis module and the is_looping cross-field invariant."""

import pytest
from conftest import make_graded_step, make_graded_trace

from inspect_degradation.analysis.loops import loop_chain_lengths, raw_loop_rate
from inspect_degradation.schema import GradedStep, Validity


# ---------------------------------------------------------------------------
# Schema cross-field invariant
# ---------------------------------------------------------------------------


def test_is_looping_true_with_pass_is_rejected():
    with pytest.raises(Exception, match="is_looping"):
        GradedStep(
            step_index=0,
            validity=Validity.pass_,
            is_looping=True,
            grader_model="t",
        )


def test_is_looping_true_with_neutral_is_allowed():
    step = GradedStep(
        step_index=0,
        validity=Validity.neutral,
        is_looping=True,
        grader_model="t",
    )
    assert step.is_looping is True


def test_is_looping_true_with_fail_is_allowed():
    # An agent repeatedly making the same wrong choice — valid combination.
    step = GradedStep(
        step_index=0,
        validity=Validity.fail,
        is_looping=True,
        grader_model="t",
    )
    assert step.is_looping is True


def test_is_looping_none_is_allowed_on_any_validity():
    # Partial references (TRAIL) do not label loops; None must round-trip.
    for v in (Validity.pass_, Validity.neutral, Validity.fail):
        step = GradedStep(
            step_index=0,
            validity=v,
            is_looping=None,
            grader_model="t",
        )
        assert step.is_looping is None


# ---------------------------------------------------------------------------
# loop_chain_lengths
# ---------------------------------------------------------------------------


def _trace_with_loop_flags(flags):
    """Build a trace with one step per entry in ``flags``.

    ``flags`` values:
      - True  : validity=neutral, is_looping=True
      - False : validity=pass, is_looping=False
      - None  : validity=neutral, is_looping=None (unlabeled)
    """
    steps = []
    for i, flag in enumerate(flags):
        if flag is True:
            steps.append(
                make_graded_step(i, validity=Validity.neutral, is_looping=True)
            )
        elif flag is False:
            steps.append(
                make_graded_step(i, validity=Validity.pass_, is_looping=False)
            )
        else:
            steps.append(
                make_graded_step(i, validity=Validity.neutral, is_looping=None)
            )
    return make_graded_trace(steps=steps)


def test_single_loop_step_contributes_chain_of_one():
    trace = _trace_with_loop_flags([False, True, False])
    assert loop_chain_lengths([trace]) == [1]


def test_contiguous_loop_steps_contribute_one_chain():
    trace = _trace_with_loop_flags([False, True, True, True, False])
    assert loop_chain_lengths([trace]) == [3]


def test_multiple_chains_per_trace():
    trace = _trace_with_loop_flags([True, False, True, True, False, True])
    assert loop_chain_lengths([trace]) == [1, 2, 1]


def test_chain_at_end_of_trace_is_counted():
    # Unlike failure runs (right-censored), loop runs at trace end are counted.
    trace = _trace_with_loop_flags([False, True, True])
    assert loop_chain_lengths([trace]) == [2]


def test_unlabeled_steps_break_chains():
    trace = _trace_with_loop_flags([True, None, True])
    # The None step terminates the first chain and does not start a new one;
    # the trailing True is its own chain.
    assert loop_chain_lengths([trace]) == [1, 1]


def test_empty_and_all_unlabeled_traces_produce_no_chains():
    assert loop_chain_lengths([]) == []
    unlabeled = _trace_with_loop_flags([None, None])
    assert loop_chain_lengths([unlabeled]) == []


# ---------------------------------------------------------------------------
# loop_rate
# ---------------------------------------------------------------------------


def test_loop_rate_ignores_unlabeled_steps():
    trace = _trace_with_loop_flags([True, False, None, True])
    # 2 looping / 3 labeled = 2/3
    assert raw_loop_rate([trace]) == pytest.approx(2 / 3)


def test_loop_rate_returns_none_when_nothing_labeled():
    trace = _trace_with_loop_flags([None, None, None])
    assert raw_loop_rate([trace]) is None


def test_loop_rate_zero_when_all_labeled_and_none_loop():
    trace = _trace_with_loop_flags([False, False, False])
    assert raw_loop_rate([trace]) == 0.0
