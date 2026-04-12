"""Tests for the bootstrap-CI variants of the chain-length mean metrics.

The existing ``test_cascade_chains.py`` and ``test_loops.py`` cover
the scalar chain-length list computation. This file covers the new
uncertainty-bearing variants:

* ``cascade_chain_length_mean_estimate`` — bootstrap CI on mean
  dependent-error chain length.
* ``mean_failing_run_length_estimate`` — bootstrap CI on mean
  failing-step-run length.
* ``loop_chain_length_mean_estimate`` — bootstrap CI on mean loop-run
  length.

All three share the same pattern: trace-level resampling with the
existing scalar statistic as the bootstrap inner function. Tests pin
point-value agreement with the scalar computation, CI bracketing,
and degenerate-input handling.
"""

from __future__ import annotations

import math

from conftest import make_graded_step, make_graded_trace
from numpy.random import default_rng

from inspect_degradation.analysis.cascade_chains import (
    cascade_chain_length_mean_estimate,
    cascade_chain_lengths,
    mean_failing_run_length_estimate,
    mean_steps_to_non_failure,
)
from inspect_degradation.analysis.loops import (
    loop_chain_length_mean_estimate,
    loop_chain_lengths,
)
from inspect_degradation.schema import Dependency, SeverityLevel, Validity


def _mixed_cascade_trace(trace_id: str) -> "object":
    """Trace with an independent error followed by two dependent ones, then a pass.

    Cascade chain: [3] (one chain of length 3).
    """
    steps = [
        make_graded_step(0, validity=Validity.pass_),
        make_graded_step(
            1,
            validity=Validity.fail,
            dependency=Dependency.independent,
            severity=SeverityLevel.medium,
        ),
        make_graded_step(
            2,
            validity=Validity.fail,
            dependency=Dependency.dependent,
            severity=SeverityLevel.medium,
        ),
        make_graded_step(
            3,
            validity=Validity.fail,
            dependency=Dependency.dependent,
            severity=SeverityLevel.medium,
        ),
        make_graded_step(4, validity=Validity.pass_),
    ]
    return make_graded_trace(trace_id=trace_id, steps=steps)


def _short_cascade_trace(trace_id: str) -> "object":
    """Trace with two short independent failures; chain list is [1, 1]."""
    steps = [
        make_graded_step(
            0,
            validity=Validity.fail,
            dependency=Dependency.independent,
            severity=SeverityLevel.low,
        ),
        make_graded_step(1, validity=Validity.pass_),
        make_graded_step(
            2,
            validity=Validity.fail,
            dependency=Dependency.independent,
            severity=SeverityLevel.low,
        ),
        make_graded_step(3, validity=Validity.pass_),
    ]
    return make_graded_trace(trace_id=trace_id, steps=steps)


def _loop_trace(trace_id: str, flags: list[bool | None]) -> "object":
    steps = [
        make_graded_step(i, validity=Validity.neutral if f else Validity.pass_, is_looping=f)
        for i, f in enumerate(flags)
    ]
    return make_graded_trace(trace_id=trace_id, steps=steps)


# ---------------------------------------------------------------------------
# Cascade chain mean estimate
# ---------------------------------------------------------------------------


class TestCascadeChainMeanEstimate:
    def test_empty_input_returns_empty_estimate(self):
        e = cascade_chain_length_mean_estimate([], rng=default_rng(0))
        assert e.method == "empty"

    def test_no_failures_in_any_trace(self):
        # All-pass traces: chain list is empty, bootstrap statistic is nan.
        ok = make_graded_trace(
            trace_id="a",
            steps=[make_graded_step(i, validity=Validity.pass_) for i in range(5)],
        )
        e = cascade_chain_length_mean_estimate(
            [ok, ok, ok], rng=default_rng(0), n_resamples=200
        )
        # Every resample produces a nan statistic → degenerate,
        # returning value=nan and no CI.
        assert math.isnan(e.value)

    def test_point_value_matches_scalar_mean(self):
        trace = _mixed_cascade_trace("a")
        # cascade_chain_lengths([trace]) = [3]; mean = 3.0
        lengths = cascade_chain_lengths([trace, trace])
        assert lengths == [3, 3]
        # Bootstrap point value should equal the raw mean.
        e = cascade_chain_length_mean_estimate(
            [trace, trace], rng=default_rng(0), n_resamples=200
        )
        assert math.isclose(e.value, 3.0)

    def test_ci_brackets_point_value_on_varied_corpus(self):
        a = _mixed_cascade_trace("a")  # chain list [3]
        b = _short_cascade_trace("b")  # chain list [1, 1]
        # Mix them so bootstrap has real variance to report.
        traces = [a, b, a, b, a, b]
        e = cascade_chain_length_mean_estimate(
            traces, rng=default_rng(42), n_resamples=400
        )
        # All chain lengths across all resamples fall in [1, 3].
        # Point value is mean of {3, 1, 1, 3, 1, 1, 3, 1, 1} = 15/9 ≈ 1.67
        # (three length-3 chains from three copies of `a` and six
        # length-1 chains from three copies of `b`).
        assert math.isclose(e.value, 15 / 9)
        assert e.ci_low <= e.value <= e.ci_high


# ---------------------------------------------------------------------------
# Failing-run length estimate
# ---------------------------------------------------------------------------


class TestFailingRunLengthEstimate:
    def test_point_matches_scalar(self):
        a = _mixed_cascade_trace("a")  # one failing run of length 3 (steps 1-3)
        b = _short_cascade_trace("b")  # two failing runs of length 1
        scalar = mean_steps_to_non_failure([a, b])
        assert scalar is not None
        est = mean_failing_run_length_estimate(
            [a, b, a, b], rng=default_rng(0), n_resamples=300
        )
        # Point value may differ if trace replication changes the mix,
        # but on the same (a, b) multiset the scalar and the bootstrap
        # point must agree.
        est_single = mean_failing_run_length_estimate(
            [a, b], rng=default_rng(0), n_resamples=300
        )
        assert math.isclose(est_single.value, scalar)

    def test_no_failing_runs(self):
        clean = make_graded_trace(
            trace_id="a",
            steps=[make_graded_step(i, validity=Validity.pass_) for i in range(4)],
        )
        e = mean_failing_run_length_estimate(
            [clean, clean], rng=default_rng(0), n_resamples=200
        )
        # Every resample gives nan → degenerate; value is nan.
        assert math.isnan(e.value)


# ---------------------------------------------------------------------------
# Loop chain length estimate
# ---------------------------------------------------------------------------


class TestLoopChainMeanEstimate:
    def test_point_value_matches_scalar(self):
        t = _loop_trace("a", [False, True, True, True, False, True])
        # Loop runs: [3, 1]; mean = 2.0
        assert loop_chain_lengths([t]) == [3, 1]
        e = loop_chain_length_mean_estimate(
            [t, t], rng=default_rng(0), n_resamples=300
        )
        assert math.isclose(e.value, 2.0)

    def test_varied_corpus_has_real_ci(self):
        short = _loop_trace("a", [True])  # chain list [1]
        long = _loop_trace(
            "b", [True, True, True, True, True]
        )  # chain list [5]
        traces = [short, long, short, long]
        e = loop_chain_length_mean_estimate(
            traces, rng=default_rng(123), n_resamples=400
        )
        assert e.ci_low < e.value < e.ci_high

    def test_empty(self):
        e = loop_chain_length_mean_estimate([], rng=default_rng(0))
        assert e.method == "empty"
