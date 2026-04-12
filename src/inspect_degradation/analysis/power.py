"""Simulation-based power analysis for degradation detection.

Closed-form power formulas for mixed-effects linear probability
models exist but they depend on assumptions (homoskedasticity,
Gaussian residuals, equal cluster sizes) that this project's data
routinely violates. A Monte Carlo simulator is slower but honest:
we draw synthetic corpora from a generative model that matches the
*actual* analysis pipeline's inputs, run the exact estimator the
report will run, and count how often it rejects the null.

The generative model here is deliberately small:

* ``n_traces`` traces, each of length ``steps_per_trace``.
* Step-level probability of error is
  ``base_rate + slope * step_index + trace_intercept``,
  clipped into ``[0, 1]``. The ``trace_intercept`` is drawn
  i.i.d. Gaussian with variance ``trace_intercept_sd ** 2`` to
  simulate between-trace heterogeneity — this is what makes the
  mixed-effects model necessary in the first place.
* Optional label flips at rate ``flip_probability`` simulate
  grader measurement error, so power estimates can be computed
  under the same noise conditions the real corpus faces.

The estimator plugged into each simulated corpus is
:func:`inspect_degradation.analysis.mixed_effects.fit_step_level_model`
with its default formula (``is_error ~ step_index``). Power is the
fraction of simulated corpora whose CI on the ``step_index``
coefficient excludes zero (i.e., the effect is detectable at the
given confidence level).

This is one of those rare cases where "run the real analyzer on
fake data" is the right implementation: it keeps power estimates
synchronized with the estimator and requires zero special-case
math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng

from inspect_degradation.analysis.statistics import (
    NINETY_FIVE,
    ConfidenceLevel,
    Estimate,
    wilson_proportion_interval,
)


@dataclass(frozen=True)
class PowerResult:
    """Result of a Monte Carlo power simulation.

    Attributes:
        power: :class:`Estimate` of the detection probability at the
            supplied ``true_slope`` under the simulated generative
            model. The CI is a Wilson interval on the binomial of
            detected / total simulations.
        mean_estimated_slope: Mean of the estimated slopes across
            simulations. Compare to ``true_slope`` to sanity-check
            the estimator's calibration under this configuration.
        fraction_converged: Fraction of simulations whose mixed-
            effects fit was marked ``fit_usable``. A low value is
            a red flag that the configuration is not actually
            estimable at the specified sample size.
        true_slope: The slope used to generate the data.
        n_simulations: Total simulations run.
        confidence_level: CI level used both for the simulator's
            "did the CI exclude zero" criterion and for the Wilson
            interval on the reported power estimate.
        config: Serialized copy of all simulation inputs, so the
            result carries its own reproduction recipe.
    """

    power: Estimate
    mean_estimated_slope: float
    fraction_converged: float
    true_slope: float
    n_simulations: int
    confidence_level: ConfidenceLevel
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "power": self.power.to_dict(),
            "mean_estimated_slope": self.mean_estimated_slope,
            "fraction_converged": self.fraction_converged,
            "true_slope": self.true_slope,
            "n_simulations": self.n_simulations,
            "confidence_level": self.confidence_level.level,
            "config": dict(self.config),
        }


def _simulate_corpus(
    *,
    n_traces: int,
    steps_per_trace: int,
    base_rate: float,
    slope: float,
    trace_intercept_sd: float,
    flip_probability: float,
    rng: Generator,
) -> pd.DataFrame:
    """Draw one synthetic step-level frame from the generative model."""
    trace_intercepts = rng.normal(0.0, trace_intercept_sd, size=n_traces)
    rows: list[dict] = []
    for t in range(n_traces):
        intercept = float(trace_intercepts[t])
        for s in range(steps_per_trace):
            p = base_rate + slope * s + intercept
            p = float(np.clip(p, 0.0, 1.0))
            is_error = bool(rng.random() < p)
            if flip_probability > 0 and rng.random() < flip_probability:
                is_error = not is_error
            rows.append(
                {
                    "trace_id": f"t{t}",
                    "task_id": f"t{t}",
                    "step_index": s,
                    "is_error": is_error,
                    "complexity": "medium",
                    "model": "sim",
                }
            )
    return pd.DataFrame(rows)


def simulate_mixed_effects_power(
    *,
    true_slope: float,
    n_traces: int,
    steps_per_trace: int,
    base_rate: float = 0.1,
    trace_intercept_sd: float = 0.05,
    flip_probability: float = 0.0,
    n_simulations: int = 200,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    rng: Generator | None = None,
) -> PowerResult:
    """Estimate power of ``fit_step_level_model`` under a generative model.

    Runs ``n_simulations`` independent simulations and counts the
    fraction whose fitted ``step_index`` coefficient CI excludes
    zero *in the same direction* as ``true_slope`` (i.e., a slope
    fit of the wrong sign does not count as a detection even if
    its CI is narrow).

    Args:
        true_slope: The slope to inject. 0 produces a Type-I error
            rate estimate under the null. A positive value produces
            a power-at-that-effect estimate.
        n_traces: Number of traces per simulated corpus.
        steps_per_trace: Steps per trace (uniform for now).
        base_rate: Baseline probability of error at step 0.
        trace_intercept_sd: SD of the between-trace random
            intercept — the "some traces are just harder" term.
        flip_probability: Per-step grader label-flip rate, matching
            the ``simex_correct`` noise model.
        n_simulations: Number of Monte Carlo replicates.
        confidence_level: Used both for the per-simulation "CI
            excludes zero" test and for the Wilson interval on the
            overall power estimate.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        :class:`PowerResult` with the estimated power as an
        :class:`Estimate` (Wilson interval on the binomial) and
        diagnostic summary statistics.
    """
    from inspect_degradation.analysis.mixed_effects import fit_step_level_model

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive")
    if n_traces <= 1:
        raise ValueError("n_traces must be >= 2 for mixed-effects")
    if steps_per_trace <= 1:
        raise ValueError("steps_per_trace must be >= 2 for slope estimation")
    if not 0.0 <= flip_probability < 0.5:
        raise ValueError("flip_probability must be in [0, 0.5)")

    rng = rng if rng is not None else default_rng()

    detections = 0
    convergences = 0
    estimated_slopes: list[float] = []

    for _ in range(n_simulations):
        df = _simulate_corpus(
            n_traces=n_traces,
            steps_per_trace=steps_per_trace,
            base_rate=base_rate,
            slope=true_slope,
            trace_intercept_sd=trace_intercept_sd,
            flip_probability=flip_probability,
            rng=rng,
        )
        result = fit_step_level_model(df, confidence_level=confidence_level)
        if not result.fit_usable:
            continue
        convergences += 1
        try:
            row = result.coefficient("step_index")
        except KeyError:
            continue
        estimated_slopes.append(row.estimate)

        if true_slope == 0.0:
            # Under the null: detection = CI excludes zero
            # (two-sided Type-I error rate).
            if row.ci_low > 0 or row.ci_high < 0:
                detections += 1
        else:
            # Under the alternative: detection = CI excludes zero
            # *and* sign matches the injected effect.
            sign = 1.0 if true_slope > 0 else -1.0
            est_sign = 1.0 if row.estimate > 0 else -1.0
            if sign == est_sign and (row.ci_low > 0 or row.ci_high < 0):
                detections += 1

    power_estimate = wilson_proportion_interval(
        detections, n_simulations, confidence_level=confidence_level
    )

    mean_est = float(np.mean(estimated_slopes)) if estimated_slopes else float("nan")
    frac_converged = convergences / n_simulations

    config = {
        "true_slope": true_slope,
        "n_traces": n_traces,
        "steps_per_trace": steps_per_trace,
        "base_rate": base_rate,
        "trace_intercept_sd": trace_intercept_sd,
        "flip_probability": flip_probability,
    }

    return PowerResult(
        power=power_estimate,
        mean_estimated_slope=mean_est,
        fraction_converged=frac_converged,
        true_slope=true_slope,
        n_simulations=n_simulations,
        confidence_level=confidence_level,
        config=config,
    )


__all__ = ["PowerResult", "simulate_mixed_effects_power"]
