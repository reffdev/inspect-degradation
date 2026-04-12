"""Inspect AI integration: scorer and metrics for within-run degradation.

Named ``integration`` rather than ``inspect`` so the subpackage doesn't
shadow Python's stdlib ``inspect`` module — every module inside this
subtree can safely ``import inspect`` if it ever needs to.

Every metric exposed here is an ``@metric``-decorated factory that
returns a scalar aggregator. The scalar form exists because Inspect AI's
metric protocol requires scalars; callers doing their own reporting
should prefer the analysis-layer APIs directly to preserve confidence
intervals.
"""

from inspect_degradation.integration.metrics import (
    cascade_chain_length_mean,
    error_rate,
    error_rate_slope,
    first_error_step_median,
    loop_chain_length_mean,
    loop_rate,
    loop_rate_slope,
    mean_failure_run_length,
    neutral_rate,
    neutral_rate_slope,
    productive_rate,
)
from inspect_degradation.integration.scorer import degradation_scorer

__all__ = [
    "cascade_chain_length_mean",
    "degradation_scorer",
    "error_rate",
    "error_rate_slope",
    "first_error_step_median",
    "loop_chain_length_mean",
    "loop_rate",
    "loop_rate_slope",
    "mean_failure_run_length",
    "neutral_rate",
    "neutral_rate_slope",
    "productive_rate",
]
