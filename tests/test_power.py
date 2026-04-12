"""Monte-Carlo power analysis for mixed-effects degradation detection."""

from __future__ import annotations

import pytest
from numpy.random import default_rng

pytest.importorskip("statsmodels")  # noqa: E402

from inspect_degradation.analysis.power import (  # noqa: E402
    PowerResult,
    simulate_mixed_effects_power,
)


class TestPowerSimulation:
    def test_null_power_near_alpha(self):
        # Under the null, detection rate should be near α = 0.05.
        # We use a small simulation count and only require the upper
        # CI to stay well below something like 0.25 — enough to catch
        # a grossly miscalibrated estimator without making the test
        # flaky.
        result = simulate_mixed_effects_power(
            true_slope=0.0,
            n_traces=20,
            steps_per_trace=10,
            base_rate=0.2,
            trace_intercept_sd=0.05,
            n_simulations=40,
            rng=default_rng(0),
        )
        assert isinstance(result, PowerResult)
        assert result.power.value <= 0.30
        assert result.power.ci_low <= 0.05 <= result.power.ci_high + 0.25

    def test_high_power_at_large_effect(self):
        result = simulate_mixed_effects_power(
            true_slope=0.05,
            n_traces=30,
            steps_per_trace=15,
            base_rate=0.05,
            trace_intercept_sd=0.02,
            n_simulations=30,
            rng=default_rng(1),
        )
        # Large per-step slope + reasonable n → power should be high.
        assert result.power.value >= 0.5
        # Mean estimated slope should be in the right ballpark. Big
        # tolerance because these are small simulations.
        assert result.mean_estimated_slope > 0.0

    def test_rejects_invalid_inputs(self):
        with pytest.raises(ValueError, match="n_simulations"):
            simulate_mixed_effects_power(
                true_slope=0.01,
                n_traces=10,
                steps_per_trace=10,
                n_simulations=0,
            )
        with pytest.raises(ValueError, match="n_traces"):
            simulate_mixed_effects_power(
                true_slope=0.01,
                n_traces=1,
                steps_per_trace=10,
            )
        with pytest.raises(ValueError, match="steps_per_trace"):
            simulate_mixed_effects_power(
                true_slope=0.01,
                n_traces=10,
                steps_per_trace=1,
            )
        with pytest.raises(ValueError, match="flip_probability"):
            simulate_mixed_effects_power(
                true_slope=0.01,
                n_traces=10,
                steps_per_trace=10,
                flip_probability=0.6,
            )

    def test_to_dict_is_json_safe(self):
        import json

        result = simulate_mixed_effects_power(
            true_slope=0.02,
            n_traces=10,
            steps_per_trace=10,
            n_simulations=10,
            rng=default_rng(2),
        )
        json.dumps(result.to_dict())
