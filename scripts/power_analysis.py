"""Phase 3 power analysis: is the planned corpus big enough?

Runs :func:`inspect_degradation.analysis.power.simulate_mixed_effects_power`
across a small grid of plausible effect sizes and corpus shapes,
and prints a power table so the experimenter can decide whether
the planned Phase 3 corpus is large enough to detect the slopes
that Phase 1 (or prior literature) suggests are realistic.

Defaults assume:

* A corpus on the order of 1000 traces with ~15 steps each. Adjust
  ``--n-traces`` and ``--steps-per-trace`` to match your real
  Phase 3 plan.
* Effect sizes from 0.001 to 0.02 errors/step. The middle of this
  range (~0.01) is the slope you'd see if a 15-step trace gains
  ~15% more failures over its course; the lower end is the
  smallest slope worth detecting.
* A grader flip rate of 0.10, the rough number Phase 1 typically
  produces against TRAIL. Adjust ``--flip-probability`` once you
  have a real Phase 1 estimate.

The script intentionally runs a *small* number of simulations per
cell (default 80) — enough to distinguish "definitely powered"
from "definitely underpowered" without burning a multi-hour
compute budget. Bump ``--n-simulations`` for tighter CIs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from numpy.random import default_rng

from inspect_degradation.analysis.power import (  # noqa: E402
    simulate_mixed_effects_power,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo power analysis for Phase 3 degradation detection.",
    )
    p.add_argument(
        "--n-traces",
        type=int,
        nargs="+",
        default=[200, 500, 1000],
        help="Trace counts to simulate at. Default: 200 500 1000.",
    )
    p.add_argument(
        "--steps-per-trace",
        type=int,
        default=15,
        help="Steps per simulated trace. Default: 15.",
    )
    p.add_argument(
        "--effect-sizes",
        type=float,
        nargs="+",
        default=[0.001, 0.005, 0.01, 0.02],
        help="True per-step slopes to inject. Default: 0.001 0.005 0.01 0.02.",
    )
    p.add_argument(
        "--base-rate",
        type=float,
        default=0.20,
        help="Baseline error rate at step 0. Default: 0.20.",
    )
    p.add_argument(
        "--trace-intercept-sd",
        type=float,
        default=0.10,
        help="SD of between-trace random intercept (heterogeneity). Default: 0.10.",
    )
    p.add_argument(
        "--flip-probability",
        type=float,
        default=0.10,
        help="Grader label-flip noise rate. Default: 0.10. Set to 0 for "
        "the no-noise upper bound; set to your Phase 1 estimate for the "
        "honest power.",
    )
    p.add_argument(
        "--n-simulations",
        type=int,
        default=80,
        help="Monte Carlo replicates per cell. Default: 80 (fast). "
        "Bump to 200+ for tighter CIs.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility.",
    )
    return p.parse_args()


def _power_emoji(value: float) -> str:
    if value >= 0.80:
        return "OK   "
    if value >= 0.50:
        return "MARG "
    return "WEAK "


def main() -> None:
    args = _parse_args()
    rng = default_rng(args.seed)

    print()
    print(
        f"Power simulation: {len(args.n_traces)}*{len(args.effect_sizes)} cells, "
        f"{args.n_simulations} reps each, "
        f"steps_per_trace={args.steps_per_trace}, base_rate={args.base_rate}, "
        f"trace_sd={args.trace_intercept_sd}, flip={args.flip_probability}"
    )
    print()

    # Header.
    cell_w = 22
    header = "n_traces \\ slope".ljust(16)
    for slope in args.effect_sizes:
        header += f"{slope:>+.4f}".ljust(cell_w)
    print(header)
    print("-" * len(header))

    for n in args.n_traces:
        row = f"{n:>10}      "
        for slope in args.effect_sizes:
            result = simulate_mixed_effects_power(
                true_slope=slope,
                n_traces=n,
                steps_per_trace=args.steps_per_trace,
                base_rate=args.base_rate,
                trace_intercept_sd=args.trace_intercept_sd,
                flip_probability=args.flip_probability,
                n_simulations=args.n_simulations,
                rng=rng,
            )
            tag = _power_emoji(result.power.value)
            cell = (
                f"{tag}{result.power.value:.2f} "
                f"[{result.power.ci_low:.2f},{result.power.ci_high:.2f}]"
            )
            row += cell.ljust(cell_w)
        print(row)
    print()
    print(
        "OK >=0.80 power · MARG 0.50-0.80 · WEAK <0.50. "
        "Cells in the OK band are reliably detectable; "
        "MARG cells need either a bigger corpus or a tighter Phase 1 grader; "
        "WEAK cells will not produce a positive Phase 3 finding regardless "
        "of how the analysis is run."
    )
    print()


if __name__ == "__main__":
    main()
