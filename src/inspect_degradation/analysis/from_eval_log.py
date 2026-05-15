"""Extract graded traces from Inspect AI eval logs and run the analysis pipeline.

This bridges the gap between running ``degradation_scorer()`` in an Inspect
eval and getting the full statistical analysis. After an eval completes,
the graded traces are embedded in ``Score.metadata``; this module extracts
them and feeds them through the analysis battery.

Usage from Python::

    from inspect_degradation.analysis.from_eval_log import analyze_eval_log

    report = analyze_eval_log("path/to/eval.log")
    print(report)

Usage from the command line::

    python -m inspect_degradation.analysis.from_eval_log path/to/eval.log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from inspect_degradation.analysis.frame import traces_to_frame
from inspect_degradation.analysis.statistics import Estimate
from inspect_degradation.integration.scorer import GRADED_TRACE_METADATA_KEY
from inspect_degradation.schema import GradedTrace

log = logging.getLogger(__name__)


def extract_graded_traces(log_path: str | Path) -> list[GradedTrace]:
    """Extract :class:`GradedTrace` objects from an Inspect eval log.

    Args:
        log_path: Path to an Inspect AI eval log file (``.json`` or
            ``.eval`` format).

    Returns:
        A list of :class:`GradedTrace` objects, one per sample that
        was graded by :func:`degradation_scorer`.
    """
    from inspect_ai.log import read_eval_log

    eval_log = read_eval_log(str(log_path))
    traces: list[GradedTrace] = []

    if not eval_log.samples:
        return traces

    for sample in eval_log.samples:
        if not sample.scores:
            continue
        for _scorer_name, score in sample.scores.items():
            meta = score.metadata or {}
            payload = meta.get(GRADED_TRACE_METADATA_KEY)
            if payload is not None:
                traces.append(GradedTrace.model_validate(payload))

    return traces


def analyze_eval_log(
    log_path: str | Path,
    *,
    use_glmm: bool = False,
) -> dict[str, Any]:
    """Run the full analysis pipeline on an Inspect eval log.

    Extracts graded traces from the eval log and runs: rates,
    slopes, mixed-effects regression, survival analysis, cascade
    chains, loop analysis, and autocorrelation diagnostics.

    Args:
        log_path: Path to an Inspect AI eval log file.
        use_glmm: If True, use the logit-link GLMM instead of the
            linear probability model. Recommended when error rates
            are below 10% or above 90%.

    Returns:
        A dict containing all analysis results, suitable for
        JSON serialization via ``json.dumps``.
    """
    traces = extract_graded_traces(log_path)
    if not traces:
        return {"error": "no graded traces found in eval log", "log_path": str(log_path)}

    return analyze_traces(traces, use_glmm=use_glmm)


def analyze_traces(
    traces: list[GradedTrace],
    *,
    use_glmm: bool = False,
) -> dict[str, Any]:
    """Run the full analysis pipeline on a list of graded traces.

    This is the core analysis function. :func:`analyze_eval_log` is a
    convenience wrapper that extracts traces from an eval log first.
    """
    from inspect_degradation.analysis import rates as _rates
    from inspect_degradation.analysis import slopes as _slopes
    from inspect_degradation.analysis.autocorrelation import (
        ljung_box_per_trace,
        per_trace_acf,
    )
    from inspect_degradation.analysis.cascade_chains import (
        cascade_chain_length_mean_estimate,
        mean_failing_run_length_estimate,
    )
    from inspect_degradation.analysis.loops import (
        loop_chain_length_mean_estimate,
        raw_loop_rate,
    )
    from inspect_degradation.analysis.mixed_effects import (
        fit_step_level_glmm,
        fit_step_level_model,
    )
    from inspect_degradation.analysis.survival import first_error_km

    df = traces_to_frame(traces)
    n_traces = len(traces)
    n_steps = len(df)
    report: dict[str, Any] = {
        "n_traces": n_traces,
        "n_steps": n_steps,
        "models": sorted(df["model"].dropna().unique().tolist()) if "model" in df.columns else [],
    }

    # -- Rates ---------------------------------------------------------------
    report["error_rate"] = _fmt(_rates.error_rate(traces))
    report["neutral_rate"] = _fmt(_rates.neutral_rate(traces))
    report["productive_rate"] = _fmt(_rates.productive_rate(traces))

    # -- Slopes --------------------------------------------------------------
    report["error_rate_slope"] = _fmt(_slopes.error_rate_slope(traces).estimate)
    report["neutral_rate_slope"] = _fmt(_slopes.neutral_rate_slope(traces).estimate)

    # -- Mixed-effects regression -------------------------------------------
    fit_fn = fit_step_level_glmm if use_glmm else fit_step_level_model
    me_result = fit_fn(df)
    report["mixed_effects"] = {
        "method": me_result.method,
        "formula": me_result.formula,
        "converged": me_result.converged,
        "fit_usable": me_result.fit_usable,
        "n_observations": me_result.n_observations,
        "step_index": _fmt(me_result.slope_estimate("step_index")),
        "coefficients": [c.to_dict() for c in me_result.coefficients],
        "random_effects": me_result.random_effects.to_dict(),
        "extras": me_result.extras,
    }

    # -- Survival ------------------------------------------------------------
    try:
        km = first_error_km(traces)
        report["survival"] = {
            "median_first_error_step": km.median_survival_time,
            "n_events": km.n_events,
            "n_censored": km.n_censored,
        }
    except Exception as e:
        log.exception("survival analysis failed")
        report["survival"] = {"error": f"{type(e).__name__}: {e}"}

    # -- Cascade chains ------------------------------------------------------
    report["cascade_chain_length_mean"] = _fmt(cascade_chain_length_mean_estimate(traces))
    report["mean_failure_run_length"] = _fmt(mean_failing_run_length_estimate(traces))

    # -- Loops ---------------------------------------------------------------
    loop_rate_val = raw_loop_rate(traces)
    report["loop_rate"] = loop_rate_val if loop_rate_val is not None else float("nan")
    try:
        report["loop_chain_length_mean"] = _fmt(loop_chain_length_mean_estimate(traces))
    except Exception as e:
        log.exception("loop_chain_length_mean_estimate failed")
        report["loop_chain_length_mean"] = {
            "value": float("nan"),
            "error": f"{type(e).__name__}: {e}",
        }

    # -- Autocorrelation -----------------------------------------------------
    try:
        acf_result = per_trace_acf(traces)
        report["mean_lag1_acf"] = acf_result.mean_acf[0] if acf_result.mean_acf else float("nan")
    except Exception:
        log.exception("per_trace_acf failed")
        report["mean_lag1_acf"] = float("nan")

    try:
        lb = ljung_box_per_trace(traces)
        report["ljung_box_rejection_rate"] = lb.rejection_rate if hasattr(lb, "rejection_rate") else None
    except Exception:
        log.exception("ljung_box_per_trace failed")
        report["ljung_box_rejection_rate"] = None

    return report


def _fmt(est: Estimate) -> dict[str, Any]:
    """Convert an Estimate to a JSON-friendly dict."""
    d: dict[str, Any] = {"value": est.value, "n": est.n, "method": est.method}
    if est.has_ci:
        d["ci_low"] = est.ci_low
        d["ci_high"] = est.ci_high
    return d


def _print_report(report: dict[str, Any]) -> None:
    """Pretty-print an analysis report to stdout."""
    print(f"Traces: {report['n_traces']}, Steps: {report['n_steps']}")
    if report.get("models"):
        print(f"Models: {', '.join(report['models'])}")
    print()

    for key in ("error_rate", "neutral_rate", "productive_rate"):
        r = report.get(key, {})
        val = r.get("value", "?")
        ci = ""
        if "ci_low" in r:
            ci = f" [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(f"  {key}: {val:.4f}{ci}" if isinstance(val, float) else f"  {key}: {val}")

    print()
    for key in ("error_rate_slope", "neutral_rate_slope"):
        r = report.get(key, {})
        val = r.get("value", "?")
        ci = ""
        if "ci_low" in r:
            ci = f" [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        print(f"  {key}: {val:+.4f}{ci}" if isinstance(val, float) else f"  {key}: {val}")

    me = report.get("mixed_effects", {})
    if me:
        print(f"\nMixed-effects ({me.get('method', '?')}):")
        print(f"  Formula: {me.get('formula', '?')}")
        print(f"  Usable: {me.get('fit_usable', '?')}")
        si = me.get("step_index", {})
        val = si.get("value", "?")
        ci = ""
        if "ci_low" in si:
            ci = f" [{si['ci_low']:+.4f}, {si['ci_high']:+.4f}]"
        print(f"  step_index: {val:+.4f}{ci}" if isinstance(val, float) else f"  step_index: {val}")
        re = me.get("random_effects", {})
        if re:
            print(f"  ICC: {re.get('icc', '?'):.3f}" if isinstance(re.get("icc"), float) else f"  ICC: {re.get('icc', '?')}")

    surv = report.get("survival", {})
    if surv and "median_first_error_step" in surv:
        print(f"\nSurvival: median first error at step {surv['median_first_error_step']}")

    cc = report.get("cascade_chain_length_mean", {})
    if cc:
        print(f"\nCascade chain length mean: {cc.get('value', '?'):.2f}" if isinstance(cc.get("value"), float) else "")

    acf = report.get("mean_lag1_acf")
    if acf is not None:
        print(f"Mean lag-1 ACF: {acf:.3f}" if isinstance(acf, float) else "")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inspect_degradation.analysis.from_eval_log",
        description=(
            "Extract graded traces from an Inspect AI eval log and run the "
            "full degradation analysis pipeline (rates, slopes, mixed-effects "
            "regression, survival, cascade chains, loops, autocorrelation)."
        ),
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to an Inspect AI eval log file (.json or .eval).",
    )
    parser.add_argument(
        "--glmm",
        dest="use_glmm",
        action="store_true",
        help=(
            "Fit a logit-link GLMM instead of the linear probability model. "
            "Recommended when the marginal error rate is below ~10%% or above ~90%%."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to write the JSON report to. Defaults to the eval log path "
            "with a '.degradation.json' suffix."
        ),
    )
    parser.add_argument(
        "--json",
        dest="json_only",
        action="store_true",
        help="Emit the JSON report to stdout and skip the human-readable summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: analyze an Inspect eval log.

    Returns the process exit code so this is also callable from tests.
    """
    args = _build_parser().parse_args(argv)

    if not args.log_path.exists():
        print(f"File not found: {args.log_path}", file=sys.stderr)
        return 1

    report = analyze_eval_log(args.log_path, use_glmm=args.use_glmm)

    if "error" in report:
        print(f"Error: {report['error']}", file=sys.stderr)
        return 1

    json_path = args.output or args.log_path.with_suffix(".degradation.json")
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if args.json_only:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_report(report)
        print(f"\nFull report written to {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
