"""Within-trace autocorrelation diagnostics for error sequences.

The linear probability model in
:mod:`inspect_degradation.analysis.mixed_effects` treats step-level
errors as conditionally independent given the random intercept and
the slope. That assumption is testable: if successive steps are
correlated *beyond* what the random intercept absorbs, the model
will understate its standard errors and overstate significance.

This module provides two diagnostics:

* :func:`per_trace_acf` — sample autocorrelation function up to a
  user-specified lag, computed per trace and averaged across
  traces. The averaged ACF is the right summary because individual
  traces are too short for the ACF at any lag to be precise on its
  own; the across-trace average tightens the estimate by ``√n_traces``.
* :func:`ljung_box_per_trace` — Ljung-Box portmanteau test for
  serial correlation, run independently on every trace, returning
  the fraction of traces that reject independence at the supplied
  significance level. A high rejection rate is a red flag that the
  i.i.d. assumption fails and that block-bootstrap or
  autocorrelation-robust standard errors should be used downstream.

Both diagnostics operate on the tidy step-level frame produced by
:func:`inspect_degradation.analysis.frame.traces_to_frame`. They
are intentionally narrow: their job is to *detect* a problem, not
to fix it. The fix (cluster-robust SEs, AR(1) error structure,
etc.) is downstream of the audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from inspect_degradation.analysis.statistics import (
    NINETY_FIVE,
    ConfidenceLevel,
    Estimate,
    _normal_ppf,
    wilson_proportion_interval,
)


@dataclass(frozen=True)
class AutocorrelationResult:
    """Sample-ACF summary across a corpus of traces.

    Attributes:
        lags: Lag indices, ``[1, 2, ..., max_lag]``.
        mean_acf: Per-lag mean of the sample ACF across traces.
        ci_low: Lower bound on the per-lag mean (normal-approx
            interval based on the across-trace SD).
        ci_high: Upper bound on the per-lag mean.
        n_traces_used: Number of traces with enough length to
            contribute to at least lag 1.
        outcome_col: Column the ACF was computed on.
        confidence_level: Confidence level for the bounds.
    """

    lags: list[int]
    mean_acf: list[float]
    ci_low: list[float]
    ci_high: list[float]
    n_traces_used: int
    outcome_col: str
    confidence_level: ConfidenceLevel

    def to_dict(self) -> dict[str, Any]:
        return {
            "lags": list(self.lags),
            "mean_acf": list(self.mean_acf),
            "ci_low": list(self.ci_low),
            "ci_high": list(self.ci_high),
            "n_traces_used": self.n_traces_used,
            "outcome_col": self.outcome_col,
            "confidence_level": self.confidence_level.level,
        }


@dataclass(frozen=True)
class LjungBoxResult:
    """Per-trace Ljung-Box test summary.

    Attributes:
        rejection_rate: Wilson CI on the fraction of traces that
            reject the white-noise null at ``alpha``.
        n_traces_tested: Traces with enough observations to run
            Ljung-Box at the requested lag.
        n_rejected: Traces that rejected the null.
        lags: The lag (or lag list) used for the portmanteau test.
        alpha: Significance level applied to each per-trace test.
        outcome_col: Column the test was run on.
        per_trace: One ``{trace_id, p_value, n}`` dict per trace
            for audit / drill-down. Empty by default to keep
            payloads small; populated if ``include_per_trace=True``.
    """

    rejection_rate: Estimate
    n_traces_tested: int
    n_rejected: int
    lags: int
    alpha: float
    outcome_col: str
    per_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rejection_rate": self.rejection_rate.to_dict(),
            "n_traces_tested": self.n_traces_tested,
            "n_rejected": self.n_rejected,
            "lags": self.lags,
            "alpha": self.alpha,
            "outcome_col": self.outcome_col,
            "per_trace": list(self.per_trace),
        }


def _trace_series(
    df: pd.DataFrame, *, outcome_col: str
) -> list[tuple[str, np.ndarray]]:
    series: list[tuple[str, np.ndarray]] = []
    for trace_id, group in df.sort_values("step_index").groupby("trace_id"):
        arr = group[outcome_col].to_numpy(dtype=float)
        series.append((str(trace_id), arr))
    return series


def _sample_acf(arr: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased sample ACF (matches statsmodels' default ``unbiased=False``).

    Returns NaN at lags where the variance is zero (constant series)
    so the across-trace average can drop them cleanly.
    """
    n = len(arr)
    centered = arr - arr.mean()
    denom = float((centered * centered).sum())
    if denom == 0.0:
        return np.full(max_lag, np.nan)
    out = np.empty(max_lag)
    for k in range(1, max_lag + 1):
        if k >= n:
            out[k - 1] = np.nan
        else:
            out[k - 1] = float((centered[:-k] * centered[k:]).sum() / denom)
    return out


def per_trace_acf(
    df: pd.DataFrame,
    *,
    outcome_col: str = "is_error",
    max_lag: int = 5,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
) -> AutocorrelationResult:
    """Average per-trace sample ACF up to ``max_lag``.

    Args:
        df: Tidy step-level frame from ``traces_to_frame``. Must
            contain ``trace_id``, ``step_index``, and ``outcome_col``.
        outcome_col: Column to compute the ACF on. Default
            ``"is_error"``.
        max_lag: Maximum lag to report.
        confidence_level: Used for the across-trace mean's CI band.

    Returns:
        :class:`AutocorrelationResult` with one entry per lag. Lags
        whose averages are computed from fewer than 2 traces report
        ``nan`` for the CI bounds.
    """
    if df.empty:
        return AutocorrelationResult(
            lags=list(range(1, max_lag + 1)),
            mean_acf=[float("nan")] * max_lag,
            ci_low=[float("nan")] * max_lag,
            ci_high=[float("nan")] * max_lag,
            n_traces_used=0,
            outcome_col=outcome_col,
            confidence_level=confidence_level,
        )

    series = _trace_series(df, outcome_col=outcome_col)
    rows: list[np.ndarray] = []
    for _, arr in series:
        if len(arr) < 2:
            continue
        rows.append(_sample_acf(arr, max_lag))

    if not rows:
        return AutocorrelationResult(
            lags=list(range(1, max_lag + 1)),
            mean_acf=[float("nan")] * max_lag,
            ci_low=[float("nan")] * max_lag,
            ci_high=[float("nan")] * max_lag,
            n_traces_used=0,
            outcome_col=outcome_col,
            confidence_level=confidence_level,
        )

    matrix = np.vstack(rows)  # shape (n_traces, max_lag)
    mean_acf: list[float] = []
    ci_low: list[float] = []
    ci_high: list[float] = []
    z = _normal_ppf(1.0 - confidence_level.alpha / 2.0)
    for k in range(max_lag):
        col = matrix[:, k]
        col = col[~np.isnan(col)]
        if len(col) == 0:
            mean_acf.append(float("nan"))
            ci_low.append(float("nan"))
            ci_high.append(float("nan"))
            continue
        m = float(col.mean())
        mean_acf.append(m)
        if len(col) < 2:
            ci_low.append(float("nan"))
            ci_high.append(float("nan"))
            continue
        sd = float(col.std(ddof=1))
        se = sd / np.sqrt(len(col))
        ci_low.append(m - z * se)
        ci_high.append(m + z * se)

    return AutocorrelationResult(
        lags=list(range(1, max_lag + 1)),
        mean_acf=mean_acf,
        ci_low=ci_low,
        ci_high=ci_high,
        n_traces_used=len(rows),
        outcome_col=outcome_col,
        confidence_level=confidence_level,
    )


def ljung_box_per_trace(
    df: pd.DataFrame,
    *,
    outcome_col: str = "is_error",
    lags: int = 5,
    alpha: float = 0.05,
    include_per_trace: bool = False,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
) -> LjungBoxResult:
    """Run Ljung-Box per trace and aggregate rejection rate.

    Args:
        df: Tidy step-level frame.
        outcome_col: Column to test (default ``"is_error"``).
        lags: Number of lags to include in the portmanteau
            statistic. Each trace must have ``> lags`` observations
            to contribute; shorter traces are skipped.
        alpha: Significance level for each per-trace test.
        include_per_trace: If true, attach per-trace p-values to
            the result for drill-down.
        confidence_level: CI level for the Wilson interval on the
            corpus-wide rejection rate.

    Returns:
        :class:`LjungBoxResult` with a Wilson CI on the fraction
        of traces that rejected white noise. Under the i.i.d.
        null this fraction should hover near ``alpha``; values
        substantially above ``alpha`` indicate that within-trace
        steps are autocorrelated and the linear-probability
        independence assumption is violated.

    Raises:
        RuntimeError: statsmodels is not installed.
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "statsmodels is required for Ljung-Box; install it via the "
            "package's main dependency group"
        ) from exc

    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if lags < 1:
        raise ValueError(f"lags must be >= 1; got {lags}")

    series = _trace_series(df, outcome_col=outcome_col) if not df.empty else []
    n_tested = 0
    n_rejected = 0
    per_trace_rows: list[dict[str, Any]] = []

    for trace_id, arr in series:
        if len(arr) <= lags:
            continue
        # Constant series have no variance and Ljung-Box returns nan;
        # skip rather than count them as either accept or reject.
        if float(np.var(arr)) == 0.0:
            continue
        try:
            result = acorr_ljungbox(arr, lags=[lags], return_df=True)
        except (ValueError, IndexError):
            # Statsmodels can complain about extreme inputs; treat
            # as untestable rather than crash the audit.
            continue
        p_value = float(result["lb_pvalue"].iloc[-1])
        if not np.isfinite(p_value):
            continue
        n_tested += 1
        rejected = p_value < alpha
        if rejected:
            n_rejected += 1
        if include_per_trace:
            per_trace_rows.append(
                {
                    "trace_id": trace_id,
                    "p_value": p_value,
                    "rejected": rejected,
                    "n": int(len(arr)),
                }
            )

    rejection_rate = wilson_proportion_interval(
        n_rejected, n_tested, confidence_level=confidence_level
    )

    return LjungBoxResult(
        rejection_rate=rejection_rate,
        n_traces_tested=n_tested,
        n_rejected=n_rejected,
        lags=lags,
        alpha=alpha,
        outcome_col=outcome_col,
        per_trace=per_trace_rows,
    )


__all__ = [
    "AutocorrelationResult",
    "LjungBoxResult",
    "ljung_box_per_trace",
    "per_trace_acf",
]
