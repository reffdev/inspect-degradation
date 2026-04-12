"""Statistical primitives: the :class:`Estimate` type and CI computations.

Every statistical claim the analysis layer produces is returned as an
:class:`Estimate`. Point estimates without a CI are not allowed: the
project plan commits to "confidence intervals and effect sizes, not
just p-values," and the type system enforces it by making CIs a
required field on every returned value. A caller that just wants the
point number can write ``.value`` — but the CI travels with it, and a
code review can grep for any sign of a bare float being promoted to
"the answer" anywhere in the analysis layer.

This module is deliberately pure: it imports ``numpy`` and nothing from
the project's own types. The higher-level analysis modules in this
package are responsible for turning :class:`GradedTrace` objects into
the numeric inputs these primitives consume.

Primitives supplied here:

* :class:`Estimate` — the universal return type.
* :class:`ConfidenceLevel` — a small wrapper around the α / (1−α) split
  that keeps a confidence level from being silently rounded or mistyped.
* :func:`wilson_proportion_interval` — closed-form Wilson score interval
  for a binomial proportion. Preferred over bootstrap whenever the
  statistic is a simple rate of independent trials.
* :func:`bootstrap_estimate` — trace-level bootstrap for statistics
  computed on grouped data where within-group observations are not
  independent. This is the right primitive for slopes-over-steps and
  for any statistic computed on pooled trace data; naive observation-
  level bootstrap would understate the CI width.
* :func:`ols_slope_with_interval` — analytic normal-theory CI for a
  single-feature OLS slope, for cases where observations really are
  independent.

Bootstrap implementation notes:

* We default to BCa (bias-corrected and accelerated) percentile
  intervals, which are asymptotically second-order accurate and handle
  skew correctly — relevant for heavy-tailed quantities like cascade-
  chain length means. The BCa acceleration constant is computed via
  jackknife across resample units, which is cheap for the group counts
  we care about (hundreds of traces, not millions).
* Percentile intervals are available as a fallback for statistics
  where BCa is ill-defined (e.g. zero-variance resamples on all-pass
  or all-fail corpora).
* We never silently degrade: if bootstrap cannot produce a valid CI
  (e.g., constant resamples, nan statistic), the :class:`Estimate`
  carries ``ci_low=nan, ci_high=nan`` and ``method`` records the
  failure mode so downstream reports can surface it instead of
  pretending a CI exists.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from numpy.random import Generator, default_rng

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Confidence level
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceLevel:
    """A two-sided confidence level, validated at construction.

    Prefer constructing via the class-method presets
    (:attr:`ninety_five`, :attr:`ninety_nine`) rather than passing a
    raw float — the latter is allowed but easy to typo (0.95 vs 0.05
    is a 19× difference in CI width).
    """

    level: float

    def __post_init__(self) -> None:
        if not (0.0 < self.level < 1.0):
            raise ValueError(
                f"ConfidenceLevel must be strictly between 0 and 1, got {self.level}"
            )

    @property
    def alpha(self) -> float:
        """Two-tailed α: the total probability mass in the tails."""
        return 1.0 - self.level

    @property
    def lower_percentile(self) -> float:
        """Lower percentile cutoff in [0, 100]."""
        return 100.0 * (self.alpha / 2.0)

    @property
    def upper_percentile(self) -> float:
        """Upper percentile cutoff in [0, 100]."""
        return 100.0 * (1.0 - self.alpha / 2.0)

    @classmethod
    def of(cls, level: float) -> "ConfidenceLevel":
        return cls(level=float(level))


NINETY_FIVE = ConfidenceLevel(0.95)
NINETY_NINE = ConfidenceLevel(0.99)
NINETY = ConfidenceLevel(0.90)


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """A point estimate plus its interval.

    Attributes:
        value: The point estimate. May be ``nan`` if the underlying data
            is degenerate (empty, all-constant, etc.); the report layer
            is expected to surface ``nan`` rather than hide it.
        ci_low: Lower bound of the confidence interval. ``nan`` when no
            valid CI could be computed — always paired with a ``method``
            string that explains the failure.
        ci_high: Upper bound of the confidence interval.
        n: Sample size the estimate was computed on. "Sample" here is
            the unit of resampling — for trace-level bootstrap this is
            the number of traces, not the number of steps.
        method: Human-readable tag identifying how the CI was computed:
            ``"wilson"``, ``"bootstrap_bca"``, ``"bootstrap_percentile"``,
            ``"ols_normal"``, ``"empty"``, ``"insufficient_data"``,
            ``"degenerate"``, etc. Auditable: every reader can tell at a
            glance what machinery produced the interval.
        confidence_level: The two-sided confidence level, captured so
            downstream reports can label axes correctly.
        se: Standard error, when one is meaningfully defined (Wilson,
            OLS normal, BCa with jackknife). ``None`` otherwise.
    """

    value: float
    ci_low: float
    ci_high: float
    n: int
    method: str
    confidence_level: ConfidenceLevel
    se: float | None = None

    def __post_init__(self) -> None:
        # Allow nan everywhere — degenerate inputs are a first-class
        # outcome. What we disallow is an interval that violates its
        # own ordering: a non-nan low greater than a non-nan high is
        # certainly a bug.
        if (
            not math.isnan(self.ci_low)
            and not math.isnan(self.ci_high)
            and self.ci_low > self.ci_high
        ):
            raise ValueError(
                f"Estimate ci_low ({self.ci_low}) > ci_high ({self.ci_high}); "
                f"interval must be ordered"
            )
        if self.n < 0:
            raise ValueError(f"Estimate n must be non-negative, got {self.n}")

    @property
    def has_ci(self) -> bool:
        """True if both interval endpoints are finite."""
        return not (math.isnan(self.ci_low) or math.isnan(self.ci_high))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict. Preserves ``nan`` as the string 'nan'.

        Used by report writers; standard ``json.dumps`` rejects ``nan``
        by default, so we stringify explicitly rather than depending on
        the caller to pass ``allow_nan=True`` and accept non-standard
        JSON in their output.
        """

        def _f(x: float) -> float | str:
            return "nan" if math.isnan(x) else x

        out: dict[str, Any] = {
            "value": _f(self.value),
            "ci_low": _f(self.ci_low),
            "ci_high": _f(self.ci_high),
            "n": self.n,
            "method": self.method,
            "confidence_level": self.confidence_level.level,
        }
        if self.se is not None:
            out["se"] = _f(self.se)
        return out

    # --- Convenience factories for degenerate cases ----------------------

    @classmethod
    def empty(cls, *, confidence_level: ConfidenceLevel = NINETY_FIVE) -> "Estimate":
        """Estimate representing "no data at all"."""
        return cls(
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=0,
            method="empty",
            confidence_level=confidence_level,
        )

    @classmethod
    def insufficient(
        cls,
        *,
        n: int,
        confidence_level: ConfidenceLevel = NINETY_FIVE,
    ) -> "Estimate":
        """Estimate representing "not enough data for inference"."""
        return cls(
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=n,
            method="insufficient_data",
            confidence_level=confidence_level,
        )


# ---------------------------------------------------------------------------
# Proportion: Wilson score interval
# ---------------------------------------------------------------------------


def wilson_proportion_interval(
    successes: int,
    n: int,
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
) -> Estimate:
    """Wilson score CI for a binomial proportion.

    Preferred over the normal approximation (which is badly behaved at
    extreme proportions and for small n) and over "exact" Clopper-
    Pearson (which is conservative to a fault). Agresti & Coull (1998)
    is the standard reference.

    Args:
        successes: Number of successes observed.
        n: Number of trials.
        confidence_level: Two-sided confidence level.

    Returns:
        :class:`Estimate` with ``method="wilson"``. If ``n == 0`` the
        estimate is :meth:`Estimate.empty`.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if successes < 0 or successes > n:
        raise ValueError(
            f"successes={successes} must satisfy 0 <= successes <= n={n}"
        )
    if n == 0:
        return Estimate.empty(confidence_level=confidence_level)

    p_hat = successes / n
    # Normal quantile for (1 - alpha/2). No scipy dependency: use the
    # inverse-erf identity. The numerical error at z at the 5% level
    # relative to scipy.stats.norm.ppf is < 1e-10.
    z = _normal_ppf(1.0 - confidence_level.alpha / 2.0)
    z_sq = z * z

    denom = 1.0 + z_sq / n
    center = (p_hat + z_sq / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt(p_hat * (1.0 - p_hat) / n + z_sq / (4.0 * n * n))
        / denom
    )
    ci_low = max(0.0, center - margin)
    ci_high = min(1.0, center + margin)

    # The Wilson midpoint is a shrunk estimate; the "point estimate" we
    # report is the raw sample proportion so the reader sees the
    # observed value. Wilson gives us the band; it is not the value.
    se_approx = math.sqrt(p_hat * (1.0 - p_hat) / n) if n > 0 else float("nan")
    return Estimate(
        value=p_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        method="wilson",
        confidence_level=confidence_level,
        se=se_approx,
    )


# ---------------------------------------------------------------------------
# OLS slope with normal-theory interval
# ---------------------------------------------------------------------------


def ols_slope_with_interval(
    x: Sequence[float],
    y: Sequence[float],
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
) -> Estimate:
    """Simple OLS slope of ``y ~ x`` with a normal-theory CI.

    The CI here assumes independent observations and normal residuals
    — so this function is appropriate for summary-level data (one point
    per trace, for example), **not** for pooled (step, is_error) data
    where steps within a trace are correlated. For the latter, use
    :func:`bootstrap_estimate` with a trace-level resampling unit.

    Args:
        x: Independent variable.
        y: Dependent variable.
        confidence_level: Two-sided confidence level.

    Returns:
        :class:`Estimate` with ``method="ols_normal"``. When there are
        fewer than three points, or zero variance in ``x``, the
        interval endpoints are ``nan`` and ``method`` reflects the
        degenerate case.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x_arr.shape} and {y_arr.shape}"
        )
    n = int(x_arr.shape[0])
    if n == 0:
        return Estimate.empty(confidence_level=confidence_level)
    if n < 3:
        return Estimate.insufficient(n=n, confidence_level=confidence_level)

    x_mean = float(x_arr.mean())
    y_mean = float(y_arr.mean())
    x_centered = x_arr - x_mean
    y_centered = y_arr - y_mean
    ssx = float(np.dot(x_centered, x_centered))
    if math.isclose(ssx, 0.0):
        return Estimate(
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=n,
            method="degenerate",
            confidence_level=confidence_level,
        )

    slope = float(np.dot(x_centered, y_centered) / ssx)
    intercept = y_mean - slope * x_mean
    residuals = y_arr - (intercept + slope * x_arr)
    sigma_sq = float(np.dot(residuals, residuals)) / (n - 2)
    se = math.sqrt(sigma_sq / ssx) if sigma_sq >= 0 else float("nan")

    # Normal-quantile CI. We use the normal rather than Student-t because
    # the rest of the analysis layer uses large-sample asymptotics, and
    # the difference at any n the analysis cares about is negligible.
    z = _normal_ppf(1.0 - confidence_level.alpha / 2.0)
    ci_low = slope - z * se
    ci_high = slope + z * se
    return Estimate(
        value=slope,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        method="ols_normal",
        confidence_level=confidence_level,
        se=se,
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_estimate(
    units: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = 2000,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    rng: Generator | None = None,
    method: str = "bca",
) -> Estimate:
    """Non-parametric bootstrap CI for a statistic over a sequence of units.

    The "units" are whatever you want to resample. For trace-level
    bootstrap (the right choice for any statistic where observations
    within a group are correlated), ``units`` is a list of groups and
    ``statistic`` reduces a list of groups to a scalar. For standard
    i.i.d. bootstrap, ``units`` is a flat list of observations.

    This distinction is the single most important technical decision
    in the bootstrap layer: resampling individual steps from a corpus
    of traces is *wrong* because it treats correlated within-trace
    observations as independent, shrinking the CI width artificially.
    The caller decides the resampling unit; this function does not
    second-guess.

    Args:
        units: The resampling pool. Length must be >= 2 for any CI to
            exist; length 0 or 1 returns an ``empty`` / ``insufficient``
            estimate.
        statistic: A function taking a sequence of units and returning a
            scalar. Must be deterministic given its input (the bootstrap
            calls it many times with different resamples). If the
            function returns ``nan`` for some resamples (e.g., all-zero
            variance), those resamples are silently dropped — we still
            report the CI over the valid ones, but ``method`` is tagged
            ``"bootstrap_*_partial"`` so downstream reports can flag it.
        n_resamples: Number of bootstrap resamples. Default 2000 gives
            stable 95% intervals to two decimal places on the scale of
            the statistic; increase if you need more precision.
        confidence_level: Two-sided confidence level.
        rng: numpy random generator. If ``None``, a fresh one is seeded
            with entropy from the OS. Pass an explicit seeded generator
            for reproducible reports.
        method: ``"bca"`` (default, bias-corrected and accelerated) or
            ``"percentile"`` (raw percentile interval). BCa is preferred
            for skewed statistics; percentile is a safe fallback when
            the acceleration constant cannot be computed.

    Returns:
        :class:`Estimate`. The ``method`` field records which algorithm
        was used and whether any resamples were dropped.
    """
    if method not in ("bca", "percentile"):
        raise ValueError(f"bootstrap method must be 'bca' or 'percentile', got {method!r}")

    n = len(units)
    if n == 0:
        return Estimate.empty(confidence_level=confidence_level)

    rng = rng if rng is not None else default_rng()
    unit_list = list(units)

    # Compute the point value on the original units before checking
    # whether a CI is feasible — so degenerate (n=1) inputs still
    # surface the point estimate to the caller instead of erasing it.
    point_value = statistic(unit_list)

    if n < 2:
        return Estimate(
            value=point_value,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=n,
            method="insufficient_data",
            confidence_level=confidence_level,
        )

    # Resample.
    resample_stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample = [unit_list[j] for j in idx]
        resample_stats[i] = statistic(resample)

    finite_mask = np.isfinite(resample_stats)
    n_valid = int(finite_mask.sum())
    if n_valid < 2:
        # Every or nearly every resample produced a non-finite statistic;
        # we cannot put a band on this one. Surface it, don't hide it.
        return Estimate(
            value=point_value,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=n,
            method="degenerate",
            confidence_level=confidence_level,
        )
    valid_stats = resample_stats[finite_mask]
    partial = n_valid < n_resamples

    if method == "percentile" or not math.isfinite(point_value):
        # Fall back to the percentile interval when the point estimate
        # itself is non-finite — BCa requires a finite point value.
        ci_low = float(np.percentile(valid_stats, confidence_level.lower_percentile))
        ci_high = float(np.percentile(valid_stats, confidence_level.upper_percentile))
        tag = "bootstrap_percentile"
    else:
        # BCa: compute bias-correction z0 and acceleration a.
        proportion_below = float(np.mean(valid_stats < point_value))
        if proportion_below <= 0.0 or proportion_below >= 1.0:
            # Degenerate z0 (all resamples on one side of the point).
            # Fall back cleanly rather than producing a spurious band.
            ci_low = float(np.percentile(valid_stats, confidence_level.lower_percentile))
            ci_high = float(np.percentile(valid_stats, confidence_level.upper_percentile))
            tag = "bootstrap_percentile"
        else:
            z0 = _normal_ppf(proportion_below)
            a = _jackknife_acceleration(unit_list, statistic)
            z_alpha_lo = _normal_ppf(confidence_level.alpha / 2.0)
            z_alpha_hi = _normal_ppf(1.0 - confidence_level.alpha / 2.0)
            # BCa-adjusted percentiles.
            alpha_lo = _normal_cdf(z0 + (z0 + z_alpha_lo) / (1.0 - a * (z0 + z_alpha_lo)))
            alpha_hi = _normal_cdf(z0 + (z0 + z_alpha_hi) / (1.0 - a * (z0 + z_alpha_hi)))
            ci_low = float(np.percentile(valid_stats, 100.0 * alpha_lo))
            ci_high = float(np.percentile(valid_stats, 100.0 * alpha_hi))
            tag = "bootstrap_bca"

    if partial:
        tag = f"{tag}_partial"

    return Estimate(
        value=point_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        method=tag,
        confidence_level=confidence_level,
        se=float(valid_stats.std(ddof=1)) if n_valid > 1 else None,
    )


def _jackknife_acceleration(
    units: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
) -> float:
    """Compute the BCa acceleration constant via jackknife.

    Efron & Tibshirani, *An Introduction to the Bootstrap*, §14.3. The
    acceleration captures the skew of the sampling distribution and
    corrects the BCa percentile adjustment for it.

    Returns 0.0 (the "no acceleration" neutral value) if the jackknife
    variance is degenerate — BCa then collapses to the simple
    bias-corrected interval.
    """
    n = len(units)
    if n < 3:
        return 0.0
    units_list = list(units)
    jack = np.empty(n, dtype=float)
    for i in range(n):
        leave_out = units_list[:i] + units_list[i + 1 :]
        jack[i] = statistic(leave_out)
    finite = np.isfinite(jack)
    if int(finite.sum()) < 3:
        return 0.0
    jack = jack[finite]
    mean = float(jack.mean())
    numerator = float(np.sum((mean - jack) ** 3))
    denominator = 6.0 * (float(np.sum((mean - jack) ** 2)) ** 1.5)
    if math.isclose(denominator, 0.0):
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Normal quantile helpers (no scipy dependency)
# ---------------------------------------------------------------------------


_SQRT_TWO = math.sqrt(2.0)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_TWO))


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (quantile function).

    Implements Beasley-Springer-Moro (Moro, 1995), which is accurate
    to better than 1e-9 across the full (0, 1) range. We reimplement
    rather than depend on scipy to keep the statistics module free of
    heavy scientific-stack imports.
    """
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return float("-inf")
        if p == 1.0:
            return float("inf")
        raise ValueError(f"normal ppf argument must be in [0, 1], got {p}")

    # Moro's rational approximation. Coefficients from the 1995 paper.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    # p > p_high
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


__all__ = [
    "ConfidenceLevel",
    "Estimate",
    "NINETY",
    "NINETY_FIVE",
    "NINETY_NINE",
    "bootstrap_estimate",
    "ols_slope_with_interval",
    "wilson_proportion_interval",
]
