"""Change-point detection on per-step error / outcome series.

The degradation hypothesis predicts a smooth slope, but real traces
sometimes contain abrupt regime shifts: an agent loses the plot at
step 12 and everything afterwards is wrong. A pure slope summary
hides that. Change-point detection complements the slope analysis
by asking "is there a single step where the local rate jumps?"

This module exposes two estimators:

* :func:`pelt_change_points` — multi-change-point detection via
  ``ruptures``' PELT (Pruned Exact Linear Time) algorithm. Optimal
  segmentation under an L2 cost with a complexity penalty. This is
  the production estimator. ``ruptures`` is an optional dependency;
  callers without it get a clear ``RuntimeError``.
* :func:`naive_change_point` — the original O(n²) single-change-
  point binary-segmentation routine, kept as a zero-dependency
  fallback for tiny series and as a baseline the PELT result can be
  cross-checked against in tests.

The :class:`ChangePointResult` carries the segmentation, the
per-segment means, and the input length so callers can always plot
or annotate without re-running the fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChangePointResult:
    """Output of a change-point segmentation.

    Attributes:
        change_points: Indices at which the series is segmented. By
            convention these are *exclusive* right endpoints in the
            ruptures style: ``[5, 12]`` on a length-20 series means
            three segments ``[0,5)``, ``[5,12)``, ``[12,20)``. The
            final value is always the series length so segment
            iteration is uniform; an empty list means no internal
            change point was found.
        segment_means: Mean of each segment, in segment order.
        method: Estimator name (e.g. ``"pelt_l2"``, ``"naive_bs"``).
        n: Length of the input series.
        cost: Final fitted cost (lower is better). May be ``nan``
            if the estimator does not expose one.
        metadata: Estimator-specific extras (penalty value, model
            type, etc.) for audit.
    """

    change_points: list[int]
    segment_means: list[float]
    method: str
    n: int
    cost: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def segments(self) -> list[tuple[int, int]]:
        """Yield ``(start, end)`` half-open intervals for each segment."""
        boundaries = [0, *self.change_points]
        if not self.change_points or self.change_points[-1] != self.n:
            boundaries.append(self.n)
        return [
            (boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_points": list(self.change_points),
            "segment_means": list(self.segment_means),
            "method": self.method,
            "n": self.n,
            "cost": self.cost,
            "metadata": dict(self.metadata),
        }


def naive_change_point(series: Sequence[float]) -> int | None:
    """Single-change-point binary segmentation under squared error.

    O(n²). Returns the index ``k`` minimizing within-segment SSE
    over splits ``[0, k) | [k, n)``. Returns ``None`` for series
    shorter than 4 — the segmentation is meaningless on three
    points or fewer.

    This is the original stub kept for use as a tie-breaker /
    cross-check in tests; callers wanting a real estimator should
    use :func:`pelt_change_points`.
    """
    n = len(series)
    if n < 4:
        return None
    best_idx: int | None = None
    best_cost = float("inf")
    total = list(series)
    for k in range(2, n - 1):
        left = total[:k]
        right = total[k:]
        ml = sum(left) / len(left)
        mr = sum(right) / len(right)
        cost = sum((x - ml) ** 2 for x in left) + sum((x - mr) ** 2 for x in right)
        if cost < best_cost:
            best_cost = cost
            best_idx = k
    return best_idx


def pelt_change_points(
    series: Sequence[float],
    *,
    penalty: float | None = None,
    model: str = "l2",
    min_size: int = 2,
    autocorrelation_adjusted: bool = False,
    autocorrelation_alpha: float = 0.05,
) -> ChangePointResult:
    """Multi-change-point segmentation via PELT.

    Args:
        series: One-dimensional numeric series. Length ≥ ``2 * min_size``.
        penalty: Complexity penalty for the BIC-style stopping
            criterion. ``None`` (the default) uses ``log(n)``,
            which is the standard BIC-flavoured choice and works
            well for noisy *independent* outcome series. Larger
            values return fewer change points. See
            ``autocorrelation_adjusted`` for the binary-error case.
        model: Cost model passed to ``ruptures`` (``"l2"`` for
            piecewise-constant means, ``"rbf"`` for distributional
            shifts, ``"l1"`` for robust median shifts). Default
            ``"l2"`` matches the squared-error baseline used by
            :func:`naive_change_point`.
        min_size: Minimum segment length. Setting this above 1
            stops PELT from declaring a regime shift on a single
            outlier.
        autocorrelation_adjusted: When True, run a Ljung-Box test
            on the input series first; if it rejects the white-
            noise null at ``autocorrelation_alpha``, multiply the
            BIC penalty by 2. This is a coarse but principled
            mitigation: the standard BIC derivation assumes i.i.d.
            residuals, and binary error sequences from agent
            traces are typically positively autocorrelated, which
            inflates the effective sample size and produces
            spurious change points. The 2× multiplier roughly
            halves the implied n_eff. Requires ``statsmodels``.
            Recorded in result metadata.
        autocorrelation_alpha: Significance level for the
            Ljung-Box decision when ``autocorrelation_adjusted``
            is True.

    Returns:
        :class:`ChangePointResult`. Series shorter than
        ``2 * min_size`` return an empty change-point list and a
        single full-series segment mean.

    Raises:
        RuntimeError: ``ruptures`` is not installed.
    """
    import math

    n = len(series)
    if n < 2 * min_size:
        mean = sum(series) / n if n else float("nan")
        return ChangePointResult(
            change_points=[],
            segment_means=[mean],
            method=f"pelt_{model}",
            n=n,
            cost=float("nan"),
            metadata={"reason": "series_too_short"},
        )

    try:
        import numpy as np
        import ruptures as rpt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ruptures is required for pelt_change_points; install via "
            "`pip install ruptures`"
        ) from exc

    arr = np.asarray(series, dtype=float)
    pen = penalty if penalty is not None else math.log(n)

    autocorr_p: float | None = None
    autocorr_adjusted_applied = False
    if autocorrelation_adjusted:
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lags = min(5, max(1, n // 4))
            if lags >= 1 and float(np.var(arr)) > 0.0 and len(arr) > lags:
                lb = acorr_ljungbox(arr, lags=[lags], return_df=True)
                autocorr_p = float(lb["lb_pvalue"].iloc[-1])
                if math.isfinite(autocorr_p) and autocorr_p < autocorrelation_alpha:
                    pen = pen * 2.0
                    autocorr_adjusted_applied = True
        except Exception:  # pragma: no cover — diagnostic, never fatal
            autocorr_p = None

    algo = rpt.Pelt(model=model, min_size=min_size).fit(arr)
    breakpoints = algo.predict(pen=pen)
    # ruptures returns the right-exclusive endpoints including n itself.
    change_points = [int(b) for b in breakpoints if b < n]

    boundaries = [0, *change_points, n]
    segment_means = [
        float(arr[boundaries[i] : boundaries[i + 1]].mean())
        for i in range(len(boundaries) - 1)
    ]

    # Total within-segment SSE for the audit field.
    cost = float(
        sum(
            ((arr[boundaries[i] : boundaries[i + 1]] - segment_means[i]) ** 2).sum()
            for i in range(len(boundaries) - 1)
        )
    )

    return ChangePointResult(
        change_points=change_points,
        segment_means=segment_means,
        method=f"pelt_{model}",
        n=n,
        cost=cost,
        metadata={
            "penalty": pen,
            "min_size": min_size,
            "autocorrelation_adjusted": autocorr_adjusted_applied,
            "autocorrelation_pvalue": autocorr_p,
        },
    )


__all__ = [
    "ChangePointResult",
    "naive_change_point",
    "pelt_change_points",
]
