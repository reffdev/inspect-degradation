"""Multiple-comparison correction for families of hypothesis tests.

Phase 3 will routinely generate families of p-values — one per
model, per task category, per rubric dimension — and any report
that cites "these ten coefficients are significant" without a
correction is making a claim it cannot back. This module wraps
``statsmodels.stats.multitest.multipletests`` so correction is
one line and its output carries the same
:class:`~inspect_degradation.analysis.statistics.Estimate` shape
the rest of the analysis layer uses.

We default to Benjamini-Hochberg FDR (``fdr_bh``) because the
research questions this project asks are exploratory and the
cost of a false negative (missing a real degradation signal) is
usually higher than the cost of a false positive. Holm-Bonferroni
is available as ``holm`` for downstream claims that need strict
family-wise error control.

The correction is *not* applied automatically anywhere in the
pipeline — it is the report author's job to decide which family
of tests belongs together. This module exists to make that step
explicit and auditable rather than buried inside a regression
helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inspect_degradation.analysis.mixed_effects import CoefficientRow
from inspect_degradation.analysis.statistics import (
    NINETY_FIVE,
    ConfidenceLevel,
)


@dataclass(frozen=True)
class AdjustedCoefficient:
    """A coefficient with its original and adjusted p-values.

    The point estimate, SE, and CI are copied from the input row
    unchanged — FDR / Holm corrections act on p-values, not on
    interval endpoints. Downstream reports should cite the
    adjusted p-value and the ``rejected`` flag alongside the
    unchanged CI.

    Attributes:
        name: Coefficient name (copied from the input).
        original: The original :class:`CoefficientRow`.
        adjusted_p_value: Corrected p-value.
        rejected: Whether the null is rejected at the family-wise
            α implied by the correction method.
        method: Correction method used (e.g. ``"fdr_bh"``).
    """

    name: str
    original: CoefficientRow
    adjusted_p_value: float
    rejected: bool
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "original": self.original.to_dict(),
            "adjusted_p_value": self.adjusted_p_value,
            "rejected": self.rejected,
            "method": self.method,
        }


@dataclass(frozen=True)
class MultipleComparisonResult:
    """Family of coefficients after multiple-comparison correction.

    Attributes:
        adjusted: One :class:`AdjustedCoefficient` per input, in
            the order supplied.
        method: Correction method applied.
        family_alpha: Family-wise α used for the rejection
            decisions (``1 - confidence_level.level``).
        n_tests: Number of tests in the family.
        n_rejected: Number of tests flagged as rejected after
            correction.
        confidence_level: Confidence level associated with
            ``family_alpha``, for audit.
    """

    adjusted: list[AdjustedCoefficient]
    method: str
    family_alpha: float
    n_tests: int
    n_rejected: int
    confidence_level: ConfidenceLevel
    metadata: dict[str, Any] = field(default_factory=dict)

    def rejected_names(self) -> list[str]:
        return [row.name for row in self.adjusted if row.rejected]

    def to_dict(self) -> dict[str, Any]:
        return {
            "adjusted": [row.to_dict() for row in self.adjusted],
            "method": self.method,
            "family_alpha": self.family_alpha,
            "n_tests": self.n_tests,
            "n_rejected": self.n_rejected,
            "confidence_level": self.confidence_level.level,
            "metadata": dict(self.metadata),
        }


def adjust_coefficients(
    coefficients: list[CoefficientRow],
    *,
    method: str = "fdr_bh",
    confidence_level: ConfidenceLevel = NINETY_FIVE,
) -> MultipleComparisonResult:
    """Apply a multiple-comparison correction to a family of coefficients.

    Args:
        coefficients: The coefficients to treat as a single family.
            Typically drawn from one or more :class:`MixedEffectsResult`
            objects; it is the caller's responsibility to decide what
            belongs in the family.
        method: Correction method. Any value accepted by
            ``statsmodels.stats.multitest.multipletests``; common
            choices are ``"fdr_bh"`` (Benjamini-Hochberg, default),
            ``"fdr_by"`` (Benjamini-Yekutieli, for dependent tests),
            ``"holm"`` (Holm-Bonferroni), and ``"bonferroni"``.
        confidence_level: Determines the family-wise α
            (``1 - level``) used for the rejection decisions.

    Returns:
        :class:`MultipleComparisonResult` with one adjusted row per
        input coefficient. Empty input yields a result with zero
        tests and no rejections (no error raised — this is the
        natural identity case).

    Raises:
        RuntimeError: statsmodels not installed.
        ValueError: any coefficient has a non-finite p-value,
            since the correction methods cannot handle NaN.
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "statsmodels is required for multiple-comparison correction"
        ) from exc

    alpha = 1.0 - confidence_level.level

    if not coefficients:
        return MultipleComparisonResult(
            adjusted=[],
            method=method,
            family_alpha=alpha,
            n_tests=0,
            n_rejected=0,
            confidence_level=confidence_level,
        )

    p_values = []
    for row in coefficients:
        p = row.p_value
        if p is None or not (0.0 <= p <= 1.0):
            raise ValueError(
                f"coefficient {row.name!r} has invalid p-value {p!r}; "
                "correction requires finite p-values in [0, 1]"
            )
        p_values.append(p)

    rejected, adjusted_p, _, _ = multipletests(
        p_values, alpha=alpha, method=method
    )

    adjusted: list[AdjustedCoefficient] = []
    for row, rej, adj_p in zip(coefficients, rejected, adjusted_p, strict=True):
        adjusted.append(
            AdjustedCoefficient(
                name=row.name,
                original=row,
                adjusted_p_value=float(adj_p),
                rejected=bool(rej),
                method=method,
            )
        )

    return MultipleComparisonResult(
        adjusted=adjusted,
        method=method,
        family_alpha=alpha,
        n_tests=len(coefficients),
        n_rejected=int(sum(rejected)),
        confidence_level=confidence_level,
    )


__all__ = [
    "AdjustedCoefficient",
    "MultipleComparisonResult",
    "adjust_coefficients",
]
