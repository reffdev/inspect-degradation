"""Per-dimension agreement reports between predicted and reference grades.

The validation pipeline asks: *for each rubric dimension, how well
does the LLM grader match human labels?* The metric for each
dimension is fixed by the dimension's measurement type:

* Nominal categorical (``validity``, ``dependency``) → Cohen's kappa.
* Ordinal categorical (``complexity``, ``severity``) → linear-weighted
  Cohen's kappa.

The mapping lives in the ``_DIMENSIONS`` table so every report
computed anywhere in the package uses the same metric for the same
dimension; two different scripts' reports are directly comparable
as long as they go through this module.

## Uncertainty

Point values for every metric carry a **bootstrap confidence
interval**. Because grade pairs within a trace are not independent
(a single grader can have systematic misjudgments at certain trace
positions or contexts), the resampling unit is **the trace**, not
the pair. Each trace's pairs travel together through bootstrap
resamples; κ is recomputed on the flattened pair list at every
resample. This is the cluster-bootstrap variant of IRR CI
estimation — a little more conservative than the IID pair-level
bootstrap, and the right choice when the question is "how reliable
is the grader on a *new* trace" rather than "how reliable is the
grader on a *new* step from the same traces."

Dimensions with no labeled pairs (e.g. severity on a corpus where
no steps are marked ``fail``) return a degenerate estimate with
``method="empty"``. This is surfaced, not hidden.

Separation of concerns between pairing and scoring is preserved:
:func:`pair_grades` returns a flat pair list for inspection and
subsetting; :func:`score_agreement` consumes the flat list. The
bootstrap machinery is invoked internally from ``score_agreement``
via :func:`_score_dimension`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from numpy.random import Generator

from inspect_degradation.analysis.statistics import (
    NINETY_FIVE,
    ConfidenceLevel,
    Estimate,
    bootstrap_estimate,
)
from inspect_degradation.schema import (
    ComplexityLevel,
    GradedStep,
    GradedTrace,
    SeverityLevel,
)
from inspect_degradation.validation.irr import (
    cohens_kappa,
    weighted_cohens_kappa,
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionAgreement:
    """Agreement on a single rubric dimension, with uncertainty.

    Attributes:
        dimension: Field name on :class:`GradedStep`.
        metric: Name of the IRR statistic used
            (``"cohens_kappa"``, ``"weighted_cohens_kappa"``).
        estimate: :class:`Estimate` carrying point value, CI, sample
            size, and method tag. The sample size is the number of
            **trace-groups**, not the number of pairs, because that's
            the cluster-bootstrap resampling unit. The number of pairs
            is preserved separately as ``n_pairs``.
        n_pairs: Number of ``(predicted, reference)`` pairs that
            contributed a non-null label on this dimension to the
            point-value computation. Distinct from ``estimate.n``
            because the resampling unit is the trace, not the pair.
    """

    dimension: str
    metric: str
    estimate: Estimate
    n_pairs: int

    @property
    def value(self) -> float:
        return self.estimate.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "metric": self.metric,
            "estimate": self.estimate.to_dict(),
            "n_pairs": self.n_pairs,
        }


@dataclass(frozen=True)
class AgreementReport:
    """Aggregate agreement across all dimensions for one grader run."""

    grader: str
    n_pairs: int
    per_dimension: dict[str, DimensionAgreement] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grader": self.grader,
            "n_pairs": self.n_pairs,
            "per_dimension": {
                k: v.to_dict() for k, v in self.per_dimension.items()
            },
        }


@dataclass(frozen=True)
class GradePair:
    """A predicted/reference pair for a single (trace, step) coordinate."""

    trace_id: str
    step_index: int
    predicted: GradedStep
    reference: GradedStep


def pair_grades(
    predicted: Iterable[GradedTrace],
    reference: Iterable[GradedTrace],
) -> list[GradePair]:
    """Pair predicted grades with reference grades by ``(trace_id, step_index)``.

    Steps that exist in only one side are dropped silently — the
    expected behavior when the grader skipped a step (e.g., a
    provider error exhausted retries) or when the reference covers
    more steps than the grader was asked to judge. Callers wanting a
    stricter contract should compare counts before and after.
    """
    ref_index: dict[tuple[str, int], GradedStep] = {}
    for trace in reference:
        for step in trace.steps:
            ref_index[(trace.trace_id, step.step_index)] = step

    pairs: list[GradePair] = []
    for trace in predicted:
        for step in trace.steps:
            key = (trace.trace_id, step.step_index)
            ref_step = ref_index.get(key)
            if ref_step is None:
                continue
            pairs.append(
                GradePair(
                    trace_id=trace.trace_id,
                    step_index=step.step_index,
                    predicted=step,
                    reference=ref_step,
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# Dimension table
# ---------------------------------------------------------------------------


def _present(value: Any) -> bool:
    return value is not None


def _ordinal_rank(value: Any) -> int:
    return value.rank  # type: ignore[no-any-return]


@dataclass(frozen=True)
class _DimensionSpec:
    name: str
    metric: str
    extract: Callable[[GradedStep], Any]
    rank: Callable[[Any], int] | None = None


_DIMENSIONS: tuple[_DimensionSpec, ...] = (
    _DimensionSpec("validity", "cohens_kappa", lambda s: s.validity.value),
    _DimensionSpec(
        "dependency",
        "cohens_kappa",
        lambda s: s.dependency.value if s.dependency is not None else None,
    ),
    _DimensionSpec(
        "complexity",
        "weighted_cohens_kappa",
        lambda s: s.complexity,
        rank=_ordinal_rank,
    ),
    _DimensionSpec(
        "severity",
        "weighted_cohens_kappa",
        lambda s: s.severity,
        rank=_ordinal_rank,
    ),
)

# Sanity check at import time: every ordinal extractor must exist on
# GradedStep, and the level enums must all expose .rank.
_ = ComplexityLevel.low.rank, SeverityLevel.medium.rank


# ---------------------------------------------------------------------------
# Bootstrap-aware scoring
# ---------------------------------------------------------------------------


def score_agreement(
    grader: str,
    pairs: Sequence[GradePair],
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_resamples: int = 2000,
    rng: Generator | None = None,
) -> AgreementReport:
    """Compute a per-dimension agreement report with bootstrap CIs.

    The bootstrap resampling unit is the **trace**: pairs are grouped
    by ``trace_id``, whole groups are resampled with replacement, the
    pair list is flattened, and κ is recomputed on the flattened
    sample. This is the cluster-bootstrap approach appropriate for
    nested data where pairs within a trace are not independent.

    Args:
        grader: Grader label to record in the report.
        pairs: Flat list of :class:`GradePair`.
        confidence_level: Two-sided CI level for every dimension.
        n_resamples: Number of bootstrap resamples per dimension.
            Kappa is a summary statistic, so a few hundred resamples
            is usually sufficient — default 2000 matches the rest of
            the analysis layer for predictability.
        rng: Optional seeded generator for reproducibility.

    Returns:
        An :class:`AgreementReport` with one :class:`DimensionAgreement`
        per rubric dimension.
    """
    # Group pairs by trace_id for cluster-bootstrap resampling.
    clusters: dict[str, list[GradePair]] = {}
    for pair in pairs:
        clusters.setdefault(pair.trace_id, []).append(pair)
    cluster_list = list(clusters.values())

    per_dim: dict[str, DimensionAgreement] = {}
    for spec in _DIMENSIONS:
        per_dim[spec.name] = _score_dimension(
            spec,
            clusters=cluster_list,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            rng=rng,
        )

    return AgreementReport(
        grader=grader, n_pairs=len(pairs), per_dimension=per_dim
    )


def _score_dimension(
    spec: _DimensionSpec,
    *,
    clusters: Sequence[Sequence[GradePair]],
    confidence_level: ConfidenceLevel,
    n_resamples: int,
    rng: Generator | None,
) -> DimensionAgreement:
    """Compute point value + cluster-bootstrap CI for one dimension."""
    # Count labeled pairs on this dimension from the flat corpus —
    # this is the ``n_pairs`` we report alongside the estimate.
    labeled_pairs = 0
    for cluster in clusters:
        for pair in cluster:
            if _present(spec.extract(pair.predicted)) and _present(
                spec.extract(pair.reference)
            ):
                labeled_pairs += 1

    if labeled_pairs == 0:
        return DimensionAgreement(
            dimension=spec.name,
            metric=spec.metric,
            estimate=Estimate.empty(confidence_level=confidence_level),
            n_pairs=0,
        )

    def statistic(unit_clusters: Sequence[Sequence[GradePair]]) -> float:
        a: list[Any] = []
        b: list[Any] = []
        for cluster in unit_clusters:
            for pair in cluster:
                pv = spec.extract(pair.predicted)
                rv = spec.extract(pair.reference)
                if not _present(pv) or not _present(rv):
                    continue
                a.append(pv)
                b.append(rv)
        if not a:
            return float("nan")
        if spec.metric == "cohens_kappa":
            return cohens_kappa(a, b)
        if spec.metric == "weighted_cohens_kappa":
            assert spec.rank is not None
            return weighted_cohens_kappa(a, b, rank=spec.rank, weights="linear")
        raise ValueError(f"unknown agreement metric: {spec.metric!r}")

    estimate = bootstrap_estimate(
        clusters,
        statistic,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        rng=rng,
    )
    return DimensionAgreement(
        dimension=spec.name,
        metric=spec.metric,
        estimate=estimate,
        n_pairs=labeled_pairs,
    )
