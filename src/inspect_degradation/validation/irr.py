"""Inter-rater reliability primitives.

Pure functions over flat sequences — no project types, no I/O. Higher-level
agreement reports (per-dimension, per-task) live in :mod:`.agreement` and
call into this module.

Why we keep our own implementations rather than depending on the
``krippendorff`` package: the project's hard dependencies are already large
(inspect-ai, lifelines, statsmodels), and these implementations are short,
testable, and have no policy (no smoothing, no missing-data heuristics).
Anything more sophisticated belongs behind an explicit opt-in dep.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal, TypeVar

import numpy as np

T = TypeVar("T")


def cohens_kappa(a: Sequence[T], b: Sequence[T]) -> float:
    """Cohen's kappa for two raters over the same items.

    Returns 1.0 on perfect agreement, 0.0 on chance agreement, and can go
    negative for systematic disagreement. Raises on empty input or
    mismatched lengths — silent fallbacks would mask real bugs in the
    pairing layer.
    """
    if len(a) != len(b):
        raise ValueError(f"rater sequences must be same length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("cohens_kappa is undefined for empty sequences")

    labels = sorted({*a, *b}, key=repr)
    idx = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=float)
    for x, y in zip(a, b):
        cm[idx[x], idx[y]] += 1.0

    n = float(len(a))
    po = float(np.trace(cm)) / n
    pe = float((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / (n * n)

    if math.isclose(pe, 1.0):
        # Both raters used a single label everywhere — kappa is undefined,
        # but agreement is trivially perfect or trivially zero.
        return 1.0 if math.isclose(po, 1.0) else 0.0
    return (po - pe) / (1.0 - pe)


def krippendorff_alpha_nominal(ratings: Sequence[Sequence[T | None]]) -> float:
    """Krippendorff's alpha for nominal data.

    ``ratings`` is a raters-by-items matrix; ``None`` marks missing values.
    Items with fewer than two raters are ignored, matching the standard
    definition.
    """
    if len(ratings) < 2:
        raise ValueError("krippendorff_alpha_nominal requires at least two raters")
    n_items = len(ratings[0])
    if any(len(r) != n_items for r in ratings):
        raise ValueError("all raters must score the same number of items")

    values = sorted({v for r in ratings for v in r if v is not None}, key=repr)
    if not values:
        raise ValueError("no non-null ratings provided")
    v_idx = {v: i for i, v in enumerate(values)}
    k = len(values)
    coincidence = np.zeros((k, k), dtype=float)

    for item in range(n_items):
        present = [r[item] for r in ratings if r[item] is not None]
        m = len(present)
        if m < 2:
            continue
        weight = 1.0 / (m - 1)
        for i, vi in enumerate(present):
            for j, vj in enumerate(present):
                if i == j:
                    continue
                coincidence[v_idx[vi], v_idx[vj]] += weight

    n_c = coincidence.sum()
    if n_c == 0:
        raise ValueError("no items had at least two raters; alpha is undefined")
    n_v = coincidence.sum(axis=1)
    do = 1.0 - float(np.trace(coincidence)) / n_c
    de = 1.0 - float((n_v * n_v).sum()) / (n_c * n_c)
    if math.isclose(de, 0.0):
        return 1.0
    return 1.0 - do / de


WeightingScheme = Literal["linear", "quadratic"]


def weighted_cohens_kappa(
    a: Sequence[T],
    b: Sequence[T],
    *,
    rank: Callable[[T], int],
    weights: WeightingScheme = "linear",
) -> float:
    """Cohen's kappa with disagreement weights, for ordinal categories.

    The standard nominal kappa treats every disagreement as equally bad.
    For ordinal categories — e.g. complexity ``low / medium / high`` —
    that throws away information: confusing ``low`` for ``high`` is a
    bigger error than confusing ``low`` for ``medium``. Weighted kappa
    rescales each off-diagonal cell by how far apart the two ratings are
    on the ordinal scale.

    Args:
        a, b: Two raters' label sequences over the same items.
        rank: Maps each label to its integer position on the ordinal
            scale. Must be defined for every label that appears in either
            sequence.
        weights: ``"linear"`` weights penalize disagreement proportionally
            to the rank distance; ``"quadratic"`` penalizes by the square
            of the rank distance, which is the more common social-science
            convention.

    Returns ``1.0`` for perfect agreement, ``0.0`` for chance, and can go
    negative for systematic disagreement.
    """
    if len(a) != len(b):
        raise ValueError(f"rater sequences must be same length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("weighted_cohens_kappa is undefined for empty sequences")

    labels = sorted({*a, *b}, key=rank)
    idx = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=float)
    for x, y in zip(a, b):
        cm[idx[x], idx[y]] += 1.0

    n = float(len(a))
    p_observed = cm / n
    p_expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / (n * n)

    # Build the disagreement weight matrix (0 on the diagonal, larger
    # off-diagonal). The classical kappa formula is
    #     1 - sum(w * p_obs) / sum(w * p_exp)
    # where w_ii = 0 and w_ij grows with |i - j|.
    ranks = np.array([rank(label) for label in labels], dtype=float)
    diff = np.abs(ranks[:, None] - ranks[None, :])
    max_diff = diff.max() if diff.max() > 0 else 1.0
    if weights == "linear":
        w = diff / max_diff
    elif weights == "quadratic":
        w = (diff / max_diff) ** 2
    else:
        raise ValueError(f"unknown weighting scheme: {weights!r}")

    numerator = float((w * p_observed).sum())
    denominator = float((w * p_expected).sum())
    if math.isclose(denominator, 0.0):
        # Both raters used a single label everywhere — agreement is
        # trivially perfect (numerator is also 0).
        return 1.0
    return 1.0 - numerator / denominator


def pearson_r(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson product-moment correlation, used for continuous dimensions.

    Returns ``nan`` when either sequence has zero variance — in that
    setting correlation is mathematically undefined and we refuse to
    paper over it with a default.
    """
    if len(a) != len(b):
        raise ValueError(f"sequences must be same length, got {len(a)} and {len(b)}")
    if len(a) < 2:
        raise ValueError("pearson_r requires at least two points")
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if np.isclose(av.std(), 0.0) or np.isclose(bv.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def accuracy(a: Sequence[T], b: Sequence[T]) -> float:
    """Exact-match agreement rate. Used for boolean dimensions like ``recovery``."""
    if len(a) != len(b):
        raise ValueError(f"sequences must be same length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("accuracy is undefined for empty sequences")
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)
