"""Grader measurement-error correction.

The LLM grader does not produce ground-truth labels. It produces a
noisy measurement, and Phase 1's TRAIL validation gives us an
estimate of the noise structure (the confusion matrix against human
labels). When we then fit a regression on the grader's outputs —
``is_error ~ step_index + ...`` — the coefficients are biased
*toward zero* (the classical attenuation bias; see Carroll et al.,
*Measurement Error in Nonlinear Models*, 2nd ed., §3).

Concretely: a true per-step degradation slope of 0.04 might appear
as 0.028 in a grader with a κ of 0.7, even with a perfectly
specified regression. Reporting 0.028 without correction
**understates real degradation** and is the kind of thing a
referee will catch. This module implements the standard
corrections.

## What's here

* :class:`ConfusionMatrix` — the structured grader-vs-human
  confusion matrix, built from per-class counts or directly from
  a sequence of :class:`~inspect_degradation.validation.agreement.GradePair`
  objects. Exposes TPR, FPR, TNR, FNR, accuracy, and the κ for
  back-reference.

* :func:`deconfound_proportion` — Method-of-Moments correction for
  a single binary rate. Given a measured proportion and a
  confusion matrix, returns the implied true proportion with a
  propagated CI.

* :func:`simex_correct` — SIMEX (Simulation-Extrapolation) slope
  correction for any regression model. Adds known amounts of
  additional noise to the outcome column, refits at each noise
  level, and extrapolates back to zero noise. Robust to the
  particular shape of the regression — works with
  :func:`~inspect_degradation.analysis.mixed_effects.fit_step_level_model`,
  a plain OLS, or anything else the caller wraps in a fit-function
  closure.

## SIMEX in one paragraph

Cook & Stefanski (1994). Let θ be the true regression coefficient.
With measurement noise variance σ², the naive estimate converges
to some biased value b(σ²). SIMEX computes b(σ²(1+λ)) for several
values of λ > 0 (adding *extra* noise on top of the existing
noise), fits a smooth curve to the b(λ) sequence, and extrapolates
to λ = −1 — which corresponds to zero total noise. The
extrapolated value is the corrected coefficient estimate. For a
binary outcome the "noise" is a label flip with probability that
matches the confusion-matrix miss rate, and λ = 1 means "add
another independent draw of the same flip process."

This implementation ships the **binary label-flip** noise model
because that's what an LLM grader produces on a categorical
label. Other noise models (additive Gaussian for continuous
outcomes) would go in a parallel function; this one stays
focused on the classification-error case.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng

from inspect_degradation.analysis.statistics import (
    NINETY_FIVE,
    ConfidenceLevel,
    Estimate,
    wilson_proportion_interval,
)


# ---------------------------------------------------------------------------
# ConfusionMatrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionMatrix:
    """Binary confusion matrix: grader (measured) vs human (true).

    Oriented as "true class in rows, predicted class in columns" in
    the canonical binary form:

    ============== ============== ==============
                   grader: fail   grader: not
    ============== ============== ==============
    truth: fail    ``tp``         ``fn``
    truth: not     ``fp``         ``tn``
    ============== ============== ==============

    The counts are the primitive; all derived metrics (TPR, FPR,
    accuracy) are computed on demand.

    Attributes:
        tp: True positives (truth=fail, grader=fail).
        fp: False positives (truth=not-fail, grader=fail).
        tn: True negatives (truth=not-fail, grader=not-fail).
        fn: False negatives (truth=fail, grader=not-fail).
        label: Short identifier for which class is the "positive"
            class — typically ``"fail"`` for the error-detection
            use case. Recorded for audit so a reader can tell at a
            glance which direction the matrix faces.
    """

    tp: int
    fp: int
    tn: int
    fn: int
    label: str = "fail"

    def __post_init__(self) -> None:
        for name, value in (
            ("tp", self.tp),
            ("fp", self.fp),
            ("tn", self.tn),
            ("fn", self.fn),
        ):
            if value < 0:
                raise ValueError(
                    f"ConfusionMatrix.{name} must be non-negative, got {value}"
                )
        if self.total == 0:
            raise ValueError(
                "ConfusionMatrix requires at least one observation"
            )

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def actual_positive(self) -> int:
        return self.tp + self.fn

    @property
    def actual_negative(self) -> int:
        return self.fp + self.tn

    @property
    def predicted_positive(self) -> int:
        return self.tp + self.fp

    @property
    def predicted_negative(self) -> int:
        return self.tn + self.fn

    @property
    def tpr(self) -> float:
        """True positive rate (sensitivity, recall, P(grader=pos | truth=pos))."""
        if self.actual_positive == 0:
            return float("nan")
        return self.tp / self.actual_positive

    @property
    def fpr(self) -> float:
        """False positive rate (P(grader=pos | truth=neg))."""
        if self.actual_negative == 0:
            return float("nan")
        return self.fp / self.actual_negative

    @property
    def tnr(self) -> float:
        """True negative rate (specificity, P(grader=neg | truth=neg))."""
        if self.actual_negative == 0:
            return float("nan")
        return self.tn / self.actual_negative

    @property
    def fnr(self) -> float:
        """False negative rate (P(grader=neg | truth=pos))."""
        if self.actual_positive == 0:
            return float("nan")
        return self.fn / self.actual_positive

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total

    def tpr_estimate(
        self, *, confidence_level: ConfidenceLevel = NINETY_FIVE
    ) -> Estimate:
        """Wilson CI on the TPR, using the actual-positive row count as n."""
        return wilson_proportion_interval(
            self.tp, self.actual_positive, confidence_level=confidence_level
        )

    def fpr_estimate(
        self, *, confidence_level: ConfidenceLevel = NINETY_FIVE
    ) -> Estimate:
        """Wilson CI on the FPR, using the actual-negative row count as n."""
        return wilson_proportion_interval(
            self.fp, self.actual_negative, confidence_level=confidence_level
        )

    def cohens_kappa(self) -> float:
        """Cohen's kappa computed from the confusion matrix.

        Useful for cross-checking against
        :func:`inspect_degradation.validation.irr.cohens_kappa` on the
        same pair list — the two should agree to floating-point
        precision when the underlying data is the same. Returns
        ``nan`` if the marginal prevalences are degenerate.
        """
        n = float(self.total)
        if n == 0:
            return float("nan")
        p_observed = (self.tp + self.tn) / n
        p_yes = (self.tp + self.fp) * (self.tp + self.fn) / (n * n)
        p_no = (self.fn + self.tn) * (self.fp + self.tn) / (n * n)
        p_expected = p_yes + p_no
        if math.isclose(p_expected, 1.0):
            return 1.0 if math.isclose(p_observed, 1.0) else 0.0
        return (p_observed - p_expected) / (1.0 - p_expected)

    def to_dict(self) -> dict[str, Any]:
        def _f(x: float) -> float | str:
            return "nan" if math.isnan(x) else x

        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "label": self.label,
            "tpr": _f(self.tpr),
            "fpr": _f(self.fpr),
            "tnr": _f(self.tnr),
            "fnr": _f(self.fnr),
            "accuracy": _f(self.accuracy),
            "cohens_kappa": _f(self.cohens_kappa()),
        }

    # ------------------------------------------------------------------ builders

    @classmethod
    def from_label_pairs(
        cls,
        predicted: Sequence[Any],
        reference: Sequence[Any],
        *,
        positive_label: Any,
        label: str | None = None,
    ) -> "ConfusionMatrix":
        """Build a confusion matrix from aligned predicted/reference sequences.

        ``positive_label`` is the value that counts as "positive"
        (typically :class:`~inspect_degradation.schema.Validity.fail`
        or its string value ``"fail"``). Everything else is
        "negative".
        """
        if len(predicted) != len(reference):
            raise ValueError(
                f"predicted and reference must have equal length; got "
                f"{len(predicted)} and {len(reference)}"
            )
        tp = fp = tn = fn = 0
        for p, r in zip(predicted, reference):
            pred_pos = p == positive_label
            true_pos = r == positive_label
            if pred_pos and true_pos:
                tp += 1
            elif pred_pos and not true_pos:
                fp += 1
            elif (not pred_pos) and true_pos:
                fn += 1
            else:
                tn += 1
        if tp + fp + tn + fn == 0:
            raise ValueError(
                "cannot build ConfusionMatrix from empty label sequences"
            )
        return cls(
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            label=label or str(positive_label),
        )

    @classmethod
    def from_grade_pairs(
        cls,
        pairs: Iterable[Any],
        *,
        extractor: Callable[[Any], Any],
        positive_label: Any,
        label: str | None = None,
    ) -> "ConfusionMatrix":
        """Build a confusion matrix from :class:`GradePair` objects.

        ``extractor`` pulls the label out of a :class:`GradedStep`;
        typically ``lambda s: s.validity`` or
        ``lambda s: s.validity.value``. Pairs where either side
        extracts to ``None`` are dropped.
        """
        predicted: list[Any] = []
        reference: list[Any] = []
        for pair in pairs:
            pv = extractor(pair.predicted)
            rv = extractor(pair.reference)
            if pv is None or rv is None:
                continue
            predicted.append(pv)
            reference.append(rv)
        return cls.from_label_pairs(
            predicted,
            reference,
            positive_label=positive_label,
            label=label,
        )


# ---------------------------------------------------------------------------
# Method-of-Moments deconfounding for a rate
# ---------------------------------------------------------------------------


def deconfound_proportion(
    measured_rate: float,
    confusion: ConfusionMatrix,
    *,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    n_measured: int | None = None,
) -> Estimate:
    """Invert a noisy measured proportion using a known confusion matrix.

    The algebra: if ``p_true`` is the true fraction of positives,
    the expected measured fraction under the confusion matrix is

        ``p_measured = TPR * p_true + FPR * (1 - p_true)``

    Solving for ``p_true``:

        ``p_true = (p_measured - FPR) / (TPR - FPR)``

    The denominator is ``TPR - FPR``, which is always positive for
    a grader better than chance (κ > 0). For a useless grader
    (TPR ≈ FPR) the correction blows up — we surface that with a
    ``method="unusable_confusion_matrix"`` tag on the returned
    estimate rather than a division by zero.

    The CI is computed by bootstrap over the confusion matrix
    counts plus the measured rate's sampling variance. This is a
    first-order approximation (delta method would be fancier) but
    it is transparent and robust for the confusion-matrix sizes we
    actually care about (~148 labeled pairs in TRAIL).

    Args:
        measured_rate: The noisy rate as observed by the grader,
            in [0, 1].
        confusion: The grader's confusion matrix, typically
            computed from Phase 1 TRAIL validation.
        confidence_level: Two-sided confidence level.
        n_measured: Optional sample size for ``measured_rate``. When
            provided, a Wilson interval on the measured rate is
            incorporated into the uncertainty on the corrected
            rate. When None, the correction is treated as a
            deterministic function of the confusion matrix alone
            (the CI reflects only confusion-matrix uncertainty).

    Returns:
        An :class:`Estimate` with the corrected rate and CI, tagged
        ``method="deconfound_proportion"`` on success.
    """
    if not (0.0 <= measured_rate <= 1.0):
        raise ValueError(f"measured_rate must be in [0, 1], got {measured_rate}")

    tpr = confusion.tpr
    fpr = confusion.fpr
    if math.isnan(tpr) or math.isnan(fpr):
        return Estimate(
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=confusion.total,
            method="unusable_confusion_matrix",
            confidence_level=confidence_level,
        )

    denom = tpr - fpr
    if denom <= 0.0 or math.isclose(denom, 0.0):
        return Estimate(
            value=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=confusion.total,
            method="unusable_confusion_matrix",
            confidence_level=confidence_level,
        )

    point = (measured_rate - fpr) / denom
    point_clipped = max(0.0, min(1.0, point))

    # Bootstrap-style uncertainty: sample TPR and FPR from their
    # Wilson intervals, sample the measured rate from its Wilson
    # interval if n_measured is provided, and recompute the
    # corrected rate on each draw. Use a moderate number of draws
    # (500) for speed; the confidence-matrix sample size caps
    # accuracy anyway.
    rng = default_rng(0)
    n_draws = 500

    def _beta_sample(successes: int, failures: int) -> float:
        # Conjugate posterior under a uniform prior; equivalent to
        # drawing from the credible-interval posterior without
        # having to construct it explicitly. Stable at extreme
        # proportions where the Wilson normal approximation is
        # loosest.
        a = successes + 1.0
        b = failures + 1.0
        return float(rng.beta(a, b))

    corrected_draws: list[float] = []
    for _ in range(n_draws):
        tpr_draw = _beta_sample(confusion.tp, confusion.fn)
        fpr_draw = _beta_sample(confusion.fp, confusion.tn)
        if n_measured and n_measured > 0:
            # Propagate the measured-rate uncertainty too.
            successes = int(round(measured_rate * n_measured))
            successes = max(0, min(n_measured, successes))
            measured_draw = _beta_sample(
                successes, n_measured - successes
            )
        else:
            measured_draw = measured_rate
        d = tpr_draw - fpr_draw
        if d <= 0.0:
            continue
        v = (measured_draw - fpr_draw) / d
        corrected_draws.append(max(0.0, min(1.0, v)))

    if len(corrected_draws) < 2:
        return Estimate(
            value=point_clipped,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=confusion.total,
            method="deconfound_proportion_degenerate",
            confidence_level=confidence_level,
        )

    lower_pct = confidence_level.lower_percentile
    upper_pct = confidence_level.upper_percentile
    ci_low = float(np.percentile(corrected_draws, lower_pct))
    ci_high = float(np.percentile(corrected_draws, upper_pct))
    return Estimate(
        value=point_clipped,
        ci_low=ci_low,
        ci_high=ci_high,
        n=confusion.total,
        method="deconfound_proportion",
        confidence_level=confidence_level,
        se=float(np.std(corrected_draws, ddof=1)) if len(corrected_draws) > 1 else None,
    )


# ---------------------------------------------------------------------------
# SIMEX
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimexPoint:
    """One (λ, coefficient) observation from the SIMEX schedule.

    λ is the additional-noise multiplier; ``coefficient`` is the
    fit's coefficient of interest at that noise level. Exposed for
    auditing — a reader can plot the sequence and see how the
    extrapolation was built.
    """

    lambda_: float
    coefficient: float


@dataclass(frozen=True)
class SimexResult:
    """The outcome of a SIMEX correction.

    Attributes:
        corrected: The extrapolated (corrected) coefficient as an
            :class:`Estimate`. The CI comes from bootstrap over
            repeats of the SIMEX schedule.
        naive: The uncorrected coefficient from the original data
            (λ = 0), as an :class:`Estimate` with the same method tag
            machinery.
        points: The full (λ, mean coefficient) sequence used for
            the extrapolation, so callers can inspect or plot it.
        flip_probability: The per-label flip probability used as
            the noise model.
        extrapolation: Short tag identifying the extrapolation
            function used (``"quadratic"`` by default).
    """

    corrected: Estimate
    naive: Estimate
    points: list[SimexPoint]
    flip_probability: float
    extrapolation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "corrected": self.corrected.to_dict(),
            "naive": self.naive.to_dict(),
            "points": [
                {"lambda": p.lambda_, "coefficient": p.coefficient}
                for p in self.points
            ],
            "flip_probability": self.flip_probability,
            "extrapolation": self.extrapolation,
        }


def simex_correct(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    flip_probability: float,
    fit_fn: Callable[[pd.DataFrame], float],
    lambdas: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    n_repeats: int = 50,
    confidence_level: ConfidenceLevel = NINETY_FIVE,
    rng: Generator | None = None,
) -> SimexResult:
    """SIMEX correction for a regression coefficient under binary label noise.

    Procedure:

    1. For each λ in ``lambdas`` (starting at 0, where 0 means
       "no added noise" and corresponds to the naive fit):

       a. For each of ``n_repeats`` repetitions:

          i. Flip each outcome label independently with probability
             ``flip_probability * λ / (1 + λ)`` — the transformation
             that composes the existing grader noise with an
             additional independent noise draw of the same rate.
             Algebraic derivation: under two independent flips with
             rate ``p``, the effective rate is ``p + p - 2*p²``
             ≈ ``2p`` for small p. The SIMEX convention uses λ as
             a linear knob on the composite noise variance; the
             formula above maps λ onto the equivalent flip rate.

          ii. Refit via ``fit_fn`` and record the coefficient.

       b. Average the ``n_repeats`` coefficients to get a smoothed
          value at this λ.

    2. Fit a quadratic function a*λ² + b*λ + c to the smoothed
       points. The extrapolated coefficient at λ = −1 is the
       measurement-error-corrected estimate.

    3. Bootstrap the CI by re-running the full schedule a small
       number of times with different random noise draws and taking
       the percentile interval over the extrapolated values. For
       Phase 1 we use a modest number of bootstrap iterations (20);
       Phase 3 can crank this up when the cost matters less than
       the tightness of the interval.

    Args:
        df: Input dataframe.
        outcome_col: Name of the binary outcome column to apply
            label noise to.
        flip_probability: Base per-label flip rate under the
            grader's measurement error. Typically the average of
            ``confusion.fpr`` and ``confusion.fnr`` from the TRAIL
            confusion matrix.
        fit_fn: A closure that takes a dataframe and returns the
            coefficient of interest as a scalar float. The caller
            controls what model is fit — this function doesn't
            prescribe a regression type.
        lambdas: Noise-level schedule. Must start at 0 and include
            at least three distinct values for the quadratic
            extrapolation to be well-defined. Default
            (0, 0.5, 1, 1.5, 2) gives a good-quality extrapolation
            for most problems.
        n_repeats: Number of repetitions per λ. More repetitions
            reduce Monte Carlo noise at each λ but scale the total
            fit cost linearly.
        confidence_level: Two-sided confidence level on the
            corrected estimate.
        rng: Seeded generator for reproducibility.

    Returns:
        A :class:`SimexResult`. If the schedule's quadratic fit is
        degenerate (e.g. all coefficients identical), the corrected
        estimate carries ``method="simex_degenerate"`` and nan CI.
    """
    if outcome_col not in df.columns:
        raise ValueError(f"outcome_col {outcome_col!r} not in dataframe columns")
    if not (0.0 <= flip_probability < 0.5):
        raise ValueError(
            f"flip_probability must be in [0, 0.5), got {flip_probability}"
        )
    if len(lambdas) < 3:
        raise ValueError(
            f"at least three lambdas required for quadratic extrapolation, "
            f"got {list(lambdas)}"
        )
    if lambdas[0] != 0.0:
        raise ValueError("lambdas must start at 0 (the naive fit)")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    rng = rng if rng is not None else default_rng()

    y_original = df[outcome_col].to_numpy(copy=True)
    if y_original.dtype != bool:
        # Convert to bool strictly — we only accept 0/1 or True/False
        # outcomes in this noise model.
        as_int = y_original.astype(float)
        if not np.all((as_int == 0.0) | (as_int == 1.0)):
            raise ValueError(
                f"{outcome_col!r} must contain binary values; "
                f"SIMEX with label-flip noise does not apply to continuous outcomes"
            )
        y_original = as_int.astype(bool)

    def _add_noise(lambda_val: float, generator: Generator) -> np.ndarray:
        """Produce a noisy outcome column for a given λ."""
        if lambda_val == 0.0:
            return y_original.copy()
        # Effective flip rate that, composed with the existing
        # grader noise, yields variance (1 + λ) × base.
        effective_flip = flip_probability * lambda_val / (1.0 + lambda_val)
        effective_flip = min(effective_flip, 0.49)
        flips = generator.random(y_original.shape) < effective_flip
        return np.where(flips, ~y_original, y_original)

    # Run the primary schedule to get the corrected estimate.
    primary_points: list[SimexPoint] = []
    for lam in lambdas:
        coefs_at_lam: list[float] = []
        for _ in range(n_repeats):
            noisy = _add_noise(lam, rng)
            df_noisy = df.copy()
            df_noisy[outcome_col] = noisy
            coef = fit_fn(df_noisy)
            if math.isfinite(coef):
                coefs_at_lam.append(coef)
        if not coefs_at_lam:
            mean_coef = float("nan")
        else:
            mean_coef = float(np.mean(coefs_at_lam))
        primary_points.append(SimexPoint(lambda_=float(lam), coefficient=mean_coef))

    naive_coef = primary_points[0].coefficient
    naive_estimate = Estimate(
        value=naive_coef,
        ci_low=float("nan"),
        ci_high=float("nan"),
        n=len(df),
        method="simex_naive",
        confidence_level=confidence_level,
    )

    corrected_point = _extrapolate_to_lambda(primary_points, target=-1.0)
    if math.isnan(corrected_point):
        return SimexResult(
            corrected=Estimate(
                value=float("nan"),
                ci_low=float("nan"),
                ci_high=float("nan"),
                n=len(df),
                method="simex_degenerate",
                confidence_level=confidence_level,
            ),
            naive=naive_estimate,
            points=primary_points,
            flip_probability=flip_probability,
            extrapolation="quadratic",
        )

    # Bootstrap the CI by running the schedule multiple times with
    # different noise seeds. Each bootstrap run is a full SIMEX
    # schedule — this is what "bootstrap the entire procedure"
    # means in the measurement-error literature.
    n_bootstrap = 20
    bootstrap_extrapolations: list[float] = []
    for boot_seed in range(n_bootstrap):
        boot_rng = default_rng(boot_seed + 1)
        boot_points: list[SimexPoint] = []
        for lam in lambdas:
            coefs: list[float] = []
            for _ in range(n_repeats):
                noisy = _add_noise(lam, boot_rng)
                df_noisy = df.copy()
                df_noisy[outcome_col] = noisy
                coef = fit_fn(df_noisy)
                if math.isfinite(coef):
                    coefs.append(coef)
            mean_c = float(np.mean(coefs)) if coefs else float("nan")
            boot_points.append(SimexPoint(lambda_=float(lam), coefficient=mean_c))
        value = _extrapolate_to_lambda(boot_points, target=-1.0)
        if math.isfinite(value):
            bootstrap_extrapolations.append(value)

    if len(bootstrap_extrapolations) < 2:
        corrected_estimate = Estimate(
            value=corrected_point,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=len(df),
            method="simex_degenerate",
            confidence_level=confidence_level,
        )
    else:
        lower = float(
            np.percentile(bootstrap_extrapolations, confidence_level.lower_percentile)
        )
        upper = float(
            np.percentile(bootstrap_extrapolations, confidence_level.upper_percentile)
        )
        corrected_estimate = Estimate(
            value=corrected_point,
            ci_low=lower,
            ci_high=upper,
            n=len(df),
            method="simex_bootstrap",
            confidence_level=confidence_level,
            se=float(np.std(bootstrap_extrapolations, ddof=1)),
        )

    return SimexResult(
        corrected=corrected_estimate,
        naive=naive_estimate,
        points=primary_points,
        flip_probability=flip_probability,
        extrapolation="quadratic",
    )


def _extrapolate_to_lambda(
    points: Sequence[SimexPoint], *, target: float
) -> float:
    """Fit a quadratic to a (λ, coefficient) sequence and evaluate at ``target``.

    SIMEX convention is quadratic extrapolation from the observed
    λ ≥ 0 values to λ = -1, which corresponds to zero total
    measurement error. Quadratic is the standard choice — Cook &
    Stefanski show it's a good compromise between bias and
    variance across a wide range of measurement-error shapes.
    """
    xs = np.array([p.lambda_ for p in points], dtype=float)
    ys = np.array([p.coefficient for p in points], dtype=float)
    mask = np.isfinite(ys)
    if int(mask.sum()) < 3:
        return float("nan")
    xs = xs[mask]
    ys = ys[mask]
    try:
        coeffs = np.polyfit(xs, ys, deg=2)
    except np.linalg.LinAlgError:
        return float("nan")
    return float(np.polyval(coeffs, target))


__all__ = [
    "ConfusionMatrix",
    "SimexPoint",
    "SimexResult",
    "deconfound_proportion",
    "simex_correct",
]
