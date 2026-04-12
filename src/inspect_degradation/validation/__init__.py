"""Grader validation: agreement against human-labeled reference traces.

This package separates two concerns that often get tangled:

* :mod:`.irr` — pure inter-rater-reliability statistics (Cohen's kappa,
  Krippendorff's alpha, Pearson r, accuracy). No project types, no I/O.
* :mod:`.agreement` — pairing predicted and reference :class:`GradedStep`
  sequences and computing per-dimension agreement reports.
* :mod:`.runner` — orchestration: takes a grader and a labeled corpus,
  produces a full :class:`AgreementReport`.

Phase 1's grader-validation pipeline lives here.
"""

from inspect_degradation.validation.agreement import (
    AgreementReport,
    DimensionAgreement,
    pair_grades,
    score_agreement,
)
from inspect_degradation.validation.invariance import (
    InvarianceReport,
    position_invariance_test,
    task_invariance_test,
)
from inspect_degradation.validation.runner import ValidationResult, run_validation
from inspect_degradation.validation.irr import (
    accuracy,
    cohens_kappa,
    krippendorff_alpha_nominal,
    pearson_r,
    weighted_cohens_kappa,
)

__all__ = [
    "AgreementReport",
    "DimensionAgreement",
    "InvarianceReport",
    "accuracy",
    "cohens_kappa",
    "krippendorff_alpha_nominal",
    "pair_grades",
    "pearson_r",
    "position_invariance_test",
    "task_invariance_test",
    "ValidationResult",
    "run_validation",
    "score_agreement",
    "weighted_cohens_kappa",
]
