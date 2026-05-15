"""inspect-degradation: within-run degradation analysis for AI agents.

Top-level re-exports cover the types most users need; layer-specific
helpers live in their submodules (``grader``, ``analysis``, ``integration``,
``validation``, ``datasets``).
"""

from inspect_degradation.schema import (
    HUMAN_GRADER,
    ComplexityLevel,
    Dependency,
    GradedStep,
    GradedTrace,
    SeverityLevel,
    Validity,
)
from inspect_degradation.trace import Trace, TraceStep

__version__ = "0.1.1"

__all__ = [
    "ComplexityLevel",
    "Dependency",
    "GradedStep",
    "GradedTrace",
    "HUMAN_GRADER",
    "SeverityLevel",
    "Trace",
    "TraceStep",
    "Validity",
    "__version__",
]
