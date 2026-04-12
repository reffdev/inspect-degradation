"""Flatten graded traces into a tidy DataFrame for downstream analysis."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from inspect_degradation.schema import GradedTrace, Validity
from inspect_degradation.step_phase import classify_step_phase


def traces_to_frame(traces: Iterable[GradedTrace]) -> pd.DataFrame:
    rows: list[dict] = []
    for t in traces:
        for s in t.steps:
            # Get the action text for phase classification. The
            # action lives on the source Trace, but by the time we
            # have a GradedTrace the raw completion may carry it.
            action_text = ""
            if s.raw:
                action_text = s.raw.get("completion", "")
                if not action_text:
                    action_text = str(s.raw)

            rows.append(
                {
                    "trace_id": t.trace_id,
                    "task_id": t.task_id,
                    "model": t.model,
                    "source": t.source,
                    "trace_success": t.success,
                    "step_index": s.step_index,
                    "validity": s.validity.value,
                    "is_error": s.validity == Validity.fail,
                    "is_productive": s.validity == Validity.pass_,
                    "is_neutral": s.validity == Validity.neutral,
                    "complexity": s.complexity,
                    "dependency": s.dependency.value if s.dependency is not None else None,
                    "is_looping": s.is_looping,
                    "severity": s.severity,
                    "grader_model": s.grader_model,
                    "step_phase": (
                        s.raw.get("step_phase") if s.raw and "step_phase" in s.raw
                        else classify_step_phase(action_text)
                    ),
                }
            )
    return pd.DataFrame(rows)
