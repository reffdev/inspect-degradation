"""Tests for the TRAIL adapter.

Two layers of test:

1. Synthetic OpenInference span tree built in-memory. Lets us pin the
   adapter's policies (step segmentation, span->step mapping, severity
   aggregation) deterministically without depending on a network.
2. Real-sample integration test that runs only when the upstream-fixture
   files exist on disk under ``data/trail_sample/``. The download is
   recorded in CONTRIBUTING-style instructions but not committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inspect_degradation.datasets.trail import (
    SEVERITY_MAP,
    _adapt_record,
    load_trail,
    load_trail_record,
)
from inspect_degradation.schema import (
    HUMAN_GRADER,
    SeverityLevel,
    Validity,
)


# ---------------------------------------------------------------------------
# Synthetic span builder
# ---------------------------------------------------------------------------


def _span(
    *,
    span_id: str,
    name: str,
    kind: str | None = None,
    input_value: str | None = None,
    output_value: str | None = None,
    parent: str | None = None,
    children: list[dict] | None = None,
) -> dict:
    attrs: dict = {}
    if kind is not None:
        attrs["openinference.span.kind"] = kind
    if input_value is not None:
        attrs["input.value"] = input_value
    if output_value is not None:
        attrs["output.value"] = output_value
    return {
        "span_id": span_id,
        "parent_span_id": parent,
        "span_name": name,
        "span_attributes": attrs,
        "child_spans": children or [],
    }


def _agent_with_steps(
    *,
    task: str,
    step_descriptions: list[tuple[str, str | None]],
) -> dict:
    """Build a CodeAgent.run AGENT span with the given step children.

    Each ``step_descriptions`` entry is ``(span_id, nested_tool_span_id)``.
    """
    children = []
    for step_id, tool_span_id in step_descriptions:
        nested = []
        if tool_span_id is not None:
            nested.append(
                _span(
                    span_id=tool_span_id,
                    name="SomeTool",
                    kind="TOOL",
                    input_value="tool input",
                    output_value="tool output",
                    parent=step_id,
                )
            )
        children.append(
            _span(
                span_id=step_id,
                name=f"Step {step_id}",
                kind="CHAIN",
                input_value=f"step {step_id} input",
                output_value=f"step {step_id} output",
                parent="agent-1",
                children=nested,
            )
        )
    return _span(
        span_id="agent-1",
        name="CodeAgent.run",
        kind="AGENT",
        input_value=json.dumps({"task": task}),
        children=children,
    )


def _wrap_root(agent_span: dict) -> dict:
    return {
        "trace_id": "synth",
        "spans": [
            _span(
                span_id="root",
                name="main",
                children=[agent_span],
            )
        ],
    }


# ---------------------------------------------------------------------------
# Synthetic-trace tests
# ---------------------------------------------------------------------------


def test_segmentation_uses_outermost_agent_children_as_steps():
    raw = _wrap_root(
        _agent_with_steps(
            task="solve the puzzle",
            step_descriptions=[("s0", None), ("s1", "tool-1"), ("s2", None)],
        )
    )
    trace, ref = _adapt_record(
        trace_id="t",
        raw_trace=raw,
        raw_annot={"errors": []},
        source="trail-test",
    )
    assert trace.task_goal == "solve the puzzle"
    assert len(trace.steps) == 3
    assert [s.metadata["span_id"] for s in trace.steps] == ["s0", "s1", "s2"]
    # Step 1 has a nested TOOL call rendered into the action body.
    assert "<TOOL SomeTool>" in trace.steps[1].action
    assert "tool input" in trace.steps[1].action
    # Reference defaults all steps to pass when there are no errors.
    assert all(s.validity == Validity.pass_ for s in ref.steps)
    assert all(s.severity is None for s in ref.steps)
    assert all(s.grader_model == HUMAN_GRADER for s in ref.steps)


def test_error_anchored_at_step_root_marks_step_failed():
    raw = _wrap_root(
        _agent_with_steps(
            task="t",
            step_descriptions=[("s0", None), ("s1", None)],
        )
    )
    annot = {
        "errors": [
            {
                "category": "Tool-related",
                "location": "s1",
                "evidence": "...",
                "description": "...",
                "impact": "HIGH",
            }
        ]
    }
    _, ref = _adapt_record(trace_id="t", raw_trace=raw, raw_annot=annot, source="trail-test")
    assert ref.steps[0].validity == Validity.pass_
    assert ref.steps[1].validity == Validity.fail
    assert ref.steps[1].severity == SeverityLevel.high
    assert ref.steps[1].raw["trail_categories"] == ["Tool-related"]


def test_error_anchored_at_descendant_walks_up_to_step():
    raw = _wrap_root(
        _agent_with_steps(
            task="t",
            step_descriptions=[("s0", "tool-deep")],
        )
    )
    annot = {
        "errors": [
            {
                "category": "Goal Deviation",
                "location": "tool-deep",
                "impact": "MEDIUM",
            }
        ]
    }
    _, ref = _adapt_record(trace_id="t", raw_trace=raw, raw_annot=annot, source="trail-test")
    assert ref.steps[0].validity == Validity.fail
    assert ref.steps[0].severity == SeverityLevel.medium


def test_multiple_errors_per_step_aggregate_to_max_severity():
    raw = _wrap_root(
        _agent_with_steps(
            task="t",
            step_descriptions=[("s0", "t0")],
        )
    )
    annot = {
        "errors": [
            {"category": "A", "location": "s0", "impact": "LOW"},
            {"category": "B", "location": "t0", "impact": "HIGH"},
            {"category": "C", "location": "s0", "impact": "MEDIUM"},
        ]
    }
    _, ref = _adapt_record(trace_id="t", raw_trace=raw, raw_annot=annot, source="trail-test")
    assert ref.steps[0].severity == SeverityLevel.high
    # Categories from all three errors are preserved for downstream inspection.
    assert set(ref.steps[0].raw["trail_categories"]) == {"A", "B", "C"}


def test_error_with_unknown_span_id_is_dropped(caplog):
    raw = _wrap_root(
        _agent_with_steps(task="t", step_descriptions=[("s0", None)])
    )
    annot = {
        "errors": [
            {"category": "X", "location": "nonexistent", "impact": "HIGH"},
        ]
    }
    with caplog.at_level("WARNING"):
        _, ref = _adapt_record(trace_id="t", raw_trace=raw, raw_annot=annot, source="trail-test")
    assert ref.steps[0].validity == Validity.pass_
    assert any("does not map" in rec.message for rec in caplog.records)


def test_severity_map_covers_trail_taxonomy_exactly():
    # Pin so a future schema change can't drift away from TRAIL.
    assert set(SEVERITY_MAP) == {"LOW", "MEDIUM", "HIGH"}
    assert SEVERITY_MAP["LOW"] == SeverityLevel.low
    assert SEVERITY_MAP["MEDIUM"] == SeverityLevel.medium
    assert SEVERITY_MAP["HIGH"] == SeverityLevel.high


def test_loader_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_trail(tmp_path / "does-not-exist")


def test_loader_rejects_missing_split_dir(tmp_path):
    (tmp_path / "data" / "GAIA").mkdir(parents=True)
    # processed_annotations_gaia is missing.
    with pytest.raises(FileNotFoundError):
        load_trail(tmp_path, splits=("gaia",))


# ---------------------------------------------------------------------------
# Real-sample integration test (only when fixture present)
# ---------------------------------------------------------------------------


_REAL_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "trail_sample"


@pytest.mark.skipif(
    not (_REAL_SAMPLE_DIR / "trace.json").exists(),
    reason="real TRAIL sample fixture not present",
)
def test_real_trail_sample_round_trips():
    trace, ref = load_trail_record(
        trace_file=_REAL_SAMPLE_DIR / "trace.json",
        annotation_file=_REAL_SAMPLE_DIR / "annot.json",
        source="trail-gaia",
    )
    assert trace.task_goal.startswith("You have one question")
    # The fixture has 3 steps; this is pinned so a future segmentation
    # change is at least visible.
    assert len(trace.steps) == 3
    assert len(ref.steps) == 3
    # The fixture's annotations: step 1 fails LOW, step 2 fails HIGH.
    assert ref.steps[0].validity == Validity.pass_
    assert ref.steps[1].validity == Validity.fail
    assert ref.steps[1].severity == SeverityLevel.low
    assert ref.steps[2].validity == Validity.fail
    assert ref.steps[2].severity == SeverityLevel.high
