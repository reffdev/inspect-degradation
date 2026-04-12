"""TRAIL benchmark adapter.

148 expert-annotated traces from GAIA and SWE-bench (Patronus AI). Used as
ground truth for grader validation in Phase 1.

https://github.com/patronus-ai/trail-benchmark

Data layout (verified against the upstream repo):

::

    benchmarking/
      data/
        GAIA/<trace_id>.json              # raw OpenInference span trees
        SWE Bench/<trace_id>.json
      processed_annotations_gaia/<trace_id>.json
      processed_annotations_swe_bench/<trace_id>.json

Each raw trace is an OpenInference (Phoenix/Arize) span tree:

* root span tree under ``trace["spans"]``,
* the agent's stepwise execution lives under a single span whose
  ``span_attributes["openinference.span.kind"] == "AGENT"``,
* its direct children, in array order, are the agent's steps — typically a
  few "planning" LLM calls followed by ``"Step 1"``, ``"Step 2"`` (CHAIN
  spans) each containing an LLM call and zero or more tool calls.

Each annotation file has ``errors`` (a list of error objects, each with
``location`` = ``span_id``, ``category``, ``impact``) and ``scores``
(rubric scores from a different evaluation we don't use here).

Mapping policy
--------------

* **Step granularity**: one :class:`TraceStep` per direct child of the
  outermost AGENT span. This commits to a single, documented unit; downstream
  analysis is comparable across traces only because we don't second-guess
  this here.
* **Step content**: the step's ``input.value`` becomes the action and its
  ``output.value`` becomes the observation; nested LLM/tool spans are
  rendered as supplementary text appended to the action so the grader sees
  the full sub-tree without the rest of the pipeline needing to know
  OpenInference exists.
* **Annotation mapping**: each error's ``location`` (a span_id) is walked
  up the parent chain until we hit a step span. Errors that anchor outside
  any step span are dropped with a logged warning — they typically
  correspond to harness setup or post-hoc spans we deliberately exclude.
* **Severity mapping**: TRAIL uses three impact levels which we map to the
  ``[0, 1]`` interval used by :class:`GradedStep.severity` so the grader's
  continuous predictions are comparable to human labels.

We grade only the dimensions TRAIL labels (``validity``, ``severity``).
Other dimensions are left ``None`` on the reference side; the agreement
harness in :mod:`inspect_degradation.validation.agreement` already drops
pairs where either side is missing the dimension under test.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_degradation.schema import (
    HUMAN_GRADER,
    GradedStep,
    GradedTrace,
    SeverityLevel,
    Validity,
)
from inspect_degradation.trace import Trace, TraceStep

log = logging.getLogger(__name__)

#: Mapping from TRAIL impact strings to our :class:`SeverityLevel`. The
#: TRAIL taxonomy aligns one-to-one with our three-level severity scale,
#: which is one of the reasons we picked that scale.
SEVERITY_MAP: dict[str, SeverityLevel] = {
    "LOW": SeverityLevel.low,
    "MEDIUM": SeverityLevel.medium,
    "HIGH": SeverityLevel.high,
}

#: OpenInference span-kind constants we look for. Stored as constants so a
#: typo can't drift between the loader and the tests.
_KIND_KEY = "openinference.span.kind"
_KIND_AGENT = "AGENT"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrailCorpus:
    """A loaded slice of TRAIL ready for grader validation.

    Invariants enforced in :meth:`__post_init__`:

    * every ``trace_id`` in ``traces`` has a matching reference and vice versa,
    * step counts agree (the reference is built from the same step segmentation
      the loader produced for the raw trace).
    """

    traces: list[Trace]
    reference: list[GradedTrace]

    def __post_init__(self) -> None:
        trace_ids = {t.trace_id for t in self.traces}
        ref_ids = {g.trace_id for g in self.reference}
        if trace_ids != ref_ids:
            missing = ref_ids ^ trace_ids
            raise ValueError(
                f"TrailCorpus traces and reference must cover the same trace_ids; "
                f"symmetric difference: {sorted(missing)}"
            )
        steps_by_id = {t.trace_id: len(t.steps) for t in self.traces}
        for ref in self.reference:
            if len(ref.steps) != steps_by_id[ref.trace_id]:
                raise ValueError(
                    f"trace {ref.trace_id!r}: reference has {len(ref.steps)} steps "
                    f"but raw trace has {steps_by_id[ref.trace_id]}"
                )


# ---------------------------------------------------------------------------
# Loader entry points
# ---------------------------------------------------------------------------


def load_trail(
    root: str | Path,
    *,
    splits: tuple[str, ...] = ("gaia", "swe_bench"),
    limit: int | None = None,
) -> TrailCorpus:
    """Load TRAIL annotated traces from a local clone of the benchmark repo.

    Args:
        root: Path to the ``benchmarking/`` directory of a local
            ``patronus-ai/trail-benchmark`` checkout.
        splits: Which TRAIL splits to load. Default is both.
        limit: Optional cap on the number of traces *per split*. Useful
            for fast iteration during rubric development.

    Returns:
        A :class:`TrailCorpus` whose ``traces`` and ``reference`` cover the
        same set of ``trace_id`` values, in load order.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"TRAIL benchmarking root not found: {root}")

    traces: list[Trace] = []
    reference: list[GradedTrace] = []

    for split in splits:
        cfg = _SPLIT_CONFIG.get(split)
        if cfg is None:
            raise ValueError(f"unknown TRAIL split: {split!r}; expected one of {sorted(_SPLIT_CONFIG)}")
        data_dir = root / cfg.data_dir
        annot_dir = root / cfg.annot_dir
        if not data_dir.is_dir():
            raise FileNotFoundError(f"TRAIL data directory missing: {data_dir}")
        if not annot_dir.is_dir():
            raise FileNotFoundError(f"TRAIL annotation directory missing: {annot_dir}")

        files = sorted(data_dir.glob("*.json"))
        if limit is not None:
            files = files[:limit]

        for trace_file in files:
            trace_id = trace_file.stem
            annot_file = annot_dir / f"{trace_id}.json"
            if not annot_file.is_file():
                log.warning("trail: trace %s has no annotation file; skipping", trace_id)
                continue
            try:
                trace, ref = load_trail_record(
                    trace_file=trace_file,
                    annotation_file=annot_file,
                    source=cfg.source_label,
                )
            except _TrailRecordError as e:
                log.warning("trail: %s in %s; skipping", e, trace_id)
                continue
            traces.append(trace)
            reference.append(ref)

    return TrailCorpus(traces=traces, reference=reference)


def load_trail_record(
    *,
    trace_file: Path,
    annotation_file: Path,
    source: str,
) -> tuple[Trace, GradedTrace]:
    """Load and adapt a single ``(trace, annotation)`` file pair.

    Exposed publicly so tests and ad-hoc tools can exercise the adapter on
    one record without scanning a directory.
    """
    raw_trace = _read_json(trace_file)
    raw_annot = _read_json(annotation_file)
    trace_id = trace_file.stem
    return _adapt_record(
        trace_id=trace_id,
        raw_trace=raw_trace,
        raw_annot=raw_annot,
        source=source,
    )


# ---------------------------------------------------------------------------
# Adapter internals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SplitConfig:
    data_dir: str
    annot_dir: str
    source_label: str


_SPLIT_CONFIG: dict[str, _SplitConfig] = {
    "gaia": _SplitConfig(
        data_dir="data/GAIA",
        annot_dir="processed_annotations_gaia",
        source_label="trail-gaia",
    ),
    "swe_bench": _SplitConfig(
        data_dir="data/SWE Bench",
        annot_dir="processed_annotations_swe_bench",
        source_label="trail-swe-bench",
    ),
}


class _TrailRecordError(ValueError):
    """Recoverable error loading a single TRAIL record (the loader skips it)."""


def _adapt_record(
    *,
    trace_id: str,
    raw_trace: dict[str, Any],
    raw_annot: dict[str, Any],
    source: str,
) -> tuple[Trace, GradedTrace]:
    spans = raw_trace.get("spans")
    if not isinstance(spans, list) or not spans:
        raise _TrailRecordError("trace has no spans")

    agent_span = _find_outermost_agent_span(spans)
    if agent_span is None:
        raise _TrailRecordError("no AGENT span found")

    task_goal = _extract_task_goal(agent_span)
    step_spans = list(agent_span.get("child_spans") or [])
    if not step_spans:
        raise _TrailRecordError("AGENT span has no children to use as steps")

    # Build a span_id -> step_index map by walking each step subtree.
    span_to_step: dict[str, int] = {}
    steps: list[TraceStep] = []
    for idx, span in enumerate(step_spans):
        for descendant_id in _iter_span_ids(span):
            # Conflicts shouldn't happen in OpenInference but if they do,
            # the first owner wins (preserves the outer step boundary).
            span_to_step.setdefault(descendant_id, idx)
        steps.append(_step_from_span(idx, span))

    trace = Trace(
        trace_id=trace_id,
        task_goal=task_goal,
        task_id=trace_id,
        source=source,
        steps=tuple(steps),
        metadata={
            "trail_agent_span_id": agent_span.get("span_id"),
        },
    )

    reference_steps = _build_reference_steps(
        n_steps=len(steps),
        errors=raw_annot.get("errors") or [],
        span_to_step=span_to_step,
        trace_id=trace_id,
    )

    reference = GradedTrace(
        trace_id=trace_id,
        task_id=trace_id,
        source=source,
        steps=reference_steps,
        metadata={
            "trail_scores": raw_annot.get("scores"),
        },
    )
    return trace, reference


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise _TrailRecordError(f"invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise _TrailRecordError(f"expected JSON object at top level, got {type(data).__name__}")
    return data


def _find_outermost_agent_span(spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    """BFS for the first span whose openinference kind is AGENT.

    BFS rather than DFS so multi-agent traces (an outer orchestrator with
    sub-agents) yield the *outermost* agent — the one whose direct children
    are the top-level steps a human would judge.
    """
    queue: list[dict[str, Any]] = list(spans)
    while queue:
        span = queue.pop(0)
        attrs = span.get("span_attributes") or {}
        if attrs.get(_KIND_KEY) == _KIND_AGENT:
            return span
        children = span.get("child_spans") or []
        queue.extend(children)
    return None


def _extract_task_goal(agent_span: dict[str, Any]) -> str:
    """Extract the original task instruction from the AGENT span's input.

    OpenInference encodes the input as a JSON string in
    ``span_attributes["input.value"]``. For smolagents the deserialized
    object is a dict with a ``"task"`` key, but we don't hardcode that —
    we look at a small set of likely keys and fall back to the raw string,
    so the loader is robust to non-smolagents harnesses.
    """
    attrs = agent_span.get("span_attributes") or {}
    raw = attrs.get("input.value")
    if not isinstance(raw, str) or not raw:
        raise _TrailRecordError("AGENT span has no input.value to extract task goal from")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, dict):
        for key in ("task", "input", "question", "prompt", "query"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value
        # Fall through: dict with no recognized key, return its JSON form
        return json.dumps(parsed, ensure_ascii=False)
    if isinstance(parsed, str):
        return parsed
    return raw


def _iter_span_ids(span: dict[str, Any]):
    span_id = span.get("span_id")
    if span_id:
        yield span_id
    for child in span.get("child_spans") or []:
        yield from _iter_span_ids(child)


def _step_from_span(index: int, span: dict[str, Any]) -> TraceStep:
    attrs = span.get("span_attributes") or {}
    name = span.get("span_name") or "(unnamed)"
    input_text = _stringify(attrs.get("input.value"))
    output_text = _stringify(attrs.get("output.value"))

    # Surface nested LLM/tool calls inside the action body so the grader
    # has the full sub-tree without us forcing a flatter trace shape.
    nested = _render_nested_calls(span.get("child_spans") or [])

    action_parts: list[str] = [f"[{name}]"]
    if input_text:
        action_parts.append(input_text)
    if nested:
        action_parts.append(nested)

    return TraceStep(
        index=index,
        action="\n\n".join(action_parts),
        observation=output_text or None,
        metadata={
            "span_id": span.get("span_id"),
            "span_name": name,
            "span_kind": attrs.get(_KIND_KEY),
        },
    )


def _render_nested_calls(children: list[dict[str, Any]]) -> str:
    """Render LLM/TOOL sub-spans of a step as readable supplementary text.

    We deliberately render only ``LLM`` and ``TOOL`` kinds — other kinds are
    framework noise (chain wrappers, internal context spans) that would
    only inflate the prompt without changing the grader's judgment.
    """
    rendered: list[str] = []
    for child in children:
        attrs = child.get("span_attributes") or {}
        kind = attrs.get(_KIND_KEY)
        if kind not in {"LLM", "TOOL"}:
            # Recurse: nested chains may still contain LLM/TOOL spans.
            inner = _render_nested_calls(child.get("child_spans") or [])
            if inner:
                rendered.append(inner)
            continue
        name = child.get("span_name") or kind
        block = [f"<{kind} {name}>"]
        inp = _stringify(attrs.get("input.value"))
        out = _stringify(attrs.get("output.value"))
        if inp:
            block.append(f"input: {_truncate(inp, 2000)}")
        if out:
            block.append(f"output: {_truncate(out, 2000)}")
        rendered.append("\n".join(block))
    return "\n\n".join(rendered)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _build_reference_steps(
    *,
    n_steps: int,
    errors: list[dict[str, Any]],
    span_to_step: dict[str, int],
    trace_id: str,
) -> list[GradedStep]:
    """Project TRAIL error annotations onto our per-step grade schema.

    Each error is a ``(category, impact, location)`` tuple where ``location``
    is a span_id. We map the span to the enclosing step via ``span_to_step``,
    then aggregate per step: a step is failing if any error maps to it, and
    its severity is the maximum impact across mapped errors. Errors whose
    span_id can't be resolved to a step are logged and dropped.
    """
    failures: dict[int, SeverityLevel] = {}
    failure_categories: dict[int, list[str]] = {}

    for err in errors:
        location = err.get("location")
        impact = err.get("impact")
        if not isinstance(location, str) or not isinstance(impact, str):
            log.warning("trail %s: malformed error %r; skipping", trace_id, err)
            continue
        step_idx = span_to_step.get(location)
        if step_idx is None:
            log.warning(
                "trail %s: error location %s does not map to any step span; skipping",
                trace_id,
                location,
            )
            continue
        severity = SEVERITY_MAP.get(impact.upper())
        if severity is None:
            log.warning("trail %s: unknown impact %r; skipping", trace_id, impact)
            continue
        prior = failures.get(step_idx)
        if prior is None or severity > prior:
            failures[step_idx] = severity
        failure_categories.setdefault(step_idx, []).append(str(err.get("category", "")))

    steps: list[GradedStep] = []
    for i in range(n_steps):
        if i in failures:
            steps.append(
                GradedStep(
                    step_index=i,
                    validity=Validity.fail,
                    severity=failures[i],
                    grader_model=HUMAN_GRADER,
                    raw={"trail_categories": failure_categories[i]},
                )
            )
        else:
            steps.append(
                GradedStep(
                    step_index=i,
                    validity=Validity.pass_,
                    grader_model=HUMAN_GRADER,
                )
            )
    return steps
