"""GLM 4.7 / terminus-2 trajectories loader.

12,410 trajectories from GLM 4.7 running on the terminus-2 agent
framework — a third scaffolding distinct from SWE-agent and OpenHands.

https://huggingface.co/datasets/DCAgent/neulab-nebius-swe-agent-trajectories-sandboxes_glm_4.7_traces_jupiter

The terminus-2 agent uses a simple user/assistant alternating
format. The assistant includes ``<think>`` blocks for reasoning
before acting.

Step granularity: one ``(assistant, user)`` pair = one step,
same as Nebius/SWE-agent.
"""

from __future__ import annotations

import logging
from typing import Any

from inspect_degradation.trace import Trace, TraceStep

log = logging.getLogger(__name__)

_MAX_MESSAGE_CHARS = 8000
_DATASET_ID = "DCAgent/neulab-nebius-swe-agent-trajectories-sandboxes_glm_4.7_traces_jupiter"


def _truncate(text: str, limit: int = _MAX_MESSAGE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 50] + f"\n\n[... truncated, {len(text) - limit + 50} chars omitted ...]"


def _extract_task_goal(messages: list[dict[str, Any]]) -> str:
    """Extract task from the first user message."""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                return _truncate(content, 4000)
    return "(no task goal found)"


def _split_think(text: str) -> tuple[str | None, str]:
    """Split ``<think>...</think>`` from the rest of the content.

    Returns (thought, action). If no think block, thought is None.
    """
    if "<think>" in text:
        parts = text.split("</think>", 1)
        if len(parts) == 2:
            thought = parts[0].replace("<think>", "").strip()
            action = parts[1].strip()
            return (thought if thought else None, action if action else text)
    return None, text


def _parse_steps(messages: list[dict[str, Any]]) -> list[TraceStep]:
    """Convert alternating user/assistant messages into TraceSteps.

    Skips the first user message (task goal). Pairs each assistant
    message with the subsequent user message (observation).
    """
    # Skip to after the first user message.
    start = 0
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            start = i + 1
            break

    steps: list[TraceStep] = []
    step_index = 0
    i = start

    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "assistant":
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content) if content else ""

            thought, action = _split_think(content)
            if thought:
                thought = _truncate(thought)
            action = _truncate(action)

            # Next user message is the observation.
            observation = None
            if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                obs_content = messages[i + 1].get("content", "")
                if isinstance(obs_content, str) and obs_content:
                    observation = _truncate(obs_content)
                i += 2
            else:
                i += 1

            steps.append(
                TraceStep(
                    index=step_index,
                    thought=thought,
                    action=action,
                    observation=observation,
                    metadata={"role": "assistant"},
                )
            )
            step_index += 1
        else:
            i += 1

    return steps


def load_terminus(
    *,
    split: str = "train",
    limit: int | None = None,
    one_per_instance: bool = False,
    min_steps: int | None = None,
    streaming: bool = True,
) -> list[Trace]:
    """Load GLM 4.7 / terminus-2 trajectories from HuggingFace.

    Args:
        split: Dataset split. Default ``"train"``.
        limit: Optional cap on total traces loaded.
        one_per_instance: If True, keep only the first trace per
            task ID.
        min_steps: If set, skip traces with fewer steps.
        streaming: Stream without downloading.

    Returns:
        List of :class:`Trace` objects ready for grading.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "the 'datasets' package is required; install via `pip install datasets`"
        ) from exc

    ds = load_dataset(_DATASET_ID, split=split, streaming=streaming)

    traces: list[Trace] = []
    n_skipped = 0
    _seen_tasks: set[str] = set()
    _id_counter: dict[str, int] = {}

    for row in ds:
        task = row.get("task", "")
        model = row.get("model", "glm-4.7")
        agent = row.get("agent", "terminus-2")

        if one_per_instance:
            if task in _seen_tasks:
                continue
            _seen_tasks.add(task)

        messages = row.get("conversations", [])
        if not messages:
            n_skipped += 1
            continue

        steps = _parse_steps(messages)
        if not steps:
            n_skipped += 1
            continue

        if min_steps is not None and len(steps) < min_steps:
            continue

        task_goal = _extract_task_goal(messages)

        base_id = f"terminus-{task}-{model}"
        _id_counter[base_id] = _id_counter.get(base_id, 0) + 1
        trace_id = base_id if _id_counter[base_id] == 1 else f"{base_id}-run{_id_counter[base_id]}"

        traces.append(
            Trace(
                trace_id=trace_id,
                task_goal=task_goal,
                task_id=task,
                model=model,
                source="terminus",
                success=None,  # result field is always None in this dataset
                steps=tuple(steps),
                metadata={
                    "agent": agent,
                    "run_id": row.get("run_id", ""),
                },
            )
        )

        if limit is not None and len(traces) >= limit:
            break

    if n_skipped:
        log.info("terminus: skipped %d traces with empty trajectories", n_skipped)
    log.info("terminus: loaded %d traces", len(traces))
    return traces


__all__ = ["load_terminus"]
