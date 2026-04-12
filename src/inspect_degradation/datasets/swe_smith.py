"""SWE-smith trajectories loader.

5,017 trajectories from Claude 3.7 Sonnet running on SWE-smith
(synthetic SWE-bench variants) via SWE-agent.

https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories

The dataset has three splits (``tool``, ``xml``, ``ticks``)
representing different SWE-agent output format configurations.
All use the same model (Claude 3.7 Sonnet). The ``tool`` split
uses function-calling style with ``tool_calls`` on assistant
messages and ``tool`` role for responses — the cleanest format
for step extraction.

Trajectory format
-----------------

The ``messages`` field is a JSON string containing a list of
chat messages:

* ``system`` — agent instructions
* ``user`` — the task (PR description / issue to fix)
* ``assistant`` — agent actions, with ``tool_calls`` containing
  bash commands, file operations, etc.
* ``tool`` — environment responses to tool calls

Step granularity: one ``(assistant, tool)`` pair = one step.
The assistant's ``content`` is the thought/reasoning, the
``tool_calls`` are the actions, and the ``tool`` response is
the observation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from inspect_degradation.trace import Trace, TraceStep

log = logging.getLogger(__name__)

_MAX_MESSAGE_CHARS = 8000


def _truncate(text: str, limit: int = _MAX_MESSAGE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 50] + f"\n\n[... truncated, {len(text) - limit + 50} chars omitted ...]"


def _extract_content(msg: dict[str, Any]) -> str:
    """Extract text content from a message, handling both string and list formats."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content) if content else ""


def _format_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Render tool_calls as readable action text."""
    parts = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "unknown")
        args = func.get("arguments", "")
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict) and "command" in parsed:
                    args = parsed["command"]
            except json.JSONDecodeError:
                pass
        parts.append(f"[{name}] {args}")
    return "\n".join(parts)


def _extract_task_goal(messages: list[dict[str, Any]]) -> str:
    """Extract the task description from the first user message."""
    for msg in messages:
        if msg.get("role") == "user":
            content = _extract_content(msg)
            if content:
                return _truncate(content, 4000)
    return "(no task goal found)"


def _parse_steps(messages: list[dict[str, Any]]) -> list[TraceStep]:
    """Convert assistant+tool message pairs into TraceStep objects.

    Each assistant message becomes an action (reasoning + tool calls).
    The subsequent tool message becomes the observation.
    """
    steps: list[TraceStep] = []
    step_index = 0

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "assistant":
            thought = _extract_content(msg)
            if thought:
                thought = _truncate(thought)

            tool_calls = msg.get("tool_calls")
            if tool_calls:
                action = _format_tool_calls(tool_calls)
            else:
                action = thought or "(no action)"
                thought = None

            # Collect all consecutive tool responses as observation.
            obs_parts: list[str] = []
            while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                i += 1
                obs_parts.append(_extract_content(messages[i]))
            observation = _truncate("\n".join(obs_parts)) if obs_parts else None
            i += 1

            steps.append(
                TraceStep(
                    index=step_index,
                    thought=thought if tool_calls else None,
                    action=action,
                    observation=observation,
                    metadata={"role": "assistant"},
                )
            )
            step_index += 1
        else:
            i += 1

    return steps


def load_swe_smith(
    *,
    split: str = "tool",
    models: list[str] | None = None,
    limit: int | None = None,
    one_per_instance: bool = False,
    min_steps: int | None = None,
    streaming: bool = True,
) -> list[Trace]:
    """Load SWE-smith trajectories from HuggingFace.

    Requires the ``datasets`` package (``pip install datasets``).

    Args:
        split: Dataset split. One of ``"tool"`` (function-calling
            format, recommended), ``"xml"``, or ``"ticks"``.
            Default ``"tool"``.
        models: Optional list of model names to include. If None,
            loads all models.
        limit: Optional cap on total traces loaded.
        one_per_instance: If True, keep only the first trace per
            ``instance_id``.
        streaming: If True (default), streams without downloading.

    Returns:
        List of :class:`Trace` objects ready for grading.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "the 'datasets' package is required; install via `pip install datasets`"
        ) from exc

    ds = load_dataset(
        "SWE-bench/SWE-smith-trajectories",
        split=split,
        streaming=streaming,
    )

    traces: list[Trace] = []
    n_skipped = 0
    _seen_instances: set[str] = set()

    for row in ds:
        instance_id = row.get("instance_id", "")
        traj_id = row.get("traj_id", "")
        model = row.get("model", "unknown")

        if models is not None and model not in models:
            continue

        if one_per_instance:
            if instance_id in _seen_instances:
                continue
            _seen_instances.add(instance_id)

        messages_raw = row.get("messages", "")
        if not messages_raw:
            n_skipped += 1
            continue

        try:
            messages = json.loads(messages_raw) if isinstance(messages_raw, str) else messages_raw
        except json.JSONDecodeError:
            n_skipped += 1
            continue

        if not messages:
            n_skipped += 1
            continue

        task_goal = _extract_task_goal(messages)
        steps = _parse_steps(messages)
        if not steps:
            n_skipped += 1
            continue

        if min_steps is not None and len(steps) < min_steps:
            continue

        resolved = row.get("resolved")
        success = bool(resolved) if isinstance(resolved, (bool, int)) else None

        traces.append(
            Trace(
                trace_id=traj_id or f"swesmith-{instance_id}-{model}",
                task_goal=task_goal,
                task_id=instance_id,
                model=model,
                source="swe-smith",
                success=success,
                steps=tuple(steps),
                metadata={
                    "split": split,
                    "has_patch": bool(row.get("patch")),
                },
            )
        )

        if limit is not None and len(traces) >= limit:
            break

    if n_skipped:
        log.info("swe-smith: skipped %d traces with empty/invalid trajectories", n_skipped)
    log.info(
        "swe-smith: loaded %d traces (%d distinct models)",
        len(traces),
        len({t.model for t in traces}),
    )
    return traces


__all__ = ["load_swe_smith"]
