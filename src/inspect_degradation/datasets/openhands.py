"""OpenHands / GPT-4o trajectory loader.

6,055 trajectories from GPT-4o running on SWE-bench via OpenHands
(function-calling agent with bash, file editor, and code tools).

https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories

This is a different scaffolding from SWE-agent (used by Nebius and
SWE-smith). OpenHands uses structured tool calls (str_replace_editor,
bash, etc.) rather than free-form shell commands. Comparing
degradation across SWE-agent and OpenHands tests whether findings
are scaffolding-specific or general.

Trajectory format
-----------------

The ``messages`` field is a list of chat-message dicts:

* ``system`` — agent instructions
* ``user`` — the task (issue to fix)
* ``assistant`` — agent actions with ``tool_calls``
* ``tool`` — tool responses

Step granularity: one ``(assistant, tool*)`` group = one step.
An assistant message may trigger multiple tool calls, each with
its own tool response; these are grouped into a single step.
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


def _format_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    parts = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "unknown")
        args_raw = func.get("arguments", "")
        if isinstance(args_raw, str):
            try:
                parsed = json.loads(args_raw)
                if isinstance(parsed, dict):
                    if "command" in parsed:
                        args_raw = parsed["command"]
                    elif len(parsed) <= 3:
                        args_raw = json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        parts.append(f"[{name}] {_truncate(str(args_raw), 2000)}")
    return "\n".join(parts)


def _extract_task_goal(messages: list[dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                return _truncate(content, 4000)
    return "(no task goal found)"


def _extract_model(run_id: str) -> str:
    """Extract model name from the run_id string.

    Format: ``gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t04-run_1``
    """
    if not run_id:
        return "unknown"
    # Take everything before the first underscore-delimited config
    parts = run_id.split("_maxiter_")
    return parts[0] if parts else run_id


def _parse_steps(messages: list[dict[str, Any]]) -> list[TraceStep]:
    """Convert assistant+tool message groups into TraceStep objects."""
    steps: list[TraceStep] = []
    step_index = 0

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "assistant":
            thought = msg.get("content") or ""
            if thought:
                thought = _truncate(thought)

            tool_calls = msg.get("tool_calls")
            if tool_calls:
                action = _format_tool_calls(tool_calls)
            else:
                action = thought or "(no action)"
                thought = None

            # Collect all consecutive tool responses.
            obs_parts: list[str] = []
            while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                i += 1
                content = messages[i].get("content", "")
                if isinstance(content, str) and content:
                    obs_parts.append(content)
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


def load_openhands(
    *,
    dataset: str = "SWE-Gym/OpenHands-Sampled-Trajectories",
    split: str | None = None,
    models: list[str] | None = None,
    limit: int | None = None,
    one_per_instance: bool = False,
    min_steps: int | None = None,
    streaming: bool = True,
) -> list[Trace]:
    """Load OpenHands trajectories from HuggingFace.

    Works with multiple OpenHands datasets that share the same
    message format:

    * ``SWE-Gym/OpenHands-Sampled-Trajectories`` — GPT-4o (6k traces)
    * ``nebius/SWE-rebench-openhands-trajectories`` — Qwen3-Coder-480B (67k traces)

    Args:
        dataset: HuggingFace dataset ID. Default is the GPT-4o set.
        split: Dataset split. Default auto-detected (``"train.raw"``
            for SWE-Gym, ``"train"`` for nebius).
        limit: Optional cap on total traces loaded.
        one_per_instance: If True, keep only the first trace per
            ``instance_id``.
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

    if split is None:
        split = "train.raw" if "SWE-Gym" in dataset else "train"

    ds = load_dataset(
        dataset,
        split=split,
        streaming=streaming,
    )

    traces: list[Trace] = []
    n_skipped = 0
    _seen_instances: set[str] = set()
    _id_counter: dict[str, int] = {}

    for row in ds:
        instance_id = row.get("instance_id", "")
        # SWE-Gym uses "run_id", nebius uses "trajectory_id".
        run_id = row.get("run_id") or row.get("trajectory_id") or ""
        model = _extract_model(run_id)
        # nebius/SWE-rebench doesn't encode model in run_id —
        # it's always Qwen3-Coder-480B.
        if model == run_id and "rebench" in dataset.lower():
            model = "qwen3-coder-480b"

        if models is not None and model not in models:
            continue

        if one_per_instance:
            if instance_id in _seen_instances:
                continue
            _seen_instances.add(instance_id)

        # SWE-Gym uses "messages", nebius uses "trajectory".
        messages = row.get("messages") or row.get("trajectory") or []
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

        resolved = row.get("resolved")
        success = bool(resolved) if isinstance(resolved, (bool, int)) else None

        base_id = f"openhands-{instance_id}-{model}"
        _id_counter[base_id] = _id_counter.get(base_id, 0) + 1
        trace_id = base_id if _id_counter[base_id] == 1 else f"{base_id}-run{_id_counter[base_id]}"

        traces.append(
            Trace(
                trace_id=trace_id,
                task_goal=task_goal,
                task_id=instance_id,
                model=model,
                source="openhands",
                success=success,
                steps=tuple(steps),
                metadata={
                    "run_id": run_id,
                    "dataset": dataset,
                },
            )
        )

        if limit is not None and len(traces) >= limit:
            break

    if n_skipped:
        log.info("openhands: skipped %d traces with empty trajectories", n_skipped)
    log.info(
        "openhands: loaded %d traces (%d distinct models)",
        len(traces),
        len({t.model for t in traces}),
    )
    return traces


__all__ = ["load_openhands"]
