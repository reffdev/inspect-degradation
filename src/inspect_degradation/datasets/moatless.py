"""Moatless SFT trajectories loader.

~3,018 trajectories from a ~32B model running on the Moatless
agent framework — a third scaffolding distinct from both SWE-agent
and OpenHands.

https://huggingface.co/datasets/swesynth/SWE-Synth_Moatless-SFT-Trajectories

Moatless uses a multi-round planning loop where each "step" is a
full conversation (system + user + assistant). The assistant output
is typically JSON with a ``scratch_pad`` field (reasoning) and
either a code edit or a search/navigation action.

Trajectory format
-----------------

The ``messages`` field is a list of *conversations*, not a flat
list of messages. Each conversation is a list of message dicts:

* ``[system, user, assistant]`` — a planning/action round
* ``[system, user, assistant, user, assistant]`` — a round with
  tool feedback

Step granularity: one conversation round = one step. The
assistant's output in each round is the action.
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


def _extract_task_goal(conversations: list[list[dict[str, Any]]]) -> str:
    """Extract the task from the first conversation's user message."""
    if not conversations or not conversations[0]:
        return "(no task goal found)"
    for msg in conversations[0]:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                return _truncate(content, 4000)
    return "(no task goal found)"


def _parse_steps(conversations: list[list[dict[str, Any]]]) -> list[TraceStep]:
    """Convert conversation rounds into TraceStep objects.

    Each conversation round becomes one step. The assistant's
    output is the action; any subsequent user message in the same
    round is the observation (tool feedback).
    """
    steps: list[TraceStep] = []

    for step_index, conv in enumerate(conversations):
        if not conv:
            continue

        # Find assistant messages in this round.
        thoughts: list[str] = []
        actions: list[str] = []
        observations: list[str] = []

        for msg in conv:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content) if content else ""

            if role == "assistant":
                # Try to parse JSON with scratch_pad.
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        scratch = parsed.pop("scratch_pad", "")
                        if scratch:
                            thoughts.append(str(scratch))
                        # The rest is the action.
                        actions.append(json.dumps(parsed, ensure_ascii=False))
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
                actions.append(content)
            elif role == "user" and step_index > 0:
                # User messages after the first conversation are
                # feedback/observations, not the task goal.
                observations.append(content)
            elif role == "user" and step_index == 0:
                # Skip — this is the task goal, handled separately.
                pass

        if not actions:
            continue

        thought = _truncate("\n".join(thoughts)) if thoughts else None
        action = _truncate("\n".join(actions))
        observation = _truncate("\n".join(observations)) if observations else None

        steps.append(
            TraceStep(
                index=step_index,
                thought=thought,
                action=action,
                observation=observation,
                metadata={"role": "assistant"},
            )
        )

    return steps


def load_moatless(
    *,
    split: str = "train",
    limit: int | None = None,
    one_per_instance: bool = False,
    min_steps: int | None = None,
    streaming: bool = True,
) -> list[Trace]:
    """Load Moatless SFT trajectories from HuggingFace.

    Args:
        split: Dataset split. Default ``"train"``.
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

    ds = load_dataset(
        "swesynth/SWE-Synth_Moatless-SFT-Trajectories",
        split=split,
        streaming=streaming,
    )

    traces: list[Trace] = []
    n_skipped = 0
    _seen_instances: set[str] = set()
    _id_counter: dict[str, int] = {}

    for row in ds:
        instance_id = row.get("instance_id", "")
        run_name = row.get("run_name", "unknown")

        if one_per_instance:
            if instance_id in _seen_instances:
                continue
            _seen_instances.add(instance_id)

        conversations = row.get("messages", [])
        if not conversations:
            n_skipped += 1
            continue

        steps = _parse_steps(conversations)
        if not steps:
            n_skipped += 1
            continue

        if min_steps is not None and len(steps) < min_steps:
            continue

        task_goal = _extract_task_goal(conversations)

        base_id = f"moatless-{instance_id}"
        _id_counter[base_id] = _id_counter.get(base_id, 0) + 1
        trace_id = base_id if _id_counter[base_id] == 1 else f"{base_id}-run{_id_counter[base_id]}"

        traces.append(
            Trace(
                trace_id=trace_id,
                task_goal=task_goal,
                task_id=instance_id,
                model=run_name,
                source="moatless",
                success=None,  # no resolved field in this dataset
                steps=tuple(steps),
                metadata={
                    "run_name": run_name,
                    "has_patch": bool(row.get("model_patch")),
                },
            )
        )

        if limit is not None and len(traces) >= limit:
            break

    if n_skipped:
        log.info("moatless: skipped %d traces with empty trajectories", n_skipped)
    log.info("moatless: loaded %d traces", len(traces))
    return traces


__all__ = ["load_moatless"]
