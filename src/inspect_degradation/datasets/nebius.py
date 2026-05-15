"""Nebius SWE-agent trajectories loader.

Loads trajectories from the ``nebius/SWE-agent-trajectories`` dataset
on HuggingFace. Each record contains a chat-message trajectory
(system/user/assistant turns) from SWE-agent running on SWE-bench
issues with Llama models.

Trajectory format
-----------------

Each trajectory is a list of chat-message dicts with keys:

* ``role``: ``"system"`` | ``"user"`` | ``"assistant"``
* ``text``: the message content
* ``mask``: bool (internal SWE-agent flag, not used here)

The mapping to :class:`TraceStep` pairs each ``ai`` message
(the agent's action) with the subsequent ``user`` message (the
environment's observation). System messages and the initial
user message (the issue description) are not steps — they form
the task context.

Note: SWE-agent uses ``"ai"`` as the role for agent messages,
not ``"assistant"``. System-prompt content lives in the
``system_prompt`` key, not ``text``.

Step granularity: one ``(ai, user)`` pair = one step.
This matches the SWE-agent execution loop where the agent
emits a command and the environment returns the output.
"""

from __future__ import annotations

import logging
from typing import Any

from inspect_degradation.trace import Trace, TraceStep

log = logging.getLogger(__name__)

#: Maximum characters of a single message to include in action/observation.
#: SWE-agent tool outputs can be enormous (full file contents, test suite
#: output); truncating keeps grader prompts within context budgets.
_MAX_MESSAGE_CHARS = 8000


def _truncate(text: str, limit: int = _MAX_MESSAGE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 50] + f"\n\n[... truncated, {len(text) - limit + 50} chars omitted ...]"


def _extract_task_goal(trajectory: list[dict[str, Any]]) -> str:
    """Extract the task/issue description from the trajectory.

    The first user message (after any system messages) contains the
    SWE-bench issue text. This is the "task goal" the grader evaluates
    each step against.
    """
    for msg in trajectory:
        if msg.get("role") == "user":
            text = msg.get("text", "")
            if text:
                return _truncate(text, 4000)
    return "(no task goal found)"


def _msg_text(msg: dict[str, Any]) -> str:
    """Extract the text content from a trajectory message.

    SWE-agent stores system-prompt content in ``system_prompt``
    and all other content in ``text``. Either can be None.
    """
    text = msg.get("text")
    if text is not None:
        return str(text)
    sp = msg.get("system_prompt")
    if sp is not None:
        return str(sp)
    return ""


def _parse_steps(trajectory: list[dict[str, Any]]) -> list[TraceStep]:
    """Convert a chat-message trajectory into TraceStep objects.

    Pairs each ``ai`` message with the subsequent ``user`` message:
    ai.text → action, next user.text → observation. The system
    prompt and initial user message (issue description) are
    skipped — they form the task context, not agent steps.

    Unpaired trailing ai messages (no subsequent user response)
    get observation=None.
    """
    # Skip system messages and the first user message (task goal).
    messages = list(trajectory)
    start = 0
    seen_first_user = False
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        if role == "system":
            start = i + 1
            continue
        if role == "user" and not seen_first_user:
            seen_first_user = True
            start = i + 1
            continue
        break
    messages = messages[start:]

    steps: list[TraceStep] = []
    step_index = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "ai":
            action_text = _truncate(_msg_text(msg))

            # Look ahead for the paired user/observation message.
            observation = None
            if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                observation = _truncate(_msg_text(messages[i + 1]))
                i += 2
            else:
                i += 1

            steps.append(
                TraceStep(
                    index=step_index,
                    action=action_text,
                    observation=observation,
                    metadata={"role": "ai"},
                )
            )
            step_index += 1
        else:
            # Orphan user message without a preceding ai message — skip.
            i += 1

    return steps


def load_nebius(
    *,
    split: str = "train",
    models: list[str] | None = None,
    limit: int | None = None,
    one_per_instance: bool = False,
    min_steps: int | None = None,
    streaming: bool = True,
) -> list[Trace]:
    """Load Nebius SWE-agent trajectories from HuggingFace.

    Requires the ``datasets`` package (``pip install datasets``).

    Args:
        split: HuggingFace split name. Default ``"train"``.
        models: Optional list of model names to include. If None,
            loads all models. Use this to filter to e.g.
            ``["swe-agent-llama-70b"]``.
        limit: Optional cap on total traces loaded. Applied after
            model filtering.
        one_per_instance: If True, keep only the first trace per
            ``(instance_id, model_name)`` pair. The Nebius dataset
            contains multiple runs of the same model on the same
            issue; this flag gives instance-diverse sampling so a
            small ``limit`` covers many different bugs rather than
            many runs of the same one.
        min_steps: If set, skip traces with fewer than this many
            agent steps (``ai`` messages). Use to filter to long
            traces for testing context-length degradation.
        streaming: If True (default), streams the dataset without
            downloading the full file. Set False to download first
            (faster iteration on repeated loads, but uses disk).

    Returns:
        List of :class:`Trace` objects ready for grading.

    Raises:
        RuntimeError: ``datasets`` package not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "the 'datasets' package is required to load Nebius trajectories; "
            "install via `pip install datasets`"
        ) from exc

    ds = load_dataset(
        "nebius/SWE-agent-trajectories",
        split=split,
        streaming=streaming,
    )

    traces: list[Trace] = []
    n_skipped = 0
    _id_counter: dict[str, int] = {}
    _seen_instances: set[str] = set()

    for row in ds:
        model_name = row.get("model_name", "unknown")
        if models is not None and model_name not in models:
            continue

        instance_id = row.get("instance_id", "")

        if one_per_instance:
            instance_key = f"{instance_id}|{model_name}"
            if instance_key in _seen_instances:
                continue
            _seen_instances.add(instance_key)
        trajectory = row.get("trajectory", [])
        if not trajectory:
            n_skipped += 1
            continue

        task_goal = _extract_task_goal(trajectory)
        steps = _parse_steps(trajectory)
        if not steps:
            n_skipped += 1
            continue

        if min_steps is not None and len(steps) < min_steps:
            continue

        # SWE-bench "target" field: True if the agent's patch resolved
        # the issue (ground-truth from test execution).
        success = row.get("target")
        if isinstance(success, (bool, int)):
            success = bool(success)
        else:
            success = None

        # Deduplicate trace IDs: same instance + model can appear
        # multiple times (multiple runs on the same issue).
        base_id = f"nebius-{instance_id}-{model_name}"
        _id_counter[base_id] = _id_counter.get(base_id, 0) + 1
        if _id_counter[base_id] > 1:
            trace_id = f"{base_id}-run{_id_counter[base_id]}"
        else:
            trace_id = base_id

        traces.append(
            Trace(
                trace_id=trace_id,
                task_goal=task_goal,
                task_id=instance_id,
                model=model_name,
                source="nebius",
                success=success,
                steps=tuple(steps),
                metadata={
                    "exit_status": row.get("exit_status"),
                    "has_patch": bool(row.get("generated_patch")),
                },
            )
        )

        if limit is not None and len(traces) >= limit:
            break

    if n_skipped:
        log.info("nebius: skipped %d traces with empty trajectories", n_skipped)
    log.info(
        "nebius: loaded %d traces (%d distinct models)",
        len(traces),
        len({t.model for t in traces}),
    )
    return traces


def load_nebius_summary(
    *,
    split: str = "train",
    sample_size: int = 2000,
) -> dict[str, Any]:
    """Quick summary of the dataset shape without loading all traces.

    Returns model counts, step-length distribution, and exit status
    counts from a streaming sample. Useful for planning Phase 3
    slice sizes and cost estimates.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "the 'datasets' package is required; install via `pip install datasets`"
        ) from exc

    from collections import Counter

    import numpy as np

    ds = load_dataset(
        "nebius/SWE-agent-trajectories",
        split=split,
        streaming=True,
    )

    model_counts: Counter[str] = Counter()
    step_counts: list[int] = []
    exit_statuses: Counter[str] = Counter()
    success_counts: Counter[bool] = Counter()

    for i, row in enumerate(ds):
        model_counts[row.get("model_name", "unknown")] += 1
        traj = row.get("trajectory", [])
        # Count assistant messages (= steps after parsing)
        n_steps = sum(1 for m in traj if m.get("role") == "assistant")
        step_counts.append(n_steps)
        exit_statuses[row.get("exit_status", "unknown")] += 1
        target = row.get("target")
        if isinstance(target, (bool, int)):
            success_counts[bool(target)] += 1
        if i >= sample_size - 1:
            break

    arr = np.array(step_counts) if step_counts else np.array([0])
    return {
        "n_sampled": len(step_counts),
        "models": dict(model_counts.most_common()),
        "steps_per_trace": {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
        },
        "exit_statuses": dict(exit_statuses.most_common()),
        "success_rate": success_counts.get(True, 0) / max(len(step_counts), 1),
    }


__all__ = ["load_nebius", "load_nebius_summary"]
