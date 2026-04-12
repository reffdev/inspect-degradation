"""Multi-SWE-bench trajectories loader.

9 models × 3 scaffoldings from ByteDance's Multi-SWE-bench leaderboard.
Each combination is a zip file containing ~500 JSON trajectory files.

https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench_trajs

Available models: Claude 3.5 Sonnet, Claude 3.7 Sonnet, GPT-4o,
DeepSeek-R1, DeepSeek-V3, Doubao-1.5-pro, OpenAI o1, OpenAI
o3-mini, Qwen 2.5-72B.

Available scaffoldings: SWE-agent, OpenHands, Agentless.

Trajectory format: each JSON file has ``fncall_messages`` (function-
calling format) or ``messages`` (plain chat). Both use the standard
``assistant``/``tool`` alternating pattern.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

from inspect_degradation.trace import Trace, TraceStep

log = logging.getLogger(__name__)

_DATASET_ID = "ByteDance-Seed/Multi-SWE-bench_trajs"
_RESOLVED_DATASET_ID = "ByteDance-Seed/Multi-SWE-bench"
_MAX_MESSAGE_CHARS = 8000

# Maps friendly names to zip file name components.
AVAILABLE_MODELS = {
    "claude-3.5-sonnet": "Claude-3.5-Sonnet(Oct)",
    "claude-3.7-sonnet": "Claude-3.7-Sonnet",
    "gpt-4o": "GPT-4o-1120",
    "deepseek-r1": "DeepSeek-R1",
    "deepseek-v3": "DeepSeek-V3",
    "doubao-1.5-pro": "Doubao-1.5-pro",
    "o1": "OpenAI-o1",
    "o3-mini": "OpenAI-o3-mini-high",
    "qwen-2.5-72b": "Qwen2.5-72B-Instruct",
}

AVAILABLE_SCAFFOLDINGS = {
    "swe-agent": "SWE-agent",
    "openhands": "OpenHands",
    # "agentless" is not step-based — it uses a structured pipeline
    # without sequential agent actions, so step-level grading doesn't
    # apply. Excluded from the available scaffoldings.
}


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
    """Extract the task description from the message history.

    SWE-agent traces include a demonstration preamble as an early user
    message (``"Here is a demonstration..."``).  The real task is the
    *last* user message before the first assistant message — that's the
    actual issue description the agent works on.  For OpenHands traces
    the first user message is typically the task itself.
    """
    # Find the last user message before the first assistant turn.
    last_user: str | None = None
    for msg in messages:
        role = msg.get("role", "")
        if role == "user":
            content = msg.get("content", "")
            # OpenHands uses [{"type": "text", "text": "..."}] format.
            if isinstance(content, list):
                text_parts = [
                    p["text"] for p in content
                    if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
                ]
                content = "\n".join(text_parts)
            if isinstance(content, str) and content:
                last_user = content
        elif role == "assistant":
            break  # stop once the agent starts acting

    if last_user:
        return _truncate(last_user, 4000)
    return "(no task goal found)"


def _parse_steps(messages: list[dict[str, Any]]) -> list[TraceStep]:
    steps: list[TraceStep] = []
    step_index = 0
    i = 0

    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "assistant":
            content = msg.get("content") or ""
            if not isinstance(content, str):
                content = str(content) if content else ""
            thought = _truncate(content) if content else None

            tool_calls = msg.get("tool_calls")
            if tool_calls:
                action = _format_tool_calls(tool_calls)
            else:
                action = thought or "(no action)"
                thought = None

            # Collect consecutive tool responses.
            obs_parts: list[str] = []
            while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                i += 1
                tc = messages[i].get("content", "")
                if isinstance(tc, str) and tc:
                    obs_parts.append(tc)
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


def _parse_traj_steps(trajectory: list[dict[str, Any]]) -> list[TraceStep]:
    """Parse SWE-agent .traj format: list of action/observation/thought dicts."""
    steps: list[TraceStep] = []
    for step_index, entry in enumerate(trajectory):
        action = entry.get("action", "")
        if isinstance(action, dict):
            action = json.dumps(action, ensure_ascii=False)
        action = _truncate(str(action)) if action else "(no action)"

        observation = entry.get("observation", "")
        if isinstance(observation, dict):
            observation = json.dumps(observation, ensure_ascii=False)
        observation = _truncate(str(observation)) if observation else None

        thought = entry.get("thought", "")
        thought = _truncate(str(thought)) if thought else None

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


def _zip_filename(model: str, scaffolding: str) -> str:
    """Build the zip filename from friendly model/scaffolding names."""
    model_part = AVAILABLE_MODELS.get(model)
    scaff_part = AVAILABLE_SCAFFOLDINGS.get(scaffolding)
    if not model_part:
        raise ValueError(
            f"Unknown model {model!r}. Available: {sorted(AVAILABLE_MODELS)}"
        )
    if not scaff_part:
        raise ValueError(
            f"Unknown scaffolding {scaffolding!r}. Available: {sorted(AVAILABLE_SCAFFOLDINGS)}"
        )
    return f"python/20250329_{scaff_part}_{model_part}.zip"


def load_resolved_status(
    model: str,
    scaffolding: str,
) -> dict[str, bool]:
    """Load per-instance resolved status from trajectory ``score`` fields.

    The Multi-SWE-bench trajectory files include a ``score`` field:
    ``1`` = resolved, ``0`` = unresolved, ``None`` = unknown.
    This function extracts scores for the given model/scaffolding
    combination.

    Returns a dict mapping ``instance_id`` to ``True``/``False``.
    Instances with ``score=None`` are omitted.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required; install via `pip install huggingface_hub`"
        ) from exc

    zip_name = _zip_filename(model, scaffolding)
    log.info("loading resolved status from %s / %s", _DATASET_ID, zip_name)
    zip_path = hf_hub_download(_DATASET_ID, zip_name, repo_type="dataset")

    resolved: dict[str, bool] = {}
    with zipfile.ZipFile(zip_path) as zf:
        traj_files = [
            n for n in zf.namelist()
            if n.endswith(".json") or n.endswith(".traj")
        ]
        for tf in traj_files:
            if "/" in tf:
                instance_id = tf.split("/")[0]
            else:
                instance_id = tf.replace(".traj", "").replace(".json", "")

            try:
                with zf.open(tf) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, KeyError):
                continue

            score = data.get("score")
            if score is not None:
                resolved[instance_id] = bool(score)

    log.info("loaded resolved status for %d instances (%d resolved)",
             len(resolved), sum(resolved.values()))
    return resolved


def load_multi_swebench(
    *,
    model: str,
    scaffolding: str,
    limit: int | None = None,
    one_per_instance: bool = False,
    min_steps: int | None = None,
    random_sample: bool = False,
    seed: int = 42,
    include_resolved: bool = False,
) -> list[Trace]:
    """Load trajectories from the Multi-SWE-bench dataset.

    Downloads the relevant zip file from HuggingFace on first call
    (cached thereafter). Requires ``huggingface_hub``.

    Args:
        model: One of the keys in :data:`AVAILABLE_MODELS`.
        scaffolding: One of the keys in :data:`AVAILABLE_SCAFFOLDINGS`.
        limit: Optional cap on traces loaded.
        one_per_instance: Keep only first trace per instance_id.
        min_steps: Skip traces with fewer steps.
        include_resolved: If True, extract resolved status from the
            trajectory ``score`` fields (1=resolved, 0=unresolved).

    Returns:
        List of :class:`Trace` objects.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required; install via `pip install huggingface_hub`"
        ) from exc

    zip_name = _zip_filename(model, scaffolding)
    log.info("downloading %s from %s", zip_name, _DATASET_ID)
    zip_path = hf_hub_download(_DATASET_ID, zip_name, repo_type="dataset")

    traces: list[Trace] = []
    n_skipped = 0
    _seen_instances: set[str] = set()

    with zipfile.ZipFile(zip_path) as zf:
        # OpenHands uses .json in subdirectories; SWE-agent uses .traj flat.
        traj_files = sorted(
            n for n in zf.namelist()
            if n.endswith(".json") or n.endswith(".traj")
        )
        if random_sample:
            import random
            rng = random.Random(seed)
            rng.shuffle(traj_files)
        log.info("found %d trajectory files in %s%s", len(traj_files), zip_name,
                 " (randomly shuffled)" if random_sample else "")

        for jf in traj_files:
            # Instance ID: from directory name (OpenHands) or filename (SWE-agent).
            if "/" in jf:
                instance_id = jf.split("/")[0]
            else:
                instance_id = jf.replace(".traj", "").replace(".json", "")

            if one_per_instance:
                if instance_id in _seen_instances:
                    continue
                _seen_instances.add(instance_id)

            try:
                with zf.open(jf) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, KeyError):
                n_skipped += 1
                continue

            # Three possible formats:
            # 1. OpenHands: fncall_messages or messages (chat format)
            # 2. SWE-agent .traj: trajectory (action/observation dicts)
            #    + history (chat messages)
            # 3. Agentless: structured pipeline, no steps — skip
            messages = data.get("fncall_messages") or data.get("messages")
            if messages:
                steps = _parse_steps(messages)
            elif "trajectory" in data and data["trajectory"]:
                steps = _parse_traj_steps(data["trajectory"])
                # Use history for task goal if available.
                messages = data.get("history", [])
            elif "history" in data and data["history"]:
                steps = _parse_steps(data["history"])
                messages = data["history"]
            else:
                n_skipped += 1
                continue
            if not steps:
                n_skipped += 1
                continue

            if min_steps is not None and len(steps) < min_steps:
                continue

            task_goal = _extract_task_goal(messages) if messages else "(no task goal found)"

            source_label = f"multi-swebench-{scaffolding}"
            trace_id = f"msb-{instance_id}-{model}-{scaffolding}"

            score = data.get("score")
            success = bool(score) if include_resolved and score is not None else None

            traces.append(
                Trace(
                    trace_id=trace_id,
                    task_goal=task_goal,
                    task_id=instance_id,
                    model=model,
                    source=source_label,
                    success=success,
                    steps=tuple(steps),
                    metadata={
                        "scaffolding": scaffolding,
                        "dataset": "Multi-SWE-bench",
                    },
                )
            )

            if limit is not None and len(traces) >= limit:
                break

    if n_skipped:
        log.info("multi-swebench: skipped %d files", n_skipped)
    log.info(
        "multi-swebench: loaded %d traces (%s on %s)",
        len(traces), model, scaffolding,
    )
    return traces


def list_available() -> list[tuple[str, str]]:
    """Return all available (model, scaffolding) combinations."""
    return [
        (m, s)
        for m in sorted(AVAILABLE_MODELS)
        for s in sorted(AVAILABLE_SCAFFOLDINGS)
    ]


__all__ = [
    "AVAILABLE_MODELS",
    "AVAILABLE_SCAFFOLDINGS",
    "list_available",
    "load_multi_swebench",
    "load_resolved_status",
]
