"""Adapter from Inspect AI ``TaskState`` to our :class:`Trace`.

This is the seam between Inspect's live message/tool-call representation
and the source-agnostic :class:`Trace` that the grader consumes. Keeping
the adapter in its own module means:

* the scorer body stays small and readable,
* offline-only users (TRAIL, Nebius) never import this code path,
* the mapping policy (which messages count as "steps", how tool calls are
  paired with results) is documented in exactly one place.

Inspect AI's :class:`TaskState` exposes a ``messages`` list and the active
``tools``; an "agent step" in our terminology is the smallest unit a human
annotator would judge — typically one assistant turn together with any
tool result that immediately answered it. Other groupings are possible
(per-tool-call, per-message), but cross-source comparability requires we
commit to one. We commit here, in this module's docstring, so that
analysis results across data sources are interpretable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inspect_degradation.trace import Trace, TraceStep

if TYPE_CHECKING:
    from inspect_ai.solver import TaskState  # pragma: no cover


def task_state_to_trace(state: "TaskState", *, task_goal: str | None = None) -> Trace:
    """Convert an Inspect AI ``TaskState`` into a graded-ready :class:`Trace`.

    Args:
        state: The current Inspect task state at scoring time.
        task_goal: Optional override for the task goal. If omitted, the
            first user message in ``state.messages`` is used.
    """
    messages = list(getattr(state, "messages", []) or [])
    if not messages:
        raise ValueError("cannot adapt a TaskState with no messages")

    goal = task_goal or _extract_task_goal(messages)

    steps: list[TraceStep] = []
    pending_assistant: Any = None
    pending_thought: str | None = None

    for msg in messages:
        role = getattr(msg, "role", None)
        if role == "user":
            # User-injected mid-trace messages become observations on the
            # most recent step if one is open, or metadata otherwise.
            if pending_assistant is not None:
                steps.append(
                    _step_from_pair(
                        index=len(steps),
                        assistant=pending_assistant,
                        observation=_message_text(msg),
                        thought=pending_thought,
                    )
                )
                pending_assistant = None
                pending_thought = None
            continue
        if role == "assistant":
            if pending_assistant is not None:
                # Two assistant messages in a row: close the prior one with
                # no observation rather than dropping it.
                steps.append(
                    _step_from_pair(
                        index=len(steps),
                        assistant=pending_assistant,
                        observation=None,
                        thought=pending_thought,
                    )
                )
                pending_thought = None
            pending_assistant = msg
            continue
        if role == "tool":
            if pending_assistant is None:
                # Orphan tool result — preserve as a standalone observation step.
                steps.append(
                    TraceStep(
                        index=len(steps),
                        action="(tool result with no preceding assistant message)",
                        observation=_message_text(msg),
                    )
                )
                continue
            steps.append(
                _step_from_pair(
                    index=len(steps),
                    assistant=pending_assistant,
                    observation=_message_text(msg),
                    thought=pending_thought,
                )
            )
            pending_assistant = None
            pending_thought = None
            continue
        # System and unknown roles are not steps.

    if pending_assistant is not None:
        steps.append(
            _step_from_pair(
                index=len(steps),
                assistant=pending_assistant,
                observation=None,
                thought=pending_thought,
            )
        )

    return Trace(
        trace_id=str(getattr(state, "sample_id", "") or getattr(state, "epoch", "") or "live"),
        task_goal=goal,
        task_id=getattr(state, "sample_id", None),
        model=str(getattr(state, "model", None)) if getattr(state, "model", None) else None,
        source="inspect-live",
        steps=tuple(steps),
        metadata={},
    )


def _extract_task_goal(messages: list[Any]) -> str:
    for msg in messages:
        if getattr(msg, "role", None) == "user":
            return _message_text(msg)
    raise ValueError("no user message found in TaskState; cannot infer task_goal")


def _message_text(msg: Any) -> str:
    """Best-effort extraction of plain text from an Inspect chat message.

    Inspect message ``content`` may be a string or a list of content parts;
    we render the parts as their text where possible and fall back to
    ``repr`` for non-text parts (images, etc.) so the grader at least sees
    that something was there.
    """
    content = getattr(msg, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered: list[str] = []
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                rendered.append(text)
            else:
                rendered.append(repr(part))
        return "\n".join(rendered)
    return str(content)


def _step_from_pair(
    *,
    index: int,
    assistant: Any,
    observation: str | None,
    thought: str | None,
) -> TraceStep:
    return TraceStep(
        index=index,
        thought=thought,
        action=_message_text(assistant),
        observation=observation,
    )
