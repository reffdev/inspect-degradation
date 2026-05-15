"""Model-drift canary: detect silent provider model updates between runs.

Providers update model weights without changing the model name
string. A run today and the same script three weeks later may
produce different grades for byte-identical inputs and there is
no first-class way to detect this from the cached results. The
canary makes it loud:

* At experiment start, send a fixed deterministic prompt to the
  grader's model and hash the response.
* Stash the hash in :class:`CanaryFingerprint` alongside the model
  name, prompt text, and timestamp.
* On a future run with the same model name and prompt, recompute
  the hash and compare. A mismatch is a *signal* (model
  responses changed) not a *crash* — a reviewer might still
  legitimately rerun a known-drifted experiment, they just need
  to know it drifted.

This is a *detection* mechanism, not a *correction* mechanism.
Drift cannot be undone after the fact; the most you can do is
flag affected experiments and re-validate them.

The canary uses temperature 0 and short max_tokens, so a typical
canary call costs less than a single grader call. It is cheap to
include in every Phase 1 / Phase 3 run start.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: A short, neutral, deterministic prompt that should produce stable
#: output across providers and over time. The point is *not* that
#: the answer is interesting, only that it is reproducible by a
#: well-behaved deterministic generation request. We deliberately
#: avoid anything model-specific (no "what model are you?"), anything
#: with a date (no "what year is it?"), and anything that providers
#: might rewrite via system prompts.
DEFAULT_CANARY_PROMPT = (
    "Output exactly the following JSON object on a single line and "
    'nothing else: {"echo": 42}'
)


@dataclass(frozen=True)
class CanaryFingerprint:
    """A reproducibility fingerprint for a single grader model.

    Attributes:
        model: Model name (the same string the grader uses).
        prompt: The exact canary prompt that was issued.
        response_sha256: SHA-256 of the model's response (after
            stripping leading/trailing whitespace, before any
            other normalization).
        response_excerpt: First 200 characters of the response,
            for human-readable diff when fingerprints disagree.
        captured_at: ISO-8601 UTC timestamp.
        metadata: Extra context (e.g. temperature, max_tokens) so
            a reader knows how the canary was generated.
    """

    model: str
    prompt: str
    response_sha256: str
    response_excerpt: str
    captured_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches(self, other: "CanaryFingerprint") -> bool:
        """Two fingerprints match iff model, prompt, and hash agree.

        Timestamp and excerpt are not compared — they are audit
        fields, not identity fields.
        """
        return (
            self.model == other.model
            and self.prompt == other.prompt
            and self.response_sha256 == other.response_sha256
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt": self.prompt,
            "response_sha256": self.response_sha256,
            "response_excerpt": self.response_excerpt,
            "captured_at": self.captured_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanaryFingerprint":
        return cls(
            model=data["model"],
            prompt=data["prompt"],
            response_sha256=data["response_sha256"],
            response_excerpt=data.get("response_excerpt", ""),
            captured_at=data.get("captured_at", ""),
            metadata=dict(data.get("metadata") or {}),
        )


def fingerprint_from_response(
    *,
    model: str,
    response: str,
    prompt: str = DEFAULT_CANARY_PROMPT,
    extra_metadata: dict[str, Any] | None = None,
) -> CanaryFingerprint:
    """Build a :class:`CanaryFingerprint` from a captured response.

    The response is whitespace-stripped before hashing so trailing
    newlines from different providers don't trip the comparison;
    everything else (capitalization, punctuation, internal
    whitespace) is preserved exactly.
    """
    normalized = response.strip()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return CanaryFingerprint(
        model=model,
        prompt=prompt,
        response_sha256=digest,
        response_excerpt=normalized[:200],
        captured_at=datetime.now(timezone.utc).isoformat(),
        metadata=dict(extra_metadata or {}),
    )


async def capture_canary_async(
    *,
    model: str,
    prompt: str = DEFAULT_CANARY_PROMPT,
    temperature: float = 0.0,
    max_tokens: int = 64,
) -> CanaryFingerprint:
    """Issue a canary call to ``model`` and return its fingerprint.

    Uses Inspect AI's :func:`inspect_ai.model.get_model` so the same
    provider routing the grader uses serves the canary call. Cheap
    by design: temperature 0, ``max_tokens=64``, no system prompt.

    Raises:
        RuntimeError: the model returned an empty response (cannot
            fingerprint nothing — and an empty response from a
            canary call is itself a sign something is wrong).
    """
    from inspect_ai.model import (
        ChatMessageUser,
        GenerateConfig,
        get_model,
    )

    handle = get_model(model)
    output = await handle.generate(
        [ChatMessageUser(content=prompt)],
        config=GenerateConfig(temperature=temperature, max_tokens=max_tokens),
    )
    response = getattr(output, "completion", None) or ""
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError(
            f"canary call to {model!r} returned an empty response; "
            "cannot fingerprint"
        )
    return fingerprint_from_response(
        model=model,
        response=response,
        prompt=prompt,
        extra_metadata={
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
    )


@dataclass(frozen=True)
class CanaryComparison:
    """Result of comparing two fingerprints from the same model.

    Attributes:
        match: True iff the fingerprints agree on hash.
        models_match: True iff the model name string is identical.
        prompts_match: True iff the canary prompt is identical.
        before: The earlier fingerprint.
        after: The later fingerprint.
        notes: Human-readable summary of any mismatches.
    """

    match: bool
    models_match: bool
    prompts_match: bool
    before: CanaryFingerprint
    after: CanaryFingerprint
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "match": self.match,
            "models_match": self.models_match,
            "prompts_match": self.prompts_match,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "notes": self.notes,
        }


def compare_fingerprints(
    before: CanaryFingerprint,
    after: CanaryFingerprint,
) -> CanaryComparison:
    """Diff two fingerprints; the returned report explains any mismatch.

    The function does not raise on mismatch — drift detection is a
    diagnostic, and the appropriate action depends on context (a
    reviewer may legitimately accept a drifted run).
    """
    models_match = before.model == after.model
    prompts_match = before.prompt == after.prompt
    hash_match = before.response_sha256 == after.response_sha256

    notes_parts: list[str] = []
    if not models_match:
        notes_parts.append(
            f"model name changed: {before.model!r} -> {after.model!r}"
        )
    if not prompts_match:
        notes_parts.append("canary prompt changed")
    if models_match and prompts_match and not hash_match:
        notes_parts.append(
            f"DRIFT: same model+prompt produced different responses "
            f"(was {before.response_sha256[:12]}..., "
            f"now {after.response_sha256[:12]}...)"
        )
    if not notes_parts:
        notes_parts.append("fingerprints match")

    return CanaryComparison(
        match=models_match and prompts_match and hash_match,
        models_match=models_match,
        prompts_match=prompts_match,
        before=before,
        after=after,
        notes="; ".join(notes_parts),
    )


__all__ = [
    "DEFAULT_CANARY_PROMPT",
    "CanaryComparison",
    "CanaryFingerprint",
    "capture_canary_async",
    "compare_fingerprints",
    "fingerprint_from_response",
]
