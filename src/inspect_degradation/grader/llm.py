"""LLM-as-judge grader backed by Inspect AI's model API.

Parse-failure handling: model providers occasionally return malformed
JSON (truncated, extra prose, broken braces). The grader handles this
in two layers:

1. **Retry inside the generation loop.** If parsing fails, the next
   retry attempt re-samples the model with the same prompt. Most
   parse failures are transient — the same prompt usually yields
   valid JSON on a second try.
2. **Fallback step on exhaustion.** If every retry attempt produces
   unparseable output, the grader returns a neutral
   :class:`GradedStep` with the parse error recorded under
   ``raw["parse_error"]``. This keeps a single bad step from
   crashing the entire trace and lets downstream analysis filter
   or count failed grades explicitly.

This is the only grader that talks to a real model. It uses
``inspect_ai.model.get_model`` so we inherit Inspect's provider routing,
caching, retry, and rate-limiting policy rather than reimplementing any
of it. The model is resolved *lazily* per call, which is the pattern
Inspect documents for use inside scorers — see PROJECT_PLAN.md
"Inspect AI Integration Layer".

This module also houses the single-model self-consistency sampling
path. When ``LLMGraderConfig.sample_n > 1`` the grader calls the model
N times at non-zero temperature, picks the majority verdict by
validity label, and records the per-sample results under
``raw["self_consistency"]`` as diagnostic metadata. This is a useful
primitive for single-model uncertainty sampling; for multi-model
uncertainty the higher-level :class:`~inspect_degradation.grader.ensemble.EnsembleGrader`
composes independent :class:`LLMGrader` instances across different
model families.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from inspect_degradation.grader.interface import Grader, GraderSnapshot, StepContext
from inspect_degradation.grader.response import (
    GradeResponse,
    GraderResponseError,
    parse_grade_response,
)
from inspect_degradation.grader.rubric import Rubric
from inspect_degradation.schema import GradedStep, Validity

if TYPE_CHECKING:  # avoid forcing inspect_ai import at type-check time only
    from inspect_ai.model import Model


#: Metadata key used to stash per-sample self-consistency results on the
#: returned :class:`GradedStep` when ``sample_n > 1``. Purely diagnostic:
#: nothing in the production pipeline gates on this value.
SELF_CONSISTENCY_KEY = "self_consistency"


@dataclass(frozen=True)
class LLMGraderConfig:
    """Tunable knobs for :class:`LLMGrader`.

    Held in a separate dataclass (rather than constructor kwargs) so
    configs can be persisted alongside grading runs for reproducibility.

    Attributes:
        model: Inspect AI model spec, e.g. ``"openai/gpt-4o-mini"`` or
            ``"anthropic/claude-haiku-4-5"``.
        temperature: Sampling temperature. Must be > 0 when
            ``sample_n > 1`` or self-consistency collapses to N identical
            samples. Checked at config construction.
        max_tokens: Per-response token cap.
        max_concurrency: Cap on concurrent step-grading calls at the
            :class:`~inspect_degradation.grader.interface.Grader` level.
            Note that when ``sample_n > 1`` each step internally
            serializes its N samples, so this cap bounds *step*
            concurrency, not raw provider call concurrency.
        max_retries: Attempts per model call on transient failures.
        sample_n: Number of independent samples per step for
            self-consistency grading. Default 1 (no sampling). Set to
            3-5 to enable self-consistency; higher values buy less
            signal per extra call.
        extra: Provider-specific overrides forwarded to ``GenerateConfig``.
    """

    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    max_concurrency: int = 8
    max_retries: int = 3
    sample_n: int = 1
    prior_context_char_budget: int | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_n < 1:
            raise ValueError(f"sample_n must be >= 1, got {self.sample_n}")
        if self.sample_n > 1 and self.temperature <= 0.0:
            raise ValueError(
                f"sample_n={self.sample_n} requires temperature > 0, "
                f"got temperature={self.temperature}. Sampling at "
                f"temperature 0 produces identical samples and defeats "
                f"self-consistency."
            )


class LLMGrader(Grader):
    """Grade trace steps with a single LLM judge using a fixed rubric.

    The grader is intentionally narrow: one model, one rubric, one
    response schema. To compose multiple judges, wrap instances of
    this class in
    :class:`~inspect_degradation.grader.ensemble.EnsembleGrader`.
    """

    def __init__(self, config: LLMGraderConfig, rubric: Rubric) -> None:
        self._config = config
        self._rubric = rubric
        self._resolved_model: "Model | None" = None
        # Per-run truncation accounting. Surfaced via
        # :meth:`truncation_summary` so end-of-run reporting can flag
        # corpora where the prior-context cap was hit often enough to
        # confound a step-index slope.
        self._n_renders = 0
        self._n_truncations = 0
        self._total_steps_dropped = 0

    # ------------------------------------------------------------------ Grader

    @property
    def name(self) -> str:
        if self._config.sample_n > 1:
            return f"{self._config.model}@sc{self._config.sample_n}"
        return self._config.model

    @property
    def max_concurrency(self) -> int:  # type: ignore[override]
        return self._config.max_concurrency

    def snapshot(self) -> GraderSnapshot:
        return GraderSnapshot(
            kind="llm",
            name=self.name,
            fields={
                "model": self._config.model,
                "temperature": self._config.temperature,
                "max_tokens": self._config.max_tokens,
                "sample_n": self._config.sample_n,
                "rubric_name": self._rubric.name,
                "rubric_version": self._rubric.version,
            },
        )

    async def grade_step(self, ctx: StepContext) -> GradedStep:
        try:
            return await self._grade_step_inner(ctx)
        except Exception as exc:
            # Last-resort catch: if anything unexpected escapes the
            # parse-failure and empty-completion handlers, we still
            # return a fallback step rather than killing the trace.
            return self._fallback_step(
                step_index=ctx.step_index,
                error=GraderResponseError(
                    f"unexpected error in grade_step: {type(exc).__name__}: {exc}",
                    raw=None,
                ),
            )

    async def _grade_step_inner(self, ctx: StepContext) -> GradedStep:
        prompt, diagnostics = self._rubric.render_user(
            task_goal=ctx.task_goal,
            step_index=ctx.step_index,
            step=ctx.step,
            prior_steps=ctx.prior_steps,
            prior_context_char_budget=self._config.prior_context_char_budget,
        )
        self._n_renders += 1
        if diagnostics.prior_steps_truncated:
            self._n_truncations += 1
            self._total_steps_dropped += diagnostics.prior_steps_dropped
        if self._config.sample_n == 1:
            return await self._grade_single(ctx, prompt)
        return await self._grade_self_consistency(ctx, prompt)

    def truncation_summary(self) -> dict[str, int | float]:
        """Return per-run prior-context truncation diagnostics.

        Callers should log this at end of a grading run; a high
        truncation rate is a red flag that the step-index slope is
        partly a context-truncation artifact, not a real degradation
        signal. ``rate`` is the fraction of step renders that hit
        the cap; ``mean_dropped_per_truncation`` is the average
        number of prior steps omitted on truncated renders.
        """
        if self._n_renders == 0:
            return {
                "n_renders": 0,
                "n_truncations": 0,
                "rate": 0.0,
                "total_steps_dropped": 0,
                "mean_dropped_per_truncation": 0.0,
            }
        return {
            "n_renders": self._n_renders,
            "n_truncations": self._n_truncations,
            "rate": self._n_truncations / self._n_renders,
            "total_steps_dropped": self._total_steps_dropped,
            "mean_dropped_per_truncation": (
                self._total_steps_dropped / self._n_truncations
                if self._n_truncations > 0
                else 0.0
            ),
        }

    # ------------------------------------------------------------------ grading paths

    async def _grade_single(self, ctx: StepContext, prompt: str) -> GradedStep:
        try:
            completion, response = await self._generate_and_parse_with_retry(prompt)
        except (GraderResponseError, _TransientGenerationError) as exc:
            return self._fallback_step(
                step_index=ctx.step_index,
                error=GraderResponseError(str(exc), raw=None),
            )
        return response.to_graded_step(
            step_index=ctx.step_index,
            grader_model=self._config.model,
            raw={"completion": completion},
        )

    async def _grade_self_consistency(
        self, ctx: StepContext, prompt: str
    ) -> GradedStep:
        """Sample the model N times and return the majority verdict.

        Samples are taken sequentially rather than concurrently so the
        Grader-level concurrency budget still bounds total provider
        calls. Each sample carries its own internal retry-on-parse-
        failure budget; samples whose retry budget is exhausted are
        dropped (with the failure recorded). If *all* samples fail
        the step falls back to a neutral grade with the parse error
        recorded — we do not crash the trace.

        The returned :class:`GradedStep` is the sample whose validity
        label matched the majority. Per-field voting (choose the modal
        value for each field independently) would be technically
        defensible but can produce incoherent cross-field states
        (majority=fail + majority-dependency=n/a), so we pick a whole
        sample instead.
        """
        samples: list[GradedStep] = []
        completions: list[str] = []
        parse_errors: list[str] = []
        for _ in range(self._config.sample_n):
            try:
                completion, response = await self._generate_and_parse_with_retry(prompt)
            except (GraderResponseError, _TransientGenerationError) as exc:
                parse_errors.append(str(exc))
                continue
            step = response.to_graded_step(
                step_index=ctx.step_index,
                grader_model=self._config.model,
                raw={"completion": completion},
            )
            samples.append(step)
            completions.append(completion)

        if not samples:
            # All N samples failed to parse — surface a single fallback
            # carrying the collected error messages so the audit trail
            # explains the gap.
            return self._fallback_step(
                step_index=ctx.step_index,
                error=GraderResponseError(
                    f"all {self._config.sample_n} self-consistency samples "
                    f"failed to parse: {parse_errors}",
                    raw=None,
                ),
            )

        consensus = _select_consensus_sample(samples)
        validity_counts = Counter(s.validity.value for s in samples)
        unanimous = len(validity_counts) == 1

        # Preserve the full sample set on the returned grade so ensemble
        # policies can read agreement signal without re-running the
        # samples, and so post-hoc analysis has per-sample detail.
        enriched_raw: dict[str, object] = dict(consensus.raw or {})
        enriched_raw[SELF_CONSISTENCY_KEY] = {
            "n_samples": self._config.sample_n,
            "n_parsed": len(samples),
            "n_parse_failures": len(parse_errors),
            "unanimous": unanimous,
            "validity_counts": dict(validity_counts),
            "sample_validities": [s.validity.value for s in samples],
            "sample_completions": completions,
            "parse_errors": parse_errors,
        }
        return consensus.model_copy(update={"raw": enriched_raw})

    # ------------------------------------------------------------------ internals

    def _fallback_step(
        self,
        *,
        step_index: int,
        error: GraderResponseError,
    ) -> GradedStep:
        """Build a neutral fallback step when parsing fails irrecoverably.

        Returns a step with ``validity=neutral`` so the trace's
        downstream rate / slope statistics treat it as
        wheel-spinning rather than as either a success or a failure.
        The parse error and any captured raw response are stashed
        under ``raw['parse_error']`` so the audit trail explains
        why a real grade is missing.
        """
        return GradedStep(
            step_index=step_index,
            validity=Validity.neutral,
            grader_model=self._config.model,
            raw={
                "parse_error": str(error),
                "parse_error_raw": error.raw,
            },
        )

    async def _generate_and_parse_with_retry(
        self, user_prompt: str
    ) -> tuple[str, GradeResponse]:
        """Generate a completion and parse it, retrying on either failure.

        Both transient generation failures (empty completions,
        provider-side output extraction problems) and parse failures
        (malformed JSON, unbalanced braces, schema violations) are
        retried by re-sampling the model with the same prompt. Most
        parse failures are transient — the same prompt usually yields
        valid JSON on a second try.

        Returns the ``(completion, response)`` pair on success, or
        raises :class:`GraderResponseError` if every retry attempt
        produced unparseable output, or
        :class:`_TransientGenerationError` if every attempt produced
        an empty completion.
        """
        from inspect_ai.model import (  # local import: see module docstring
            ChatMessageSystem,
            ChatMessageUser,
            GenerateConfig,
            get_model,
        )

        if self._resolved_model is None:
            self._resolved_model = get_model(self._config.model)

        gen_config = GenerateConfig(
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
            **self._config.extra,  # type: ignore[arg-type]
        )

        messages = [
            ChatMessageSystem(content=self._rubric.system),
            ChatMessageUser(content=user_prompt),
        ]

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._config.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
            retry=retry_if_exception_type(
                (_TransientGenerationError, GraderResponseError)
            ),
            reraise=True,
        ):
            with attempt:
                output = await self._resolved_model.generate(
                    messages, config=gen_config
                )
                completion = _completion_text(output)
                if not completion.strip():
                    raise _TransientGenerationError("model returned empty completion")
                # parse_grade_response raises GraderResponseError on
                # malformed output; tenacity catches it and we re-sample.
                response = parse_grade_response(completion)
                return completion, response

        # Unreachable: AsyncRetrying with reraise=True either returns or raises.
        raise RuntimeError("retry loop exited without result")  # pragma: no cover


def _select_consensus_sample(samples: list[GradedStep]) -> GradedStep:
    """Return the sample whose validity matches the modal validity label.

    Tie-breaking (when multiple samples share the modal label): return
    the first such sample in the original sample order. This preserves
    determinism across runs with the same sampled text, which matters
    for the sample-caching layer.
    """
    if not samples:
        raise ValueError("cannot select consensus from empty sample list")
    counts = Counter(s.validity.value for s in samples)
    modal_validity, _ = counts.most_common(1)[0]
    for sample in samples:
        if sample.validity.value == modal_validity:
            return sample
    raise RuntimeError("unreachable: modal validity must appear in sample list")


class _TransientGenerationError(RuntimeError):
    """Internal marker for retry-eligible failures."""


def _completion_text(output: object) -> str:
    """Extract the assistant text from an Inspect ``ModelOutput``.

    Isolated so the LLMGrader body doesn't get coupled to Inspect's exact
    output object shape — only this helper does.
    """
    completion = getattr(output, "completion", None)
    if isinstance(completion, str):
        return completion
    # Fallback: walk choices[0].message.content for older shapes.
    choices = getattr(output, "choices", None) or []
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if isinstance(content, str):
            return content
    raise _TransientGenerationError(
        f"could not extract completion text from model output of type {type(output).__name__}"
    )
