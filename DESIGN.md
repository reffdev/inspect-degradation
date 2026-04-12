# inspect-degradation -- Design Document

*Architecture and design rationale. For results, see the [degradation study](https://github.com/reffdev/inspect-degradation-study). For usage, see [README.md](README.md).*

A Python package that extends [Inspect AI](https://inspect.aisi.org.uk/) with **within-run degradation analysis** for AI agents: does the agent get worse over the course of a task, and if so, *why*?

## The problem

Current agent benchmarks measure final outcomes -- did the task succeed. They don't measure what happens *during* a task. An agent that makes one early mistake and then flawlessly executes the wrong plan for 20 steps looks identical to an agent that independently fails every step. These are very different failure modes requiring very different fixes, and no existing tooling distinguishes them.

The package measures each step of a trace along a small set of dimensions, then applies established statistical methods to decompose degradation from confounds (task complexity, cascading errors, model identity).

## Per-step measurements

Each step is graded by an LLM along five dimensions. Every dimension is a closed-set categorical -- no floats, no free text -- because structured classification is what LLM graders are actually good at.

- **Validity**: Does the step push the task forward? `fail` / `neutral` / `pass`. `neutral` specifically covers wheel-spinning and idle exploration: not wrong, but not advancing. Distinguishing neutral from fail is what lets the analysis separate "agent is breaking" from "agent is flailing."
- **Complexity**: How clear was the correct next move *before* the agent took this step? `low` / `medium` / `high`. A control variable for the confound that later steps may simply be harder than earlier ones.
- **Dependency**: When a step fails, is the failure independent or a consequence of a prior failure? `independent` / `dependent` / `n/a`. Enables cascade-chain analysis.
- **Severity**: For failing steps, how consequential? `low` / `medium` / `high`, anchored to recovery cost.
- **Looping** (`is_looping`): Occurrence-based boolean -- is this step at least the *second* occurrence of a substantially identical action or reasoning pattern within the prior steps? Mechanical rather than impressionistic: the grader must be able to point to the earlier step.

Every measurement is answerable from the current step plus prior context alone. Nothing requires the grader to peek at future steps or to do the kind of multi-step reasoning the agent is itself being evaluated on.

## Statistical analysis

The step-level data feeds into established techniques applied to agent evaluation:

- **Survival / hazard analysis** -- Kaplan-Meier curves and Cox proportional hazards for time-to-first-error.
- **Change-point detection** -- identifying the step where agent behavior shifts.
- **Cascade-chain analysis** -- distribution of contiguous-dependent-failure chain lengths; mean failing-run length (the honest reformulation of "steps to recovery").
- **Loop-rate slope** -- distinguishes "more errors late" from "more flailing late" from "stuck in a cycle late" as three separate degradation signatures.
- **Mixed-effects models** -- decomposing error rate into contributions from step position, step complexity, task identity, and model, to isolate true degradation from confounds.

All statistical claims are reported with confidence intervals and effect sizes, not just p-values.

## Grader architecture

**Ensemble grading.** Each step is judged by N independent graders, and the ensemble reduces them to a single verdict by majority vote on validity. The intent is heterogeneous ensembles -- Haiku + Sonnet + GPT-4o-mini + Gemini Flash, etc. -- because errors across model families are less correlated than errors within a single family. Single-model self-consistency (same model, N samples at non-zero temperature) is available as a primitive, and single-member ensembles are the identity, so every configuration from baseline to production runs through the same code path.

**Why ensemble, not cascade.** A cheap→expensive cascade assumes the cheap grader's uncertainty signal reliably flags the cases the expensive grader would get wrong. That assumption is not established for LLM-judge settings; Kapoor et al. 2024 specifically tested it and found cheap-judge confidence unreliable as an escalation gate. The ensemble doesn't require the assumption -- every step gets every grader's verdict, so a single grader's error is bounded by the agreement of the others. Self-consistency across samples is the well-supported signal (Wang et al. 2022, Kadavath et al. 2022); verbalized self-confidence is not (Xiong et al. 2023), which is why no `confidence` field exists on the rubric.

**Provider neutrality.** The grader layer goes through Inspect AI's `get_model`, so provider routing is a deployment concern, not a package concern. Direct per-provider keys, unified gateways (OpenRouter, LiteLLM), or local inference servers all work through the same configuration -- see [README.md](README.md#provider-routing).

**Reproducibility.** Every validation run writes an `ExperimentConfig` envelope recording the grader snapshot (model, rubric version, ensemble shape), the dataset slice, package version, git commit, and timestamps. Re-running an experiment from a config file is the baseline guarantee.

## Included dataset loaders

- **TRAIL** -- [patronus-ai/trail-benchmark](https://github.com/patronus-ai/trail-benchmark): 148 expert-annotated traces from GAIA and SWE-bench with step-level error labels. Ground truth for grader validation.
- **SWE-smith trajectories** -- [SWE-bench/SWE-smith-trajectories](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories): 5,017 agent trajectories.
- **Nebius SWE-agent trajectories** -- [nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories): 80,036 trajectories across multiple models.
- **Multi-SWE-bench** -- ByteDance's multi-model, multi-scaffolding trajectories.
- **OpenHands** -- OpenHands/SWE-Gym sampled trajectories.
- **terminus** -- terminus-2 agent trajectories.
- **Auto-SWE** -- Custom multi-agent pipeline traces loaded from SQLite or JSONL.

## Layer separation

Three layers, independently usable:

1. **Inspect AI integration** -- `@scorer` and `@metric` implementations that plug into Inspect's eval pipeline, for users running the grader alongside a live eval.
2. **Grader** -- `LLMGrader`, `EnsembleGrader`, rubric loader, response parser. Usable against any source of traces, not just Inspect runs.
3. **Analysis** -- survival, change-point, cascade-chain, loop, and regression analyses that operate on graded traces regardless of where they came from.

The grader depends on `inspect_ai.model` for its model API, so `inspect_ai` is a hard dependency. What's decoupled is the rest of Inspect's surface (tasks, solvers, evals): you can grade pre-existing traces and run the analysis without ever constructing an Inspect task.

## References

- Inspect AI: https://inspect.aisi.org.uk/ · [GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)
- Anthropic, *Demystifying Evals*: https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- Wang et al. 2022, *Self-Consistency Improves Chain of Thought Reasoning in Language Models*: https://arxiv.org/abs/2203.11171
- Kadavath et al. 2022, *Language Models (Mostly) Know What They Know*: https://arxiv.org/abs/2207.05221
- Xiong et al. 2023, *Can LLMs Express Their Uncertainty?*: https://arxiv.org/abs/2306.13063
- Kapoor et al. 2024, *On Scalable Oversight with Weak LLMs Judging Strong Models*: https://arxiv.org/abs/2407.04622

