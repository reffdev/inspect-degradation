# inspect-degradation

[![PyPI](https://img.shields.io/pypi/v/inspect-degradation)](https://pypi.org/project/inspect-degradation/)
[![Python](https://img.shields.io/pypi/pyversions/inspect-degradation)](https://pypi.org/project/inspect-degradation/)
[![Tests](https://github.com/reffdev/inspect-degradation/actions/workflows/test.yml/badge.svg)](https://github.com/reffdev/inspect-degradation/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Within-run degradation analysis for AI agents, built as an [Inspect AI](https://inspect.aisi.org.uk/) extension.

Does the agent get worse as it works longer? Agent benchmarks measure final outcomes -- did the task succeed -- but not what happens *during* a task. An agent that makes one early mistake and flawlessly executes the wrong plan for 20 steps looks identical to one that independently fails every step. These are different failure modes, and no existing tooling distinguishes them.

This package measures each step of an agent trace along structured dimensions, validates the grader against human labels, corrects downstream statistics for grader noise, and decomposes degradation from confounds (task complexity, phase composition, cascading errors, model identity). The pipeline is designed around a specific lesson: naive step-level regressions produce convincing artifacts. The [companion study](https://github.com/reffdev/inspect-degradation-study) ran 15 configurations in total; of the 14 graded under full context (one was dropped after a context-cap pilot was found to introduce position-correlated bias), 5 show degradation (3 surviving Benjamini-Hochberg correction), 2 show improvement, and 7 are null. Phase composition (agents shift from exploration to action over a task) attenuates the raw slope by roughly half but does not drive it to zero. The study also characterizes two properties of the grading instrument - length-dependent accuracy and a construct mismatch with hindsight-informed human labels - that affect any LLM-as-judge temporal analysis.

For full results (8+ models, 4 scaffoldings, ~24,000 graded steps), see [inspect-degradation-study](https://github.com/reffdev/inspect-degradation-study).

## How it works

### Step-level grading

Each step is graded by an LLM along five dimensions using a structured rubric:

| Dimension | Values | Purpose |
|---|---|---|
| **Validity** | fail / neutral / pass | Did the step push the task forward? |
| **Complexity** | low / medium / high | How clear was the correct move? (confound control) |
| **Dependency** | independent / dependent / n/a | Is a failure caused by a prior failure? (cascade analysis) |
| **Severity** | low / medium / high | How consequential is a failing step? |
| **Looping** | true / false | Is this step a repeat of a prior action? |

Every dimension is a closed-set categorical -- no floats, no free text, no confidence scores. The rubric is designed around two constraints: LLM graders are good at structured classification but bad at verbalized self-confidence (Xiong et al. 2023), and surface plausibility over-credits stuck agents, so "pass" requires *positive evidence of progress* rather than absence of visible error.

### Statistical analysis

The step-level grades feed into established methods, each returning an `Estimate` with confidence intervals:

| Analysis | What it answers |
|---|---|
| **Mixed-effects regression** | Does degradation survive after controlling for task complexity, model identity, and eventual outcome? |
| **Survival analysis** | Kaplan-Meier and Cox PH for time-to-first-error |
| **Cascade-chain analysis** | Are errors independent or do they cascade? |
| **Loop analysis** | How long are contiguous repeated-action runs? |
| **SIMEX correction** | How much does grader noise bias the degradation slope? |
| **Change-point detection** | Is there a step where behavior shifts abruptly? |
| **Autocorrelation diagnostics** | Are successive errors correlated? (assumption check) |
| **Power analysis** | Is the corpus large enough to detect a given effect size? |
| **FDR correction** | Multiple-comparison adjustment on coefficient families |

All results carry confidence intervals computed via trace-level bootstrap, which correctly accounts for within-trace correlation.

### Grader validation

Built-in validation against [TRAIL benchmark](https://github.com/patronus-ai/trail-benchmark) (148 expert-annotated traces). SIMEX noise correction uses the measured flip rate from validation to correct regression coefficients for label noise.

## Install

```bash
pip install inspect-degradation
```

For development or to use dataset loaders (TRAIL, Nebius, SWE-smith, etc.):

```bash
pip install inspect-degradation[datasets]

# Or from source:
git clone https://github.com/reffdev/inspect-degradation.git
pip install -e ".[dev,datasets]"
```

## Quick start

### Add degradation analysis to an Inspect eval

```python
from inspect_ai import eval, task, Task
from inspect_ai.solver import generate
from inspect_degradation.integration import (
    degradation_scorer,
    error_rate,
    error_rate_slope,
    neutral_rate,
)

@task
def my_agent_task():
    return Task(
        dataset=my_dataset,
        solver=my_agent_solver(),
        scorer=[
            my_correctness_scorer(),       # your primary scorer
            degradation_scorer(),           # adds per-step grading
        ],
    )

# error_rate, productive_rate, and error_rate_slope are
# included by default. Add more with the metrics parameter:
results = eval(my_agent_task(), metrics=[neutral_rate()])
```

The scorer grades each agent step along five dimensions (validity, complexity, dependency, severity, looping), stashes the results in `Score.metadata`, and reports the default metrics in the eval viewer. If another scorer runs first, the degradation scorer reads its result to control for task success in downstream analysis.

### Analyze an eval log

After running an eval with `degradation_scorer()`, extract the graded traces and run the full analysis pipeline (mixed-effects regression, survival analysis, cascade chains, autocorrelation):

```bash
python -m inspect_degradation.analysis.from_eval_log path/to/eval.log
```

Or from Python:

```python
from inspect_degradation.analysis.from_eval_log import analyze_eval_log

report = analyze_eval_log("path/to/eval.log")

# Use --glmm for low error-rate models (< 10%):
report = analyze_eval_log("path/to/eval.log", use_glmm=True)
```

### Validate a grader against TRAIL

```bash
git clone https://github.com/patronus-ai/trail-benchmark.git

python scripts/validate_grader.py \
  --trail-root ./trail-benchmark/benchmarking \
  --grader minimax=openai/minimax/minimax-m2.5 \
  --output-dir results/phase1
```

### Run a power analysis

```bash
python scripts/power_analysis.py --n-traces 50 --steps-per-trace 20
```

## Provider routing

Provider-agnostic via Inspect AI's `get_model`. Any OpenAI-compatible endpoint works - direct provider APIs, unified gateways (OpenRouter, LiteLLM proxy, Portkey), or local inference servers - by setting the standard `OPENAI_API_KEY` / `OPENAI_BASE_URL` pair before invoking the grader. There is no provider-specific code in this package; routing is a deployment concern.

```bash
# Direct
export ANTHROPIC_API_KEY=sk-ant-...
--grader haiku=anthropic/claude-haiku-4-5

# OpenRouter
export OPENAI_API_KEY=sk-or-v1-...
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
--grader haiku=openai/anthropic/claude-haiku-4.5

# Local
export OPENAI_BASE_URL=http://localhost:8000/v1
--grader local=openai/meta-llama/Llama-3.3-70B-Instruct
```

## Design decisions

**Outcome stratification.** The canonical formula includes `C(trace_success)` to partial out survivorship bias -- failed traces are systematically longer, so "errors increase with step index" partly reflects "long traces are the failed ones."

**Step-phase control.** A computed column classifies each step as exploration or action from the agent's command text, using framework-aware detection layers (Auto-SWE tool calls, OpenHands subcommands, SWE-agent XML blocks, shell command fallback). This covariate is critical -- without it, the natural shift from exploration to action over a task mimics degradation.

**Grader noise is first-class.** Validation measures exactly how wrong the grader is. SIMEX corrects regression coefficients for label noise. Confusion-matrix deconfounding corrects rate estimates.

**Trace-level resampling everywhere.** Steps within a trace are not independent. All bootstrap CIs resample whole traces -- the correct unit for nested data.

**No confidence scores.** LLM verbalized confidence is unreliable (Xiong et al. 2023). The rubric uses only closed-set categoricals. Uncertainty is measured empirically via validation, not asked for.

**Ensemble over cascade.** A cheap-then-expensive cascade assumes cheap-judge uncertainty reliably flags hard cases. Kapoor et al. 2024 found it doesn't. The ensemble gives every step every grader's verdict, bounding any single grader's error by agreement.

## Project layout

```
src/inspect_degradation/
  grader/          # LLM-as-judge: rubric, per-step grading, ensemble, drift canary
  analysis/        # Mixed-effects, survival, SIMEX, cascade, loops, change-point, power
  validation/      # Grader-vs-human agreement, invariance falsification tests
  datasets/        # Loaders for TRAIL, Nebius, SWE-smith, OpenHands, Multi-SWE-bench, terminus, Auto-SWE
  integration/     # @scorer / @metric for Inspect AI
  prompts/         # Rubric YAML (code-free iteration)
  schema.py        # Pydantic models with cross-field invariants
data/
  trail_sample/    # TRAIL test fixtures for offline tests
tests/             # 400+ tests including ground-truth recovery for SIMEX, mixed-effects, Cox
scripts/
  validate_grader.py       # Grader validation entry point
  power_analysis.py        # Monte Carlo power table
```

## Related work

**Agent benchmarks** ([SWE-bench](https://www.swebench.com/), [GAIA](https://huggingface.co/gaia-benchmark), [WebArena](https://webarena.dev/)) measure final task outcomes. They answer "did the agent succeed?" but not "did the agent get worse over time?" or "why did it fail at step 15?" This tool operates on the within-run trajectory that those benchmarks discard.

**LLM-as-judge frameworks** ([MT-Bench](https://arxiv.org/abs/2306.05685), [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval), [Chatbot Arena](https://arena.lmsys.org/)) evaluate single responses or conversations. They don't handle multi-step agent traces where each step depends on prior context, and they don't correct for the measurement error their own judges introduce. This tool's SIMEX correction and confusion-matrix deconfounding treat grader noise as a first-class statistical concern rather than assuming the judge is ground truth.

**TRAIL** ([Patronus AI](https://github.com/patronus-ai/trail-benchmark)) provides expert-annotated step-level labels for agent traces -- the closest prior work. TRAIL is a benchmark (labels + evaluation), not a measurement pipeline: it reports full-trace accuracy, not temporal error trajectories, and doesn't include the statistical machinery for decomposing degradation from confounds. This tool uses TRAIL as validation ground truth and extends it with temporal analysis.

**Anthropic's "Demystifying Evals"** (2025) discusses per-step evaluation methodology for agents but does not provide reusable tooling or report degradation measurements. The step-phase confound this tool controls for -- agents shifting from exploration to action over a task, mimicking degradation -- is not addressed in that work.

**Long-context retrieval** ([Liu et al. 2024](https://arxiv.org/abs/2307.03172), needle-in-a-haystack) measures how reliably models retrieve planted facts from long prompts. These are real effects, but they test a different capability: retrieving information from context is not the same as generating a correct next action given a history of prior actions. This tool measures the latter.

## Limitations

A few things to keep in mind when interpreting results.

**"Step validity" is not a single construct, and true validity is undecidable from trace data alone.** The grader judges each step using only the agent's prior context (no hindsight), which measures one interpretation of validity: was this a reasonable decision given available information? But whether a step was *actually correct* - whether that edit introduced a subtle bug, whether that API call returned stale data, whether the agent's mental model of the codebase matched reality - requires the full codebase state, the test suite, and the eventual outcome. That information is not in current trace formats, and no public benchmark the author is aware of exports it alongside step data. An edit that compiles and looks correct at step 3 but turns out wrong when a test fails at step 60 is "valid" under the grader's interpretation and "invalid" under an equally defensible outcome-contribution interpretation. These are different constructs, not a correct one and a wrong one, and no amount of improved grading can collapse them. The [companion study](https://github.com/reffdev/inspect-degradation-study) finds this construct ambiguity is the dominant source of grader-vs-human disagreement, ahead of conventional grading errors. "Validity" as reported by this tool means "appeared reasonable in context," not "was actually correct in the real-world scenario the agent ran in."

**Grader validation corpus is small.** The built-in TRAIL integration provides 148 expert-annotated traces. That is enough to estimate per-dimension grader accuracy with usable but not narrow CIs, and it is the largest expert-labeled set the package can plausibly ship. Downstream noise corrections (SIMEX, confusion-matrix deconfounding) inherit the uncertainty of the validation estimates; reporting an upper-bound disagreement rate alongside any temporal claim is the load-bearing move.

**SIMEX assumes a uniform flip rate.** The label-noise correction estimates a single per-dimension flip probability and treats it as constant across step position, complexity, and trace length. If grader noise is itself position-dependent - e.g. accuracy degrades on long traces, which the companion study finds - SIMEX will under- or over-correct in directions that depend on the unmeasured covariate. Treat SIMEX-corrected slopes as a sensitivity check, not a debiased point estimate.

**Step-phase classification is heuristic.** The exploration-vs-action covariate that controls the dominant confound is computed from agent commands using a layered detector (Auto-SWE tool calls, OpenHands subcommands, SWE-agent XML blocks, shell-command fallback). It works well on the scaffoldings the package has loaders for; it has not been validated on novel agent frameworks and will silently mislabel them. Adding a new framework means writing a step-phase rule for it.

**Grader cost scales linearly.** Grading 1,000 traces of 30 steps each is 30,000 model calls per grader. An ensemble of three is 90,000. Self-consistency at sample_n=3 is another 3×. Budget accordingly; the resumable cache makes interruptions cheap to recover from but does not change total cost.

**Rubric is calibrated against TRAIL-style errors.** The five-dimension closed-set rubric was tuned to match the kinds of mistakes TRAIL annotators flag (hallucinations, wrong tool calls, looping, wrong-task drift). It will probably underperform on novel error modes - e.g. specification-gaming, deceptive-step patterns, or domain-specific failures the rubric has no language for. The rubric is a YAML file in `prompts/`; iterating it against a new error taxonomy is the supported extension path.

**Change-point detection's autocorrelation adjustment is heuristic.** When the optional autocorrelation adjustment is enabled, the BIC penalty is doubled if a Ljung-Box test rejects white noise. The 2× factor is a coarse approximation, not a derived constant; results sensitive to the exact change-point list should be cross-validated on a hold-out segment.

## References

- Inspect AI: [inspect.aisi.org.uk](https://inspect.aisi.org.uk/) / [GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)
- TRAIL benchmark: [patronus-ai/trail-benchmark](https://github.com/patronus-ai/trail-benchmark)
- Xiong et al. 2023, [Can LLMs Express Their Uncertainty?](https://arxiv.org/abs/2306.13063)
- Kapoor et al. 2024, [On Scalable Oversight with Weak LLMs Judging Strong Models](https://arxiv.org/abs/2407.04622)
- Cook & Stefanski 1994, [Simulation-Extrapolation Estimation](https://doi.org/10.1080/01621459.1994.10476871) (SIMEX)
- Liu et al. 2024, [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- Zheng et al. 2023, [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
