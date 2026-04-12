# inspect-degradation

Within-run degradation analysis for AI agents, built as an [Inspect AI](https://inspect.aisi.org.uk/) extension.

Does the agent get worse as it works longer? This package provides a statistical pipeline to measure it: per-step LLM grading with a structured rubric, grader validation against human labels, noise correction, and a full analysis battery. Instead of assuming degradation exists (or doesn't), you can measure it on your own traces.

For a study that used this tool across 15 configurations and found no robust degradation, see [inspect-degradation-study](https://github.com/reffdev/inspect-degradation-study).

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

Provider-agnostic via Inspect AI's `get_model`. Works with direct API keys, OpenRouter, LiteLLM, Portkey, or any OpenAI-compatible endpoint:

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

## References

- Inspect AI: [inspect.aisi.org.uk](https://inspect.aisi.org.uk/) / [GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)
- TRAIL benchmark: [patronus-ai/trail-benchmark](https://github.com/patronus-ai/trail-benchmark)
- Xiong et al. 2023, [Can LLMs Express Their Uncertainty?](https://arxiv.org/abs/2306.13063)
- Kapoor et al. 2024, [On Scalable Oversight with Weak LLMs Judging Strong Models](https://arxiv.org/abs/2407.04622)
- Cook & Stefanski 1994, [Simulation-Extrapolation Estimation](https://doi.org/10.1080/01621459.1994.10476871) (SIMEX)
