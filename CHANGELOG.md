# Changelog

All notable changes to `inspect-degradation` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-14

Initial release.

### Added

- **Step-level grading** - LLM-as-judge rubric over five closed-set categorical
  dimensions (validity, complexity, dependency, severity, looping). Rubric lives
  in `prompts/step_grader_v1.yaml` for code-free iteration.
- **Inspect AI integration** - `degradation_scorer()` and a family of
  `@metric`-decorated factories (`error_rate`, `error_rate_slope`,
  `productive_rate`, `neutral_rate`, `loop_rate`, `cascade_chain_length_mean`,
  `first_error_step_median`, ...). Discovered by Inspect via the `inspect_ai`
  entry-point group.
- **Statistical analysis pipeline** with `Estimate` + bootstrap CIs:
  mixed-effects regression with outcome and step-phase controls, Kaplan-Meier
  and Cox PH survival fits, cascade-chain analysis, contiguous-loop analysis,
  change-point detection, autocorrelation diagnostics, FDR correction, and
  Monte Carlo power analysis.
- **SIMEX measurement-error correction** - uses the grader's empirical flip
  rate (measured against TRAIL) to correct regression coefficients for
  label noise.
- **Grader validation** - `validation/runner.py` and `scripts/validate_grader.py`
  for grading a labeled corpus and reporting per-dimension agreement,
  confusion matrices, and IRR. Built-in TRAIL loader.
- **Dataset loaders** - TRAIL, Nebius, SWE-smith, OpenHands, Multi-SWE-bench,
  terminus, Auto-SWE.
- **CLI entry points** - `python -m inspect_degradation.analysis.from_eval_log`
  to run the full analysis on an Inspect eval log; `scripts/power_analysis.py`
  for a Monte Carlo power table.
- **410+ tests** including ground-truth recovery for SIMEX, mixed-effects, and
  Cox PH, plus invariance falsification tests for the grader.

[Unreleased]: https://github.com/reffdev/inspect-degradation/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/reffdev/inspect-degradation/releases/tag/v0.1.0
