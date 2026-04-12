"""Phase 1 entry point: validate the LLM step grader against TRAIL.

Loads a TRAIL corpus, runs one or more grader configurations against
it, and writes per-grader results plus a side-by-side comparison
report.

Each grader run produces three sibling files in ``--output-dir``:

* ``<run_name>.config.json`` — the :class:`ExperimentConfig` envelope
  (grader spec, dataset slice, package version, git commit, timestamps).
* ``<run_name>.cache.jsonl`` — the :class:`GradedTraceStore` cache,
  one :class:`GradedTrace` per line. Reused on resume; safe to delete
  to force a fresh run.
* ``<run_name>.report.json`` — the :class:`AgreementReport` for the
  run, one entry per rubric dimension.

After all runs complete, ``comparison.json`` summarizes every grader's
per-dimension agreement in one place.

Concrete graders are specified on the command line as ``label=spec``
pairs. The spec supports single-model self-consistency suffixes:

    --grader haiku=anthropic/claude-haiku-4-5
    --grader haiku_sc3=anthropic/claude-haiku-4-5@sc3

Ensembles are composed from labels already defined by ``--grader``
entries, using ``--ensemble``:

    --ensemble trio=haiku,sonnet,gpt4o

The script intentionally does NOT hardcode model identifiers — that
choice belongs to whoever runs the experiment, not to the package.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Allow running directly from a checkout without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from inspect_degradation.datasets.trail import TrailCorpus, load_trail  # noqa: E402
from inspect_degradation.experiment import (  # noqa: E402
    DatasetSlice,
    ExperimentConfig,
)
from inspect_degradation.grader.ensemble import EnsembleGrader  # noqa: E402
from inspect_degradation.grader.interface import Grader  # noqa: E402
from inspect_degradation.grader.llm import LLMGrader, LLMGraderConfig  # noqa: E402
from inspect_degradation.grader.rubric import Rubric  # noqa: E402
from inspect_degradation.store import GradedTraceStore  # noqa: E402
from inspect_degradation.validation import run_validation  # noqa: E402
from inspect_degradation.validation.runner import ValidationResult  # noqa: E402

log = logging.getLogger("validate_grader")


def _parse_grader_spec(spec: str) -> tuple[str, str]:
    """Parse a ``label=model_spec`` argument.

    The model_spec may carry an optional ``@sc<N>`` suffix to enable
    single-model self-consistency sampling: ``claude-haiku@sc3`` means
    three samples at non-zero temperature. The suffix is parsed later
    when the grader is built.
    """
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--grader value must be 'label=model_spec', got {spec!r}"
        )
    label, model = spec.split("=", 1)
    label = label.strip()
    model = model.strip()
    if not label or not model:
        raise argparse.ArgumentTypeError(
            f"--grader label and model must be non-empty, got {spec!r}"
        )
    return label, model


def _parse_ensemble_spec(spec: str) -> tuple[str, list[str]]:
    """Parse a ``label=member1,member2,...`` ensemble argument."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--ensemble value must be 'label=member1,member2,...', got {spec!r}"
        )
    label, members_str = spec.split("=", 1)
    label = label.strip()
    members = [m.strip() for m in members_str.split(",") if m.strip()]
    if not label or not members:
        raise argparse.ArgumentTypeError(
            f"--ensemble must have a label and at least one member, got {spec!r}"
        )
    return label, members


def _split_sc_suffix(model_spec: str) -> tuple[str, int]:
    """Split ``model@scN`` into ``(model, N)``. Returns N=1 if no suffix."""
    if "@sc" not in model_spec:
        return model_spec, 1
    base, suffix = model_spec.rsplit("@sc", 1)
    try:
        n = int(suffix)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"invalid @sc suffix in model spec {model_spec!r}: {e}"
        ) from e
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"@sc suffix must be >= 1, got {n} in {model_spec!r}"
        )
    return base, n


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate the inspect-degradation step grader against TRAIL.",
    )
    p.add_argument(
        "--trail-root",
        type=Path,
        required=True,
        help="Path to the benchmarking/ directory of a local trail-benchmark checkout.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["gaia", "swe_bench"],
        choices=["gaia", "swe_bench"],
        help="TRAIL splits to load.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on traces per split.",
    )
    p.add_argument(
        "--rubric",
        default="step_grader_v1",
        help="Rubric YAML stem (loaded from inspect_degradation/prompts/).",
    )
    p.add_argument(
        "--grader",
        action="append",
        type=_parse_grader_spec,
        required=True,
        metavar="LABEL=MODEL_SPEC",
        help="Grader to evaluate. Repeat to evaluate multiple. "
        "Model spec may carry an '@scN' suffix for single-model "
        "self-consistency (e.g. claude-haiku-4-5@sc3). "
        "Example: --grader fast=openai/gpt-4o-mini",
    )
    p.add_argument(
        "--ensemble",
        action="append",
        type=_parse_ensemble_spec,
        default=[],
        metavar="LABEL=MEMBER1,MEMBER2,...",
        help="Also evaluate an ensemble composed of already-defined "
        "--grader labels. Example: --ensemble trio=haiku,sonnet,gpt4o",
    )
    p.add_argument(
        "--sample-temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for graders with @scN suffix. "
        "Must be > 0. Graders without the suffix run at temperature 0.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write per-run config/cache/report files.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Per-grader maximum concurrent grading calls.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _build_llm_grader(
    *,
    model_spec: str,
    rubric: Rubric,
    max_concurrency: int,
    sample_temperature: float,
) -> LLMGrader:
    """Build an :class:`LLMGrader` from a ``model[@scN]`` spec."""
    base_model, sample_n = _split_sc_suffix(model_spec)
    temperature = sample_temperature if sample_n > 1 else 0.0
    return LLMGrader(
        config=LLMGraderConfig(
            model=base_model,
            max_concurrency=max_concurrency,
            sample_n=sample_n,
            temperature=temperature,
        ),
        rubric=rubric,
    )


async def _run_one(
    *,
    name: str,
    grader: Grader,
    corpus: TrailCorpus,
    dataset: DatasetSlice,
    output_dir: Path,
) -> ValidationResult:
    cache_path = output_dir / f"{name}.cache.jsonl"
    config_path = output_dir / f"{name}.config.json"
    report_path = output_dir / f"{name}.report.json"

    config = ExperimentConfig.from_grader(
        name=name,
        grader=grader,
        dataset=dataset,
    )
    config.write_json(config_path)

    cache = GradedTraceStore(cache_path)
    log.info(
        "running grader %s (%d cached traces in %s)",
        name,
        len(cache),
        cache_path,
    )

    result = await run_validation(
        grader=grader,
        traces=corpus.traces,
        reference=corpus.reference,
        cache=cache,
    )

    report_path.write_text(
        json.dumps(result.report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    log.info(
        "  done: %d cached, %d freshly graded, n_pairs=%d",
        result.n_from_cache,
        result.n_freshly_graded,
        result.report.n_pairs,
    )
    return result


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading TRAIL from %s", args.trail_root)
    corpus = load_trail(
        args.trail_root,
        splits=tuple(args.splits),
        limit=args.limit,
    )
    log.info(
        "loaded %d traces (%d total reference grades)",
        len(corpus.traces),
        sum(len(g.steps) for g in corpus.reference),
    )

    rubric = Rubric.from_package(args.rubric)
    dataset = DatasetSlice(
        name="trail",
        path=str(args.trail_root),
        splits=tuple(args.splits),
        limit=args.limit,
    )

    # Build the standalone LLM graders. Each label gets its own
    # independent LLMGrader instance so ensembles that reference the
    # label get a distinct object (duplicate-instance check in
    # EnsembleGrader would otherwise fire if we reused the same object).
    grader_specs: list[tuple[str, str]] = list(args.grader)
    llm_specs_by_label: dict[str, str] = {}
    graders: dict[str, Grader] = {}
    for label, model_spec in grader_specs:
        if label in graders:
            log.error("duplicate grader label %r; aborting", label)
            return 2
        llm_specs_by_label[label] = model_spec
        graders[label] = _build_llm_grader(
            model_spec=model_spec,
            rubric=rubric,
            max_concurrency=args.max_concurrency,
            sample_temperature=args.sample_temperature,
        )

    # Build any ensembles from the labeled graders. Each ensemble gets
    # freshly-built member instances so a single LLMGrader instance is
    # never shared between a standalone run and an ensemble run — that
    # keeps state (lazy model handle, counters) from leaking across
    # runs, and lets the ensemble's duplicate-instance invariant catch
    # genuine misconfigurations.
    for ensemble_label, member_labels in args.ensemble:
        if ensemble_label in graders:
            log.error("ensemble label %r collides with a grader label; aborting", ensemble_label)
            return 2
        unknown = [m for m in member_labels if m not in llm_specs_by_label]
        if unknown:
            log.error(
                "ensemble %r references unknown grader labels: %s (known: %s)",
                ensemble_label,
                unknown,
                sorted(llm_specs_by_label),
            )
            return 2
        members = [
            _build_llm_grader(
                model_spec=llm_specs_by_label[m],
                rubric=rubric,
                max_concurrency=args.max_concurrency,
                sample_temperature=args.sample_temperature,
            )
            for m in member_labels
        ]
        graders[ensemble_label] = EnsembleGrader(members, name=f"ensemble:{ensemble_label}")

    # Run all grader configs in parallel. Each grader has its own
    # instance-level semaphore capping its in-flight API calls, so
    # concurrent graders compete for provider bandwidth independently.
    # The cache files are per-grader (no sharing), so there is no
    # data dependency between runs.
    async def _run_named(name: str, g: Grader) -> tuple[str, ValidationResult]:
        result = await _run_one(
            name=name,
            grader=g,
            corpus=corpus,
            dataset=dataset,
            output_dir=args.output_dir,
        )
        return name, result

    completed = await asyncio.gather(
        *(_run_named(name, grader) for name, grader in graders.items())
    )
    results: dict[str, ValidationResult] = dict(completed)

    # Write the side-by-side comparison.
    comparison_path = args.output_dir / "comparison.json"
    comparison = {
        "n_traces": len(corpus.traces),
        "graders": {
            name: result.report.to_dict() for name, result in results.items()
        },
    }
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    _print_comparison_table(results)
    return 0


def _print_comparison_table(results: dict[str, ValidationResult]) -> None:
    if not results:
        return
    dimensions = list(next(iter(results.values())).report.per_dimension.keys())
    name_w = max(len(n) for n in results) + 2
    dim_w = max(max(len(d) for d in dimensions), 32) + 2

    header = "grader".ljust(name_w) + "".join(d.ljust(dim_w) for d in dimensions)
    print()
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        cells: list[str] = [name.ljust(name_w)]
        for d in dimensions:
            agreement = result.report.per_dimension[d]
            est = agreement.estimate
            cells.append(
                f"{est.value:+.3f} [{est.ci_low:+.2f},{est.ci_high:+.2f}] (n={agreement.n_pairs})".ljust(dim_w)
            )
        print("".join(cells))
    print()


def main() -> None:
    args = _parse_args()
    sys.exit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
