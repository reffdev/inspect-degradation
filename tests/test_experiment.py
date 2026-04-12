"""Tests for the experiment-config snapshot type.

These tests use the real ``LLMGrader`` and ``EnsembleGrader``
constructors, not mocks, because the snapshot logic lives on the
grader subclasses themselves — that's the whole point of the design.
We never call the underlying model, so no API key is needed.
"""

import json

from inspect_degradation.experiment import DatasetSlice, ExperimentConfig
from inspect_degradation.grader.ensemble import EnsembleGrader
from inspect_degradation.grader.llm import LLMGrader, LLMGraderConfig
from inspect_degradation.grader.rubric import Rubric


def _llm_grader(model: str = "test/model") -> LLMGrader:
    return LLMGrader(
        config=LLMGraderConfig(model=model, max_concurrency=2),
        rubric=Rubric.from_package("step_grader_v1"),
    )


def test_llm_grader_snapshot_captures_config():
    grader = _llm_grader("openai/gpt-4o-mini")
    snap = grader.snapshot()
    assert snap.kind == "llm"
    assert snap.fields["model"] == "openai/gpt-4o-mini"
    assert snap.fields["rubric_name"] == "step_grader_v1"
    assert snap.fields["rubric_version"] == 1
    assert snap.fields["sample_n"] == 1


def test_llm_grader_snapshot_reflects_sample_n():
    grader = LLMGrader(
        config=LLMGraderConfig(
            model="anthropic/claude-haiku-4-5",
            sample_n=3,
            temperature=0.7,
        ),
        rubric=Rubric.from_package("step_grader_v1"),
    )
    snap = grader.snapshot()
    assert snap.fields["sample_n"] == 3
    assert snap.fields["temperature"] == 0.7


def test_ensemble_snapshot_nests_member_snapshots():
    member_a = _llm_grader("haiku")
    member_b = _llm_grader("sonnet")
    member_c = _llm_grader("gpt-4o-mini")
    ensemble = EnsembleGrader([member_a, member_b, member_c])
    snap = ensemble.snapshot()
    assert snap.kind == "ensemble"
    assert snap.fields["n_members"] == 3
    assert set(snap.children) == {"member_0", "member_1", "member_2"}
    assert snap.children["member_0"].fields["model"] == "haiku"
    assert snap.children["member_1"].fields["model"] == "sonnet"
    assert snap.children["member_2"].fields["model"] == "gpt-4o-mini"


def test_experiment_config_round_trips_llm_grader_json(tmp_path):
    grader = _llm_grader("test/model")
    cfg = ExperimentConfig.from_grader(
        name="phase1-smoke",
        grader=grader,
        dataset=DatasetSlice(
            name="trail",
            path="/some/path",
            splits=("gaia",),
            limit=10,
        ),
        seed=42,
        notes="smoke test",
    )
    out = tmp_path / "config.json"
    cfg.write_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["name"] == "phase1-smoke"
    assert data["dataset"]["splits"] == ["gaia"]
    assert data["dataset"]["limit"] == 10
    assert data["grader"]["kind"] == "llm"
    assert data["grader"]["fields"]["model"] == "test/model"
    assert data["seed"] == 42
    assert data["package_version"]
    assert data["python_version"]
    assert data["created_at"]


def test_experiment_config_round_trips_ensemble_json(tmp_path):
    # Snapshot of an ensemble must include every member's config so the
    # full grader shape is auditable from the on-disk envelope alone.
    ensemble = EnsembleGrader(
        [_llm_grader("haiku"), _llm_grader("sonnet")],
        name="phase1-ensemble",
    )
    cfg = ExperimentConfig.from_grader(
        name="phase1-ensemble",
        grader=ensemble,
        dataset=DatasetSlice(name="trail", path="/p", splits=("gaia",)),
    )
    out = tmp_path / "ens.json"
    cfg.write_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["grader"]["kind"] == "ensemble"
    assert data["grader"]["fields"]["n_members"] == 2
    assert data["grader"]["children"]["member_0"]["fields"]["model"] == "haiku"
    assert data["grader"]["children"]["member_1"]["fields"]["model"] == "sonnet"
