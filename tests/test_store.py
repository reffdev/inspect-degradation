"""Tests for the JSONL graded-trace store."""

from conftest import make_graded_step, make_graded_trace

from inspect_degradation.store import GradedTraceStore


def test_append_and_iterate(tmp_path):
    store = GradedTraceStore(tmp_path / "out.jsonl")
    a = make_graded_trace(trace_id="a")
    b = make_graded_trace(trace_id="b")
    store.append(a)
    store.append(b)

    loaded = list(store)
    assert [t.trace_id for t in loaded] == ["a", "b"]
    assert len(store) == 2


def test_completed_trace_ids_skips_already_done(tmp_path):
    store = GradedTraceStore(tmp_path / "out.jsonl")
    store.append(make_graded_trace(trace_id="x"))
    store.append(make_graded_trace(trace_id="y"))
    assert store.completed_trace_ids() == {"x", "y"}


def test_skips_corrupt_lines(tmp_path):
    path = tmp_path / "out.jsonl"
    store = GradedTraceStore(path)
    store.append(make_graded_trace(trace_id="good1"))
    # Manually corrupt the file mid-stream.
    with path.open("a", encoding="utf-8") as f:
        f.write("not-json\n")
        f.write("{also not valid}\n")
    store.append(make_graded_trace(trace_id="good2"))

    loaded = list(store)
    assert [t.trace_id for t in loaded] == ["good1", "good2"]


def test_resume_preserves_existing_records_across_instances(tmp_path):
    path = tmp_path / "out.jsonl"
    s1 = GradedTraceStore(path)
    s1.append(make_graded_trace(trace_id="a"))
    # Re-open: nothing should be truncated.
    s2 = GradedTraceStore(path)
    assert s2.completed_trace_ids() == {"a"}
    s2.append(make_graded_trace(trace_id="b"))
    assert {t.trace_id for t in s2} == {"a", "b"}


def test_empty_store_iterates_to_empty(tmp_path):
    store = GradedTraceStore(tmp_path / "fresh.jsonl")
    assert list(store) == []
    assert store.completed_trace_ids() == set()
    assert len(store) == 0


def test_appending_trace_with_newlines_in_metadata_is_safe(tmp_path):
    # GradedTrace.model_dump_json escapes newlines inside JSON string
    # values, so even free-form text in metadata round-trips safely
    # without breaking JSONL line boundaries.
    from inspect_degradation.schema import GradedTrace

    store = GradedTraceStore(tmp_path / "n.jsonl")
    trace = GradedTrace(
        trace_id="nl",
        steps=[make_graded_step(0)],
        metadata={"note": "line1\nline2\nline3"},
    )
    store.append(trace)
    loaded = list(store)
    assert loaded[0].metadata["note"] == "line1\nline2\nline3"
    # And the on-disk file is still exactly one record (no stray newlines).
    assert len(store) == 1
