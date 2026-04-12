"""Auto-SWE custom agent framework loader.

Loads agent traces from a local SQLite database produced by a custom
multi-stage agent framework. The framework uses a pipeline of stages
(scout, implement, test_write, review) where each stage is a
separate "run" with its own sequence of LLM calls.

The natural unit for degradation analysis is the **run** (one stage
execution), not the issue (which spans multiple stages and retries).
Each run becomes one Trace with its LLM calls as steps.

Supports three levels of granularity:
- **run-level** (default): each pipeline stage execution = one trace
- **issue-level**: all LLM calls for an issue = one trace (very long)
- **foreman-level**: each foreman task = one trace

Also supports filtering by issue type (worker, foreman, director,
verifier), pipeline stage, and model.
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from pathlib import Path
from typing import Any

from inspect_degradation.trace import Trace, TraceStep

log = logging.getLogger(__name__)

_MAX_MESSAGE_CHARS = 8000


def _truncate(text: str, limit: int = _MAX_MESSAGE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 50] + f"\n\n[... truncated, {len(text) - limit + 50} chars omitted ...]"


def _classify_issue_type(issue_id: str | None) -> str:
    if issue_id is None:
        return "unassigned"
    if issue_id.startswith("director-planner:"):
        return "director-planner"
    if issue_id.startswith("director:"):
        return "director"
    if issue_id.startswith("foreman:"):
        return "foreman"
    if issue_id.startswith("verifier:"):
        return "verifier"
    return "worker"


def load_autoswe(
    db_path: str | Path,
    *,
    granularity: str = "run",
    stages: list[str] | None = None,
    issue_types: list[str] | None = None,
    models: list[str] | None = None,
    limit: int | None = None,
    min_steps: int | None = None,
    random_sample: bool = False,
    seed: int = 42,
) -> list[Trace]:
    """Load traces from an Auto-SWE SQLite database.

    Args:
        db_path: Path to the SQLite database file.
        granularity: How to group LLM calls into traces:
            ``"run"`` (default) — one pipeline stage execution per trace.
            ``"issue"`` — all calls for an issue as one trace.
            ``"foreman"`` — each foreman task as one trace.
        stages: Filter to specific pipeline stages (e.g.
            ``["implement", "scout"]``). Only applies when
            ``granularity="run"``.
        issue_types: Filter to specific issue types (e.g.
            ``["worker", "foreman"]``).
        models: Filter to specific model IDs.
        limit: Cap on total traces returned.
        min_steps: Skip traces with fewer steps.
        random_sample: Shuffle traces before applying limit.
        seed: RNG seed for random sampling.

    Returns:
        List of :class:`Trace` objects.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if granularity == "run":
        traces = _load_by_run(cur, stages=stages, issue_types=issue_types, models=models)
    elif granularity == "issue":
        traces = _load_by_issue(cur, issue_types=issue_types, models=models)
    elif granularity == "foreman":
        traces = _load_by_issue(cur, issue_types=["foreman"], models=models)
    else:
        raise ValueError(f"Unknown granularity {granularity!r}; expected 'run', 'issue', or 'foreman'")

    conn.close()

    # Filter by min_steps.
    if min_steps is not None:
        traces = [t for t in traces if len(t.steps) >= min_steps]

    # Random sample.
    if random_sample:
        rng = random.Random(seed)
        rng.shuffle(traces)

    # Apply limit.
    if limit is not None:
        traces = traces[:limit]

    log.info(
        "autoswe: loaded %d traces (granularity=%s, %d total steps)",
        len(traces), granularity, sum(len(t.steps) for t in traces),
    )
    return traces


def _rows_to_steps(rows: list[sqlite3.Row]) -> list[TraceStep]:
    """Convert LLM request rows into TraceStep objects.

    Mapping:
    - action = output_text (what the model DID — its reasoning + tool calls)
    - observation = next row's input_text prefix (what came back from the
      environment — tool results). For the last step, observation is None.
    - The current row's input_text is the prompt context, not used directly.
    """
    steps: list[TraceStep] = []
    for i, row in enumerate(rows):
        output_text = row["output_text"] or ""

        action = _truncate(output_text) if output_text else "(no response)"

        # Observation = the tool results that came back, which appear
        # at the start of the next row's input_text.
        observation = None
        if i + 1 < len(rows):
            next_input = rows[i + 1]["input_text"] or ""
            if next_input:
                observation = _truncate(next_input)

        steps.append(
            TraceStep(
                index=i,
                action=action,
                observation=observation,
                metadata={
                    "model_id": row["model_id"],
                    "duration_ms": row["duration_ms"],
                    "prompt_tokens": row["prompt_tokens"],
                    "completion_tokens": row["completion_tokens"],
                    "created_at": row["created_at"],
                    "request_id": row["id"],
                },
            )
        )
    return steps


def _load_by_run(
    cur: sqlite3.Cursor,
    *,
    stages: list[str] | None,
    issue_types: list[str] | None,
    models: list[str] | None,
) -> list[Trace]:
    """Load one trace per pipeline run (stage execution)."""
    query = """
        SELECT r.id as run_id, r.stage, r.issue_id, r.status,
               i.title as issue_title
        FROM runs r
        LEFT JOIN issues i ON i.id = r.issue_id
        WHERE 1=1
    """
    params: list[Any] = []

    if stages:
        placeholders = ",".join("?" for _ in stages)
        query += f" AND r.stage IN ({placeholders})"
        params.extend(stages)

    query += " ORDER BY r.created_at"
    cur.execute(query, params)
    runs = cur.fetchall()

    traces: list[Trace] = []
    for run in runs:
        run_id = run["run_id"]
        issue_id = run["issue_id"]
        issue_type = _classify_issue_type(issue_id)

        if issue_types and issue_type not in issue_types:
            continue

        cur.execute(
            "SELECT * FROM llm_requests WHERE run_id = ? ORDER BY created_at",
            (run_id,),
        )
        rows = cur.fetchall()
        if not rows:
            continue

        if models:
            rows = [r for r in rows if r["model_id"] in models]
            if not rows:
                continue

        steps = _rows_to_steps(rows)
        if not steps:
            continue

        # Determine model — use the most common model in this run.
        model_counts: dict[str, int] = {}
        for r in rows:
            m = r["model_id"] or "unknown"
            model_counts[m] = model_counts.get(m, 0) + 1
        model = max(model_counts, key=model_counts.get)  # type: ignore

        stage = run["stage"] or "unknown"
        issue_title = run["issue_title"] or issue_id or "unknown"
        task_goal = f"[{stage}] {issue_title}"

        has_output = any(r["output_text"] for r in rows)

        traces.append(
            Trace(
                trace_id=f"autoswe-run-{run_id}",
                task_goal=_truncate(task_goal, 4000),
                task_id=issue_id,
                model=model,
                source="autoswe",
                success=run["status"] == "completed" if run["status"] else None,
                steps=tuple(steps),
                metadata={
                    "stage": stage,
                    "issue_type": issue_type,
                    "run_status": run["status"],
                    "framework": "autoswe",
                },
            )
        )

    return traces


def _load_by_issue(
    cur: sqlite3.Cursor,
    *,
    issue_types: list[str] | None,
    models: list[str] | None,
) -> list[Trace]:
    """Load one trace per issue (all LLM calls grouped)."""
    cur.execute(
        "SELECT DISTINCT issue_id FROM llm_requests WHERE issue_id IS NOT NULL"
    )
    issue_ids = [r["issue_id"] for r in cur.fetchall()]

    traces: list[Trace] = []
    for issue_id in issue_ids:
        issue_type = _classify_issue_type(issue_id)
        if issue_types and issue_type not in issue_types:
            continue

        cur.execute(
            "SELECT * FROM llm_requests WHERE issue_id = ? ORDER BY created_at",
            (issue_id,),
        )
        rows = cur.fetchall()
        if not rows:
            continue

        if models:
            rows = [r for r in rows if r["model_id"] in models]
            if not rows:
                continue

        steps = _rows_to_steps(rows)
        if not steps:
            continue

        model_counts: dict[str, int] = {}
        for r in rows:
            m = r["model_id"] or "unknown"
            model_counts[m] = model_counts.get(m, 0) + 1
        model = max(model_counts, key=model_counts.get)  # type: ignore

        # Try to get issue title.
        title = issue_id
        if issue_type == "worker":
            cur.execute("SELECT title FROM issues WHERE id = ?", (issue_id,))
            row = cur.fetchone()
            if row:
                title = row["title"] or issue_id

        traces.append(
            Trace(
                trace_id=f"autoswe-issue-{issue_id}",
                task_goal=_truncate(title, 4000),
                task_id=issue_id,
                model=model,
                source="autoswe",
                success=None,
                steps=tuple(steps),
                metadata={
                    "issue_type": issue_type,
                    "framework": "autoswe",
                },
            )
        )

    return traces


def load_autoswe_jsonl(
    jsonl_path: str | Path,
    *,
    stages: list[str] | None = None,
    models: list[str] | None = None,
    limit: int | None = None,
    min_steps: int | None = None,
    random_sample: bool = False,
    seed: int = 42,
) -> list[Trace]:
    """Load traces from an exported JSONL file (no DB needed).

    The JSONL is produced by ``scripts/export_autoswe.py`` and
    contains sanitized traces ready for public distribution.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    traces: list[Trace] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            if stages and record.get("metadata", {}).get("stage") not in stages:
                continue
            if models and record.get("model") not in models:
                continue

            steps = tuple(
                TraceStep(
                    index=s["index"],
                    action=s.get("action", ""),
                    observation=s.get("observation"),
                    thought=s.get("thought"),
                    metadata={
                        "model_id": s.get("model_id"),
                        "duration_ms": s.get("duration_ms"),
                        "prompt_tokens": s.get("prompt_tokens"),
                        "completion_tokens": s.get("completion_tokens"),
                    },
                )
                for s in record["steps"]
            )

            if min_steps is not None and len(steps) < min_steps:
                continue

            traces.append(
                Trace(
                    trace_id=record["trace_id"],
                    task_goal=record.get("task_goal", ""),
                    task_id=record.get("task_id"),
                    model=record.get("model"),
                    source=record.get("source", "autoswe"),
                    success=record.get("success"),
                    steps=steps,
                    metadata=record.get("metadata", {}),
                )
            )

    if random_sample:
        rng = random.Random(seed)
        rng.shuffle(traces)

    if limit is not None:
        traces = traces[:limit]

    log.info("autoswe-jsonl: loaded %d traces", len(traces))
    return traces


def summarize_db(db_path: str | Path) -> dict[str, Any]:
    """Quick summary of what's in the database."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM llm_requests")
    total_requests = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT issue_id) FROM llm_requests WHERE issue_id IS NOT NULL")
    n_issues = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT run_id) FROM llm_requests WHERE run_id IS NOT NULL")
    n_runs = cur.fetchone()[0]

    cur.execute("SELECT model_id, COUNT(*) FROM llm_requests GROUP BY model_id ORDER BY COUNT(*) DESC")
    models = {r[0]: r[1] for r in cur.fetchall()}

    cur.execute("SELECT stage, COUNT(*) FROM runs GROUP BY stage ORDER BY COUNT(*) DESC")
    stages = {r[0]: r[1] for r in cur.fetchall()}

    conn.close()
    return {
        "total_requests": total_requests,
        "n_issues": n_issues,
        "n_runs": n_runs,
        "models": models,
        "stages": stages,
    }


__all__ = ["load_autoswe", "load_autoswe_jsonl", "summarize_db"]
