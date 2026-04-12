"""On-disk store for graded traces.

Phase 1 uses this for *resumability*: rerunning the validation script
after a crash, an interrupted API call, or a rubric tweak should not
re-grade traces that already have results on disk. Phase 3 will use the
same store to grade 80k Nebius trajectories incrementally.

The format is JSONL — one :class:`GradedTrace` per line, serialized via
``model_dump_json``. JSONL was chosen because:

* it's append-only: a crash mid-run leaves a valid prefix that the next
  run can read and skip,
* it streams: we never need to hold the entire corpus in memory,
* every line is independently parseable, so a single corrupted line
  doesn't take the file with it (we surface and skip).

The store is intentionally a *single-writer* store. Concurrent processes
appending to the same file would interleave bytes and corrupt records;
that contention belongs at a higher layer (a job scheduler) and is
explicitly out of scope.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path

from inspect_degradation.schema import GradedTrace


class GradedTraceStore:
    """Append-only JSONL store of :class:`GradedTrace` records.

    Thread-safe within a single process via an internal lock; not
    multi-process safe (see module docstring).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Touch the file so callers can iterate even before the first
        # append. Open in append mode (not write) so existing data is
        # preserved if the file already exists.
        self._path.touch(exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------ writes

    def append(self, trace: GradedTrace) -> None:
        """Append one :class:`GradedTrace` to the store, durably.

        Each call flushes and ``fsync``s so a process kill between calls
        cannot lose a record. The cost is one syscall per trace, which is
        negligible relative to LLM grading latency.
        """
        line = trace.model_dump_json()
        if "\n" in line:
            # model_dump_json never produces newlines for our schema, but
            # if a future field allows arbitrary text we want to fail loud
            # rather than corrupt the format.
            raise ValueError("serialized GradedTrace contains a newline; refusing to append")
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())

    # ------------------------------------------------------------------ reads

    def __iter__(self) -> Iterator[GradedTrace]:
        """Stream every record. Skips corrupted lines with a warning."""
        with self._path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    yield GradedTrace.model_validate_json(stripped)
                except (json.JSONDecodeError, ValueError) as e:
                    # Don't crash a Phase 3 run because line 47,238 is
                    # corrupt; surface it and keep going. Production
                    # callers can audit the file separately.
                    import logging

                    logging.getLogger(__name__).warning(
                        "GradedTraceStore: skipping corrupt line %d in %s: %s",
                        line_no,
                        self._path,
                        e,
                    )
                    continue

    def __len__(self) -> int:
        # Count only non-empty lines so a trailing newline doesn't lie.
        with self._path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def completed_trace_ids(self) -> set[str]:
        """Return the set of ``trace_id`` values already in the store.

        Callers use this to skip work on resume. Reads are O(stored bytes)
        but parse only the ``trace_id`` field per line, so this stays cheap
        even on large stores.
        """
        seen: set[str] = set()
        with self._path.open("r", encoding="utf-8") as f:
            for raw in f:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                tid = obj.get("trace_id") if isinstance(obj, dict) else None
                if isinstance(tid, str):
                    seen.add(tid)
        return seen

    def load_all(self) -> list[GradedTrace]:
        """Materialize every record. Convenience for small stores; for
        large stores prefer iterating."""
        return list(self)
