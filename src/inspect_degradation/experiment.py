"""Experiment configuration: the reproducibility envelope around a run.

Every Phase 1 (and later Phase 3) experiment writes one of these next to
its results so a reader can answer:

* which grader (and which rubric version) produced these grades?
* which dataset slice was used?
* what ensemble policy, if any?
* when was it run, against which package version, against which git commit?

The dataclass is a *snapshot*: once constructed, it does not reach back
into live grader objects, so persisted configs round-trip through JSON
without any reference to runtime state. Grader-specific shape lives in
:class:`~inspect_degradation.grader.interface.GraderSnapshot`, which each
grader produces from its own ``snapshot()`` method — this module never
introspects grader internals.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_degradation import __version__
from inspect_degradation.grader.drift_canary import CanaryFingerprint
from inspect_degradation.grader.interface import Grader, GraderSnapshot


@dataclass(frozen=True)
class DatasetSlice:
    """Identifier for the dataset slice an experiment was run on."""

    name: str
    """Logical name (e.g. ``"trail"``)."""

    path: str
    """Filesystem path to the dataset root, recorded as-given (not resolved)."""

    splits: tuple[str, ...] = ()
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "splits": list(self.splits),
            "limit": self.limit,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level reproducibility envelope for one validation run."""

    name: str
    grader: GraderSnapshot
    dataset: DatasetSlice
    seed: int | None = None
    notes: str = ""
    canary: CanaryFingerprint | None = None

    package_version: str = __version__
    python_version: str = field(default_factory=lambda: platform.python_version())
    git_commit: str | None = field(default_factory=lambda: _git_commit())
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_grader(
        cls,
        *,
        name: str,
        grader: Grader,
        dataset: DatasetSlice,
        seed: int | None = None,
        notes: str = "",
        canary: CanaryFingerprint | None = None,
    ) -> "ExperimentConfig":
        return cls(
            name=name,
            grader=grader.snapshot(),
            dataset=dataset,
            seed=seed,
            notes=notes,
            canary=canary,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "grader": self.grader.to_dict(),
            "dataset": self.dataset.to_dict(),
            "seed": self.seed,
            "notes": self.notes,
            "canary": self.canary.to_dict() if self.canary is not None else None,
            "package_version": self.package_version,
            "python_version": self.python_version,
            "git_commit": self.git_commit,
            "created_at": self.created_at,
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _git_commit() -> str | None:
    """Best-effort current git commit; ``None`` outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None
