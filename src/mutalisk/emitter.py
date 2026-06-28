"""Concrete candidate sinks.

`FileCandidateEmitter` is the first concrete `CandidateEmitter`: it writes
fail-closed-validated candidate JSON to a local, gitignored `candidates/`
directory. This is the seam between the offline Python optimizer and the Effect
online authority.

TODO (build-out, see open issues): an `R2CandidateEmitter` / object-store sink
that writes under the public-safe candidate schema agreed with the Effect side
and indexes the candidate in D1. The Effect authority reads candidates, runs its
own acceptance gate, and only then promotes. The shared schema must be agreed
first (Candidate here <-> the Action Submission / candidate-manifest shape on the
Effect side). Keep the `CandidateEmitter.emit(candidate) -> str` interface
stable so swapping sinks is a config change, not a rewrite.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .candidate import Candidate, CandidateEmitter


def _slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in value)


class FileCandidateEmitter(CandidateEmitter):
    """Writes validated candidates as JSON files into a local directory.

    The output dir defaults to ``candidates/`` (gitignored). Candidate outputs,
    traces, and artifacts are never committed (see ``.gitignore``).
    """

    def __init__(self, out_dir: str | Path = "candidates") -> None:
        self.out_dir = Path(out_dir)

    def emit(self, candidate: Candidate) -> str:
        # Fail closed before any write: to_json() calls validate().
        payload = candidate.to_json()
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{_slug(candidate.signature)}-{int(time.time())}-{digest}.json"
        path = self.out_dir / name
        path.write_text(payload, encoding="utf-8")
        return str(path)
