"""Concrete candidate sinks.

`FileCandidateEmitter` is the first concrete `CandidateEmitter`: it writes
fail-closed-validated candidate JSON to a local, gitignored `candidates/`
directory. This is the seam between the offline Python optimizer and the Effect
online authority.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Protocol

from .candidate import (
    MUTALISK_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    PROBE_GEPA_CANDIDATE_MANIFEST_SCHEMA_VERSION,
    Candidate,
    CandidateEmitter,
)


class CandidateObjectStore(Protocol):
    """Minimal R2-compatible object store surface used by the batch emitter."""

    def put(
        self,
        key: str,
        body: bytes,
        *,
        content_type: str,
        metadata: Mapping[str, str],
    ) -> None: ...


class CandidateIndexSink(Protocol):
    """D1-style index surface for candidate manifest summary rows."""

    def upsert_candidate_manifest(
        self,
        summary: Mapping[str, Any],
        *,
        manifest_object_ref: str,
        artifact_object_ref: str,
    ) -> None: ...


_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
        self.out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{_slug(candidate.candidate_manifest_ref)}.json"
        path = self.out_dir / name
        path.write_text(payload, encoding="utf-8")
        return str(path)


class D1CandidateIndexSink:
    """SQLite/D1-compatible candidate manifest index.

    Cloudflare D1 is SQLite-shaped. This adapter intentionally depends only on a
    small DB-API connection surface so tests can use sqlite3 while production can
    route the same row shape through a Worker/D1 binding or HTTP bridge.
    """

    def __init__(
        self,
        connection,
        *,
        table: str = "probe_gepa_candidate_manifests",
    ) -> None:
        if not _SQL_IDENTIFIER_RE.match(table):
            raise ValueError("D1 candidate index table must be a SQL identifier")
        self.connection = connection
        self.table = table
        self.setup()

    def setup(self) -> None:
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
              candidate_manifest_ref TEXT PRIMARY KEY,
              candidate_ref TEXT NOT NULL,
              schema_version TEXT NOT NULL,
              signature TEXT NOT NULL,
              metric_name TEXT NOT NULL,
              metric_value_bps INTEGER NOT NULL,
              base_module_ref TEXT NOT NULL,
              optimized_module_ref TEXT NOT NULL,
              manifest_object_ref TEXT NOT NULL,
              artifact_object_ref TEXT NOT NULL,
              eval_evidence_refs_json TEXT NOT NULL,
              trace_provenance_refs_json TEXT NOT NULL,
              manifest_json TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def upsert_candidate_manifest(
        self,
        summary: Mapping[str, Any],
        *,
        manifest_object_ref: str,
        artifact_object_ref: str,
    ) -> None:
        self.connection.execute(
            f"""
            INSERT INTO {self.table} (
              candidate_manifest_ref,
              candidate_ref,
              schema_version,
              signature,
              metric_name,
              metric_value_bps,
              base_module_ref,
              optimized_module_ref,
              manifest_object_ref,
              artifact_object_ref,
              eval_evidence_refs_json,
              trace_provenance_refs_json,
              manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_manifest_ref) DO UPDATE SET
              candidate_ref = excluded.candidate_ref,
              schema_version = excluded.schema_version,
              signature = excluded.signature,
              metric_name = excluded.metric_name,
              metric_value_bps = excluded.metric_value_bps,
              base_module_ref = excluded.base_module_ref,
              optimized_module_ref = excluded.optimized_module_ref,
              manifest_object_ref = excluded.manifest_object_ref,
              artifact_object_ref = excluded.artifact_object_ref,
              eval_evidence_refs_json = excluded.eval_evidence_refs_json,
              trace_provenance_refs_json = excluded.trace_provenance_refs_json,
              manifest_json = excluded.manifest_json
            """,
            (
                summary["candidateManifestRef"],
                summary["candidateRef"],
                summary["schemaVersion"],
                summary["signature"],
                summary["metricName"],
                int(summary["metricValueBps"]),
                summary["baseModuleRef"],
                summary["optimizedModuleRef"],
                manifest_object_ref,
                artifact_object_ref,
                json.dumps(summary["evalEvidenceRefs"], sort_keys=True),
                json.dumps(summary["traceProvenanceRefs"], sort_keys=True),
                json.dumps(dict(summary), separators=(",", ":"), sort_keys=True),
            ),
        )
        self.connection.commit()


class R2CandidateEmitter(CandidateEmitter):
    """Write candidate manifests/artifacts to an R2-compatible store and D1 index."""

    def __init__(
        self,
        object_store: CandidateObjectStore,
        index_sink: CandidateIndexSink,
        *,
        bucket_ref: str = "r2://openagents-probe-gepa-candidates",
        key_prefix: str = "mutalisk",
    ) -> None:
        self.object_store = object_store
        self.index_sink = index_sink
        self.bucket_ref = bucket_ref.rstrip("/")
        self.key_prefix = key_prefix.strip("/")

    def _object_ref(self, key: str) -> str:
        return f"{self.bucket_ref}/{key}"

    def emit(self, candidate: Candidate) -> str:
        # Fail closed before any object or index write.
        manifest_json = candidate.to_manifest_json()
        artifact_json = candidate.to_json()
        summary = candidate.to_manifest_summary()
        signature = _slug(candidate.signature)
        digest = candidate.candidate_hash_short
        base_key = f"{self.key_prefix}/{signature}/{digest}"
        manifest_key = f"{base_key}/candidate-manifest.json"
        artifact_key = f"{base_key}/candidate-artifact.json"
        manifest_ref = self._object_ref(manifest_key)
        artifact_ref = self._object_ref(artifact_key)

        self.object_store.put(
            manifest_key,
            manifest_json.encode("utf-8"),
            content_type="application/json",
            metadata={
                "schemaVersion": PROBE_GEPA_CANDIDATE_MANIFEST_SCHEMA_VERSION,
                "candidateManifestRef": str(summary["candidateManifestRef"]),
                "candidateRef": str(summary["candidateRef"]),
            },
        )
        self.object_store.put(
            artifact_key,
            artifact_json.encode("utf-8"),
            content_type="application/json",
            metadata={
                "schemaVersion": MUTALISK_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
                "candidateManifestRef": str(summary["candidateManifestRef"]),
                "candidateRef": str(summary["candidateRef"]),
            },
        )
        self.index_sink.upsert_candidate_manifest(
            summary,
            manifest_object_ref=manifest_ref,
            artifact_object_ref=artifact_ref,
        )
        return manifest_ref
