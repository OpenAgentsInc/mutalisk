import json
import sqlite3

import pytest

from mutalisk.candidate import Candidate
from mutalisk.emitter import D1CandidateIndexSink, FileCandidateEmitter, R2CandidateEmitter


class MemoryObjectStore:
    def __init__(self):
        self.writes = []

    def put(self, key, body, *, content_type, metadata):
        self.writes.append(
            {
                "body": body,
                "content_type": content_type,
                "key": key,
                "metadata": dict(metadata),
            }
        )


def _valid_candidate(**kw) -> Candidate:
    base = dict(
        signature="mutalisk.sentiment_classify.v1",
        base_module={
            "ref": "module://base@1",
            "components": {"positive_cues": "good"},
        },
        optimized_module={
            "components": {"positive_cues": "great"},
            "optimizer": "gepa@0.1.1",
        },
        metric={
            "name": "val_accuracy",
            "value": 0.83,
            "base_value": 0.50,
            "eval_dataset_ref": "trace-eval://fixture",
        },
        eval_evidence_refs=["eval://x"],
        trace_provenance={
            "refs": ["trace://y"],
            "source": "fixture",
            "record_count": 1,
        },
    )
    base.update(kw)
    return Candidate(**base)


def test_file_emitter_writes_valid_json(tmp_path):
    emitter = FileCandidateEmitter(tmp_path)
    path = emitter.emit(_valid_candidate())
    data = json.loads(open(path).read())
    assert data["candidateManifest"]["schemaVersion"] == (
        "psionic.probe_gepa_candidate_manifest.v1"
    )
    assert data["optimized_module"]["optimizer"] == "gepa@0.1.1"
    assert data["signature"] == "mutalisk.sentiment_classify.v1"
    assert data["metric"]["value"] == 0.83


def test_file_emitter_fails_closed_and_writes_nothing(tmp_path):
    emitter = FileCandidateEmitter(tmp_path)
    bad = _valid_candidate(trace_provenance={"refs": [], "source": "fixture", "record_count": 1})
    with pytest.raises(ValueError):
        emitter.emit(bad)
    # Fail-closed: no partial artifact left behind.
    assert list(tmp_path.glob("*.json")) == []


def test_file_emitter_creates_missing_dir(tmp_path):
    out = tmp_path / "nested" / "candidates"
    path = FileCandidateEmitter(out).emit(_valid_candidate())
    assert path.startswith(str(out))


def test_r2_emitter_writes_manifest_artifact_and_d1_index():
    objects = MemoryObjectStore()
    connection = sqlite3.connect(":memory:")
    index = D1CandidateIndexSink(connection)
    emitter = R2CandidateEmitter(
        objects,
        index,
        bucket_ref="r2://test-bucket",
        key_prefix="gd2",
    )

    ref = emitter.emit(
        _valid_candidate(
            signature="khala.fleet.delegation",
            base_module={
                "ref": "module://mutalisk/khala_fleet_delegate_seed@0.0.1",
                "components": {"positive_cues": "good"},
            },
            metric={
                "name": "khala.fleet.delegation",
                "value": 0.8123,
                "base_value": 0.50,
                "eval_dataset_ref": "eval://mutalisk/fixtures/khala-delegation@v1",
            },
            eval_evidence_refs=["eval://mutalisk/khala-delegation/no-capacity"],
            trace_provenance={
                "refs": ["trace://mutalisk/khala-delegation/no-capacity"],
                "source": "fixture",
                "record_count": 1,
            },
        )
    )

    assert ref.startswith("r2://test-bucket/gd2/khala.fleet.delegation/")
    assert len(objects.writes) == 2
    manifest = json.loads(objects.writes[0]["body"].decode("utf-8"))
    artifact = json.loads(objects.writes[1]["body"].decode("utf-8"))
    assert set(manifest) == {
        "baseModuleRef",
        "candidateManifestRef",
        "candidateRef",
        "evalEvidenceRefs",
        "metricName",
        "metricValueBps",
        "optimizedModuleRef",
        "schemaVersion",
        "signature",
        "traceProvenanceRefs",
    }
    assert manifest["schemaVersion"] == "psionic.probe_gepa_candidate_manifest.v1"
    assert manifest["metricValueBps"] == 8123
    assert artifact["candidateManifest"] == manifest
    assert objects.writes[0]["metadata"]["candidateManifestRef"] == (
        manifest["candidateManifestRef"]
    )

    row = connection.execute(
        """
        SELECT candidate_manifest_ref, candidate_ref, metric_value_bps,
               manifest_object_ref, artifact_object_ref, manifest_json
        FROM probe_gepa_candidate_manifests
        """
    ).fetchone()
    assert row[0] == manifest["candidateManifestRef"]
    assert row[1] == manifest["candidateRef"]
    assert row[2] == 8123
    assert row[3] == ref
    assert row[4].endswith("/candidate-artifact.json")
    assert json.loads(row[5]) == manifest


def test_r2_emitter_fails_closed_before_writing():
    objects = MemoryObjectStore()
    connection = sqlite3.connect(":memory:")
    emitter = R2CandidateEmitter(objects, D1CandidateIndexSink(connection))
    bad = _valid_candidate(trace_provenance={"refs": [], "source": "fixture", "record_count": 1})

    with pytest.raises(ValueError):
        emitter.emit(bad)

    assert objects.writes == []
    assert connection.execute("SELECT count(*) FROM probe_gepa_candidate_manifests").fetchone()[0] == 0
