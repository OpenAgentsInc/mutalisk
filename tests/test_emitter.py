import json

import pytest

from mutalisk.candidate import Candidate
from mutalisk.emitter import FileCandidateEmitter


def _valid_candidate(**kw) -> Candidate:
    base = dict(
        signature="mutalisk.sentiment_classify.v1",
        base_module_ref="module://base@1",
        optimized_module={"components": {"positive_cues": "great"}},
        metric_name="val_accuracy",
        metric_value=0.83,
        optimizer="gepa@0.1.1",
        eval_evidence_refs=["eval://x"],
        trace_provenance_refs=["trace://y"],
    )
    base.update(kw)
    return Candidate(**base)


def test_file_emitter_writes_valid_json(tmp_path):
    emitter = FileCandidateEmitter(tmp_path)
    path = emitter.emit(_valid_candidate())
    data = json.loads(open(path).read())
    assert data["optimizer"] == "gepa@0.1.1"
    assert data["signature"] == "mutalisk.sentiment_classify.v1"
    assert data["metric_value"] == 0.83


def test_file_emitter_fails_closed_and_writes_nothing(tmp_path):
    emitter = FileCandidateEmitter(tmp_path)
    bad = _valid_candidate(trace_provenance_refs=[])
    with pytest.raises(ValueError):
        emitter.emit(bad)
    # Fail-closed: no partial artifact left behind.
    assert list(tmp_path.glob("*.json")) == []


def test_file_emitter_creates_missing_dir(tmp_path):
    out = tmp_path / "nested" / "candidates"
    path = FileCandidateEmitter(out).emit(_valid_candidate())
    assert path.startswith(str(out))
