import json

import pytest

from mutalisk.trace_eval import TraceEvalDataset, TraceEvalRecord


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_trace_eval_dataset_loads_public_safe_records(tmp_path):
    path = tmp_path / "records.jsonl"
    _write_jsonl(
        path,
        [
            {
                "public_text": "great release",
                "label": "positive",
                "split": "train",
                "trace_ref": "trace://run/1",
                "eval_ref": "eval://run/1",
            },
            {
                "public_text": "broken release",
                "label": "negative",
                "split": "val",
                "trace_ref": "trace://run/2",
                "eval_ref": "eval://run/2",
            },
        ],
    )

    dataset = TraceEvalDataset.from_jsonl(path)

    assert dataset.dataset_ref.startswith("trace-eval://sha256:")
    assert dataset.trainset[0].label == "positive"
    assert dataset.valset[0].label == "negative"
    assert dataset.eval_evidence_refs == ["eval://run/1", "eval://run/2"]
    assert dataset.trace_provenance["refs"] == ["trace://run/1", "trace://run/2"]


def test_trace_eval_rejects_missing_refs():
    with pytest.raises(ValueError, match="trace_ref"):
        TraceEvalRecord.from_mapping(
            {
                "public_text": "great release",
                "label": "positive",
                "split": "train",
                "eval_ref": "eval://run/1",
            }
        )


def test_trace_eval_rejects_forbidden_raw_fields():
    with pytest.raises(ValueError, match="raw field"):
        TraceEvalRecord.from_mapping(
            {
                "public_text": "great release",
                "prompt": "raw private prompt",
                "label": "positive",
                "split": "train",
                "trace_ref": "trace://run/1",
                "eval_ref": "eval://run/1",
            }
        )


def test_trace_eval_requires_train_and_val(tmp_path):
    path = tmp_path / "records.jsonl"
    _write_jsonl(
        path,
        [
            {
                "public_text": "great release",
                "label": "positive",
                "split": "train",
                "trace_ref": "trace://run/1",
                "eval_ref": "eval://run/1",
            }
        ],
    )

    with pytest.raises(ValueError, match="val split"):
        TraceEvalDataset.from_jsonl(path)
