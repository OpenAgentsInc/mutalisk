import json
from pathlib import Path

from mutalisk.candidate import PROBE_GEPA_CANDIDATE_MANIFEST_SCHEMA_VERSION
from mutalisk.demo import (
    KHALA_FLEET_DELEGATION_FEEDBACK_DIMENSIONS_SCHEMA_VERSION,
    load_khala_fleet_delegation_demo_dataset,
    run_khala_fleet_delegation_demo,
)

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "khala_fleet_delegation_demo.json"


def test_demo_fixture_loads_public_safe_train_and_val_examples():
    dataset = load_khala_fleet_delegation_demo_dataset(FIXTURE)

    assert dataset.dataset_ref == "eval://mutalisk/fixtures/khala-fleet-delegation-demo@v1"
    assert dataset.train
    assert dataset.val
    assert all(example.eval_ref.startswith("eval://mutalisk/fixtures/") for example in dataset.examples)
    assert all(example.trace_ref.startswith("trace://mutalisk/fixtures/") for example in dataset.examples)


def test_khala_fleet_delegation_demo_emits_openagents_summary_shape(tmp_path):
    summary_path = tmp_path / "khala-fleet-delegation-summary.json"
    out = run_khala_fleet_delegation_demo(
        candidate_out_dir=tmp_path / "candidates",
        dataset_path=FIXTURE,
        max_metric_calls=8,
        summary_path=summary_path,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifact = json.loads(out.candidate_artifact_path.read_text(encoding="utf-8"))

    assert summary == artifact["candidateManifest"]
    assert summary["schemaVersion"] == PROBE_GEPA_CANDIDATE_MANIFEST_SCHEMA_VERSION
    assert summary["signature"] == "khala.fleet.delegation"
    assert summary["metricName"] == "khala.fleet.delegation"
    assert isinstance(summary["metricValueBps"], int)
    assert summary["metricValueBps"] > 0
    assert summary["evalEvidenceRefs"]
    assert summary["traceProvenanceRefs"]
    assert artifact["signature"] == "khala.fleet.delegation"
    assert artifact["metric"]["eval_dataset_ref"] == (
        "eval://mutalisk/fixtures/khala-fleet-delegation-demo@v1"
    )
    dimensions = artifact["metric"]["feedback_dimensions"]
    assert dimensions["schemaVersion"] == KHALA_FLEET_DELEGATION_FEEDBACK_DIMENSIONS_SCHEMA_VERSION
    assert {item["name"] for item in dimensions["dimensions"]} >= {
        "capacity_recovery",
        "typed_blockers",
        "verifier_closeout",
    }
