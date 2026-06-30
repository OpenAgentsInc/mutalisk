import pytest

from mutalisk.candidate import PROBE_GEPA_CANDIDATE_MANIFEST_SCHEMA_VERSION, Candidate


def _c(**kw):
    base = dict(
        signature="sig.example",
        base_module={"ref": "module.base@1", "components": {"cue": "old"}},
        optimized_module={"components": {"cue": "new"}, "optimizer": "gepa@0.1.1"},
        metric={
            "name": "pass_rate",
            "value": 0.91,
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


def test_valid_candidate_serializes():
    data = _c().to_dict()
    assert data["schemaVersion"] == "mutalisk.candidate_artifact.v1"
    assert data["candidateHash"].startswith("sha256:")
    assert data["manifestHash"].startswith("sha256:")
    assert data["candidateManifest"]["schemaVersion"] == (
        PROBE_GEPA_CANDIDATE_MANIFEST_SCHEMA_VERSION
    )
    assert data["optimized_module"]["optimizer"] == "gepa@0.1.1"


def test_candidate_manifest_summary_matches_effect_khala_fields():
    summary = _c(
        signature="khala.fleet.delegation",
        base_module={
            "ref": "module://mutalisk/khala_fleet_delegate_seed@0.0.1",
            "components": {"cue": "old"},
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
    ).to_manifest_summary()

    assert set(summary) == {
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
    assert summary["candidateManifestRef"].startswith(
        "candidate_manifest.khala.fleet.delegation."
    )
    assert summary["candidateRef"].startswith("candidate.khala.fleet.delegation.")
    assert summary["metricName"] == "khala.fleet.delegation"
    assert summary["metricValueBps"] == 8123
    assert summary["baseModuleRef"] == "module.mutalisk.khala_fleet_delegate_seed.0.0.1"
    assert summary["evalEvidenceRefs"] == [
        "eval_result.eval.mutalisk.khala-delegation.no-capacity"
    ]
    assert summary["traceProvenanceRefs"] == [
        "trace.public.trace.mutalisk.khala-delegation.no-capacity"
    ]


def test_fails_closed_without_evidence():
    with pytest.raises(ValueError):
        _c(eval_evidence_refs=[]).validate()
    with pytest.raises(ValueError):
        _c(trace_provenance={"refs": [], "source": "fixture", "record_count": 1}).validate()
    with pytest.raises(ValueError):
        _c(optimized_module={"components": {"cue": "new"}, "optimizer": ""}).validate()


def test_fails_closed_without_metric_dataset_ref():
    with pytest.raises(ValueError):
        _c(metric={"name": "pass_rate", "value": 0.9, "base_value": 0.5}).validate()


def test_manifest_summary_rejects_unsafe_refs_and_component_text():
    with pytest.raises(ValueError):
        _c(eval_evidence_refs=["/Users/christopherdavid/raw_trace.json"]).to_manifest_summary()
    with pytest.raises(ValueError):
        _c(
            optimized_module={
                "components": {"cue": "request_new_runtime_authority"},
                "optimizer": "gepa@0.1.1",
            }
        ).validate()
