import pytest

from mutalisk.candidate import Candidate


def _c(**kw):
    base = dict(
        signature="sig.example",
        base_module_ref="module.base@1",
        optimized_module={"prompt": "..."},
        metric_name="pass_rate",
        metric_value=0.91,
        optimizer="gepa@0.1.1",
        eval_evidence_refs=["eval://x"],
        trace_provenance_refs=["trace://y"],
    )
    base.update(kw)
    return Candidate(**base)


def test_valid_candidate_serializes():
    assert '"optimizer": "gepa@0.1.1"' in _c().to_json()


def test_fails_closed_without_evidence():
    with pytest.raises(ValueError):
        _c(eval_evidence_refs=[]).validate()
    with pytest.raises(ValueError):
        _c(trace_provenance_refs=[]).validate()
    with pytest.raises(ValueError):
        _c(optimizer="").validate()
