import json

from mutalisk import eval_set
from mutalisk.emitter import FileCandidateEmitter
from mutalisk.optimizer import LocalSearchOptimizer, OptimizeResult
from mutalisk.program import SEED_CANDIDATE
from mutalisk.runner import build_candidate, run_optimization
from mutalisk.signatures import SENTIMENT_SIGNATURE_ID


class FakeOptimizer:
    """A dependency-free optimizer to test runner wiring in isolation."""

    @property
    def optimizer_id(self) -> str:
        return "fake.optimizer@9.9.9"

    def optimize(self, seed_candidate, trainset, valset):
        return OptimizeResult(
            optimized_candidate={"positive_cues": "great", "negative_cues": "bad"},
            metric_name="val_accuracy",
            metric_value=0.99,
            base_metric_value=0.50,
            optimizer_id=self.optimizer_id,
        )


def test_runner_with_fake_optimizer_emits_valid_candidate(tmp_path):
    out = run_optimization(FakeOptimizer(), FileCandidateEmitter(tmp_path))
    assert out.candidate.signature == SENTIMENT_SIGNATURE_ID
    assert out.candidate.optimizer == "fake.optimizer@9.9.9"
    # candidate landed on disk and round-trips as the validated JSON
    data = json.loads(open(out.sink_path).read())
    assert data["optimizer"] == "fake.optimizer@9.9.9"
    assert data["metric_value"] == 0.99


def test_runner_records_optimizer_version_and_provenance():
    result = LocalSearchOptimizer().optimize(
        dict(SEED_CANDIDATE), eval_set.trainset(), eval_set.valset()
    )
    candidate = build_candidate(result)
    # AGENTS.md invariant: optimizer@version + evidence + provenance present.
    candidate.validate()
    assert "@" in candidate.optimizer
    assert candidate.eval_evidence_refs == [eval_set.EVAL_EVIDENCE_REF]
    assert candidate.trace_provenance_refs == [eval_set.TRACE_PROVENANCE_REF]


def test_runner_end_to_end_local(tmp_path):
    out = run_optimization(LocalSearchOptimizer(), FileCandidateEmitter(tmp_path))
    assert out.result.metric_value >= out.result.base_metric_value
    assert out.candidate.optimizer.startswith("mutalisk.local_search@")
