"""The optimization runner: program + eval set + optimizer -> Candidate.

This is the orchestration seam. It runs an `Optimizer` over the in-repo
fixtures, builds a fully-provenanced `Candidate` (fail-closed via
`Candidate.validate`), and hands it to a `CandidateEmitter`. It never mutates
production state -- it only emits a candidate proposal + evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from .candidate import Candidate, CandidateEmitter
from .eval_set import (
    EVAL_EVIDENCE_REF,
    TRACE_PROVENANCE_REF,
    Example,
    trainset,
    valset,
)
from .optimizer import Optimizer, OptimizeResult
from .program import SEED_CANDIDATE
from .signatures import SENTIMENT_SIGNATURE_ID
from .trace_eval import TraceEvalDataset

# Provenance of the base module being improved (the weak seed program).
BASE_MODULE_REF = "module://mutalisk/sentiment_seed@0.0.1"
FIXTURE_TRACE_PROVENANCE = {
    "refs": [TRACE_PROVENANCE_REF],
    "source": "mutalisk.synthetic_fixture@v1",
    "record_count": len(trainset()) + len(valset()),
}


@dataclass(frozen=True)
class RunOutput:
    """Result of an end-to-end run."""

    candidate: Candidate
    result: OptimizeResult
    sink_path: str


def build_candidate(
    result: OptimizeResult,
    *,
    signature_id: str = SENTIMENT_SIGNATURE_ID,
    base_module_ref: str = BASE_MODULE_REF,
    base_candidate: dict[str, str] | None = None,
    eval_dataset_ref: str = EVAL_EVIDENCE_REF,
    eval_evidence_refs: list[str] | None = None,
    trace_provenance: dict[str, object] | None = None,
) -> Candidate:
    """Assemble a fail-closed-validated Candidate from an optimize result."""
    candidate = Candidate(
        signature=signature_id,
        base_module={
            "ref": base_module_ref,
            "signature": signature_id,
            "components": dict(base_candidate if base_candidate is not None else SEED_CANDIDATE),
        },
        optimized_module={
            "components": result.optimized_candidate,
            "optimizer": result.optimizer_id,
        },
        metric={
            "name": result.metric_name,
            "value": result.metric_value,
            "base_value": result.base_metric_value,
            "eval_dataset_ref": eval_dataset_ref,
            "higher_is_better": True,
        },
        eval_evidence_refs=(
            eval_evidence_refs if eval_evidence_refs is not None else [EVAL_EVIDENCE_REF]
        ),
        trace_provenance=(
            trace_provenance if trace_provenance is not None else dict(FIXTURE_TRACE_PROVENANCE)
        ),
    )
    candidate.validate()  # fail closed before anyone sees it
    return candidate


def run_optimization(
    optimizer: Optimizer,
    emitter: CandidateEmitter,
    *,
    seed_candidate: dict[str, str] | None = None,
    train: list[Example] | None = None,
    val: list[Example] | None = None,
    trace_eval_dataset: TraceEvalDataset | None = None,
    signature_id: str = SENTIMENT_SIGNATURE_ID,
    base_module_ref: str = BASE_MODULE_REF,
    eval_evidence_refs: list[str] | None = None,
    trace_provenance: dict[str, object] | None = None,
) -> RunOutput:
    """Run the optimizer over the fixtures and emit a validated candidate."""
    if trace_eval_dataset is not None and (train is not None or val is not None):
        raise ValueError("pass either trace_eval_dataset or explicit train/val sets, not both")

    seed = seed_candidate if seed_candidate is not None else dict(SEED_CANDIDATE)
    train_items = train if train is not None else trainset()
    val_items = val if val is not None else valset()
    eval_dataset_ref = EVAL_EVIDENCE_REF

    if trace_eval_dataset is not None:
        trace_eval_dataset.validate()
        train_items = trace_eval_dataset.trainset
        val_items = trace_eval_dataset.valset
        eval_dataset_ref = trace_eval_dataset.dataset_ref
        if eval_evidence_refs is None:
            eval_evidence_refs = trace_eval_dataset.eval_evidence_refs
        if trace_provenance is None:
            trace_provenance = trace_eval_dataset.trace_provenance

    result = optimizer.optimize(
        seed,
        train_items,
        val_items,
    )
    candidate = build_candidate(
        result,
        signature_id=signature_id,
        base_module_ref=base_module_ref,
        base_candidate=seed,
        eval_dataset_ref=eval_dataset_ref,
        eval_evidence_refs=eval_evidence_refs,
        trace_provenance=trace_provenance,
    )
    sink_path = emitter.emit(candidate)
    return RunOutput(candidate=candidate, result=result, sink_path=sink_path)
