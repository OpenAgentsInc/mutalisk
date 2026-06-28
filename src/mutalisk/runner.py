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

# Provenance of the base module being improved (the weak seed program).
BASE_MODULE_REF = "module://mutalisk/sentiment_seed@0.0.1"


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
    eval_evidence_refs: list[str] | None = None,
    trace_provenance_refs: list[str] | None = None,
) -> Candidate:
    """Assemble a fail-closed-validated Candidate from an optimize result."""
    candidate = Candidate(
        signature=signature_id,
        base_module_ref=base_module_ref,
        optimized_module={
            "components": result.optimized_candidate,
            "base_metric_value": result.base_metric_value,
        },
        metric_name=result.metric_name,
        metric_value=result.metric_value,
        optimizer=result.optimizer_id,
        eval_evidence_refs=eval_evidence_refs or [EVAL_EVIDENCE_REF],
        trace_provenance_refs=trace_provenance_refs or [TRACE_PROVENANCE_REF],
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
    signature_id: str = SENTIMENT_SIGNATURE_ID,
    base_module_ref: str = BASE_MODULE_REF,
    eval_evidence_refs: list[str] | None = None,
    trace_provenance_refs: list[str] | None = None,
) -> RunOutput:
    """Run the optimizer over the fixtures and emit a validated candidate."""
    result = optimizer.optimize(
        seed_candidate if seed_candidate is not None else dict(SEED_CANDIDATE),
        train if train is not None else trainset(),
        val if val is not None else valset(),
    )
    candidate = build_candidate(
        result,
        signature_id=signature_id,
        base_module_ref=base_module_ref,
        eval_evidence_refs=eval_evidence_refs,
        trace_provenance_refs=trace_provenance_refs,
    )
    sink_path = emitter.emit(candidate)
    return RunOutput(candidate=candidate, result=result, sink_path=sink_path)
