"""Mutalisk: offline DSPy/GEPA optimization that emits candidate artifacts.

Mutalisk is OFFLINE/LEAF compute. It produces untrusted candidate proposals
plus the evidence behind them. The Effect online authority (Khala/Artanis)
selects, gates, and admits candidates via the Blueprint signature/evidence/
receipt model. Mutalisk never mutates production state.
"""

__version__ = "0.0.1"

from .candidate import Candidate, CandidateEmitter
from .delegation import (
    DELEGATION_CANDIDATE_COMPONENTS,
    DELEGATION_MODULE_PIPELINE,
    DELEGATION_SEED_CANDIDATE,
    KHALA_FLEET_DELEGATE_PROGRAM_ID,
    KHALA_FLEET_DELEGATION_CANDIDATE_SIGNATURE_ID,
    DelegationExample,
    DelegationProgramRun,
    delegation_score,
    delegation_trainset,
    delegation_valset,
    run_delegation_program,
    seed_delegation_candidate,
)
from .emitter import FileCandidateEmitter
from .trace_eval import TraceEvalDataset, TraceEvalRecord

__all__ = [
    "Candidate",
    "CandidateEmitter",
    "DELEGATION_CANDIDATE_COMPONENTS",
    "DELEGATION_MODULE_PIPELINE",
    "DELEGATION_SEED_CANDIDATE",
    "DelegationExample",
    "DelegationProgramRun",
    "FileCandidateEmitter",
    "KHALA_FLEET_DELEGATE_PROGRAM_ID",
    "KHALA_FLEET_DELEGATION_CANDIDATE_SIGNATURE_ID",
    "TraceEvalDataset",
    "TraceEvalRecord",
    "__version__",
    "delegation_score",
    "delegation_trainset",
    "delegation_valset",
    "run_delegation_program",
    "seed_delegation_candidate",
]
