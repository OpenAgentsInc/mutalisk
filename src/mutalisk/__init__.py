"""Mutalisk: offline DSPy/GEPA optimization that emits candidate artifacts.

Mutalisk is OFFLINE/LEAF compute. It produces untrusted candidate proposals
plus the evidence behind them. The Effect online authority (Khala/Artanis)
selects, gates, and admits candidates via the Blueprint signature/evidence/
receipt model. Mutalisk never mutates production state.
"""

__version__ = "0.0.1"

from .candidate import Candidate, CandidateEmitter
from .emitter import FileCandidateEmitter
from .trace_eval import TraceEvalDataset, TraceEvalRecord

__all__ = [
    "Candidate",
    "CandidateEmitter",
    "FileCandidateEmitter",
    "TraceEvalDataset",
    "TraceEvalRecord",
    "__version__",
]
