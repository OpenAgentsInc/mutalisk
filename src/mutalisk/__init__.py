"""Mutalisk: offline DSPy/GEPA optimization that emits candidate artifacts.

Mutalisk is OFFLINE/LEAF compute. It produces untrusted candidate proposals
plus the evidence behind them. The Effect online authority (Khala/Artanis)
selects, gates, and admits candidates via the Blueprint signature/evidence/
receipt model. Mutalisk never mutates production state.
"""

from .candidate import Candidate, CandidateEmitter

__all__ = ["Candidate", "CandidateEmitter"]
