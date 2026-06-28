"""The candidate-artifact contract — mutalisk's only output (the seam).

A Candidate is a structured, public-safe proposal. It is NOT an authorized
write; the Effect online authority gates it. See AGENTS.md invariants.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json


@dataclass(frozen=True)
class Candidate:
    """An optimized DSPy module proposal + the evidence behind it."""

    signature: str  # the Blueprint/DSPy signature id this targets
    base_module_ref: str  # provenance of the module being improved
    optimized_module: dict  # the proposed optimized module (prompts/policies)
    metric_name: str
    metric_value: float
    optimizer: str  # e.g. "gepa@0.1.1" or "dspy.MIPROv2@2.5"
    eval_evidence_refs: list[str] = field(default_factory=list)
    trace_provenance_refs: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Fail closed if provenance or metric evidence is missing."""
        if not self.signature:
            raise ValueError("candidate missing signature")
        if not self.optimizer:
            raise ValueError("candidate missing optimizer version")
        if not self.eval_evidence_refs:
            raise ValueError("candidate missing eval evidence refs")
        if not self.trace_provenance_refs:
            raise ValueError("candidate missing trace provenance refs")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), sort_keys=True)


class CandidateEmitter:
    """Writes validated candidates to a sink. Implementations: object/R2 store.

    NOTE: this is a skeleton. The real emitter (R2/object-store wiring + the
    shared candidate schema agreed with the Effect side) is the build-out task.
    """

    def emit(self, candidate: Candidate) -> str:
        candidate.validate()
        raise NotImplementedError("wire the candidate sink in the build-out")
