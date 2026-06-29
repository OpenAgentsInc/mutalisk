"""The candidate-artifact contract — mutalisk's only output (the seam).

A Candidate is a structured, public-safe proposal. It is NOT an authorized
write; the Effect online authority gates it. See AGENTS.md invariants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any
import json


def _require_mapping(value: dict[str, Any], name: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"candidate missing {name}")


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate missing {name}")
    return value


def _require_ref_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"candidate missing {name}")
    refs = [_require_string(item, name) for item in value]
    return refs


@dataclass(frozen=True)
class Candidate:
    """An optimized DSPy module proposal + the evidence behind it."""

    signature: str  # the Blueprint/DSPy signature id this targets
    base_module: dict[str, Any]  # provenance + public-safe seed module metadata
    optimized_module: dict  # the proposed optimized module (prompts/policies)
    metric: dict[str, Any]  # name/value/base/eval dataset/optimizer evidence
    eval_evidence_refs: list[str] = field(default_factory=list)
    trace_provenance: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Fail closed if provenance or metric evidence is missing."""
        if not self.signature:
            raise ValueError("candidate missing signature")

        _require_mapping(self.base_module, "base_module")
        _require_string(self.base_module.get("ref"), "base_module.ref")

        _require_mapping(self.optimized_module, "optimized_module")
        optimizer = _require_string(
            self.optimized_module.get("optimizer"), "optimized_module.optimizer"
        )
        if "@" not in optimizer or optimizer.endswith("@"):
            raise ValueError("candidate optimizer must include an explicit version")
        components = self.optimized_module.get("components")
        if not isinstance(components, dict) or not components:
            raise ValueError("candidate missing optimized_module.components")

        _require_mapping(self.metric, "metric")
        _require_string(self.metric.get("name"), "metric.name")
        _require_string(self.metric.get("eval_dataset_ref"), "metric.eval_dataset_ref")
        for key in ("value", "base_value"):
            value = self.metric.get(key)
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"candidate missing metric.{key}")

        _require_ref_list(self.eval_evidence_refs, "eval evidence refs")

        _require_mapping(self.trace_provenance, "trace_provenance")
        _require_ref_list(self.trace_provenance.get("refs"), "trace provenance refs")
        _require_string(self.trace_provenance.get("source"), "trace_provenance.source")
        record_count = self.trace_provenance.get("record_count")
        if not isinstance(record_count, int) or record_count <= 0:
            raise ValueError("candidate missing trace_provenance.record_count")

    @property
    def optimizer(self) -> str:
        """Return the explicit optimizer id, e.g. ``gepa@0.1.1``."""
        return str(self.optimized_module.get("optimizer", ""))

    @property
    def metric_name(self) -> str:
        return str(self.metric.get("name", ""))

    @property
    def metric_value(self) -> float:
        return float(self.metric.get("value", 0.0))

    @property
    def base_module_ref(self) -> str:
        return str(self.base_module.get("ref", ""))

    @property
    def trace_provenance_refs(self) -> list[str]:
        refs = self.trace_provenance.get("refs", [])
        return list(refs) if isinstance(refs, list) else []

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class CandidateEmitter:
    """Writes validated candidates to a sink. Implementations: object/R2 store.

    NOTE: this is a skeleton. The real emitter (R2/object-store wiring + the
    shared candidate schema agreed with the Effect side) is the build-out task.
    """

    def emit(self, candidate: Candidate) -> str:
        candidate.validate()
        raise NotImplementedError("wire the candidate sink in the build-out")
