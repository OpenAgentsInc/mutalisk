"""The candidate-artifact contract — mutalisk's only output (the seam).

A Candidate is a structured, public-safe proposal. It is NOT an authorized
write; the Effect online authority gates it. See AGENTS.md invariants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from math import isfinite
from typing import Any
import json
import re


PROBE_GEPA_CANDIDATE_MANIFEST_SCHEMA_VERSION = (
    "psionic.probe_gepa_candidate_manifest.v1"
)
MUTALISK_CANDIDATE_ARTIFACT_SCHEMA_VERSION = "mutalisk.candidate_artifact.v1"

_UNSAFE_TEXT_RE = re.compile(
    r"access_token|refresh_token|bearer |mdk_mnemonic|wallet_mnemonic|"
    r"private-repo://|bypass_release_gate|ignore_release_gate|disable_release_gate|"
    r"public_claim_upgrade_authority|request_new_runtime_authority|"
    r"new_runtime_authority|grant_runtime_authority",
    re.IGNORECASE,
)
_UNSAFE_REF_RE = re.compile(
    r"(@|/Users/|/home/|access[_-]?token|auth\.json|bearer|callback[_-]?token|"
    r"cookie|credential|customer[_-]?(email|name|value)|email[_-]?(address|body)|"
    r"fixture[_-]?body|gho_[A-Za-z0-9_]+|ghp_[A-Za-z0-9_]+|"
    r"github\.com/[^:/]+/private|invoice|lnbc|lntb|lnbcrt|lno1|"
    r"mdk[_-]?(access[_-]?token|mnemonic|webhook[_-]?secret)|mnemonic|oauth|"
    r"opencode_auth_content|payment[_-]?(hash|id|preimage|proof)|"
    r"payout[_-]?(address|destination|target)|preimage|private[_-]?(channel|key|repo)|"
    r"provider[_-]?(account|grant|payload|secret|token)|"
    r"raw[_-]?(auth|benchmark|email|fixture|invoice|payment|payload|prompt|provider|"
    r"runner|run[_-]?log|source[_-]?archive|trace|traces)|runner[_-]?log|"
    r"secret|sk-[a-z0-9]|source[_-]?archive|token|wallet)",
    re.IGNORECASE,
)
_RAW_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
_SAFE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,260}$")
_PUBLIC_REF_SEGMENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


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


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_ref(value: str | bytes) -> str:
    body = value.encode("utf-8") if isinstance(value, str) else value
    return f"sha256:{sha256(body).hexdigest()}"


def _short_hash(hash_ref: str) -> str:
    return hash_ref.removeprefix("sha256:")[:16]


def _public_ref_segment(value: str, fallback: str) -> str:
    segment = (
        value.strip()
        .replace("://", ".")
        .replace("/", ".")
        .replace("@", ".")
    )
    segment = _PUBLIC_REF_SEGMENT_RE.sub("_", segment).strip("._-")[:160]
    if not segment:
        segment = fallback
    if not segment[0].isalnum():
        segment = f"{fallback}.{segment}"
    return segment


def _public_ref(value: str, *, prefix: str) -> str:
    original = value.strip()
    if _UNSAFE_REF_RE.search(original.replace("@", "")) or _RAW_TIMESTAMP_RE.search(original):
        raise ValueError(f"candidate manifest ref is not public-safe: {prefix}")
    segment = _public_ref_segment(value, prefix)
    if segment.startswith(f"{prefix}."):
        ref = segment
    else:
        ref = f"{prefix}.{segment}"
    if len(ref) > 261:
        ref = ref[:261]
    if (
        not _SAFE_REF_RE.match(ref)
        or _UNSAFE_REF_RE.search(ref)
        or _RAW_TIMESTAMP_RE.search(ref)
    ):
        raise ValueError(f"candidate manifest ref is not public-safe: {prefix}")
    return ref


def _assert_safe_component_text(value: str, name: str) -> None:
    normalized = value.lower()
    if (
        _UNSAFE_TEXT_RE.search(value)
        or normalized.startswith("sk-")
        or " sk-" in normalized
    ):
        raise ValueError(f"candidate optimized_module.components.{name} is not public-safe")


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
        for key, value in components.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"candidate missing optimized_module.components.{key}")
            _assert_safe_component_text(value, str(key))

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

    @property
    def candidate_hash(self) -> str:
        """Content hash of the public-safe candidate payload."""
        return _sha256_ref(_stable_json(self._content_payload()))

    @property
    def candidate_hash_short(self) -> str:
        return _short_hash(self.candidate_hash)

    @property
    def metric_value_bps(self) -> int:
        """Return the normalized metric in basis points for the Effect seam."""
        value = self.metric_value
        if value < 0 or value > 1:
            raise ValueError("candidate metric.value must be normalized to [0, 1]")
        return int(round(value * 10_000))

    @property
    def candidate_ref(self) -> str:
        signature = _public_ref_segment(self.signature, "candidate")
        return f"candidate.{signature}.{self.candidate_hash_short}"

    @property
    def candidate_manifest_ref(self) -> str:
        signature = _public_ref_segment(self.signature, "candidate_manifest")
        return f"candidate_manifest.{signature}.{self.candidate_hash_short}"

    @property
    def optimized_module_ref(self) -> str:
        signature = _public_ref_segment(self.signature, "module")
        return f"module.{signature}.optimized.{self.candidate_hash_short}"

    def _content_payload(self) -> dict[str, Any]:
        return {
            "base_module": self.base_module,
            "eval_evidence_refs": list(self.eval_evidence_refs),
            "metric": self.metric,
            "optimized_module": self.optimized_module,
            "signature": self.signature,
            "trace_provenance": self.trace_provenance,
        }

    def to_manifest_summary(self) -> dict[str, Any]:
        """Return the field-for-field GD-3 Khala candidate manifest summary.

        This mirrors `KhalaFleetDelegationCandidateManifestSummary` on the
        Effect side. Refs are normalized to public projection-safe ref strings
        because the admission loop rejects raw `trace://`, `eval://`, local path,
        provider, credential, wallet, and timestamp-shaped values.
        """
        self.validate()
        return {
            "baseModuleRef": _public_ref(self.base_module_ref, prefix="module"),
            "candidateManifestRef": self.candidate_manifest_ref,
            "candidateRef": self.candidate_ref,
            "evalEvidenceRefs": [
                _public_ref(ref, prefix="eval_result") for ref in self.eval_evidence_refs
            ],
            "metricName": self.metric_name,
            "metricValueBps": self.metric_value_bps,
            "optimizedModuleRef": self.optimized_module_ref,
            "schemaVersion": PROBE_GEPA_CANDIDATE_MANIFEST_SCHEMA_VERSION,
            "signature": self.signature,
            "traceProvenanceRefs": [
                _public_ref(ref, prefix="trace.public")
                for ref in self.trace_provenance_refs
            ],
        }

    def to_manifest_json(self) -> str:
        return _stable_json(self.to_manifest_summary())

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "candidateHash": self.candidate_hash,
            "candidateManifest": self.to_manifest_summary(),
            "evalEvidenceRefs": list(self.eval_evidence_refs),
            "manifestHash": _sha256_ref(self.to_manifest_json()),
            "schemaVersion": MUTALISK_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
            **asdict(self),
        }

    def to_json(self) -> str:
        return _stable_json(self.to_dict())


class CandidateEmitter:
    """Writes validated candidates to a sink."""

    def emit(self, candidate: Candidate) -> str:
        candidate.validate()
        raise NotImplementedError("wire the candidate sink in the build-out")
