"""Recording-friendly demo runners.

The demo surface is intentionally a thin wrapper over the same Candidate
contract used by production-intent batch jobs. It stays offline: no LM, no
network, no OpenAgents writes.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Any

from .candidate import Candidate
from .delegation import (
    FEATURE_BLOCKERS,
    FeatureName,
    KHALA_FLEET_DELEGATION_CANDIDATE_SIGNATURE_ID,
    DelegationExample,
    run_delegation_program,
)
from .emitter import FileCandidateEmitter
from .optimizer import DelegationGepaOptimizer
from .runner import (
    DELEGATION_BASE_MODULE_REF,
    build_candidate,
    _delegation_refs,
)

KHALA_FLEET_DELEGATION_DEMO_DATASET_SCHEMA_VERSION = (
    "mutalisk.khala_fleet_delegation_demo_dataset.v1"
)
KHALA_FLEET_DELEGATION_FEEDBACK_DIMENSIONS_SCHEMA_VERSION = (
    "mutalisk.khala_fleet_delegation_feedback_dimensions.v1"
)
DEFAULT_KHALA_FLEET_DELEGATION_DEMO_DATASET = Path(
    "fixtures/khala_fleet_delegation_demo.json"
)
DEFAULT_KHALA_FLEET_DELEGATION_SUMMARY = Path(
    "out/khala-fleet-delegation-summary.json"
)

_DIMENSION_FEATURES: dict[str, tuple[FeatureName, ...]] = {
    "capacity_recovery": (
        "dispatch_per_account_capacity",
        "dispatch_no_capacity_advertise",
        "dispatch_stale_heartbeat_retry",
    ),
    "typed_blockers": (
        "dispatch_duplicate_backoff",
        "dispatch_load_gate",
    ),
    "verifier_closeout": (
        "objective_issue_repo_verify",
        "verifier_required_for_repo",
    ),
    "merge_recovery": ("merge_rebase_preserve_verify",),
}


@dataclass(frozen=True)
class KhalaFleetDelegationDemoDataset:
    dataset_ref: str
    train: list[DelegationExample]
    val: list[DelegationExample]

    @property
    def examples(self) -> list[DelegationExample]:
        return [*self.train, *self.val]


@dataclass(frozen=True)
class KhalaFleetDelegationDemoOutput:
    candidate: Candidate
    candidate_artifact_path: Path
    summary_path: Path
    summary: dict[str, Any]


def load_khala_fleet_delegation_demo_dataset(
    dataset_path: str | Path,
) -> KhalaFleetDelegationDemoDataset:
    """Load the public-safe JSON fixture dataset used by the Part 2 demo."""
    path = Path(dataset_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != KHALA_FLEET_DELEGATION_DEMO_DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported Khala fleet delegation demo dataset schema")
    dataset_ref = payload.get("datasetRef")
    if not isinstance(dataset_ref, str) or not dataset_ref:
        raise ValueError("demo dataset missing datasetRef")
    raw_examples = payload.get("examples")
    if not isinstance(raw_examples, list) or not raw_examples:
        raise ValueError("demo dataset missing examples")

    train: list[DelegationExample] = []
    val: list[DelegationExample] = []
    for idx, raw in enumerate(raw_examples):
        if not isinstance(raw, dict):
            raise ValueError(f"demo dataset example {idx} must be an object")
        split = raw.get("split")
        if split not in ("train", "val"):
            raise ValueError(f"demo dataset example {idx} has invalid split")
        required_features = raw.get("requiredFeatures")
        if not isinstance(required_features, list):
            raise ValueError(f"demo dataset example {idx} missing requiredFeatures")
        unknown_features = [
            item for item in required_features if item not in FEATURE_BLOCKERS
        ]
        if unknown_features:
            raise ValueError(
                f"demo dataset example {idx} has unknown requiredFeatures: "
                f"{', '.join(str(item) for item in unknown_features)}"
            )
        example = DelegationExample(
            capacity_context=_require_str(raw, "capacityContext", idx),
            eval_ref=_require_str(raw, "evalRef", idx),
            example_ref=_require_str(raw, "exampleRef", idx),
            issue=_require_str(raw, "issue", idx),
            objective=_require_str(raw, "objective", idx),
            repo=_require_str(raw, "repo", idx),
            required_features=tuple(required_features),  # type: ignore[arg-type]
            trace_ref=_require_str(raw, "traceRef", idx),
            verify=_require_str(raw, "verify", idx),
        )
        if split == "train":
            train.append(example)
        else:
            val.append(example)

    if not train or not val:
        raise ValueError("demo dataset must contain train and val examples")
    return KhalaFleetDelegationDemoDataset(
        dataset_ref=dataset_ref,
        train=train,
        val=val,
    )


def run_khala_fleet_delegation_demo(
    *,
    dataset_path: str | Path = DEFAULT_KHALA_FLEET_DELEGATION_DEMO_DATASET,
    summary_path: str | Path = DEFAULT_KHALA_FLEET_DELEGATION_SUMMARY,
    candidate_out_dir: str | Path | None = None,
    max_metric_calls: int = 8,
    seed: int = 0,
) -> KhalaFleetDelegationDemoOutput:
    """Run the bounded Part 2 demo and emit both artifact and manifest summary."""
    dataset = load_khala_fleet_delegation_demo_dataset(dataset_path)
    optimizer = DelegationGepaOptimizer(max_metric_calls=max_metric_calls, seed=seed)
    with redirect_stdout(io.StringIO()):
        result = optimizer.optimize(
            dict(_seed_from_runner()),
            dataset.train,
            dataset.val,
        )
    eval_refs, trace_provenance = _delegation_refs(dataset.examples)
    feedback_dimensions = _feedback_dimensions(result.optimized_candidate, dataset.examples)
    candidate = build_candidate(
        result,
        signature_id=KHALA_FLEET_DELEGATION_CANDIDATE_SIGNATURE_ID,
        base_module_ref=DELEGATION_BASE_MODULE_REF,
        base_candidate=dict(_seed_from_runner()),
        eval_dataset_ref=dataset.dataset_ref,
        eval_evidence_refs=eval_refs,
        metric_extras={
            "feedback_dimensions": feedback_dimensions,
            "max_metric_calls": max_metric_calls,
        },
        trace_provenance=trace_provenance,
    )

    summary_out = Path(summary_path)
    artifact_dir = Path(candidate_out_dir) if candidate_out_dir else summary_out.parent / "candidates"
    candidate_artifact = Path(FileCandidateEmitter(artifact_dir).emit(candidate))
    summary = candidate.to_manifest_summary()
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return KhalaFleetDelegationDemoOutput(
        candidate=candidate,
        candidate_artifact_path=candidate_artifact,
        summary_path=summary_out,
        summary=summary,
    )


def _require_str(raw: dict[str, Any], name: str, idx: int) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"demo dataset example {idx} missing {name}")
    return value


def _seed_from_runner() -> dict[str, str]:
    from .delegation import DELEGATION_SEED_CANDIDATE

    return dict(DELEGATION_SEED_CANDIDATE)


def _feedback_dimensions(
    candidate: dict[str, str],
    examples: list[DelegationExample],
) -> dict[str, Any]:
    runs = [(example, run_delegation_program(candidate, example)) for example in examples]
    dimensions: list[dict[str, Any]] = []
    for name, features in _DIMENSION_FEATURES.items():
        feature_set = set(features)
        relevant: list[tuple[DelegationExample, set[FeatureName], set[FeatureName]]] = []
        for example, run in runs:
            required = set(example.required_features).intersection(feature_set)
            if not required:
                continue
            satisfied = set(run.satisfied_features).intersection(required)
            relevant.append((example, required, satisfied))
        if not relevant:
            continue
        score = sum(
            len(satisfied) / len(required)
            for _, required, satisfied in relevant
        ) / len(relevant)
        missing_features = sorted(
            feature
            for _, required, satisfied in relevant
            for feature in required.difference(satisfied)
        )
        dimensions.append(
            {
                "blockerRefs": sorted(
                    dict.fromkeys(FEATURE_BLOCKERS[feature] for feature in missing_features)
                ),
                "evalEvidenceRefs": sorted(
                    dict.fromkeys(example.eval_ref for example, _, _ in relevant)
                ),
                "metricValueBps": int(round(score * 10_000)),
                "name": name,
                "traceProvenanceRefs": sorted(
                    dict.fromkeys(example.trace_ref for example, _, _ in relevant)
                ),
            }
        )
    return {
        "dimensions": dimensions,
        "schemaVersion": KHALA_FLEET_DELEGATION_FEEDBACK_DIMENSIONS_SCHEMA_VERSION,
    }
