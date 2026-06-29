"""Public-safe executed trace/eval ingestion.

Mutalisk optimizes over executed traces and evals, but candidate artifacts must
carry provenance and evidence refs rather than raw records. This module accepts
sanitized JSONL records and turns them into optimizer examples plus the refs the
candidate emitter needs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from .eval_set import Example

Split = Literal["train", "val"]

_FORBIDDEN_RAW_FIELDS = frozenset(
    {
        "prompt",
        "raw_prompt",
        "customer_prompt",
        "private_source",
        "secret",
        "credential",
        "wallet_material",
    }
)


@dataclass(frozen=True)
class TraceEvalRecord:
    """One sanitized executed trace + eval result for optimizer input."""

    public_text: str
    label: str
    split: Split
    trace_ref: str
    eval_ref: str

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "TraceEvalRecord":
        forbidden = _FORBIDDEN_RAW_FIELDS.intersection(raw)
        if forbidden:
            fields = ", ".join(sorted(forbidden))
            raise ValueError(f"trace/eval record includes forbidden raw field(s): {fields}")

        public_text = _require_string(raw, "public_text")
        label = _require_string(raw, "label")
        split = _require_string(raw, "split")
        trace_ref = _require_string(raw, "trace_ref")
        eval_ref = _require_string(raw, "eval_ref")

        if label not in {"positive", "negative"}:
            raise ValueError("trace/eval record label must be positive or negative")
        if split not in {"train", "val"}:
            raise ValueError("trace/eval record split must be train or val")

        return cls(
            public_text=public_text,
            label=label,
            split=split,  # type: ignore[arg-type]
            trace_ref=trace_ref,
            eval_ref=eval_ref,
        )

    def to_example(self) -> Example:
        return Example(text=self.public_text, label=self.label)


@dataclass(frozen=True)
class TraceEvalDataset:
    """A public-safe dataset derived from executed trace/eval records."""

    records: tuple[TraceEvalRecord, ...]
    dataset_ref: str
    source: str

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TraceEvalDataset":
        raw_bytes = Path(path).read_bytes()
        digest = hashlib.sha256(raw_bytes).hexdigest()
        records: list[TraceEvalRecord] = []
        for line_no, line in enumerate(raw_bytes.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL on line {line_no}") from exc
            if not isinstance(decoded, dict):
                raise ValueError(f"trace/eval JSONL line {line_no} must be an object")
            records.append(TraceEvalRecord.from_mapping(decoded))

        dataset = cls(
            records=tuple(records),
            dataset_ref=f"trace-eval://sha256:{digest}",
            source=f"trace-eval-jsonl@sha256:{digest[:12]}",
        )
        dataset.validate()
        return dataset

    def validate(self) -> None:
        if not self.records:
            raise ValueError("trace/eval dataset is empty")
        if not any(record.split == "train" for record in self.records):
            raise ValueError("trace/eval dataset missing train split")
        if not any(record.split == "val" for record in self.records):
            raise ValueError("trace/eval dataset missing val split")
        if not self.eval_evidence_refs:
            raise ValueError("trace/eval dataset missing eval evidence refs")
        if not self.trace_refs:
            raise ValueError("trace/eval dataset missing trace provenance refs")

    @property
    def trainset(self) -> list[Example]:
        return [record.to_example() for record in self.records if record.split == "train"]

    @property
    def valset(self) -> list[Example]:
        return [record.to_example() for record in self.records if record.split == "val"]

    @property
    def eval_evidence_refs(self) -> list[str]:
        return _dedupe(record.eval_ref for record in self.records)

    @property
    def trace_refs(self) -> list[str]:
        return _dedupe(record.trace_ref for record in self.records)

    @property
    def trace_provenance(self) -> dict[str, object]:
        return {
            "refs": self.trace_refs,
            "source": self.source,
            "record_count": len(self.records),
        }


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _require_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"trace/eval record missing {key}")
    return value
