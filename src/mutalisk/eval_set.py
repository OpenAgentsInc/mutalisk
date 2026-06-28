"""Tiny in-repo, public-safe eval fixtures.

This is the optimizer's offline training/validation set. It is SYNTHETIC and
public-safe by construction: no customer prompts, no production traces, no
secrets. Real trace/eval ingestion (public-safe provenance only) is a later
build-out task (see docs/ARCHITECTURE.md and the open issues).
"""

from __future__ import annotations

from dataclasses import dataclass

# Stable provenance refs for the in-repo synthetic fixtures. These are honest:
# they point at this fixture module, not at any customer/production data.
FIXTURE_VERSION = "v1"
EVAL_EVIDENCE_REF = f"eval://mutalisk/fixtures/sentiment@{FIXTURE_VERSION}"
TRACE_PROVENANCE_REF = f"trace://mutalisk/fixtures/synthetic-sentiment@{FIXTURE_VERSION}"


@dataclass(frozen=True)
class Example:
    """One labeled example for the toy sentiment task."""

    text: str
    label: str  # "positive" | "negative"


# Synthetic, public-safe sentiment phrases. Deliberately tiny and obvious so the
# optimizer can demonstrably improve a weak seed prompt offline.
_TRAIN: list[Example] = [
    Example("this release is great work", "positive"),
    Example("the build passed and tests are green", "positive"),
    Example("a delightful and fast experience", "positive"),
    Example("clean code, well structured and solid", "positive"),
    Example("the docs are clear and helpful", "positive"),
    Example("this is broken and slow", "negative"),
    Example("the deploy failed with an error", "negative"),
    Example("a confusing and buggy mess", "negative"),
    Example("flaky tests and poor structure", "negative"),
    Example("the latency is terrible and unreliable", "negative"),
]

_VAL: list[Example] = [
    Example("a solid and helpful improvement", "positive"),
    Example("fast, clean and green pipeline", "positive"),
    Example("clear structure with great docs", "positive"),
    Example("broken deploy with a confusing error", "negative"),
    Example("slow, flaky and unreliable", "negative"),
    Example("buggy and terrible experience", "negative"),
]


def trainset() -> list[Example]:
    """Return a copy of the training fixtures."""
    return list(_TRAIN)


def valset() -> list[Example]:
    """Return a copy of the validation fixtures."""
    return list(_VAL)
