"""The offline program under optimization (zero-dependency, fully offline).

The "program" is a tiny, deterministic sentiment classifier whose behaviour is
entirely a function of an optimizable text candidate -- a mapping from named
component -> component text, exactly the shape GEPA optimizes
(`dict[str, str]`). The two components are comma-separated cue-word lists:

    {"positive_cues": "great, good", "negative_cues": "bad, broken"}

This keeps optimization real and offline: no LLM, no network, deterministic.
A *real* DSPy program/signature mirror (for when an LM is configured) lives in
`mutalisk.signatures`; this module is the offline-runnable substrate the
optimizer actually searches over.
"""

from __future__ import annotations

from collections.abc import Iterable

from .eval_set import Example

# Component names of the candidate the optimizer searches over.
POSITIVE_COMPONENT = "positive_cues"
NEGATIVE_COMPONENT = "negative_cues"

# A deliberately weak seed so optimization has room to improve offline.
SEED_CANDIDATE: dict[str, str] = {
    POSITIVE_COMPONENT: "good",
    NEGATIVE_COMPONENT: "bad",
}

POSITIVE = "positive"
NEGATIVE = "negative"


def parse_cues(text: str) -> list[str]:
    """Parse a comma-separated cue list into normalized tokens."""
    return [tok.strip().lower() for tok in text.split(",") if tok.strip()]


def _tokens(text: str) -> set[str]:
    return {tok.strip(".,!?;:").lower() for tok in text.split() if tok.strip()}


def classify(candidate: dict[str, str], text: str) -> str:
    """Classify one text given a candidate. Deterministic, never raises."""
    words = _tokens(text)
    pos = sum(1 for cue in parse_cues(candidate.get(POSITIVE_COMPONENT, "")) if cue in words)
    neg = sum(1 for cue in parse_cues(candidate.get(NEGATIVE_COMPONENT, "")) if cue in words)
    # Tie / no signal -> negative (fail-closed toward the non-positive label).
    return POSITIVE if pos > neg else NEGATIVE


def score_example(candidate: dict[str, str], example: Example) -> float:
    """Per-example score in [0, 1]; higher is better (GEPA convention)."""
    return 1.0 if classify(candidate, example.text) == example.label else 0.0


def accuracy(candidate: dict[str, str], examples: Iterable[Example]) -> float:
    """Mean accuracy of a candidate over examples."""
    items = list(examples)
    if not items:
        return 0.0
    return sum(score_example(candidate, ex) for ex in items) / len(items)
