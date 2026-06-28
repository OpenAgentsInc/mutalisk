"""DSPy signature/program mirror for the offline sentiment task.

This is the real DSPy surface: a typed `dspy.Signature` (the conceptual cousin
of a Blueprint Program Signature) plus a `dspy.Predict` module builder. It is
illustrative of the program registry described in docs/ARCHITECTURE.md
(component 1). Executing the Predict module requires an LM to be configured
(`dspy.configure(lm=...)`), so mutalisk's *offline* optimization searches over
the zero-dependency program in `mutalisk.program` instead. dspy is imported
lazily so the rest of mutalisk stays import-light and offline-testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# The Blueprint-style signature id stamped onto emitted candidates.
SENTIMENT_SIGNATURE_ID = "mutalisk.sentiment_classify.v1"

if TYPE_CHECKING:  # pragma: no cover - typing only
    import dspy


def sentiment_signature() -> type[Any]:
    """Build the DSPy Signature class for the sentiment task (lazy import)."""
    import dspy

    class SentimentSignature(dspy.Signature):
        """Classify the sentiment of a short, public-safe phrase."""

        text: str = dspy.InputField(desc="a short phrase to classify")
        label: str = dspy.OutputField(desc='either "positive" or "negative"')

    return SentimentSignature


def build_predict() -> "dspy.Module":
    """Return a `dspy.Predict` module over the sentiment signature (lazy import).

    Requires an LM to be configured before calling the returned module; mutalisk
    does not configure one (offline, no provider credentials).
    """
    import dspy

    return dspy.Predict(sentiment_signature())
