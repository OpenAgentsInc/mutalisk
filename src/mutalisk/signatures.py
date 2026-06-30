"""DSPy signature/program mirrors for Mutalisk offline tasks.

These are the real DSPy surfaces: typed `dspy.Signature` classes (the
conceptual cousin of Blueprint Program Signatures) plus `dspy.Predict` module
builders. Executing Predict modules requires an LM to be configured
(`dspy.configure(lm=...)`), so Mutalisk's offline optimization searches over the
zero-dependency substrates in `mutalisk.program` and `mutalisk.delegation`
instead. dspy is imported lazily so the rest of mutalisk stays import-light and
offline-testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# The Blueprint-style signature id stamped onto emitted candidates.
SENTIMENT_SIGNATURE_ID = "mutalisk.sentiment_classify.v1"
KHALA_FLEET_DELEGATION_SIGNATURE_ID = "khala.fleet.delegation"
KHALA_FLEET_DELEGATE_PROGRAM_ID = "khala.fleet.delegate"

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


def khala_fleet_delegate_signature() -> type[Any]:
    """Build the DSPy Signature for Khala fleet-delegation policy search.

    The control-flow modules are fixed online (`ensure_pylon ->
    advertise_capacity -> select_account -> prepare_work -> dispatch ->
    verify_closeout`). The optimizable outputs here are the policy/text
    components that Mutalisk proposes as an untrusted candidate.
    """
    import dspy

    class KhalaFleetDelegateSignature(dspy.Signature):
        """Propose public-safe parameters for the Khala fleet delegate program."""

        delegation_example: str = dspy.InputField(
            desc="public-safe GD-0 delegation example summary and metric feedback"
        )
        objective_template: str = dspy.OutputField(
            desc="worker objective template using {objective}, {issue}, {repo}, and {verify}"
        )
        verifier_selection: str = dspy.OutputField(
            desc="public-safe verifier selection policy for repo and fixture work"
        )
        dispatch_policy: str = dspy.OutputField(
            desc=(
                "capacity advertisement, account ranking, heartbeat, retry, "
                "duplicate-assignment backoff, and load-gating policy"
            )
        )
        merge_resolution_template: str = dspy.OutputField(
            desc="public-safe conflict-resolution worker prompt template"
        )

    return KhalaFleetDelegateSignature


def build_khala_fleet_delegate_predict() -> "dspy.Module":
    """Return a `dspy.Predict` module for fleet-delegation policy search."""
    import dspy

    return dspy.Predict(khala_fleet_delegate_signature())
