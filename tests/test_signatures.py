"""DSPy signature/program mirror test, gated behind the dspy dependency."""

import pytest

pytest.importorskip("dspy")

from mutalisk.signatures import (  # noqa: E402
    build_khala_fleet_delegate_predict,
    build_predict,
    khala_fleet_delegate_signature,
    sentiment_signature,
)


def test_sentiment_signature_has_typed_fields():
    sig = sentiment_signature()
    assert "text" in sig.input_fields
    assert "label" in sig.output_fields


def test_build_predict_returns_dspy_module():
    import dspy

    module = build_predict()
    assert isinstance(module, dspy.Predict)


def test_khala_fleet_delegate_signature_exposes_policy_fields():
    sig = khala_fleet_delegate_signature()
    assert "delegation_example" in sig.input_fields
    assert "objective_template" in sig.output_fields
    assert "verifier_selection" in sig.output_fields
    assert "dispatch_policy" in sig.output_fields
    assert "merge_resolution_template" in sig.output_fields


def test_build_khala_fleet_delegate_predict_returns_dspy_module():
    import dspy

    module = build_khala_fleet_delegate_predict()
    assert isinstance(module, dspy.Predict)
