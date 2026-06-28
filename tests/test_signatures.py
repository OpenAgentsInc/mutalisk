"""DSPy signature/program mirror test, gated behind the dspy dependency."""

import pytest

pytest.importorskip("dspy")

from mutalisk.signatures import build_predict, sentiment_signature  # noqa: E402


def test_sentiment_signature_has_typed_fields():
    sig = sentiment_signature()
    assert "text" in sig.input_fields
    assert "label" in sig.output_fields


def test_build_predict_returns_dspy_module():
    import dspy

    module = build_predict()
    assert isinstance(module, dspy.Predict)
