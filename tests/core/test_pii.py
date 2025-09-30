import importlib

import pytest

pytestmark = pytest.mark.core_headless


def test_redaction_patterns():
    pii = importlib.import_module("utils.pii")
    sample = "Contact me at alice@example.com or +1-555-123-4567"
    redacted = pii.redact_text(sample)
    assert "[REDACTED]" in redacted
    assert "alice@example.com" not in redacted
    assert "+1-555" not in redacted
