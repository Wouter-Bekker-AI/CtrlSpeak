from __future__ import annotations

import pytest

from app.languages import choose_allowed_language, normalize_language_policy


def test_language_policy_preserves_order_and_supports_legacy_single_value() -> None:
    assert normalize_language_policy(None, None) == ()
    assert normalize_language_policy("en, af,en", None) == ("en", "af")
    assert normalize_language_policy(None, "EN") == ("en",)


def test_language_policy_rejects_unknown_excessive_and_conflicting_values() -> None:
    with pytest.raises(ValueError, match="unsupported Whisper language"):
        normalize_language_policy("en,xx", None)
    with pytest.raises(ValueError, match="at most 5"):
        normalize_language_policy("en,af,de,fr,es,it", None)
    with pytest.raises(ValueError, match="must be included"):
        normalize_language_policy("en,af", "de")


def test_allowed_detection_wins_otherwise_first_language_is_the_fallback() -> None:
    assert choose_allowed_language((), "zh") is None
    assert choose_allowed_language(("en",), "zh") == "en"
    assert choose_allowed_language(("en", "af"), "af") == "af"
    assert choose_allowed_language(("en", "af"), "zh") == "en"
