from __future__ import annotations

import pytest

from utils.languages import (
    choose_allowed_language,
    language_policy_display,
    normalize_allowed_output_languages,
)


pytestmark = pytest.mark.core_headless


def test_language_policy_is_ordered_deduplicated_and_accepts_names() -> None:
    assert normalize_allowed_output_languages(None) == ()
    assert normalize_allowed_output_languages("English, afrikaans, EN") == ("en", "af")
    assert language_policy_display(("en", "af")) == "English (en), Afrikaans (af)"


def test_language_policy_rejects_unknown_or_excessive_values() -> None:
    with pytest.raises(ValueError, match="Unsupported Whisper language"):
        normalize_allowed_output_languages("en,klingon")
    with pytest.raises(ValueError, match="at most 5"):
        normalize_allowed_output_languages("en,af,de,fr,es,it")


def test_multiple_language_selection_uses_allowed_detection_or_first_fallback() -> None:
    assert choose_allowed_language((), "zh") is None
    assert choose_allowed_language(("en",), "zh") == "en"
    assert choose_allowed_language(("en", "af"), "af") == "af"
    assert choose_allowed_language(("en", "af"), "zh") == "en"
