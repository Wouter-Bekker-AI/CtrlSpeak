import pytest

from tools import keywords

pytestmark = pytest.mark.core_headless


def test_detect_vision_keyword_screen():
    match = keywords.detect_vision_keyword("Please look at my screen now")
    assert match is not None
    assert match.keyword.payload == "screen"


def test_detect_vision_keyword_clipboard():
    match = keywords.detect_vision_keyword("Could you look at my clipboard instead?")
    assert match is not None
    assert match.keyword.payload == "clipboard"


def test_detect_vision_keyword_clipboard_fuzzy_variants():
    for phrase in [
        "look at my slipboard",
        "look at my clupboard",
        "look at my clip board",
    ]:
        match = keywords.detect_vision_keyword(phrase)
        assert match is not None, phrase
        assert match.keyword.payload == "clipboard"


def test_detect_vision_keyword_clipboard_rejects_different_words():
    match = keywords.detect_vision_keyword("please look at my keyboard")
    assert match is None


def test_detect_vision_keyword_empty():
    match = keywords.detect_vision_keyword("   ")
    assert match is None


def test_detect_memory_refresh_keyword_variants():
    match = keywords.detect_memory_refresh_keyword("could you update documentation now?")
    assert match is not None
    assert match.keyword.payload == "documentation"

    fuzzy = keywords.detect_memory_refresh_keyword("please refresh documentation for me")
    assert fuzzy is not None
    assert fuzzy.keyword.payload == "documentation"


def test_detect_memory_refresh_keyword_datetime_variants():
    match = keywords.detect_memory_refresh_keyword("could you update datetime now?")
    assert match is not None
    assert match.keyword.payload == "datetime"

    fuzzy = keywords.detect_memory_refresh_keyword("please refresh date time for me")
    assert fuzzy is not None
    assert fuzzy.keyword.payload == "datetime"


def test_detect_memory_refresh_keyword_ignores_other_phrases():
    match = keywords.detect_memory_refresh_keyword("update the itinerary")
    assert match is None


def test_get_vision_keyword():
    keyword = keywords.get_vision_keyword("SCREEN")
    assert keyword is not None
    assert keyword.payload == "screen"


def test_iter_keyword_matches_multiple():
    text = "look at my screen and also look at my clipboard"
    matches = list(keywords.iter_keyword_matches(text, keywords.VISION_KEYWORDS))
    assert [m.keyword.payload for m in matches] == ["screen", "clipboard"]


def test_configure_identity_keywords_registers_conversation_triggers():
    keywords.configure_identity_keywords(["vision", "reception", "einstein"])

    start_match = keywords.detect_conversation_start_keyword("please chat with vision right now")
    assert start_match is not None
    assert start_match.keyword.payload == "vision"

    reception_match = keywords.detect_conversation_start_keyword("chat with receptionist for me")
    assert reception_match is not None
    assert reception_match.keyword.payload == "reception"

    end_match = keywords.detect_conversation_end_keyword("goodbye reception")
    assert end_match is not None
    assert end_match.keyword.payload == "reception"

    einstein_match = keywords.detect_conversation_start_keyword("chat with einstein for me")
    assert einstein_match is not None
    assert einstein_match.keyword.payload == "einstein"

    keywords.configure_identity_keywords([])


def test_configure_identity_keywords_handles_duplicates_and_spacing():
    keywords.configure_identity_keywords(["Vision", "vision", "my_helper_bot"])

    start_match = keywords.detect_conversation_start_keyword("chat with my helper bot")
    assert start_match is not None
    assert start_match.keyword.payload == "my_helper_bot"

    assert keywords.detect_conversation_end_keyword("goodbye vision") is not None

    keywords.configure_identity_keywords([])


def test_conversation_keywords_support_fuzzy_variants():
    keywords.configure_identity_keywords(["reception"])
    try:
        assert keywords.detect_conversation_end_keyword("goodbye, receptionist") is not None
        start_match = keywords.detect_conversation_start_keyword("please chat was receptionist today")
        assert start_match is not None
        assert start_match.keyword.payload == "reception"
    finally:
        keywords.configure_identity_keywords([])
