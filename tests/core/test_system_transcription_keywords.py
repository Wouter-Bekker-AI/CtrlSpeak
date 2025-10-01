import sys
import types

import pytest

from tools import keywords
import utils.system as system


pytestmark = pytest.mark.core_headless


@pytest.fixture(autouse=True)
def configure_conversation_keywords():
    keywords.configure_identity_keywords(["assistant", "default"])
    yield
    keywords.configure_identity_keywords([])


@pytest.fixture
def bot_module(monkeypatch):
    stub = types.SimpleNamespace(
        get_active_identity=lambda: "assistant",
        request_goodbye=lambda **_kwargs: True,
        start_bot=lambda **_kwargs: True,
        stop_bot=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "utils.bot_integration", stub)
    return stub


def test_handle_transcription_keyword_stops_active_identity(bot_module):
    calls: list[tuple[str, str | None]] = []

    def _request_goodbye(*, identity: str | None = None, timeout: float = 3.0):
        calls.append(("request", identity))
        return True

    bot_module.request_goodbye = _request_goodbye

    handled, text = system.handle_transcription_keyword("Goodbye Assistant")

    assert handled is True
    assert text == ""
    assert calls == [("request", "assistant")]


def test_handle_transcription_keyword_allows_punctuation(bot_module):
    handled, text = system.handle_transcription_keyword("goodbye, assistant")

    assert handled is True
    assert text == ""


def test_handle_transcription_keyword_falls_back_to_stop(bot_module):
    request_calls: list[tuple[str, str | None]] = []
    stop_calls: list[str] = []

    def _request_goodbye(*, identity: str | None = None, timeout: float = 3.0):
        request_calls.append(("request", identity))
        return False

    def _stop_bot():
        stop_calls.append("stop")

    bot_module.request_goodbye = _request_goodbye
    bot_module.stop_bot = _stop_bot

    handled, text = system.handle_transcription_keyword("goodbye assistant")

    assert handled is True
    assert text == ""
    assert request_calls == [("request", "assistant")]
    assert stop_calls == ["stop"]


def test_handle_transcription_keyword_switches_identity(bot_module):
    bot_module.get_active_identity = lambda: "default"

    request_calls: list[tuple[str, float]] = []
    start_calls: list[str | None] = []

    def _request_goodbye(*, identity: str | None = None, timeout: float = 3.0):
        request_calls.append((identity or None, timeout))
        return True

    def _start_bot(*, identity: str | None = None, **_kwargs):
        start_calls.append(identity)
        return True

    bot_module.request_goodbye = _request_goodbye
    bot_module.start_bot = _start_bot

    handled, text = system.handle_transcription_keyword("Chat with assistant")

    assert handled is True
    assert text == ""
    assert request_calls == [("default", 3.0)]
    assert start_calls == ["assistant"]


def test_handle_transcription_keyword_switches_identity_with_fallback(bot_module):
    bot_module.get_active_identity = lambda: "default"

    request_calls: list[tuple[str, float]] = []
    stop_calls: list[str] = []
    start_calls: list[str | None] = []

    def _request_goodbye(*, identity: str | None = None, timeout: float = 3.0):
        request_calls.append((identity or None, timeout))
        return False

    def _stop_bot():
        stop_calls.append("stop")

    def _start_bot(*, identity: str | None = None, **_kwargs):
        start_calls.append(identity)
        return True

    bot_module.request_goodbye = _request_goodbye
    bot_module.stop_bot = _stop_bot
    bot_module.start_bot = _start_bot

    handled, text = system.handle_transcription_keyword("chat with assistant")

    assert handled is True
    assert text == ""
    assert request_calls == [("default", 3.0)]
    assert stop_calls == ["stop"]
    assert start_calls == ["assistant"]


def test_handle_transcription_keyword_starts_when_no_identity_active(bot_module):
    bot_module.get_active_identity = lambda: None

    start_calls: list[str | None] = []

    def _start_bot(*, identity: str | None = None, **_kwargs):
        start_calls.append(identity)
        return True

    bot_module.start_bot = _start_bot

    handled, text = system.handle_transcription_keyword("chat with assistant")

    assert handled is True
    assert text == ""
    assert start_calls == ["assistant"]


def test_handle_transcription_keyword_supports_fuzzy_chat(bot_module):
    bot_module.get_active_identity = lambda: None

    start_calls: list[str | None] = []

    def _start_bot(*, identity: str | None = None, **_kwargs):
        start_calls.append(identity)
        return True

    bot_module.start_bot = _start_bot

    handled, text = system.handle_transcription_keyword("could you chat was assistant now")

    assert handled is True
    assert text == ""
    assert start_calls == ["assistant"]


def test_handle_transcription_keyword_ignores_start_for_active_identity(bot_module):
    bot_module.get_active_identity = lambda: "assistant"

    start_calls: list[str | None] = []

    def _start_bot(*, identity: str | None = None, **_kwargs):
        start_calls.append(identity)
        return True

    bot_module.start_bot = _start_bot

    handled, text = system.handle_transcription_keyword("chat with Assistant")

    assert handled is True
    assert text == ""
    assert start_calls == []


def test_handle_transcription_keyword_ignores_other_identity(bot_module):
    handled, text = system.handle_transcription_keyword("goodbye default")

    assert handled is False
    assert text == "goodbye default"


def test_handle_transcription_keyword_ignores_when_no_bot_active(bot_module):
    bot_module.get_active_identity = lambda: None

    handled, text = system.handle_transcription_keyword("goodbye assistant")

    assert handled is False
    assert text == "goodbye assistant"
