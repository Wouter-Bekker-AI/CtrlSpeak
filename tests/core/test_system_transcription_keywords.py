import sys
import types

import pytest

from tools import keywords
import utils.system as system


pytestmark = pytest.mark.core_headless


@pytest.fixture(autouse=True)
def configure_conversation_keywords():
    keywords.configure_identity_keywords(["vision", "reception"])
    yield
    keywords.configure_identity_keywords([])


@pytest.fixture
def bot_module(monkeypatch):
    stub = types.SimpleNamespace(
        get_active_identity=lambda: "vision",
        request_goodbye=lambda **_kwargs: True,
        start_bot=lambda **_kwargs: True,
        stop_bot=lambda: None,
    )
    goodbye_calls: list[str] = []

    def _record_goodbye(identity: str) -> None:
        goodbye_calls.append(identity)

    monkeypatch.setattr(system, "_speak_conversation_goodbye", _record_goodbye)
    stub.goodbye_calls = goodbye_calls
    monkeypatch.setitem(sys.modules, "utils.bot_integration", stub)
    return stub


def test_handle_transcription_keyword_stops_active_identity(bot_module):
    calls: list[tuple[str, str | None]] = []

    def _request_goodbye(*, identity: str | None = None, timeout: float = 3.0):
        calls.append(("request", identity))
        return True

    bot_module.request_goodbye = _request_goodbye

    handled, text = system.handle_transcription_keyword("Goodbye Vision")

    assert handled is True
    assert text == ""
    assert calls == [("request", "vision")]
    assert bot_module.goodbye_calls == ["vision"]


def test_handle_transcription_keyword_allows_punctuation(bot_module):
    handled, text = system.handle_transcription_keyword("goodbye, vision")

    assert handled is True
    assert text == ""
    assert bot_module.goodbye_calls == ["vision"]


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

    handled, text = system.handle_transcription_keyword("goodbye vision")

    assert handled is True
    assert text == ""
    assert request_calls == [("request", "vision")]
    assert stop_calls == ["stop"]
    assert bot_module.goodbye_calls == ["vision"]


def test_handle_transcription_keyword_switches_identity(bot_module):
    bot_module.get_active_identity = lambda: "reception"

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

    handled, text = system.handle_transcription_keyword("Chat with vision")

    assert handled is True
    assert text == ""
    assert request_calls == [("reception", 3.0)]
    assert start_calls == ["vision"]
    assert bot_module.goodbye_calls == []


def test_handle_transcription_keyword_switches_identity_with_fallback(bot_module):
    bot_module.get_active_identity = lambda: "reception"

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

    handled, text = system.handle_transcription_keyword("chat with vision")

    assert handled is True
    assert text == ""
    assert request_calls == [("reception", 3.0)]
    assert stop_calls == ["stop"]
    assert start_calls == ["vision"]
    assert bot_module.goodbye_calls == []


def test_handle_transcription_keyword_starts_when_no_identity_active(bot_module):
    bot_module.get_active_identity = lambda: None

    start_calls: list[str | None] = []

    def _start_bot(*, identity: str | None = None, **_kwargs):
        start_calls.append(identity)
        return True

    bot_module.start_bot = _start_bot

    handled, text = system.handle_transcription_keyword("chat with vision")

    assert handled is True
    assert text == ""
    assert start_calls == ["vision"]
    assert bot_module.goodbye_calls == []


def test_handle_transcription_keyword_supports_fuzzy_chat(bot_module):
    bot_module.get_active_identity = lambda: None

    start_calls: list[str | None] = []

    def _start_bot(*, identity: str | None = None, **_kwargs):
        start_calls.append(identity)
        return True

    bot_module.start_bot = _start_bot

    handled, text = system.handle_transcription_keyword("could you chat was vision now")

    assert handled is True
    assert text == ""
    assert start_calls == ["vision"]
    assert bot_module.goodbye_calls == []


def test_handle_transcription_keyword_normalizes_defunct(bot_module):
    bot_module.get_active_identity = lambda: None

    start_calls: list[str | None] = []

    def _start_bot(*, identity: str | None = None, **_kwargs):
        start_calls.append(identity)
        return True

    bot_module.start_bot = _start_bot

    handled, text = system.handle_transcription_keyword("chat with defunct")

    assert handled is True
    assert text == ""
    assert start_calls == ["reception"]
    assert bot_module.goodbye_calls == []


def test_handle_transcription_keyword_quit_requests_shutdown(monkeypatch, bot_module):
    calls: list[str] = []
    goodbye_calls: list[int] = []

    def _request(reason: str = "unspecified") -> None:
        calls.append(reason)

    monkeypatch.setattr(system, "_speak_lobby_goodbye", lambda: goodbye_calls.append(1))
    monkeypatch.setattr(system, "request_application_shutdown", _request)
    bot_module.get_active_identity = lambda: None

    handled, text = system.handle_transcription_keyword("Quit Control Speak")

    assert handled is True
    assert text == ""
    assert calls == ["transcription keyword"]
    assert goodbye_calls == [1]
    assert bot_module.goodbye_calls == []


def test_handle_transcription_keyword_quit_ignored_in_conversation(monkeypatch, bot_module):
    calls: list[str] = []
    goodbye_calls: list[int] = []

    def _request(reason: str = "unspecified") -> None:
        calls.append(reason)

    monkeypatch.setattr(system, "_speak_lobby_goodbye", lambda: goodbye_calls.append(1))
    monkeypatch.setattr(system, "request_application_shutdown", _request)
    bot_module.get_active_identity = lambda: "vision"

    handled, text = system.handle_transcription_keyword("Quit Control Speak")

    assert handled is False
    assert text.lower() == "quit control speak"
    assert calls == []
    assert goodbye_calls == []
    assert bot_module.goodbye_calls == []


def test_handle_transcription_keyword_ignores_start_for_active_identity(bot_module):
    bot_module.get_active_identity = lambda: "vision"

    start_calls: list[str | None] = []

    def _start_bot(*, identity: str | None = None, **_kwargs):
        start_calls.append(identity)
        return True

    bot_module.start_bot = _start_bot

    handled, text = system.handle_transcription_keyword("chat with Assistant")

    assert handled is True
    assert text == ""
    assert start_calls == []
    assert bot_module.goodbye_calls == []


def test_handle_transcription_keyword_ignores_other_identity(bot_module):
    handled, text = system.handle_transcription_keyword("goodbye reception")

    assert handled is False
    assert text == "goodbye reception"
    assert bot_module.goodbye_calls == []


def test_handle_transcription_keyword_ignores_when_no_bot_active(bot_module):
    bot_module.get_active_identity = lambda: None

    handled, text = system.handle_transcription_keyword("goodbye vision")

    assert handled is False
    assert text == "goodbye vision"
    assert bot_module.goodbye_calls == []


def test_handle_transcription_keyword_returns_normalized_text():
    handled, text = system.handle_transcription_keyword("please look at my clubboard")

    assert handled is False
    assert text == "please look at my clipboard"
