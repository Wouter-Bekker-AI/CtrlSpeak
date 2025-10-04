"""Tests for keyword handling in the push-to-talk workflow."""

from __future__ import annotations

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
        get_active_identity=lambda: None,
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


def test_hotkey_start_launches_requested_identity(bot_module):
    calls: list[tuple[str, str | None]] = []

    bot_module.get_active_identity = lambda: None
    bot_module.stop_bot = lambda: calls.append(("stop", None))

    def _start_bot(*, identity: str | None = None, **_kwargs) -> bool:
        calls.append(("start", identity))
        return True

    bot_module.start_bot = _start_bot

    handled = system.handle_transcribed_text_from_hotkey("Chat with Vision")

    assert handled is True
    assert calls == [("start", "vision")]


def test_hotkey_start_ignores_active_identity(bot_module):
    calls: list[tuple[str, str | None]] = []

    bot_module.get_active_identity = lambda: "vision"
    bot_module.stop_bot = lambda: calls.append(("stop", None))

    def _start_bot(**_kwargs):
        calls.append(("start", _kwargs.get("identity")))
        return True

    bot_module.start_bot = _start_bot

    handled = system.handle_transcribed_text_from_hotkey("chat with vision")

    assert handled is True
    assert calls == []


def test_hotkey_switches_identity_after_stopping_current(bot_module):
    sequence: list[str] = []

    bot_module.get_active_identity = lambda: "vision"
    bot_module.stop_bot = lambda: sequence.append("stop")

    def _start_bot(*, identity: str | None = None, **_kwargs) -> bool:
        sequence.append(f"start:{identity}")
        return True

    bot_module.start_bot = _start_bot

    handled = system.handle_transcribed_text_from_hotkey("chat with reception")

    assert handled is True
    assert sequence == ["stop", "start:reception"]


def test_hotkey_goodbye_stops_active_identity(bot_module):
    stop_calls: list[str] = []
    start_calls: list[str] = []

    bot_module.get_active_identity = lambda: "vision"
    bot_module.stop_bot = lambda: stop_calls.append("stop")
    bot_module.start_bot = lambda **_kwargs: start_calls.append(_kwargs.get("identity")) or True

    handled = system.handle_transcribed_text_from_hotkey("goodbye vision")

    assert handled is True
    assert stop_calls == ["stop"]
    assert start_calls == []
    assert bot_module.goodbye_calls == ["vision"]


def test_hotkey_goodbye_handles_punctuation(bot_module):
    stop_calls: list[str] = []

    bot_module.get_active_identity = lambda: "vision"
    bot_module.stop_bot = lambda: stop_calls.append("stop")

    handled = system.handle_transcribed_text_from_hotkey("goodbye, vision!")

    assert handled is True
    assert stop_calls == ["stop"]
    assert bot_module.goodbye_calls == ["vision"]


def test_hotkey_goodbye_for_other_identity_is_ignored(bot_module):
    stop_calls: list[str] = []
    start_calls: list[str] = []

    bot_module.get_active_identity = lambda: "vision"
    bot_module.stop_bot = lambda: stop_calls.append("stop")
    bot_module.start_bot = lambda **_kwargs: start_calls.append(_kwargs.get("identity")) or True

    handled = system.handle_transcribed_text_from_hotkey("goodbye reception")

    assert handled is True
    assert stop_calls == []
    assert start_calls == []
    assert bot_module.goodbye_calls == []


def test_hotkey_non_keyword_returns_false(bot_module):
    stop_calls: list[str] = []
    start_calls: list[str] = []

    bot_module.get_active_identity = lambda: None
    bot_module.stop_bot = lambda: stop_calls.append("stop")
    bot_module.start_bot = lambda **_kwargs: start_calls.append(_kwargs.get("identity")) or True

    handled = system.handle_transcribed_text_from_hotkey("hello world")

    assert handled is False
    assert stop_calls == []
    assert start_calls == []
    assert bot_module.goodbye_calls == []


def test_hotkey_supports_fuzzy_chat(bot_module):
    start_calls: list[str] = []

    bot_module.get_active_identity = lambda: None

    def _start_bot(*, identity: str | None = None, **_kwargs) -> bool:
        start_calls.append(identity)
        return True

    bot_module.start_bot = _start_bot

    handled = system.handle_transcribed_text_from_hotkey("please chat was vision right now")

    assert handled is True
    assert start_calls == ["vision"]
    assert bot_module.goodbye_calls == []


def test_hotkey_normalizes_defunct_identity(bot_module):
    start_calls: list[str] = []

    bot_module.get_active_identity = lambda: None

    def _start_bot(*, identity: str | None = None, **_kwargs) -> bool:
        start_calls.append(identity)
        return True

    bot_module.start_bot = _start_bot

    handled = system.handle_transcribed_text_from_hotkey("chat with defunct")

    assert handled is True
    assert start_calls == ["reception"]
    assert bot_module.goodbye_calls == []


def test_hotkey_quit_requests_shutdown(monkeypatch, bot_module):
    calls: list[str] = []
    goodbye_calls: list[int] = []

    def _request(reason: str = "unspecified") -> None:
        calls.append(reason)

    monkeypatch.setattr(system, "_speak_lobby_goodbye", lambda: goodbye_calls.append(1))
    monkeypatch.setattr(system, "request_application_shutdown", _request)

    handled = system.handle_transcribed_text_from_hotkey("Quit Control Speak")

    assert handled is True
    assert calls == ["hotkey keyword"]
    assert goodbye_calls == [1]
    assert bot_module.goodbye_calls == []


def test_hotkey_quit_ignored_during_conversation(monkeypatch, bot_module):
    calls: list[str] = []
    goodbye_calls: list[int] = []

    bot_module.get_active_identity = lambda: "vision"

    def _request(reason: str = "unspecified") -> None:
        calls.append(reason)

    monkeypatch.setattr(system, "_speak_lobby_goodbye", lambda: goodbye_calls.append(1))
    monkeypatch.setattr(system, "request_application_shutdown", _request)

    handled = system.handle_transcribed_text_from_hotkey("Quit Control Speak")

    assert handled is False
    assert calls == []
    assert goodbye_calls == []
    assert bot_module.goodbye_calls == []


def test_hotkey_update_documentation_uses_active_identity(monkeypatch, bot_module):
    calls: list[tuple[str, bool, str | None]] = []

    def _refresh(identity: str, *, force: bool = False, reason: str | None = None) -> bool:
        calls.append((identity, force, reason))
        return True

    stub_module = types.SimpleNamespace(refresh_document_memory=_refresh)
    monkeypatch.setitem(sys.modules, "background_agents.document_memory_agent", stub_module)
    monkeypatch.setitem(
        sys.modules,
        "background_agents.datetime_memory_agent",
        types.SimpleNamespace(refresh_datetime_memory=lambda *args, **kwargs: True),
    )
    bot_module.get_active_identity = lambda: "reception"

    handled = system.handle_transcribed_text_from_hotkey("update documentation")

    assert handled is True
    assert calls == [("reception", True, "hotkey")]


def test_hotkey_update_datetime_uses_active_identity(monkeypatch, bot_module):
    calls: list[tuple[str, bool, str | None]] = []

    def _refresh(identity: str, *, force: bool = False, reason: str | None = None) -> bool:
        calls.append((identity, force, reason))
        return True

    monkeypatch.setitem(
        sys.modules,
        "background_agents.datetime_memory_agent",
        types.SimpleNamespace(refresh_datetime_memory=_refresh),
    )
    monkeypatch.setitem(
        sys.modules,
        "background_agents.document_memory_agent",
        types.SimpleNamespace(refresh_document_memory=lambda *args, **kwargs: True),
    )
    bot_module.get_active_identity = lambda: "vision"

    handled = system.handle_transcribed_text_from_hotkey("please update date time now")

    assert handled is True
    assert calls == [("vision", True, "hotkey")]
