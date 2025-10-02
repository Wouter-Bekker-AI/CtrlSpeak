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
    keywords.configure_identity_keywords(["assistant", "default"])
    yield
    keywords.configure_identity_keywords([])


@pytest.fixture
def bot_module(monkeypatch):
    stub = types.SimpleNamespace(
        get_active_identity=lambda: None,
        start_bot=lambda **_kwargs: True,
        stop_bot=lambda: None,
    )
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

    handled = system.handle_transcribed_text_from_hotkey("Chat with Assistant")

    assert handled is True
    assert calls == [("start", "assistant")]


def test_hotkey_start_ignores_active_identity(bot_module):
    calls: list[tuple[str, str | None]] = []

    bot_module.get_active_identity = lambda: "assistant"
    bot_module.stop_bot = lambda: calls.append(("stop", None))

    def _start_bot(**_kwargs):
        calls.append(("start", _kwargs.get("identity")))
        return True

    bot_module.start_bot = _start_bot

    handled = system.handle_transcribed_text_from_hotkey("chat with assistant")

    assert handled is True
    assert calls == []


def test_hotkey_switches_identity_after_stopping_current(bot_module):
    sequence: list[str] = []

    bot_module.get_active_identity = lambda: "assistant"
    bot_module.stop_bot = lambda: sequence.append("stop")

    def _start_bot(*, identity: str | None = None, **_kwargs) -> bool:
        sequence.append(f"start:{identity}")
        return True

    bot_module.start_bot = _start_bot

    handled = system.handle_transcribed_text_from_hotkey("chat with default")

    assert handled is True
    assert sequence == ["stop", "start:default"]


def test_hotkey_goodbye_stops_active_identity(bot_module):
    stop_calls: list[str] = []
    start_calls: list[str] = []

    bot_module.get_active_identity = lambda: "assistant"
    bot_module.stop_bot = lambda: stop_calls.append("stop")
    bot_module.start_bot = lambda **_kwargs: start_calls.append(_kwargs.get("identity")) or True

    handled = system.handle_transcribed_text_from_hotkey("goodbye assistant")

    assert handled is True
    assert stop_calls == ["stop"]
    assert start_calls == []


def test_hotkey_goodbye_handles_punctuation(bot_module):
    stop_calls: list[str] = []

    bot_module.get_active_identity = lambda: "assistant"
    bot_module.stop_bot = lambda: stop_calls.append("stop")

    handled = system.handle_transcribed_text_from_hotkey("goodbye, assistant!")

    assert handled is True
    assert stop_calls == ["stop"]


def test_hotkey_goodbye_for_other_identity_is_ignored(bot_module):
    stop_calls: list[str] = []
    start_calls: list[str] = []

    bot_module.get_active_identity = lambda: "assistant"
    bot_module.stop_bot = lambda: stop_calls.append("stop")
    bot_module.start_bot = lambda **_kwargs: start_calls.append(_kwargs.get("identity")) or True

    handled = system.handle_transcribed_text_from_hotkey("goodbye default")

    assert handled is True
    assert stop_calls == []
    assert start_calls == []


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


def test_hotkey_supports_fuzzy_chat(bot_module):
    start_calls: list[str] = []

    bot_module.get_active_identity = lambda: None

    def _start_bot(*, identity: str | None = None, **_kwargs) -> bool:
        start_calls.append(identity)
        return True

    bot_module.start_bot = _start_bot

    handled = system.handle_transcribed_text_from_hotkey("please chat was assistant right now")

    assert handled is True
    assert start_calls == ["assistant"]


def test_hotkey_normalizes_defunct_identity(bot_module):
    start_calls: list[str] = []

    bot_module.get_active_identity = lambda: None

    def _start_bot(*, identity: str | None = None, **_kwargs) -> bool:
        start_calls.append(identity)
        return True

    bot_module.start_bot = _start_bot

    handled = system.handle_transcribed_text_from_hotkey("chat with defunct")

    assert handled is True
    assert start_calls == ["default"]


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
    bot_module.get_active_identity = lambda: "default"

    handled = system.handle_transcribed_text_from_hotkey("update documentation")

    assert handled is True
    assert calls == [("default", True, "hotkey")]


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
    bot_module.get_active_identity = lambda: "assistant"

    handled = system.handle_transcribed_text_from_hotkey("please update date time now")

    assert handled is True
    assert calls == [("assistant", True, "hotkey")]
