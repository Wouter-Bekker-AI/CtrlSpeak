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
