"""Trusted live separators reach actual adapters; all OS input is mocked."""
from __future__ import annotations

import ctypes
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.core_headless


def test_router_only_forwards_explicit_live_option(monkeypatch):
    from utils import winio
    calls = []
    monkeypatch.setattr(winio, "_adapter", lambda: SimpleNamespace(
        insert_text_into_focus=lambda text, **options: calls.append((text, options))))
    winio.insert_text_into_focus("normal")
    winio.insert_text_into_focus("live", append_space=True)
    assert calls == [
        ("normal", {"preserve_formatting": False, "cancel_check": None}),
        ("live", {"preserve_formatting": False, "cancel_check": None, "append_space": True}),
    ]


@pytest.fixture
def win(monkeypatch):
    if not sys.platform.startswith("win"):
        pytest.skip("Windows INPUT layout")
    from utils import windows_input
    monkeypatch.setattr(windows_input, "get_focused_control", lambda: 101)
    monkeypatch.setattr(windows_input, "_check_insertion_target", lambda *args: None)
    monkeypatch.setattr(windows_input, "is_console_window", lambda _: False)
    monkeypatch.setattr(windows_input, "is_remote_host_window", lambda _: True)
    monkeypatch.setattr(windows_input, "is_anydesk_window", lambda _: True)
    monkeypatch.setattr(windows_input, "_FORCE_SENDINPUT", False)
    monkeypatch.setattr(windows_input.time, "sleep", lambda _: None)

    def forbidden(*args, **kwargs):
        pytest.fail("Test reached an unmocked OS input route")

    monkeypatch.setattr(windows_input, "_send_input", forbidden)
    monkeypatch.setattr(windows_input, "try_direct_text_insert", forbidden)
    monkeypatch.setattr(windows_input, "try_sendinput_paste", forbidden)
    monkeypatch.setattr(windows_input, "try_clipboard_paste", forbidden)
    monkeypatch.setattr(windows_input.pyautogui, "write", forbidden, raising=False)
    monkeypatch.setattr(windows_input.pyautogui, "hotkey", forbidden, raising=False)
    return windows_input


def test_actual_remote_batches_keep_one_safe_separator_between_live_phrases(win, monkeypatch):
    units = []

    def send(count, pointer, _size):
        packet = ctypes.cast(pointer, ctypes.POINTER(win.INPUT))
        units.extend(packet[i].union.ki.wScan for i in range(count)
                     if not packet[i].union.ki.dwFlags & win.KEYEVENTF_KEYUP)
        return count

    monkeypatch.setattr(win, "_send_input", send)
    win.insert_text_into_focus("first phrase\n\t\x1b", append_space=True)
    win.insert_text_into_focus("next café 🌱\r\n", append_space=True)
    received = b"".join(unit.to_bytes(2, "little") for unit in units).decode("utf-16-le")
    assert received == "first phrase next café 🌱 "
    assert not any(unit < 32 or 127 <= unit < 160 for unit in units)


def test_trusted_space_does_not_turn_empty_model_text_into_input(win):
    win.insert_text_into_focus("\r\n\t\x1b", append_space=True)
    assert win.send_unicode_input("\r\n\t\x1b", append_space=True, remote=True)


def test_zero_event_remote_fallback_keeps_separator_exactly_once(win, monkeypatch):
    pasted = []
    monkeypatch.setattr(win, "_send_input", lambda *args: 0)
    monkeypatch.setattr(win, "try_sendinput_paste", lambda text, **kwargs: pasted.append(text) or True)
    win.insert_text_into_focus("fallback\n", append_space=True)
    assert pasted == ["fallback "]


def test_live_local_editor_preserves_caret_not_legacy_move_to_end(win, monkeypatch):
    pasted = []
    monkeypatch.setattr(win, "is_remote_host_window", lambda _: False)
    monkeypatch.setattr(win, "is_anydesk_window", lambda _: False)
    monkeypatch.setattr(win, "try_sendinput_paste", lambda text, **kwargs: pasted.append(text) or True)
    win.insert_text_into_focus("at current caret", append_space=True)
    assert pasted == ["at current caret "]


def test_default_local_editor_keeps_legacy_route_and_no_added_space(win, monkeypatch):
    direct = []
    monkeypatch.setattr(win, "is_remote_host_window", lambda _: False)
    monkeypatch.setattr(win, "is_anydesk_window", lambda _: False)
    monkeypatch.setattr(win, "try_direct_text_insert", lambda text, hwnd, **kwargs: direct.append(text) or True)
    win.insert_text_into_focus("ordinary hold\n")
    assert direct == ["ordinary hold"]


@pytest.mark.parametrize("clipboard_available", [True, False])
def test_linux_live_space_survives_clipboard_and_tk_routes(monkeypatch, clipboard_available):
    from utils import linux_input as lin
    delivered = []
    monkeypatch.setattr(lin, "ensure_desktop_automation_supported", lambda: None)
    monkeypatch.setattr(lin, "clipboard_tool_available", lambda: clipboard_available)
    monkeypatch.setattr(lin, "clipboard_contains_non_text_data", lambda: False)
    monkeypatch.setattr(lin, "get_clipboard_text", lambda: "prior clipboard")
    monkeypatch.setattr(lin, "set_clipboard_text", lambda text: delivered.append(text) or True)
    monkeypatch.setattr(lin, "restore_clipboard_text", lambda _: None)
    monkeypatch.setattr(lin, "_insert_text_with_tk_fallback", lambda text, **kwargs: delivered.append(text))
    monkeypatch.setattr(lin, "get_pyautogui", lambda: SimpleNamespace(hotkey=lambda *args: None))
    monkeypatch.setattr(lin.time, "sleep", lambda _: None)
    lin.insert_text_into_focus("first\n", append_space=True)
    lin.insert_text_into_focus("second\t\x1b\r", append_space=True)
    lin.insert_text_into_focus("\n\t\x1b", append_space=True)
    assert delivered == ["first ", "second "]
