"""Input delivery tests use mocks only: never type into the user's desktop."""
from __future__ import annotations

import ctypes
import sys
from types import SimpleNamespace

import pytest

from utils.input_errors import InputInsertionCancelled, InputInsertionError


pytestmark = pytest.mark.core_headless


@pytest.fixture
def win(monkeypatch):
    if not sys.platform.startswith("win"):
        pytest.skip("Win32 INPUT layout is Windows-only")
    from utils import windows_input

    monkeypatch.setattr(windows_input, "get_focused_control", lambda: 101)
    monkeypatch.setattr(windows_input, "is_console_window", lambda _: False)
    monkeypatch.setattr(windows_input, "is_remote_host_window", lambda _: True)
    monkeypatch.setattr(windows_input, "is_anydesk_window", lambda _: True)
    monkeypatch.setattr(windows_input, "_FORCE_SENDINPUT", False)
    monkeypatch.setattr(windows_input, "get_clipboard_text", lambda: "previous clipboard")
    monkeypatch.setattr(windows_input, "set_clipboard_text", lambda _: True)
    monkeypatch.setattr(windows_input, "restore_clipboard_text", lambda _: None)
    monkeypatch.setattr(windows_input, "clipboard_contains_non_text_data", lambda: False)
    monkeypatch.setattr(windows_input.time, "sleep", lambda _: None)
    # A test forgetting to replace the syscall must fail rather than type.
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Unexpected real input path in mock test")
    monkeypatch.setattr(windows_input, "_send_input", forbidden)
    monkeypatch.setattr(windows_input.pyautogui, "write", forbidden, raising=False)
    monkeypatch.setattr(windows_input.pyautogui, "hotkey", forbidden, raising=False)
    return windows_input


def _events(win, count, pointer):
    packet = ctypes.cast(pointer, ctypes.POINTER(win.INPUT))
    return [(packet[i].union.ki.wVk, packet[i].union.ki.wScan,
             packet[i].union.ki.dwFlags) for i in range(count)]


def test_remote_insertion_uses_small_paced_batches_not_per_character_delay(win, monkeypatch):
    batches, sleeps = [], []
    def send(count, pointer, _size):
        batches.append(_events(win, count, pointer))
        return count
    monkeypatch.setattr(win, "_send_input", send)
    monkeypatch.setattr(win.time, "sleep", sleeps.append)
    assert win.send_unicode_input("a" * 4911, remote=True)
    assert len(batches) == 307
    assert all(len(batch) <= 32 for batch in batches)
    assert sum(len(batch) for batch in batches) == 4911 * 2
    assert sum(sleeps) == pytest.approx(6.14)
    assert 0.01 not in sleeps  # Former 49.11 seconds of deliberate sleeping.


def test_surrogate_pair_stays_in_one_packet_at_boundary(win, monkeypatch):
    batches = []
    def send(count, pointer, _size):
        batches.append(_events(win, count, pointer))
        return count
    monkeypatch.setattr(win, "_send_input", send)
    text = "a" * 15 + "\U0001000d" + "b"
    assert win.send_unicode_input(text, remote=True)
    downs = [[scan for _, scan, flags in batch if not flags & win.KEYEVENTF_KEYUP]
             for batch in batches]
    assert downs == [[ord("a")] * 15, [0xD800, 0xDC0D, ord("b")]]
    assert all(scan != 13 for units in downs for scan in units)


@pytest.mark.parametrize("accepted", [1, 2, 9, 31])
def test_partial_unicode_stops_without_full_transcript_fallback(win, monkeypatch, accepted):
    calls, fallback = [], []
    def send(count, pointer, _size):
        calls.append(_events(win, count, pointer))
        return accepted if len(calls) == 1 else count
    monkeypatch.setattr(win, "_send_input", send)
    monkeypatch.setattr(win, "try_sendinput_paste", lambda *a, **kw: fallback.append("paste"))
    monkeypatch.setattr(win, "try_clipboard_paste", lambda *a, **kw: fallback.append("clipboard"))
    with pytest.raises(InputInsertionError) as error:
        win.insert_text_into_focus("x" * 40)
    assert error.value.may_have_inserted
    assert error.value.events_inserted == accepted
    assert fallback == []
    if accepted % 2:
        assert len(calls) == 2
        assert calls[-1][0][2] == win.KEYEVENTF_UNICODE | win.KEYEVENTF_KEYUP
    else:
        assert len(calls) == 1


def test_zero_events_after_prior_success_is_partial_and_never_replayed(win, monkeypatch):
    calls = []
    def send(count, pointer, _size):
        calls.append(_events(win, count, pointer))
        return count if len(calls) == 1 else 0
    monkeypatch.setattr(win, "_send_input", send)
    with pytest.raises(InputInsertionError) as error:
        win.send_unicode_input("x" * 40, remote=True)
    assert error.value.events_inserted == 32
    assert error.value.may_have_inserted
    assert len(calls) == 2


def test_zero_events_initially_allows_one_safe_paste_fallback(win, monkeypatch):
    pasted = []
    monkeypatch.setattr(win, "_send_input", lambda *_: 0)
    monkeypatch.setattr(win, "try_sendinput_paste", lambda text, **kw: pasted.append(text) or True)
    win.insert_text_into_focus("only once")
    assert pasted == ["only once"]


def test_ambiguous_syscall_exception_is_not_replayed(win, monkeypatch):
    def send(*_args):
        raise OSError("ambiguous delivery")
    monkeypatch.setattr(win, "_send_input", send)
    with pytest.raises(InputInsertionError) as error:
        win.insert_text_into_focus("do not duplicate")
    assert error.value.may_have_inserted


def test_cancel_before_input_never_touches_windows(win):
    with pytest.raises(InputInsertionCancelled) as error:
        win.insert_text_into_focus("cancelled", cancel_check=lambda: True)
    assert not error.value.may_have_inserted
    assert error.value.events_inserted == 0


def test_cancel_between_batches_reports_partial_without_replay(win, monkeypatch):
    calls = []
    def send(count, pointer, _size):
        calls.append(_events(win, count, pointer))
        return count
    monkeypatch.setattr(win, "_send_input", send)
    with pytest.raises(InputInsertionCancelled) as error:
        win.insert_text_into_focus("x" * 80, cancel_check=lambda: len(calls) > 0)
    assert len(calls) == 1
    assert error.value.events_inserted == 32
    assert error.value.may_have_inserted


def test_focus_change_between_batches_stops_without_reclaiming_focus(win, monkeypatch):
    calls = []
    def send(count, pointer, _size):
        calls.append(_events(win, count, pointer))
        return count
    monkeypatch.setattr(win, "_send_input", send)
    monkeypatch.setattr(win, "get_focused_control", lambda: 202 if calls else 101)
    with pytest.raises(InputInsertionError, match="focused window changed") as error:
        win.insert_text_into_focus("x" * 80)
    assert len(calls) == 1
    assert error.value.may_have_inserted


def test_insertion_deadline_stops_partial_delivery(win, monkeypatch):
    calls = []
    def send(count, pointer, _size):
        calls.append(_events(win, count, pointer))
        return count
    monkeypatch.setattr(win, "_send_input", send)
    monkeypatch.setattr(win.time, "monotonic", lambda: 31.0 if calls else 0.0)
    with pytest.raises(InputInsertionError, match="safety deadline") as error:
        win.send_unicode_input("x" * 80, remote=True)
    assert len(calls) == 1
    assert error.value.may_have_inserted


@pytest.mark.parametrize("accepted,may_paste", [(0, False), (1, False), (2, True), (3, True)])
def test_partial_paste_releases_keys_and_never_claims_safe_retry_after_v_down(
    win, monkeypatch, accepted, may_paste,
):
    calls, restored = [], []
    def send(count, pointer, _size):
        calls.append(_events(win, count, pointer))
        return accepted if len(calls) == 1 else count
    monkeypatch.setattr(win, "_send_input", send)
    monkeypatch.setattr(win, "restore_clipboard_text", restored.append)
    if may_paste:
        with pytest.raises(InputInsertionError) as error:
            win.try_sendinput_paste("paste only once")
        assert error.value.may_have_inserted
    else:
        assert win.try_sendinput_paste("paste only once") is False
    assert restored == ["previous clipboard"]
    assert len(calls) == (3 if accepted else 1)
    assert all(event[2] == win.KEYEVENTF_KEYUP for batch in calls[1:] for event in batch)


@pytest.mark.parametrize("method", ["try_sendinput_paste", "try_clipboard_paste"])
def test_cancel_after_clipboard_staging_restores_without_paste(win, monkeypatch, method):
    staged, restored = [], []
    monkeypatch.setattr(win, "set_clipboard_text", lambda text: staged.append(text) or True)
    monkeypatch.setattr(win, "restore_clipboard_text", restored.append)
    with pytest.raises(InputInsertionCancelled) as error:
        getattr(win, method)("not pasted", cancel_check=lambda: bool(staged))
    assert not error.value.may_have_inserted
    assert staged == ["not pasted"]
    assert restored == ["previous clipboard"]


def test_direct_editor_timeout_is_uncertain_no_fallback(win, monkeypatch):
    messages = []
    def send_message(_hwnd, message, *_args):
        messages.append(message)
        return int(message == win.EM_SETSEL)
    monkeypatch.setattr(win, "user32", SimpleNamespace(SendMessageTimeoutW=send_message))
    monkeypatch.setattr(win, "get_class_name", lambda _: "edit")
    monkeypatch.setattr(win, "is_remote_host_window", lambda _: False)
    monkeypatch.setattr(win, "is_anydesk_window", lambda _: False)
    with pytest.raises(InputInsertionError) as error:
        win.insert_text_into_focus("maybe delivered")
    assert error.value.may_have_inserted
    assert messages == [win.EM_SETSEL, win.EM_REPLACESEL]


@pytest.mark.parametrize("cause", ["cancel", "focus"])
def test_direct_editor_checks_cancel_and_focus_again_after_selection(win, monkeypatch, cause):
    messages = []
    def send_message(_hwnd, message, *_args):
        messages.append(message)
        return 1
    monkeypatch.setattr(win, "user32", SimpleNamespace(SendMessageTimeoutW=send_message))
    monkeypatch.setattr(win, "get_class_name", lambda _: "edit")
    if cause == "focus":
        monkeypatch.setattr(win, "get_focused_control", lambda: 202 if messages else 101)
    with pytest.raises(InputInsertionError) as error:
        win.try_direct_text_insert("not inserted", 101,
                                   cancel_check=lambda: cause == "cancel" and bool(messages))
    assert not error.value.may_have_inserted
    assert messages == [win.EM_SETSEL]


def test_linux_cancel_after_staging_restores_clipboard_without_paste(monkeypatch):
    from utils import linux_input as lin
    staged, restored, pasted = [], [], []
    monkeypatch.setattr(lin, "ensure_desktop_automation_supported", lambda: None)
    monkeypatch.setattr(lin, "clipboard_tool_available", lambda: True)
    monkeypatch.setattr(lin, "clipboard_contains_non_text_data", lambda: False)
    monkeypatch.setattr(lin, "get_clipboard_text", lambda: "prior")
    monkeypatch.setattr(lin, "set_clipboard_text", lambda text: staged.append(text) or True)
    monkeypatch.setattr(lin, "restore_clipboard_text", restored.append)
    monkeypatch.setattr(lin, "get_pyautogui", lambda: SimpleNamespace(hotkey=lambda *keys: pasted.append(keys)))
    with pytest.raises(InputInsertionCancelled) as error:
        lin.insert_text_into_focus("not pasted", cancel_check=lambda: bool(staged))
    assert not error.value.may_have_inserted
    assert pasted == []
    assert restored == ["prior"]


def test_linux_direct_typing_cancellation_is_partial(monkeypatch):
    from utils import linux_input as lin
    typed = []
    monkeypatch.setattr(lin, "get_pyautogui", lambda: SimpleNamespace(write=lambda text, **kw: typed.append(text)))
    with pytest.raises(InputInsertionCancelled) as error:
        lin._type_directly_or_raise("x" * 100, reason="test", cancel_check=lambda: bool(typed))
    assert typed == ["x" * 32]
    assert error.value.events_inserted == 64


def test_winio_forwards_cancellation_without_invoking_native_adapter(monkeypatch):
    from utils import winio
    forwarded = []
    monkeypatch.setattr(winio, "_adapter", lambda: SimpleNamespace(
        insert_text_into_focus=lambda text, **kwargs: forwarded.append((text, kwargs))))
    cancel = lambda: False
    winio.insert_text_into_focus("text", preserve_formatting=True, cancel_check=cancel)
    assert forwarded == [("text", {"preserve_formatting": True, "cancel_check": cancel})]
