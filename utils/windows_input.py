# -*- coding: utf-8 -*-
"""Windows input, clipboard, and focused-control implementation.

This module is imported lazily by :mod:`utils.winio` only on Windows. Keeping
the Win32 ctypes declarations here prevents Linux processes from importing or
initialising Windows-only APIs.
"""
from __future__ import annotations

import sys
import time
import ctypes
import threading
import uuid
from ctypes import wintypes
from pathlib import Path
from typing import Optional

import pyautogui

from utils.config_paths import get_logger
from utils.dictation_text import prepare_dictation_text
from utils.input_errors import CancelCheck, InputInsertionError, check_insertion_cancelled

logger = get_logger(__name__)

# Windows GUI helpers
user32 = ctypes.windll.user32 if sys.platform.startswith("win") else None
kernel32 = ctypes.windll.kernel32 if sys.platform.startswith("win") else None
psapi = ctypes.windll.psapi if sys.platform.startswith("win") else None

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
HWND_MESSAGE = -3
EM_SETSEL = 0x00B1
EM_REPLACESEL = 0x00C2

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_V = 0x56
KEYEVENTF_UNICODE = 0x0004
SMTO_BLOCK = 0x0001
SMTO_ABORTIFHUNG = 0x0002

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# ---------------- Text insertion helpers ----------------
class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)] if sys.platform.startswith("win") else []


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
        ("rcCaret", RECT),
    ] if sys.platform.startswith("win") else []


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", wintypes.LPARAM)] if sys.platform.startswith("win") else []


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", wintypes.LPARAM)] if sys.platform.startswith("win") else []


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)] if sys.platform.startswith("win") else []


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)] if sys.platform.startswith("win") else []


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)] if sys.platform.startswith("win") else []


if sys.platform.startswith("win"):
    # ctypes otherwise assumes a 32-bit ``c_int`` return value. Clipboard and
    # global-memory handles are pointer-sized on 64-bit Windows, so leaving
    # these declarations implicit truncates valid HGLOBAL values and makes
    # SetClipboardData fail even though allocation succeeded.
    user32.OpenClipboard.argtypes = (wintypes.HWND,)
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = ()
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = ()
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.IsClipboardFormatAvailable.argtypes = (wintypes.UINT,)
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = (wintypes.UINT,)
    user32.GetClipboardData.restype = wintypes.HANDLE
    user32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.EnumClipboardFormats.argtypes = (wintypes.UINT,)
    user32.EnumClipboardFormats.restype = wintypes.UINT
    user32.CreateWindowExW.argtypes = (
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        wintypes.LPVOID,
    )
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DestroyWindow.argtypes = (wintypes.HWND,)
    user32.DestroyWindow.restype = wintypes.BOOL
    kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalLock.restype = wintypes.LPVOID
    kernel32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = (wintypes.HGLOBAL,)
    kernel32.GlobalFree.restype = wintypes.HGLOBAL

    _send_input = user32.SendInput
    _send_input.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    _send_input.restype = wintypes.UINT

    user32.SendMessageTimeoutW.argtypes = (
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
        wintypes.UINT, wintypes.UINT, ctypes.POINTER(wintypes.WPARAM),
    )
    user32.SendMessageTimeoutW.restype = wintypes.LPARAM

_FORCE_SENDINPUT = False
_CLIPBOARD_LOCK = threading.RLock()
_CLIPBOARD_OPEN_ATTEMPTS = 16
_CLIPBOARD_RETRY_SECONDS = 0.02
# Remote viewers can lose/reorder very large bursts even when SendInput accepts
# them. Small, paced packets are deliberately more conservative than an entire
# transcript in one call, while removing the former 10 ms per UTF-16 unit delay.
_REMOTE_BATCH_UNITS = 16
_REMOTE_BATCH_PAUSE_SECONDS = 0.020
_LOCAL_BATCH_UNITS = 64
_LOCAL_BATCH_PAUSE_SECONDS = 0.005
_INSERTION_DEADLINE_SECONDS = 30.0
_DIRECT_INSERT_TIMEOUT_MS = 250


def set_force_sendinput(flag: bool) -> None:
    global _FORCE_SENDINPUT
    _FORCE_SENDINPUT = bool(flag)


def get_focused_control() -> Optional[int]:
    if not sys.platform.startswith("win"):
        return None
    try:
        foreground = user32.GetForegroundWindow()
        if not foreground:
            return None
        thread_id = user32.GetWindowThreadProcessId(foreground, None)
        current_thread_id = kernel32.GetCurrentThreadId()
        attached = False
        if thread_id != current_thread_id:
            if user32.AttachThreadInput(current_thread_id, thread_id, True):
                attached = True
        info = GUITHREADINFO()
        info.cbSize = ctypes.sizeof(GUITHREADINFO)
        hwnd_focus = None
        if user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
            hwnd_focus = info.hwndFocus or info.hwndCaret or info.hwndActive
        if attached:
            user32.AttachThreadInput(current_thread_id, thread_id, False)
        return hwnd_focus or foreground
    except Exception:
        logger.exception("Failed to query focused control")
        return None


def get_class_name(hwnd: int) -> str:
    if not sys.platform.startswith("win") or not hwnd:
        return ""
    buffer = ctypes.create_unicode_buffer(256)
    try:
        if user32.GetClassNameW(hwnd, buffer, len(buffer)) == 0:
            return ""
    except Exception:
        logger.exception("Failed to determine window class name for hwnd=%s", hwnd)
        return ""
    return buffer.value.lower()


def get_window_text(hwnd: int) -> str:
    if not sys.platform.startswith("win"):
        return ""
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    if user32.GetWindowTextW(hwnd, buffer, len(buffer)) == 0:
        return ""
    return buffer.value


def get_window_process_name(hwnd: int) -> str:
    if not sys.platform.startswith("win"):
        return ""
    pid = wintypes.DWORD()
    if user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)) == 0:
        return ""
    access = PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | PROCESS_QUERY_LIMITED_INFORMATION
    process_handle = kernel32.OpenProcess(access, False, pid.value)
    if not process_handle:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(260)
        length = psapi.GetModuleFileNameExW(process_handle, None, buffer, len(buffer))
        if length == 0:
            return ""
        return Path(buffer.value).name.lower()
    except Exception:
        logger.exception("Failed to resolve process name for hwnd=%s", hwnd)
        return ""
    finally:
        kernel32.CloseHandle(process_handle)


def is_console_window(hwnd: int) -> bool:
    if not sys.platform.startswith("win") or not hwnd:
        return False
    class_name = get_class_name(hwnd)
    if class_name in {"consolewindowclass"}:
        return True
    if "cascadia" in class_name:
        return True
    process_name = get_window_process_name(hwnd)
    return process_name in {"conhost.exe", "openconsole.exe", "wt.exe", "windowsterminal.exe", "ubuntu.exe", "wsl.exe"}


# --- Remote host awareness ---
# Treat remote-control host windows as "hosted" targets that need SendInput-first
_REMOTE_HOST_PROCESSES = {
    "anydesk.exe",
    "rustdesk.exe",
    "teamviewer.exe",
    "parsec.exe",
    "mstsc.exe",                 # RDP classic
    "msrdc.exe",                 # Remote Desktop (UWP)
    "chrome_remote_desktop_host.exe",
    "vncviewer.exe",
}


def is_remote_host_window(hwnd: int) -> bool:
    if not sys.platform.startswith("win") or not hwnd:
        return False
    try:
        proc = (get_window_process_name(hwnd) or "").lower()
        return proc in _REMOTE_HOST_PROCESSES
    except Exception:
        logger.exception("Failed to detect remote host window for hwnd=%s", hwnd)
        return False


# --- Explicit AnyDesk detection (restored from working build) ---
def window_matches_anydesk(hwnd: int) -> bool:
    class_name = get_class_name(hwnd)
    if "anydesk" in class_name:
        return True
    process_name = get_window_process_name(hwnd)
    if "anydesk" in process_name:
        return True
    title = get_window_text(hwnd).lower()
    if "anydesk" in title:
        return True
    return False


def is_anydesk_window(hwnd: int) -> bool:
    if not sys.platform.startswith("win"):
        return False
    current = hwnd
    visited = set()
    # Walk up a few parents because the focused control can be nested
    for _ in range(6):
        if not current or current in visited:
            break
        visited.add(current)
        if window_matches_anydesk(current):
            return True
        parent = user32.GetParent(current)
        if not parent:
            break
        current = parent
    return False


def open_clipboard(owner_hwnd: int | None = None) -> bool:
    if not sys.platform.startswith("win"):
        return False
    for attempt in range(_CLIPBOARD_OPEN_ATTEMPTS):
        if user32.OpenClipboard(owner_hwnd):
            return True
        if attempt + 1 < _CLIPBOARD_OPEN_ATTEMPTS:
            time.sleep(_CLIPBOARD_RETRY_SECONDS)
    return False


def _create_clipboard_owner_window() -> int | None:
    """Create a short-lived, process-owned HWND for EmptyClipboard.

    Win32 requires a real owner window when EmptyClipboard is followed by
    SetClipboardData. Opening with ``NULL`` makes EmptyClipboard assign no
    owner and can make SetClipboardData fail. A message-only STATIC window is
    invisible, belongs to this process/thread, and needs no long-lived UI.
    """

    if not sys.platform.startswith("win"):
        return None
    hwnd = user32.CreateWindowExW(
        0,
        "STATIC",
        "CtrlSpeakClipboardOwner",
        0,
        0,
        0,
        0,
        0,
        wintypes.HWND(HWND_MESSAGE),
        None,
        None,
        None,
    )
    return int(hwnd) if hwnd else None


def get_clipboard_text() -> Optional[str]:
    if not sys.platform.startswith("win"):
        return None
    with _CLIPBOARD_LOCK:
        if not open_clipboard():
            return None
        try:
            if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
                return None
            handle = user32.GetClipboardData(CF_UNICODETEXT)
            if not handle:
                return None
            pointer = kernel32.GlobalLock(handle)
            if not pointer:
                return None
            try:
                return ctypes.wstring_at(pointer)
            finally:
                kernel32.GlobalUnlock(handle)
        finally:
            user32.CloseClipboard()


def set_clipboard_text(text: str) -> bool:
    if not sys.platform.startswith("win"):
        return False
    with _CLIPBOARD_LOCK:
        # Prepare the movable block before clearing the clipboard.  If memory
        # allocation fails, the user's existing clipboard contents remain
        # untouched.
        data = ctypes.create_unicode_buffer(text)
        size = ctypes.sizeof(data)
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            return False
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            kernel32.GlobalFree(handle)
            return False
        ctypes.memmove(pointer, ctypes.addressof(data), size)
        kernel32.GlobalUnlock(handle)

        owner_hwnd = _create_clipboard_owner_window()
        if owner_hwnd is None:
            kernel32.GlobalFree(handle)
            return False
        try:
            if not open_clipboard(owner_hwnd):
                kernel32.GlobalFree(handle)
                return False
            try:
                if not user32.EmptyClipboard():
                    kernel32.GlobalFree(handle)
                    return False
                if not user32.SetClipboardData(CF_UNICODETEXT, handle):
                    kernel32.GlobalFree(handle)
                    return False
                # SetClipboardData owns the HGLOBAL after success. It must not
                # be freed; Windows keeps it pasteable after this function.
                return True
            finally:
                user32.CloseClipboard()
        finally:
            user32.DestroyWindow(owner_hwnd)


def restore_clipboard_text(previous: Optional[str]) -> None:
    if not sys.platform.startswith("win"):
        return
    with _CLIPBOARD_LOCK:
        if previous is None:
            if open_clipboard():
                try:
                    user32.EmptyClipboard()
                finally:
                    user32.CloseClipboard()
            return
        set_clipboard_text(previous)


def clipboard_contains_non_text_data() -> bool:
    """Return True when capture would overwrite a non-text-only clipboard."""
    if not sys.platform.startswith("win"):
        return False
    with _CLIPBOARD_LOCK:
        if not open_clipboard():
            return False
        try:
            first_format = user32.EnumClipboardFormats(0)
            return bool(first_format) and not bool(
                user32.IsClipboardFormatAvailable(CF_UNICODETEXT)
            )
        finally:
            user32.CloseClipboard()


def snapshot_active_text_field(*, copy_wait_seconds: float = 0.08) -> Optional[str]:
    """Best-effort Ctrl+A/C snapshot of the focused Windows field.

    The original Enter is owned by the caller's non-suppressing keyboard hook;
    this helper does not send Enter. A prior Unicode-text clipboard value (or an
    empty clipboard) is restored before returning. A non-text-only clipboard is
    left untouched and capture is skipped because it cannot be restored safely
    through the lightweight Win32 text helpers.
    """
    if not sys.platform.startswith("win"):
        return None
    if clipboard_contains_non_text_data():
        logger.info("Skipping active-field capture to preserve non-text clipboard data")
        return None

    previous = get_clipboard_text()
    marker = f"CtrlSpeak-field-capture-{uuid.uuid4()}"
    try:
        if not set_clipboard_text(marker):
            return None
        pyautogui.hotkey("ctrl", "a")
        pyautogui.hotkey("ctrl", "c")
        if copy_wait_seconds > 0:
            time.sleep(copy_wait_seconds)
        captured = get_clipboard_text()
        return captured if captured is not None and captured != marker else None
    except Exception:
        logger.exception("Failed to snapshot the active text field")
        return None
    finally:
        restore_clipboard_text(previous)


def send_input_key(vk: int, keyup: bool = False) -> bool:
    if not sys.platform.startswith("win"):
        return False
    flags = KEYEVENTF_KEYUP if keyup else 0
    union = _INPUTUNION()
    union.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
    inp = INPUT(type=INPUT_KEYBOARD, union=union)
    return _send_input(1, ctypes.byref(inp), ctypes.sizeof(inp)) == 1


def ensure_foreground(hwnd: int) -> None:
    if not sys.platform.startswith("win") or not hwnd:
        return
    try:
        pid = wintypes.DWORD()
        thread_id = user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        current_thread_id = kernel32.GetCurrentThreadId()
        attached = False
        if thread_id and thread_id != current_thread_id:
            if user32.AttachThreadInput(current_thread_id, thread_id, True):
                attached = True
        try:
            user32.SetForegroundWindow(hwnd)
            user32.SetFocus(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(current_thread_id, thread_id, False)
    except Exception:
        logger.exception("Failed to bring window %s to foreground", hwnd)


def _unicode_batches(text: str, max_units: int):
    """Yield bounded UTF-16 batches without splitting surrogate pairs."""
    units: list[int] = []
    for character in text:
        # SendInput wScan is a UTF-16 WORD, not a Unicode codepoint. Truncating
        # supplementary characters can manufacture a CR/ESC/control character.
        encoded = character.encode("utf-16-le")
        character_units = [
            int.from_bytes(encoded[offset:offset + 2], "little")
            for offset in range(0, len(encoded), 2)
        ]
        if units and len(units) + len(character_units) > max_units:
            yield units
            units = []
        units.extend(character_units)
    if units:
        yield units


def _keyboard_event(*, code: int = 0, vk: int = 0, flags: int = 0) -> INPUT:
    union = _INPUTUNION()
    union.ki = KEYBDINPUT(wVk=vk, wScan=code, dwFlags=flags, time=0, dwExtraInfo=0)
    return INPUT(type=INPUT_KEYBOARD, union=union)


def _check_insertion_target(hwnd: int | None, events_inserted: int = 0) -> None:
    # Never reclaim focus between batches: the user may have deliberately
    # switched windows. A partial prefix is safer than typing into a new target.
    if not hwnd or get_focused_control() != hwnd:
        raise InputInsertionError(
            "The focused window changed during insertion. Use Copy last transcription "
            "to recover the text; it has not been automatically replayed.",
            events_inserted=events_inserted,
        )


def _release_paste_keys() -> None:
    for vk in (VK_V, VK_CONTROL):
        try:
            if not send_input_key(vk, True):
                logger.warning("Could not release paste key %s after partial delivery", vk)
        except Exception:
            logger.warning("Could not release paste key %s after partial delivery", vk)


def send_unicode_input(
    text: str, *, preserve_formatting: bool = False,
    cancel_check: CancelCheck = None, target_hwnd: int | None = None,
    remote: bool | None = None,
) -> bool:
    """Send paced Unicode batches, returning False ONLY before any accepted input.

    Partial or uncertain delivery raises rather than enabling a full-text
    clipboard fallback, which would duplicate an already inserted prefix.
    """
    if not sys.platform.startswith("win"):
        return False
    check_insertion_cancelled(cancel_check)
    text = prepare_dictation_text(text, preserve_formatting=preserve_formatting)
    if not text:
        return True
    hwnd = target_hwnd if target_hwnd is not None else get_focused_control()
    if not hwnd:
        raise InputInsertionError("There is no focused window for text insertion.")
    if remote is None:
        remote = is_remote_host_window(hwnd) or is_anydesk_window(hwnd)
    batch_units = _REMOTE_BATCH_UNITS if remote else _LOCAL_BATCH_UNITS
    pause = _REMOTE_BATCH_PAUSE_SECONDS if remote else _LOCAL_BATCH_PAUSE_SECONDS
    inserted = 0
    started = time.monotonic()
    for units in _unicode_batches(text.replace("\n", "\r"), batch_units):
        check_insertion_cancelled(cancel_check, events_inserted=inserted)
        _check_insertion_target(hwnd, inserted)
        if time.monotonic() - started >= _INSERTION_DEADLINE_SECONDS:
            raise InputInsertionError(
                "Text insertion exceeded its safety deadline. Copy last transcription "
                "to recover the text; no automatic replay was attempted.",
                events_inserted=inserted,
            )
        events = [event for code in units for event in (
            _keyboard_event(code=code, flags=KEYEVENTF_UNICODE),
            _keyboard_event(code=code, flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP),
        )]
        packet = (INPUT * len(events))(*events)
        try:
            sent = int(_send_input(len(packet), packet, ctypes.sizeof(INPUT)))
        except Exception as exc:
            raise InputInsertionError(
                "Windows input delivery could not be confirmed. No automatic replay was attempted.",
                events_inserted=inserted, may_have_inserted=True,
            ) from exc
        if not 0 <= sent <= len(packet):
            raise InputInsertionError(
                "Windows returned an invalid input count. No automatic replay was attempted.",
                events_inserted=inserted, may_have_inserted=True,
            )
        inserted += sent
        if sent != len(packet):
            if sent % 2:
                # The final key-down may have been accepted without its key-up.
                # Release it best-effort; never resend its key-down or text.
                release = _keyboard_event(
                    code=units[sent // 2], flags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                )
                try:
                    _send_input(1, ctypes.byref(release), ctypes.sizeof(INPUT))
                except Exception:
                    logger.warning("Could not release the last partially accepted Unicode input")
            if inserted:
                raise InputInsertionError(
                    "Windows accepted only part of the transcript. Use Copy last transcription; "
                    "automatic fallback was stopped to avoid duplicating text.",
                    events_inserted=inserted,
                )
            return False
        time.sleep(pause)
    logger.info("Unicode insertion accepted %s input events in %.3fs (remote=%s)",
                inserted, time.monotonic() - started, remote)
    return True


def try_sendinput_paste(text: str, *, cancel_check: CancelCheck = None,
                        target_hwnd: int | None = None) -> bool:
    if not sys.platform.startswith("win"):
        return False
    check_insertion_cancelled(cancel_check)
    hwnd = target_hwnd if target_hwnd is not None else get_focused_control()
    _check_insertion_target(hwnd)
    if clipboard_contains_non_text_data():
        return False
    previous = get_clipboard_text()
    if not set_clipboard_text(text):
        return False
    try:
        check_insertion_cancelled(cancel_check)
        _check_insertion_target(hwnd)
        packet = (INPUT * 4)(
            _keyboard_event(vk=VK_CONTROL), _keyboard_event(vk=VK_V),
            _keyboard_event(vk=VK_V, flags=KEYEVENTF_KEYUP),
            _keyboard_event(vk=VK_CONTROL, flags=KEYEVENTF_KEYUP),
        )
        try:
            sent = int(_send_input(4, packet, ctypes.sizeof(INPUT)))
        except Exception as exc:
            # The call might have sent Ctrl+V before failing. Never replay it.
            _release_paste_keys()
            raise InputInsertionError("Paste delivery could not be confirmed.",
                                      may_have_inserted=True) from exc
        if sent != 4:
            if sent:
                _release_paste_keys()
            if sent >= 2 or sent < 0:
                raise InputInsertionError(
                    "Paste may already have been delivered. No automatic replay was attempted.",
                    may_have_inserted=True,
                )
            return False  # Zero events, or only Ctrl-down, cannot paste text.
        time.sleep(0.15)
        return True
    finally:
        restore_clipboard_text(previous)


def try_clipboard_paste(text: str, *, cancel_check: CancelCheck = None,
                        target_hwnd: int | None = None) -> bool:
    if not sys.platform.startswith("win"):
        return False
    check_insertion_cancelled(cancel_check)
    hwnd = target_hwnd if target_hwnd is not None else get_focused_control()
    _check_insertion_target(hwnd)
    if clipboard_contains_non_text_data():
        return False
    previous = get_clipboard_text()
    if not set_clipboard_text(text):
        return False
    try:
        check_insertion_cancelled(cancel_check)
        _check_insertion_target(hwnd)
        try:
            pyautogui.hotkey("ctrl", "v")
        except Exception as exc:
            raise InputInsertionError("Paste delivery could not be confirmed.",
                                      may_have_inserted=True) from exc
        time.sleep(0.05)
        return True
    finally:
        restore_clipboard_text(previous)


def try_direct_text_insert(text: str, hwnd: Optional[int] = None, *,
                           cancel_check: CancelCheck = None) -> bool:
    if not sys.platform.startswith("win"):
        return False
    if hwnd is None:
        hwnd = get_focused_control()
    if not hwnd:
        return False
    class_name = get_class_name(hwnd)
    if not class_name:
        return False
    if any(token in class_name for token in ("edit", "richedit", "notepad", "textarea", "tmemo", "scintilla")):
        check_insertion_cancelled(cancel_check)
        _check_insertion_target(hwnd)
        result = wintypes.WPARAM()
        flags = SMTO_BLOCK | SMTO_ABORTIFHUNG
        # A hung editor must not hold CtrlSpeak's dictation thread indefinitely.
        if not user32.SendMessageTimeoutW(
            hwnd, EM_SETSEL, 0xFFFFFFFF, 0xFFFFFFFF, flags,
            _DIRECT_INSERT_TIMEOUT_MS, ctypes.byref(result),
        ):
            return False  # Selection-only failure has not inserted text.
        # Selection can take the full timeout; Cancel/focus may have changed
        # while the target editor was responding. Recheck before sending text.
        check_insertion_cancelled(cancel_check)
        _check_insertion_target(hwnd)
        payload = ctypes.create_unicode_buffer(text)
        try:
            delivered = user32.SendMessageTimeoutW(
                hwnd, EM_REPLACESEL, True, ctypes.addressof(payload), flags,
                _DIRECT_INSERT_TIMEOUT_MS, ctypes.byref(result),
            )
        except Exception as exc:
            raise InputInsertionError("Direct editor insertion could not be confirmed.",
                                      may_have_inserted=True) from exc
        if not delivered:
            # Timeout does not prove the editor failed to process the message.
            raise InputInsertionError(
                "The editor did not acknowledge insertion in time. No automatic replay was attempted.",
                may_have_inserted=True,
            )
        return True
    return False


def insert_text_into_focus(text: str, *, preserve_formatting: bool = False,
                           cancel_check: CancelCheck = None) -> None:
    check_insertion_cancelled(cancel_check)
    text = prepare_dictation_text(text, preserve_formatting=preserve_formatting)
    if not text:
        return
    if not sys.platform.startswith("win"):
        check_insertion_cancelled(cancel_check)
        pyautogui.write(text)
        return

    hwnd = get_focused_control()
    console_window = bool(hwnd and is_console_window(hwnd))
    hosted_remote = bool(hwnd and is_remote_host_window(hwnd))
    anydesk_window = bool(hwnd and is_anydesk_window(hwnd))

    if not hwnd:
        raise InputInsertionError("There is no focused window for text insertion.")

    # A False return guarantees zero text events; partial/uncertain delivery
    # raises and must never fall through into another whole-transcript route.
    if hwnd and (_FORCE_SENDINPUT or anydesk_window or console_window or hosted_remote):
        if send_unicode_input(
            text, preserve_formatting=preserve_formatting,
            cancel_check=cancel_check, target_hwnd=hwnd,
            remote=anydesk_window or hosted_remote,
        ):
            return

    # For normal local GUI controls, try direct insertion first
    check_insertion_cancelled(cancel_check)
    _check_insertion_target(hwnd)
    if (not console_window and not anydesk_window and not hosted_remote
            and try_direct_text_insert(text, hwnd, cancel_check=cancel_check)):
        return

    # Fallbacks are cancellation- and focus-aware immediately before delivery.
    if try_sendinput_paste(text, cancel_check=cancel_check, target_hwnd=hwnd):
        return
    if try_clipboard_paste(text, cancel_check=cancel_check, target_hwnd=hwnd):
        return
    check_insertion_cancelled(cancel_check)
    _check_insertion_target(hwnd)
    # pyautogui.write silently drops unsupported Unicode; do not claim success
    # for a damaged transcript. ASCII last-resort typing is bounded/cancellable.
    if not text.isascii() or "\n" in text or "\r" in text:
        raise InputInsertionError("No safe Unicode insertion method is available. Use Copy last transcription.")
    accepted = 0
    started = time.monotonic()
    for offset in range(0, len(text), _LOCAL_BATCH_UNITS):
        check_insertion_cancelled(cancel_check, events_inserted=accepted)
        _check_insertion_target(hwnd, accepted)
        if time.monotonic() - started >= _INSERTION_DEADLINE_SECONDS:
            raise InputInsertionError("Direct typing exceeded its safety deadline.",
                                      events_inserted=accepted)
        chunk = text[offset:offset + _LOCAL_BATCH_UNITS]
        try:
            pyautogui.write(chunk)
        except Exception as exc:
            raise InputInsertionError("Direct typing delivery could not be confirmed.",
                                      events_inserted=accepted, may_have_inserted=True) from exc
        accepted += len(chunk) * 2
