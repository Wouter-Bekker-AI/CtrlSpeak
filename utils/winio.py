# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import time
import ctypes
from ctypes import wintypes
from pathlib import Path
from typing import Optional

import pyautogui

from utils.config_paths import get_logger

logger = get_logger(__name__)

# Windows GUI helpers
user32 = ctypes.windll.user32 if sys.platform.startswith("win") else None
kernel32 = ctypes.windll.kernel32 if sys.platform.startswith("win") else None
psapi = ctypes.windll.psapi if sys.platform.startswith("win") else None

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
EM_SETSEL = 0x00B1
EM_REPLACESEL = 0x00C2

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_V = 0x56
KEYEVENTF_UNICODE = 0x0004

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
    _send_input = user32.SendInput
    _send_input.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    _send_input.restype = wintypes.UINT

_FORCE_SENDINPUT = False


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


def open_clipboard() -> bool:
    if not sys.platform.startswith("win"):
        return False
    for _ in range(5):
        if user32.OpenClipboard(None):
            return True
        time.sleep(0.01)
    return False


def get_clipboard_text() -> Optional[str]:
    if not sys.platform.startswith("win"):
        return None
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
    if not open_clipboard():
        return False
    try:
        user32.EmptyClipboard()
        data = ctypes.create_unicode_buffer(text)
        size = ctypes.sizeof(ctypes.c_wchar) * len(data)
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            return False
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            kernel32.GlobalFree(handle)
            return False
        ctypes.memmove(pointer, ctypes.byref(data), size)
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)
            return False
        return True
    finally:
        user32.CloseClipboard()


def restore_clipboard_text(previous: Optional[str]) -> None:
    if not sys.platform.startswith("win"):
        return
    if previous is None:
        if open_clipboard():
            try:
                user32.EmptyClipboard()
            finally:
                user32.CloseClipboard()
        return
    set_clipboard_text(previous)


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


def send_unicode_input(text: str) -> bool:
    if not sys.platform.startswith("win"):
        return False
    hwnd = get_focused_control()
    ensure_foreground(hwnd)
    for ch in text.replace("\n", "\r"):
        code = ord(ch)
        union = _INPUTUNION()
        union.ki = KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE, time=0, dwExtraInfo=0)
        inp = INPUT(type=INPUT_KEYBOARD, union=union)
        if _send_input(1, ctypes.byref(inp), ctypes.sizeof(inp)) != 1:
            return False
        union = _INPUTUNION()
        union.ki = KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=0)
        inp = INPUT(type=INPUT_KEYBOARD, union=union)
        if _send_input(1, ctypes.byref(inp), ctypes.sizeof(inp)) != 1:
            return False
        time.sleep(0.01)
    time.sleep(0.02)
    return True


def try_sendinput_paste(text: str) -> bool:
    if not sys.platform.startswith("win"):
        return False
    hwnd = get_focused_control()
    ensure_foreground(hwnd)
    previous = get_clipboard_text()
    if not set_clipboard_text(text):
        return False
    try:
        if not (send_input_key(VK_CONTROL) and send_input_key(VK_V) and send_input_key(VK_V, True) and send_input_key(VK_CONTROL, True)):
            return False
        time.sleep(0.15)
        return True
    finally:
        restore_clipboard_text(previous)


def try_clipboard_paste(text: str) -> bool:
    if not sys.platform.startswith("win"):
        return False
    previous = get_clipboard_text()
    if not set_clipboard_text(text):
        return False
    try:
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.05)
        return True
    finally:
        restore_clipboard_text(previous)


def try_direct_text_insert(text: str, hwnd: Optional[int] = None) -> bool:
    if not sys.platform.startswith("win"):
        return False
    if hwnd is None:
        hwnd = get_focused_control()
    if not hwnd:
        return False
    class_name = get_class_name(hwnd)
    if not class_name:
        return False
    try:
        if any(token in class_name for token in ("edit", "richedit", "notepad", "textarea", "tmemo", "scintilla")):
            user32.SendMessageW(hwnd, EM_SETSEL, 0xFFFFFFFF, 0xFFFFFFFF)
            user32.SendMessageW(hwnd, EM_REPLACESEL, True, text)
            return True
    except Exception:
        logger.exception("Direct text insertion failed for hwnd=%s", hwnd)
        return False
    return False


def insert_text_into_focus(text: str) -> None:
    if not text:
        return
    if not sys.platform.startswith("win"):
        pyautogui.write(text)
        return

    hwnd = get_focused_control()
    console_window = bool(hwnd and is_console_window(hwnd))
    hosted_remote = bool(hwnd and is_remote_host_window(hwnd))
    anydesk_window = bool(hwnd and is_anydesk_window(hwnd))

    # Prefer SendInput for consoles/WSL, AnyDesk (explicit), and remote-hosted windows, or when forced by CLI
    if hwnd and (_FORCE_SENDINPUT or anydesk_window or console_window or hosted_remote):
        if send_unicode_input(text):
            return
        if try_sendinput_paste(text):
            return

    # For normal local GUI controls, try direct insertion first
    if not console_window and not anydesk_window and not hosted_remote and try_direct_text_insert(text, hwnd):
        return

    # If forced or remote-hosted / AnyDesk (but not console), try SendInput again
    if (_FORCE_SENDINPUT or anydesk_window or hosted_remote) and not console_window:
        if send_unicode_input(text):
            return

    # Fallbacks
    if try_sendinput_paste(text):
        return
    if try_clipboard_paste(text):
        return
    pyautogui.write(text)
