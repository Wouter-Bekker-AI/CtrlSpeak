"""Lazy platform router for desktop text input and clipboard operations."""
from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Optional


_WINDOWS_ADAPTER: ModuleType | None = None
if sys.platform.startswith("win"):
    # Static guarded import keeps the unchanged Windows PyInstaller spec able
    # to discover the adapter. Linux never executes this import, and its v0.4
    # spec also excludes the module explicitly.
    from utils import windows_input as _WINDOWS_ADAPTER


class UnsupportedDesktopPlatformError(RuntimeError):
    """CtrlSpeak has no desktop input adapter for the current platform."""


def platform_adapter_module_name(platform_name: str | None = None) -> str:
    platform_value = (platform_name or sys.platform).lower()
    if platform_value.startswith("win"):
        return "utils.windows_input"
    if platform_value.startswith("linux"):
        return "utils.linux_input"
    raise UnsupportedDesktopPlatformError(
        f"CtrlSpeak desktop input is unsupported on platform {platform_value!r}."
    )


def _adapter() -> ModuleType:
    if sys.platform.startswith("win") and _WINDOWS_ADAPTER is not None:
        return _WINDOWS_ADAPTER
    return importlib.import_module(platform_adapter_module_name())


def set_force_sendinput(flag: bool) -> None:
    _adapter().set_force_sendinput(flag)


def get_focused_control() -> Optional[int]:
    return _adapter().get_focused_control()


def is_console_window(hwnd: int | None) -> bool:
    return bool(_adapter().is_console_window(hwnd))


def get_clipboard_text() -> Optional[str]:
    return _adapter().get_clipboard_text()


def set_clipboard_text(text: str) -> bool:
    return bool(_adapter().set_clipboard_text(text))


def restore_clipboard_text(previous: Optional[str]) -> None:
    _adapter().restore_clipboard_text(previous)


def clipboard_contains_non_text_data() -> bool:
    return bool(_adapter().clipboard_contains_non_text_data())


def snapshot_active_text_field(*, copy_wait_seconds: float = 0.08) -> Optional[str]:
    return _adapter().snapshot_active_text_field(copy_wait_seconds=copy_wait_seconds)


def insert_text_into_focus(text: str) -> None:
    _adapter().insert_text_into_focus(text)
