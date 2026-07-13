"""X11 text injection and active-field capture for CtrlSpeak on Linux.

The adapter deliberately refuses native Wayland sessions. Clipboard-based
operations prefer the external ``xclip`` utility. When it is unavailable, a
short-lived withdrawn tkinter root owns and services the X11 clipboard until
the paste/copy operation and prior-text restoration are complete. When the
clipboard contains non-text formats, CtrlSpeak avoids overwriting it and falls
back to direct ASCII typing where possible.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
import uuid
from typing import Any, Optional

from utils.config_paths import get_logger
from utils.hotkeys import DesktopSessionError, ensure_desktop_automation_supported


logger = get_logger(__name__)

_XCLIP_TIMEOUT_SECONDS = 3.0
_TK_EVENT_POLL_SECONDS = 0.01
_CLIPBOARD_LOCK = threading.RLock()
_TEXT_TARGETS = {
    "UTF8_STRING",
    "STRING",
    "TEXT",
    "COMPOUND_TEXT",
    "text/plain",
    "text/plain;charset=utf-8",
}
_METADATA_TARGETS = {
    "TARGETS",
    "TIMESTAMP",
    "MULTIPLE",
    "SAVE_TARGETS",
    "LENGTH",
    "LIST_LENGTH",
    "DELETE",
    "INSERT_SELECTION",
}


def set_force_sendinput(flag: bool) -> None:
    if flag:
        logger.warning("--force-sendinput is Windows-only; using the Linux X11 adapter")


def get_focused_control() -> None:
    return None


def is_console_window(_hwnd: int | None) -> bool:
    return False


def get_pyautogui() -> Any:
    try:
        import pyautogui
    except Exception as exc:
        raise DesktopSessionError(
            "CtrlSpeak could not initialise pyautogui for X11 text input. Launch it inside "
            f"Ubuntu on Xorg and verify the Python dependencies. {exc.__class__.__name__}: {exc}"
        ) from exc
    return pyautogui


def _xclip_path() -> str | None:
    return shutil.which("xclip")


def clipboard_tool_available() -> bool:
    """Return whether the preferred external X11 clipboard tool is installed."""
    return _xclip_path() is not None


def _run_xclip(*args: str, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    executable = _xclip_path()
    if executable is None:
        raise DesktopSessionError(
            "CtrlSpeak needs xclip for clipboard-preserving paste and edit feedback on X11. "
            "Install it with 'sudo apt install xclip', then restart CtrlSpeak."
        )
    try:
        return subprocess.run(
            [executable, "-selection", "clipboard", *args],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_XCLIP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DesktopSessionError(f"xclip could not access the X11 clipboard: {exc}") from exc


def _clipboard_targets() -> set[str]:
    result = _run_xclip("-o", "-target", "TARGETS")
    if result.returncode != 0:
        return set()
    return {
        line.strip()
        for line in result.stdout.decode("utf-8", errors="replace").splitlines()
        if line.strip()
    }


def _is_text_target(target: str) -> bool:
    return target in _TEXT_TARGETS or target.lower().startswith("text/")


def _tk_clipboard_error(action: str, exc: BaseException) -> DesktopSessionError:
    return DesktopSessionError(
        f"CtrlSpeak could not {action} because xclip is unavailable and the Tk X11 "
        "clipboard fallback could not access DISPLAY. Verify DISPLAY points to an "
        "active Ubuntu on Xorg session and use a Python/packaged build with tkinter "
        f"support. {exc.__class__.__name__}: {exc}"
    )


def _create_tk_clipboard_root() -> Any:
    """Create a dedicated Tk interpreter without importing it at module startup."""
    try:
        import tkinter as tk
    except Exception as exc:
        raise _tk_clipboard_error("load tkinter", exc) from exc

    try:
        return tk.Tk()
    except Exception as exc:
        raise _tk_clipboard_error("open the X11 clipboard", exc) from exc


class _TkClipboardSession:
    """Own an invisible Tk root for one complete X11 clipboard transaction."""

    def __init__(self, root: Any) -> None:
        self.root = root
        self._closed = False

    def __enter__(self) -> "_TkClipboardSession":
        try:
            # Withdraw before the first event update so the root is never mapped.
            self.root.withdraw()
            self.update_events()
        except DesktopSessionError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise _tk_clipboard_error("initialise the hidden clipboard window", exc) from exc
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.root.destroy()
        except Exception as exc:
            logger.warning("Failed to clean up the hidden Tk clipboard root: %s", exc)

    def update_events(self) -> None:
        """Service X11 selection requests while this root owns the clipboard."""
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception as exc:
            raise _tk_clipboard_error("service X11 clipboard events", exc) from exc

    def wait_with_events(self, seconds: float) -> None:
        """Wait briefly without starving Tk's X11 selection event handling."""
        self.update_events()
        remaining = max(0.0, seconds)
        while remaining > 0:
            interval = min(_TK_EVENT_POLL_SECONDS, remaining)
            time.sleep(interval)
            self.update_events()
            remaining -= interval

    def targets(self) -> set[str]:
        try:
            raw_targets = self.root.tk.call(
                "selection",
                "get",
                "-selection",
                "CLIPBOARD",
                "-type",
                "TARGETS",
            )
            values = self.root.tk.splitlist(raw_targets)
        except Exception:
            # No selection owner (an empty clipboard) commonly reports an error.
            return set()
        return {str(value).strip() for value in values if str(value).strip()}

    def contains_non_text_data(self) -> bool:
        meaningful = {
            target
            for target in self.targets()
            if target not in _METADATA_TARGETS and not _is_text_target(target)
        }
        return bool(meaningful)

    def get_text(self) -> Optional[str]:
        try:
            value = self.root.clipboard_get()
        except Exception as exc:
            targets = self.targets()
            if any(_is_text_target(target) for target in targets):
                raise _tk_clipboard_error("read text from the X11 clipboard", exc) from exc
            return None
        return value if isinstance(value, str) else str(value)

    def set_text(self, text: str) -> None:
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            # clipboard_append only establishes ownership; update must run so
            # the paste target can request the Unicode selection payload.
            self.update_events()
        except DesktopSessionError:
            raise
        except Exception as exc:
            raise _tk_clipboard_error("set text on the X11 clipboard", exc) from exc

    def restore_text(self, previous: Optional[str]) -> None:
        self.set_text(previous or "")


def _new_tk_clipboard_session() -> _TkClipboardSession:
    return _TkClipboardSession(_create_tk_clipboard_root())


def clipboard_contains_non_text_data() -> bool:
    ensure_desktop_automation_supported()
    if clipboard_tool_available():
        targets = _clipboard_targets()
        meaningful = {
            target
            for target in targets
            if target not in _METADATA_TARGETS and not _is_text_target(target)
        }
        return bool(meaningful)
    with _CLIPBOARD_LOCK, _new_tk_clipboard_session() as clipboard:
        return clipboard.contains_non_text_data()


def get_clipboard_text() -> Optional[str]:
    ensure_desktop_automation_supported()
    if clipboard_tool_available():
        targets = _clipboard_targets()
        target_order = (
            "UTF8_STRING",
            "text/plain;charset=utf-8",
            "text/plain",
            "COMPOUND_TEXT",
            "STRING",
            "TEXT",
        )
        for target in target_order:
            if targets and target not in targets:
                continue
            result = _run_xclip("-o", "-target", target)
            if result.returncode == 0:
                return result.stdout.decode("utf-8", errors="replace")
        return None
    with _CLIPBOARD_LOCK, _new_tk_clipboard_session() as clipboard:
        return clipboard.get_text()


def set_clipboard_text(text: str) -> bool:
    ensure_desktop_automation_supported()
    if clipboard_tool_available():
        result = _run_xclip(
            "-in",
            "-target",
            "UTF8_STRING",
            input_bytes=text.encode("utf-8"),
        )
        if result.returncode != 0:
            logger.warning(
                "xclip failed to set clipboard text: %s",
                result.stderr.decode("utf-8", errors="replace").strip(),
            )
            return False
        return True
    with _CLIPBOARD_LOCK, _new_tk_clipboard_session() as clipboard:
        clipboard.set_text(text)
        clipboard.wait_with_events(_TK_EVENT_POLL_SECONDS)
    return True


def restore_clipboard_text(previous: Optional[str]) -> None:
    if not set_clipboard_text(previous or ""):
        logger.warning("Failed to restore the previous X11 text clipboard")


def _type_directly_or_raise(text: str, *, reason: str) -> None:
    if text.isascii() and "\n" not in text and "\r" not in text:
        get_pyautogui().write(text, interval=0.001)
        return
    raise DesktopSessionError(
        f"{reason} CtrlSpeak preserved the existing clipboard but cannot safely type this "
        "Unicode or multiline transcript directly. Use xclip or the included Tk fallback "
        "inside an Ubuntu on Xorg desktop."
    )


def _insert_text_with_tk_fallback(text: str) -> None:
    try:
        clipboard_session = _new_tk_clipboard_session()
    except DesktopSessionError:
        # Preserve the pre-v0.4 safe last resort if Tk itself cannot start.
        if text.isascii() and "\n" not in text and "\r" not in text:
            get_pyautogui().write(text, interval=0.001)
            return
        raise

    with _CLIPBOARD_LOCK, clipboard_session as clipboard:
        if clipboard.contains_non_text_data():
            _type_directly_or_raise(
                text,
                reason="The clipboard contains non-text data that CtrlSpeak will not overwrite.",
            )
            return

        previous = clipboard.get_text()
        try:
            clipboard.set_text(text)
            get_pyautogui().hotkey("ctrl", "v")
            clipboard.wait_with_events(0.08)
        finally:
            clipboard.restore_text(previous)
            clipboard.wait_with_events(_TK_EVENT_POLL_SECONDS)


def insert_text_into_focus(text: str) -> None:
    if not text:
        return
    ensure_desktop_automation_supported()
    if not clipboard_tool_available():
        _insert_text_with_tk_fallback(text)
        return
    if clipboard_contains_non_text_data():
        _type_directly_or_raise(
            text,
            reason="The clipboard contains non-text data that CtrlSpeak will not overwrite.",
        )
        return

    previous = get_clipboard_text()
    try:
        if not set_clipboard_text(text):
            _type_directly_or_raise(text, reason="CtrlSpeak could not stage the X11 clipboard.")
            return
        get_pyautogui().hotkey("ctrl", "v")
        time.sleep(0.08)
    finally:
        restore_clipboard_text(previous)


def snapshot_active_text_field(*, copy_wait_seconds: float = 0.08) -> Optional[str]:
    """Best-effort X11 Ctrl+A/C capture that never sends Enter."""
    try:
        ensure_desktop_automation_supported()
        if not clipboard_tool_available():
            with _CLIPBOARD_LOCK, _new_tk_clipboard_session() as clipboard:
                if clipboard.contains_non_text_data():
                    logger.info(
                        "Skipping active-field capture to preserve non-text clipboard data"
                    )
                    return None

                previous = clipboard.get_text()
                marker = f"CtrlSpeak-field-capture-{uuid.uuid4()}"
                try:
                    clipboard.set_text(marker)
                    automation = get_pyautogui()
                    automation.hotkey("ctrl", "a")
                    automation.hotkey("ctrl", "c")
                    clipboard.wait_with_events(copy_wait_seconds)
                    captured = clipboard.get_text()
                    return captured if captured is not None and captured != marker else None
                finally:
                    clipboard.restore_text(previous)
                    clipboard.wait_with_events(_TK_EVENT_POLL_SECONDS)
        if clipboard_contains_non_text_data():
            logger.info("Skipping active-field capture to preserve non-text clipboard data")
            return None

        previous = get_clipboard_text()
        marker = f"CtrlSpeak-field-capture-{uuid.uuid4()}"
        try:
            if not set_clipboard_text(marker):
                return None
            automation = get_pyautogui()
            automation.hotkey("ctrl", "a")
            automation.hotkey("ctrl", "c")
            if copy_wait_seconds > 0:
                time.sleep(copy_wait_seconds)
            captured = get_clipboard_text()
            return captured if captured is not None and captured != marker else None
        finally:
            restore_clipboard_text(previous)
    except DesktopSessionError as exc:
        logger.warning("Active-field feedback capture is unavailable: %s", exc)
        return None
    except Exception:
        logger.exception("Failed to snapshot the active X11 text field")
        return None
