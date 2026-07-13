"""Cross-platform global-hotkey setup with explicit Linux session checks."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Mapping, Any

from utils.config_paths import get_logger


logger = get_logger(__name__)


class DesktopSessionError(RuntimeError):
    """Desktop automation cannot operate reliably in the current session."""


@dataclass(frozen=True)
class DesktopSessionStatus:
    supported: bool
    session_type: str
    detail: str


def get_desktop_session_status(
    *,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> DesktopSessionStatus:
    """Describe whether CtrlSpeak can use global hooks and injection here."""
    env = os.environ if environ is None else environ
    platform_value = (platform_name or sys.platform).lower()
    if platform_value.startswith("win"):
        return DesktopSessionStatus(True, "windows", "Windows desktop automation is available.")
    if not platform_value.startswith("linux"):
        return DesktopSessionStatus(
            False,
            platform_value,
            f"CtrlSpeak desktop automation is unsupported on {platform_value!r}.",
        )

    declared = str(env.get("XDG_SESSION_TYPE", "")).strip().lower()
    wayland_display = str(env.get("WAYLAND_DISPLAY", "")).strip()
    x_display = str(env.get("DISPLAY", "")).strip()
    if declared == "wayland" or (wayland_display and declared != "x11"):
        return DesktopSessionStatus(
            False,
            "wayland",
            "CtrlSpeak global hotkeys, active-field feedback, and text injection are not "
            "reliable in a native Wayland session. Log out, select the gear icon, choose "
            "Ubuntu on Xorg, then sign in again.",
        )
    if declared == "x11" and not x_display:
        return DesktopSessionStatus(
            False,
            "x11",
            "CtrlSpeak detected X11 but DISPLAY is unset. Start CtrlSpeak inside the signed-in "
            "desktop session instead of a headless shell or service.",
        )
    if x_display:
        return DesktopSessionStatus(
            True,
            "x11",
            "X11 global hotkeys and desktop automation are available.",
        )
    return DesktopSessionStatus(
        False,
        declared or "headless",
        "CtrlSpeak could not find an X11 DISPLAY. Run it inside an Ubuntu on Xorg desktop "
        "session; headless shells and services cannot provide the active-field workflow.",
    )


def ensure_desktop_automation_supported(
    *,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> DesktopSessionStatus:
    status = get_desktop_session_status(
        environ=environ,
        platform_name=platform_name,
    )
    if not status.supported:
        raise DesktopSessionError(status.detail)
    return status


def create_global_listener(
    *,
    on_press: Callable[[Any], None],
    on_release: Callable[[Any], None],
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    listener_factory: Callable[..., Any] | None = None,
) -> Any:
    """Create a non-suppressing pynput listener after validating the session."""
    ensure_desktop_automation_supported(
        environ=environ,
        platform_name=platform_name,
    )
    if listener_factory is None:
        try:
            from pynput import keyboard
        except Exception as exc:
            raise DesktopSessionError(
                "CtrlSpeak could not initialise pynput for global hotkeys. Ensure the Python "
                "dependencies are installed and launch CtrlSpeak inside Ubuntu on Xorg. "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc
        listener_factory = keyboard.Listener
    return listener_factory(
        on_press=on_press,
        on_release=on_release,
        suppress=False,
    )


def key_name(key: Any) -> str:
    """Return stable pynput-style names without importing a platform backend."""
    char = getattr(key, "char", None)
    if isinstance(char, str):
        return char.lower()
    value = str(key).strip().lower()
    return value[4:] if value.startswith("key.") else value


def is_right_control(key: Any) -> bool:
    return key_name(key) == "ctrl_r"
