from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from utils import midnight_overlay


@pytest.mark.core_headless
def test_negative_monitor_geometry_is_sent_to_win32_as_absolute_coordinates(
    monkeypatch,
) -> None:
    geometry = midnight_overlay.overlay_geometry(
        midnight_overlay.MonitorBounds(-1920, -1080, 0, 0),
        1.0,
    )
    calls: list[tuple[object, ...]] = []

    class FakeWindow:
        update_count = 0

        def update_idletasks(self) -> None:
            self.update_count += 1

    window = FakeWindow()
    monkeypatch.setattr(midnight_overlay.sys, "platform", "win32")

    placed = midnight_overlay._place_windows_overlay_absolute(
        window,
        geometry.x,
        geometry.y,
        hwnd=7301,
        set_window_pos=lambda *args: calls.append(args) or 1,
    )

    assert placed is True
    assert geometry.x < 0 and geometry.y < 0
    assert window.update_count == 1
    assert calls == [
        (7301, 0, geometry.x, geometry.y, 0, 0, 0x0015),
    ]


@pytest.mark.core_headless
def test_native_overlay_placement_is_a_safe_noop_off_windows(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeWindow:
        def update_idletasks(self) -> None:
            raise AssertionError("non-Windows placement must not touch Tk")

    monkeypatch.setattr(midnight_overlay.sys, "platform", "linux")

    assert midnight_overlay._place_windows_overlay_absolute(
        FakeWindow(),
        -400,
        -200,
        hwnd=99,
        set_window_pos=lambda *args: calls.append(args) or 1,
    ) is False
    assert calls == []


@pytest.mark.core_headless
def test_overlay_is_focus_hardened_and_native_placed_while_still_hidden() -> None:
    source = inspect.getsource(midnight_overlay.MidnightSignalOverlay.show)

    focus_safe = source.index("_make_focus_safe(window)")
    absolute_placement = source.index("_place_windows_overlay_absolute(")
    first_show = source.index("window.deiconify()")
    assert focus_safe < absolute_placement < first_show


@pytest.mark.full_gui
def test_native_hidden_overlay_maps_at_negative_absolute_x_without_activation() -> None:
    if not sys.platform.startswith("win"):
        pytest.skip("native Windows Tk geometry check")
    script = textwrap.dedent(
        """
        import ctypes
        import tkinter as tk
        from utils.midnight_overlay import (
            _make_focus_safe,
            _place_windows_overlay_absolute,
        )

        root = tk.Tk()
        root.withdraw()
        window = tk.Toplevel(root, class_="CtrlSpeakOverlayGeometryTest")
        window.withdraw()
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        window.attributes("-alpha", 0.0)
        # Tk initially interprets this as a right-edge offset.  The native
        # placement must replace that interpretation before the hidden window
        # is mapped.
        window.geometry("300x100-100-80")
        hwnd = _make_focus_safe(window)
        assert hwnd
        foreground_before = int(ctypes.windll.user32.GetForegroundWindow() or 0)
        assert _place_windows_overlay_absolute(window, -100, -80, hwnd=hwnd)
        window.deiconify()
        root.update()
        foreground_after = int(ctypes.windll.user32.GetForegroundWindow() or 0)

        assert window.winfo_x() == -100
        assert window.winfo_y() == -80
        assert foreground_after == foreground_before

        get_style = getattr(
            ctypes.windll.user32,
            "GetWindowLongPtrW",
            ctypes.windll.user32.GetWindowLongW,
        )
        get_style.restype = ctypes.c_ssize_t
        exstyle = int(get_style(hwnd, -20))
        assert exstyle & 0x08000000  # WS_EX_NOACTIVATE
        window.destroy()
        root.destroy()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
