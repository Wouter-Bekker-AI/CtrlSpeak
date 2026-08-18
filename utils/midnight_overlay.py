"""Midnight Signal recording/transcription capsule.

The overlay is deliberately a small, focus-safe status surface.  Provider
telemetry is only shown after the gateway returns it; while a request is in
flight the animation communicates activity without pretending that a
particular provider has already been selected.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import tkinter as tk

from utils.ui_state import UiPhase, UiSnapshot


CHROMA_KEY = "#010203"
SURFACE = "#111820"
SURFACE_RAISED = "#17212B"
OUTLINE = "#30404D"
TEXT = "#F3F7FA"
TEXT_MUTED = "#91A2AF"
CYAN = "#49D7E8"
MINT = "#71E6BA"
AMBER = "#F5C56B"
CORAL = "#FF7A79"


@dataclass(frozen=True)
class MonitorBounds:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(1, self.right - self.left)

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)


def _active_monitor_bounds(root: tk.Misc) -> MonitorBounds:
    """Return the active/cursor monitor work area, with a portable fallback."""

    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT),
                    ("dwFlags", wintypes.DWORD),
                ]

            point = wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
            monitor = ctypes.windll.user32.MonitorFromPoint(point, 2)
            info = MONITORINFO()
            info.cbSize = ctypes.sizeof(info)
            if monitor and ctypes.windll.user32.GetMonitorInfoW(
                monitor, ctypes.byref(info)
            ):
                work = info.rcWork
                return MonitorBounds(work.left, work.top, work.right, work.bottom)
        except Exception:
            pass
    return MonitorBounds(0, 0, int(root.winfo_screenwidth()), int(root.winfo_screenheight()))


def active_monitor_bounds(root: tk.Misc) -> MonitorBounds:
    """Public wrapper used by the tray flyout and UI screenshot tests."""

    return _active_monitor_bounds(root)


def display_scale(widget: tk.Misc) -> float:
    """Return the Tk point-to-pixel scale relative to a 96-DPI display."""

    try:
        value = float(widget.tk.call("tk", "scaling")) / (96.0 / 72.0)
    except Exception:
        value = 1.0
    return max(0.75, min(3.0, value))


def flyout_geometry(
    bounds: MonitorBounds,
    scale: float,
) -> tuple[int, int, int, int, bool]:
    """Return a fully visible tray-flyout geometry and whether it must scroll."""

    scale = max(0.75, min(3.0, float(scale)))
    desired_width = max(1, int(round(430 * scale)))
    desired_height = max(1, int(round(558 * scale)))
    max_margin = max(0, (min(bounds.width, bounds.height) - 1) // 2)
    margin = min(max(8, int(round(18 * scale))), max_margin)
    available_width = max(1, bounds.width - (2 * margin))
    available_height = max(1, bounds.height - (2 * margin))
    width = min(desired_width, available_width)
    height = min(desired_height, available_height)
    x = max(bounds.left, bounds.right - width - margin)
    y = max(bounds.top, bounds.bottom - height - margin)
    return x, y, width, height, (width < desired_width or height < desired_height)


def _rounded_rectangle(
    canvas: tk.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: float,
    **kwargs,
) -> int:
    radius = max(1.0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    points: Sequence[float] = (
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    )
    return canvas.create_polygon(points, smooth=True, splinesteps=24, **kwargs)


def _resolve_windows_toplevel_hwnd(
    window: tk.Toplevel,
    *,
    get_root: Callable[[int], int] | None = None,
) -> int:
    """Resolve Tk's outer Win32 wrapper rather than its client child HWND."""

    hwnd = int(window.winfo_id())
    if get_root is None:
        import ctypes
        from ctypes import wintypes

        # GA_ROOT follows only the parent chain.  Unlike GetParent/GetWindow it
        # cannot climb from an owned WS_POPUP into CtrlSpeak's management root.
        get_ancestor = ctypes.windll.user32.GetAncestor
        get_ancestor.argtypes = (wintypes.HWND, wintypes.UINT)
        get_ancestor.restype = wintypes.HWND
        get_root = lambda value: int(get_ancestor(value, 2) or 0)
    root_hwnd = int(get_root(hwnd) or 0)
    return root_hwnd or hwnd


def _make_focus_safe(window: tk.Toplevel) -> int | None:
    """Use Windows extended styles so the capsule never steals keyboard focus."""

    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        window.update_idletasks()
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        get_ancestor = user32.GetAncestor
        get_ancestor.argtypes = (wintypes.HWND, wintypes.UINT)
        get_ancestor.restype = wintypes.HWND
        hwnd = _resolve_windows_toplevel_hwnd(
            window,
            get_root=lambda value: int(get_ancestor(value, 2) or 0),
        )
        get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        get_style.argtypes = (wintypes.HWND, ctypes.c_int)
        get_style.restype = ctypes.c_ssize_t
        set_style.argtypes = (wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t)
        set_style.restype = ctypes.c_ssize_t
        exstyle = get_style(hwnd, -20)
        # WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_TRANSPARENT
        set_style(hwnd, -20, exstyle | 0x00000080 | 0x08000000 | 0x00000020)
        return hwnd
    except Exception:
        return None


class MidnightSignalOverlay:
    """Render a compact, state-driven overlay on the active monitor."""

    WIDTH = 548
    HEIGHT = 88

    def __init__(
        self,
        root: tk.Misc,
        *,
        snapshot_provider: Callable[[], UiSnapshot],
        waveform_provider: Callable[[], object] | None = None,
        device_label_provider: Callable[[], str | None] | None = None,
        reduced_motion: bool = False,
    ) -> None:
        self.root = root
        self.snapshot_provider = snapshot_provider
        self.waveform_provider = waveform_provider
        self.device_label_provider = device_label_provider
        self.reduced_motion = bool(reduced_motion)
        self.scale = display_scale(root)
        self.window: tk.Toplevel | None = None
        self.canvas: tk.Canvas | None = None
        self._job: str | None = None
        self._close_job: str | None = None
        self._started = time.monotonic()
        self._closing = False

    def is_open(self) -> bool:
        try:
            return bool(self.window and self.window.winfo_exists())
        except Exception:
            return False

    def show(self) -> None:
        if self.is_open():
            return
        window = tk.Toplevel(self.root, class_="CtrlSpeakOverlay")
        window.title("CtrlSpeak overlay")
        window.withdraw()
        window.overrideredirect(True)
        window.attributes("-topmost", True)
        try:
            window.attributes("-alpha", 0.97)
        except tk.TclError:
            pass
        try:
            if sys.platform.startswith("win"):
                window.wm_attributes("-transparentcolor", CHROMA_KEY)
        except tk.TclError:
            pass
        bounds = _active_monitor_bounds(self.root)
        width = min(int(round(self.WIDTH * self.scale)), max(360, bounds.width - 40))
        height = int(round(self.HEIGHT * self.scale))
        x = bounds.left + (bounds.width - width) // 2
        y = bounds.bottom - height - max(int(34 * self.scale), int(bounds.height * 0.045))
        window.geometry(f"{width}x{height}+{x}+{y}")
        canvas = tk.Canvas(
            window,
            width=width,
            height=height,
            background=CHROMA_KEY,
            highlightthickness=0,
            borderwidth=0,
        )
        canvas.pack(fill=tk.BOTH, expand=True)
        self.window = window
        self.canvas = canvas
        self._closing = False
        self._started = time.monotonic()
        # The no-activate style must be on Tk's outer wrapper before the first
        # visible presentation.  Applying it after deiconify can momentarily
        # steal focus from the field that should receive the transcription.
        _make_focus_safe(window)
        window.deiconify()
        self._tick()

    def set_waveform_provider(self, provider: Callable[[], object] | None) -> None:
        self.cancel_pending_close()
        self._closing = False
        self._started = time.monotonic()
        self.waveform_provider = provider

    def cancel_pending_close(self) -> None:
        window, close_job = self.window, self._close_job
        self._close_job = None
        if window is not None and close_job:
            try:
                window.after_cancel(close_job)
            except tk.TclError:
                pass

    def close(self, *, delay_ms: int = 0) -> None:
        if not self.is_open():
            return
        if delay_ms > 0:
            assert self.window is not None
            self.cancel_pending_close()

            def finish_close() -> None:
                self._close_job = None
                self.close()

            self._close_job = self.window.after(delay_ms, finish_close)
            return
        self.cancel_pending_close()
        self._closing = True
        window, canvas, job = self.window, self.canvas, self._job
        self._job = None
        if canvas is not None and job:
            try:
                canvas.after_cancel(job)
            except tk.TclError:
                pass
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass
        self.window = None
        self.canvas = None

    def _tick(self) -> None:
        if self._closing or not self.is_open() or self.canvas is None:
            return
        try:
            snapshot = self.snapshot_provider()
            self._draw(snapshot)
        finally:
            if not self._closing and self.canvas is not None:
                self._job = self.canvas.after(50 if self.reduced_motion else 33, self._tick)

    def _draw(self, state: UiSnapshot) -> None:
        canvas = self.canvas
        if canvas is None:
            return
        width = max(1, canvas.winfo_width()) / self.scale
        height = max(1, canvas.winfo_height()) / self.scale
        canvas.delete("all")
        _rounded_rectangle(
            canvas, 3, 3, width - 3, height - 3, 24,
            fill=SURFACE, outline=OUTLINE, width=1,
        )
        phase = state.phase
        if phase is UiPhase.RECORDING:
            self._draw_recording(canvas, width, height, state)
        elif phase is UiPhase.PROCESSING:
            self._draw_processing(canvas, width, height, state)
        elif phase is UiPhase.SUCCESS:
            self._draw_result(canvas, width, height, state, success=True)
        elif phase is UiPhase.ERROR:
            self._draw_result(canvas, width, height, state, success=False)
        elif phase is UiPhase.CANCELLED:
            self._draw_cancelled(canvas, width, height, state)
        else:
            self._draw_idle(canvas, width, height)
        if self.scale != 1.0:
            canvas.scale("all", 0, 0, self.scale, self.scale)

    @staticmethod
    def _text(canvas: tk.Canvas, x: float, y: float, text: str, **kwargs) -> int:
        options = {
            "anchor": "w",
            "fill": TEXT,
            "font": ("Segoe UI", 11),
        }
        options.update(kwargs)
        return canvas.create_text(x, y, text=text, **options)

    def _draw_recording(
        self, canvas: tk.Canvas, width: int, height: int, state: UiSnapshot
    ) -> None:
        canvas.create_oval(22, 21, 66, 65, fill=SURFACE_RAISED, outline=CYAN, width=2)
        canvas.create_line(44, 31, 44, 48, fill=CYAN, width=4, capstyle=tk.ROUND)
        canvas.create_arc(35, 37, 53, 56, start=180, extent=180, style=tk.ARC, outline=CYAN, width=2)
        canvas.create_line(44, 56, 44, 60, fill=CYAN, width=2)
        device_label = "System default"
        if self.device_label_provider is not None:
            try:
                device_label = self.device_label_provider() or device_label
            except Exception:
                pass
        if len(device_label) > 26:
            device_label = device_label[:25] + "…"
        self._text(canvas, 82, 31, f"LISTENING · {device_label}", fill=CYAN, font=("Segoe UI Semibold", 9))
        self._text(canvas, 82, 55, "Release Right Ctrl to transcribe", font=("Segoe UI Semibold", 12))

        level = max(0.0, min(1.0, state.level_fraction))
        bar_x = max(300, width - 198)
        for index in range(7):
            bx = bar_x + index * 12
            activity = max(0.12, level - abs(index - 3) * 0.08)
            bh = 8 + activity * 31
            canvas.create_line(
                bx, height / 2 - bh / 2, bx, height / 2 + bh / 2,
                fill=CYAN if activity > 0.23 else OUTLINE,
                width=5, capstyle=tk.ROUND,
            )
        self._text(canvas, width - 92, 32, state.elapsed_label, anchor="e", font=("Segoe UI Semibold", 11))
        self._text(canvas, width - 22, 56, state.level_label, anchor="e", fill=TEXT_MUTED, font=("Segoe UI", 9))

    def _draw_processing(
        self, canvas: tk.Canvas, width: int, height: int, state: UiSnapshot
    ) -> None:
        cx, cy = 48, height / 2
        rx, ry = 26, 14
        canvas.create_oval(cx - rx, cy - ry, cx + rx, cy + ry, outline=OUTLINE, width=2)
        if self.reduced_motion:
            angle = 330.0
        else:
            angle = ((time.monotonic() - self._started) * 235.0) % 360.0
        radians = math.radians(angle)
        dot_x = cx + math.cos(radians) * rx
        dot_y = cy + math.sin(radians) * ry
        canvas.create_oval(dot_x - 4, dot_y - 4, dot_x + 4, dot_y + 4, fill=CYAN, outline="")
        canvas.create_arc(
            cx - rx, cy - ry, cx + rx, cy + ry,
            start=angle - 85, extent=70, style=tk.ARC, outline=CYAN, width=3,
        )
        self._text(canvas, 88, 31, "TRANSCRIBING", fill=CYAN, font=("Segoe UI Semibold", 9))
        self._text(canvas, 88, 55, "Routing through your preferred provider…", font=("Segoe UI Semibold", 12))
        self._text(canvas, width - 22, 32, state.elapsed_label, anchor="e", font=("Segoe UI Semibold", 11))
        self._text(canvas, width - 22, 56, "Cancel from tray", anchor="e", fill=TEXT_MUTED, font=("Segoe UI", 9))

    def _draw_result(
        self, canvas: tk.Canvas, width: int, height: int, state: UiSnapshot, *, success: bool
    ) -> None:
        colour = MINT if success else CORAL
        cx, cy = 45, height / 2
        canvas.create_oval(cx - 20, cy - 20, cx + 20, cy + 20, fill=SURFACE_RAISED, outline=colour, width=2)
        if success:
            canvas.create_line(cx - 9, cy, cx - 2, cy + 7, cx + 11, cy - 9, fill=colour, width=3, capstyle=tk.ROUND, joinstyle=tk.ROUND)
        else:
            canvas.create_line(cx - 8, cy - 8, cx + 8, cy + 8, fill=colour, width=3, capstyle=tk.ROUND)
            canvas.create_line(cx + 8, cy - 8, cx - 8, cy + 8, fill=colour, width=3, capstyle=tk.ROUND)
        self._text(canvas, 82, 31, state.headline.upper(), fill=colour, font=("Segoe UI Semibold", 9))
        detail = state.detail
        if success and state.provider:
            timing = state.provider.latency_label
            detail = f"Inserted · {state.provider.display_name} · {timing}"
            if state.degraded:
                detail += " · fallback"
        self._text(canvas, 82, 55, detail, font=("Segoe UI Semibold", 12))
        self._text(canvas, width - 22, 43, state.elapsed_label, anchor="e", fill=TEXT_MUTED, font=("Segoe UI", 9))

    def _draw_cancelled(
        self, canvas: tk.Canvas, width: int, height: int, state: UiSnapshot
    ) -> None:
        self._text(canvas, 28, 34, "CANCELLED", fill=AMBER, font=("Segoe UI Semibold", 10))
        self._text(canvas, 28, 58, state.detail, font=("Segoe UI", 11))

    def _draw_idle(self, canvas: tk.Canvas, width: int, height: int) -> None:
        self._text(canvas, 28, height / 2, "CtrlSpeak is ready", font=("Segoe UI Semibold", 12))


__all__ = [
    "MidnightSignalOverlay",
    "MonitorBounds",
    "active_monitor_bounds",
    "display_scale",
    "flyout_geometry",
]
