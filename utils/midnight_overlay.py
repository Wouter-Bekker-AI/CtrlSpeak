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
WAVEFORM_BAR_COUNT = 10


@dataclass(frozen=True)
class RecordingCapsuleLayout:
    """Logical (pre-DPI) anchors for the approved recording capsule."""

    mic_x: float
    divider_one_x: float
    timer_x: float
    divider_two_x: float
    waveform_left: float
    waveform_right: float
    status_x: float
    dbfs_x: float
    compact: bool = False


@dataclass(frozen=True)
class ProcessingCapsuleLayout:
    """Logical anchors for the generic, provider-neutral processing comet."""

    mic_x: float
    divider_x: float
    orbit_cx: float
    orbit_cy: float
    orbit_rx: float
    orbit_ry: float
    status_x: float
    elapsed_x: float
    compact: bool = False


@dataclass(frozen=True)
class OverlayGeometry:
    """A focus-safe capsule rectangle constrained to one monitor work area."""

    x: int
    y: int
    width: int
    height: int
    compact: bool


@dataclass(frozen=True)
class ResultCapsuleLayout:
    """Text and timer columns for success, error, and cancellation states."""

    icon_x: float
    icon_radius: float
    text_x: float
    detail_right: float
    elapsed_left: float
    elapsed_x: float
    compact: bool

    @property
    def detail_width(self) -> float:
        return max(1.0, self.detail_right - self.text_x)


def recording_capsule_layout(width: float, height: float) -> RecordingCapsuleLayout:
    """Return a stable horizontal hierarchy that scales down without overlap."""

    width = max(160.0, float(width))
    height = max(48.0, float(height))
    compact = width < 470.0 or height < 76.0
    if compact:
        # On a narrow/high-DPI work area the second line carries the instruction,
        # leaving the timer, real waveform, and dBFS meter their own top-row lanes.
        waveform_right = max(137.0, width - 91.0)
        return RecordingCapsuleLayout(
            mic_x=26.0,
            divider_one_x=50.0,
            timer_x=61.0,
            divider_two_x=111.0,
            waveform_left=126.0,
            waveform_right=waveform_right,
            status_x=61.0,
            dbfs_x=width - 12.0,
            compact=True,
        )
    horizontal_scale = min(1.0, width / 548.0)
    return RecordingCapsuleLayout(
        mic_x=31.0 * horizontal_scale,
        divider_one_x=62.0 * horizontal_scale,
        timer_x=77.0 * horizontal_scale,
        divider_two_x=129.0 * horizontal_scale,
        waveform_left=148.0 * horizontal_scale,
        waveform_right=274.0 * horizontal_scale,
        status_x=292.0 * horizontal_scale,
        dbfs_x=width - 20.0,
        compact=False,
    )


def processing_capsule_layout(width: float, height: float) -> ProcessingCapsuleLayout:
    """Return the elongated Midnight Signal orbit and adjacent copy anchors."""

    width = max(160.0, float(width))
    height = max(48.0, float(height))
    compact = width < 470.0 or height < 76.0
    if compact:
        orbit_rx = max(27.0, min(52.0, (width - 92.0) * 0.24))
        return ProcessingCapsuleLayout(
            mic_x=26.0,
            divider_x=50.0,
            orbit_cx=58.0 + orbit_rx,
            orbit_cy=height * 0.43,
            orbit_rx=orbit_rx,
            orbit_ry=max(11.0, min(17.0, height * 0.2)),
            status_x=61.0,
            elapsed_x=width - 12.0,
            compact=True,
        )
    horizontal_scale = min(1.0, width / 548.0)
    return ProcessingCapsuleLayout(
        mic_x=31.0 * horizontal_scale,
        divider_x=62.0 * horizontal_scale,
        orbit_cx=157.0 * horizontal_scale,
        orbit_cy=height / 2.0,
        orbit_rx=74.0 * horizontal_scale,
        orbit_ry=20.0,
        status_x=250.0 * horizontal_scale,
        elapsed_x=width - 20.0,
        compact=False,
    )


def overlay_geometry(bounds: "MonitorBounds", scale: float) -> OverlayGeometry:
    """Fit the capsule wholly inside the active monitor's physical work area.

    The returned dimensions are physical pixels.  Compactness is determined
    against the corresponding DPI-scaled design dimensions, so an 800-pixel
    work area at 200% scaling selects the narrow composition instead of
    drawing a clipped 1096-pixel capsule.
    """

    scale = max(0.75, min(3.0, float(scale)))
    desired_width = max(1, int(round(548 * scale)))
    desired_height = max(1, int(round(88 * scale)))
    max_horizontal_margin = max(0, (bounds.width - 1) // 2)
    horizontal_margin = min(
        max(8, int(round(20 * scale))), max_horizontal_margin
    )
    available_width = max(1, bounds.width - (2 * horizontal_margin))
    width = min(desired_width, available_width)

    # The preferred bottom lift mirrors the original tray-adjacent position,
    # but is reduced as necessary rather than pushing the capsule above rcWork.
    preferred_bottom_gap = max(
        int(round(34 * scale)), int(round(bounds.height * 0.045))
    )
    height = min(desired_height, bounds.height)
    bottom_gap = min(preferred_bottom_gap, max(0, bounds.height - height))

    x = bounds.left + max(0, (bounds.width - width) // 2)
    y = bounds.bottom - height - bottom_gap
    x = min(max(bounds.left, x), max(bounds.left, bounds.right - width))
    y = min(max(bounds.top, y), max(bounds.top, bounds.bottom - height))
    compact = width < int(round(470 * scale)) or height < int(round(76 * scale))
    return OverlayGeometry(x=x, y=y, width=width, height=height, compact=compact)


def overlay_geometry_spec(geometry: OverlayGeometry) -> str:
    """Format a Tk geometry safely for monitors with negative coordinates."""

    return (
        f"{geometry.width}x{geometry.height}"
        f"{geometry.x:+d}{geometry.y:+d}"
    )


def result_capsule_layout(width: float, height: float) -> ResultCapsuleLayout:
    """Reserve a non-overlapping elapsed column for every terminal state."""

    width = max(120.0, float(width))
    height = max(40.0, float(height))
    compact = width < 470.0 or height < 76.0
    icon_x = 28.0 if compact else 45.0
    icon_radius = 14.0 if compact else 20.0
    text_x = 51.0 if compact else 82.0
    elapsed_x = width - (12.0 if compact else 22.0)
    elapsed_column_width = 66.0 if compact else 82.0
    elapsed_left = max(text_x + 12.0, elapsed_x - elapsed_column_width)
    detail_right = max(text_x + 1.0, elapsed_left - 10.0)
    return ResultCapsuleLayout(
        icon_x=icon_x,
        icon_radius=icon_radius,
        text_x=text_x,
        detail_right=detail_right,
        elapsed_left=elapsed_left,
        elapsed_x=elapsed_x,
        compact=compact,
    )


def fit_overlay_text(value: object, *, pixel_width: float) -> str:
    """Return a deterministic, single-line rendering that fits its lane.

    The immutable :class:`UiSnapshot` retains the complete safe detail for the
    tray and management window; only this transient focus-safe capsule is
    condensed.
    """

    text = " ".join(str(value or "").split())
    max_characters = max(1, int(max(1.0, float(pixel_width)) // 6.8))
    if len(text) <= max_characters:
        return text
    if max_characters == 1:
        return "…"
    return text[: max_characters - 1].rstrip() + "…"


def waveform_bar_levels(
    samples: object,
    *,
    count: int = WAVEFORM_BAR_COUNT,
    fallback: float = 0.0,
) -> tuple[float, ...]:
    """Downsample a transient waveform into bounded display-only bar levels.

    The input is consumed immediately and never retained.  Invalid values fall
    back to the already-sanitized dBFS fraction from :class:`UiSnapshot`.
    """

    count = max(1, int(count))
    fallback = max(0.0, min(1.0, float(fallback)))
    try:
        flattened = samples.ravel() if hasattr(samples, "ravel") else samples
        values = [abs(float(value)) for value in flattened]  # type: ignore[union-attr]
        values = [value for value in values if math.isfinite(value)]
    except (TypeError, ValueError, OverflowError):
        values = []
    if not values:
        # A gentle deterministic contour reads as a signal rather than a meter.
        centre = (count - 1) / 2.0
        return tuple(
            max(0.08, min(1.0, fallback * (1.0 - abs(index - centre) * 0.055)))
            for index in range(count)
        )
    peak = max(values)
    if peak > 1.0:
        values = [value / peak for value in values]
    bucket_size = max(1, math.ceil(len(values) / count))
    result: list[float] = []
    for index in range(count):
        bucket = values[index * bucket_size : (index + 1) * bucket_size]
        amplitude = max(bucket) if bucket else 0.0
        # Raise quiet detail without allowing a full-height solid wall.
        result.append(max(0.08, min(1.0, math.sqrt(amplitude))))
    return tuple(result)


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
    """Return the foreground-window monitor work area, with safe fallbacks.

    Dictation belongs to the application that currently owns keyboard focus,
    not necessarily the monitor where the mouse happens to be resting.  On
    Windows, prefer the foreground window and use the cursor only when Windows
    cannot resolve a usable foreground handle (for example during logon or a
    shell transition).
    """

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

            user32 = ctypes.windll.user32
            monitor = None
            foreground = user32.GetForegroundWindow()
            if foreground:
                monitor = user32.MonitorFromWindow(foreground, 2)
            if not monitor:
                point = wintypes.POINT()
                if user32.GetCursorPos(ctypes.byref(point)):
                    monitor = user32.MonitorFromPoint(point, 2)
            info = MONITORINFO()
            info.cbSize = ctypes.sizeof(info)
            if monitor and user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
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
    desired_height = max(1, int(round(680 * scale)))
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
        self._render_scale = self.scale
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
        geometry = overlay_geometry(bounds, self.scale)
        # When physical height is unusually constrained, lower the canvas
        # render scale enough to retain a usable 72-logical-pixel composition.
        # Normal and narrow/high-DPI monitors continue to use the real DPI.
        self._render_scale = max(
            0.25,
            min(
                self.scale,
                geometry.width / 160.0,
                geometry.height / 72.0,
            ),
        )
        window.geometry(overlay_geometry_spec(geometry))
        canvas = tk.Canvas(
            window,
            width=geometry.width,
            height=geometry.height,
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
        render_scale = max(0.25, float(getattr(self, "_render_scale", self.scale)))
        width = max(1, canvas.winfo_width()) / render_scale
        height = max(1, canvas.winfo_height()) / render_scale
        canvas.delete("all")
        # Two quiet outlines give the capsule the instrument-panel depth from
        # the approved Midnight Signal concept without relying on OS shadows.
        _rounded_rectangle(
            canvas, 2, 2, width - 2, height - 2, 25,
            fill="#091017", outline="#20323E", width=1,
        )
        _rounded_rectangle(
            canvas, 4, 4, width - 4, height - 4, 22,
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
        if render_scale != 1.0:
            canvas.scale("all", 0, 0, render_scale, render_scale)

    @staticmethod
    def _text(canvas: tk.Canvas, x: float, y: float, text: str, **kwargs) -> int:
        options = {
            "anchor": "w",
            "fill": TEXT,
            "font": ("Segoe UI", 11),
        }
        options.update(kwargs)
        return canvas.create_text(x, y, text=text, **options)

    @staticmethod
    def _draw_microphone(canvas: tk.Canvas, x: float, y: float) -> None:
        """Draw the same restrained cyan microphone across capsule states."""

        canvas.create_line(x, y - 11, x, y + 4, fill=CYAN, width=3, capstyle=tk.ROUND)
        canvas.create_arc(
            x - 8, y - 3, x + 8, y + 13,
            start=180, extent=180, style=tk.ARC, outline=CYAN, width=2,
        )
        canvas.create_line(x, y + 12, x, y + 17, fill=CYAN, width=2)
        canvas.create_line(x - 6, y + 17, x + 6, y + 17, fill=CYAN, width=2, capstyle=tk.ROUND)

    def _draw_recording(
        self, canvas: tk.Canvas, width: int, height: int, state: UiSnapshot
    ) -> None:
        layout = recording_capsule_layout(width, height)
        centre_y = height / 2.0
        self._draw_microphone(canvas, layout.mic_x, centre_y - 2)
        canvas.create_line(
            layout.divider_one_x, 19, layout.divider_one_x, height - 19,
            fill=OUTLINE, width=1,
        )
        timer_y = height * 0.34 if layout.compact else centre_y
        self._text(
            canvas, layout.timer_x, timer_y, state.elapsed_label,
            font=("Segoe UI Semibold", 10 if layout.compact else 11), anchor="w",
        )
        canvas.create_line(
            layout.divider_two_x, 19, layout.divider_two_x, height - 19,
            fill=OUTLINE, width=1,
        )

        samples: object = ()
        if self.waveform_provider is not None:
            try:
                samples = self.waveform_provider()
            except Exception:
                samples = ()
        levels = waveform_bar_levels(samples, fallback=state.level_fraction)
        span = max(1.0, layout.waveform_right - layout.waveform_left)
        step = span / max(1, len(levels) - 1)
        waveform_y = height * 0.34 if layout.compact else centre_y
        for index, activity in enumerate(levels):
            bx = layout.waveform_left + index * step
            bh = (6 + activity * 18) if layout.compact else (8 + activity * 29)
            canvas.create_line(
                bx, waveform_y - bh / 2, bx, waveform_y + bh / 2,
                fill=CYAN if activity > 0.16 else OUTLINE,
                width=3 if layout.compact else 4,
                capstyle=tk.ROUND, tags=("waveform",),
            )
        if layout.compact:
            instruction = fit_overlay_text(
                "LISTENING · release Right Ctrl",
                pixel_width=max(1.0, width - layout.status_x - 12.0),
            )
            self._text(
                canvas, layout.status_x, height * 0.72, instruction,
                fill=CYAN, font=("Segoe UI Semibold", 9),
                width=max(1.0, width - layout.status_x - 12.0),
            )
            self._text(
                canvas, layout.dbfs_x, timer_y, state.level_label,
                anchor="e", fill=TEXT_MUTED, font=("Segoe UI", 8),
            )
            return
        self._text(
            canvas, layout.status_x, 31, "LISTENING",
            fill=CYAN, font=("Segoe UI Semibold", 9),
        )
        self._text(
            canvas, layout.status_x, 55, "Release Right Ctrl to transcribe",
            font=("Segoe UI Semibold", 10),
        )
        self._text(
            canvas, layout.dbfs_x, 31, state.level_label,
            anchor="e", fill=TEXT_MUTED, font=("Segoe UI", 9),
        )

    def _draw_processing(
        self, canvas: tk.Canvas, width: int, height: int, state: UiSnapshot
    ) -> None:
        layout = processing_capsule_layout(width, height)
        self._draw_microphone(canvas, layout.mic_x, layout.orbit_cy - 2)
        canvas.create_line(
            layout.divider_x, 19, layout.divider_x, height - 19,
            fill=OUTLINE, width=1,
        )
        cx, cy = layout.orbit_cx, layout.orbit_cy
        rx, ry = layout.orbit_rx, layout.orbit_ry
        # The animation represents activity only.  It deliberately contains no
        # provider icon/name because routing has not completed yet.
        canvas.create_oval(
            cx - rx - 4, cy - ry - 3, cx + rx + 4, cy + ry + 3,
            outline="#1F3945", width=2, tags=("processing-orbit",),
        )
        canvas.create_oval(
            cx - rx, cy - ry, cx + rx, cy + ry,
            outline="#28505C", width=1, tags=("processing-orbit",),
        )
        if self.reduced_motion:
            angle = 330.0
        else:
            angle = ((time.monotonic() - self._started) * 155.0) % 360.0
        radians = math.radians(angle)
        dot_x = cx + math.cos(radians) * rx
        dot_y = cy + math.sin(radians) * ry
        # Layered comet: a dim halo, bright core and two differently paced arcs.
        canvas.create_oval(
            dot_x - 7, dot_y - 7, dot_x + 7, dot_y + 7,
            fill="#173641", outline="", tags=("processing-comet",),
        )
        canvas.create_oval(
            dot_x - 3, dot_y - 3, dot_x + 3, dot_y + 3,
            fill="#B9F7FF", outline="", tags=("processing-comet",),
        )
        canvas.create_arc(
            cx - rx, cy - ry, cx + rx, cy + ry,
            start=angle - 104, extent=88, style=tk.ARC, outline=CYAN, width=2,
            tags=("processing-comet",),
        )
        canvas.create_arc(
            cx - rx + 10, cy - ry + 5, cx + rx - 10, cy + ry - 5,
            start=angle + 76, extent=74, style=tk.ARC, outline="#6EEAF5", width=1,
            tags=("processing-comet",),
        )
        strategy_label = self._processing_strategy_label(state)
        if layout.compact:
            compact_status = fit_overlay_text(
                f"TRANSCRIBING · {strategy_label}",
                pixel_width=max(1.0, width - layout.status_x - 12.0),
            )
            self._text(
                canvas, layout.status_x, height * 0.74, compact_status,
                fill=CYAN, font=("Segoe UI Semibold", 9),
                width=max(1.0, width - layout.status_x - 12.0),
            )
            self._text(
                canvas, layout.elapsed_x, height * 0.31, state.elapsed_label,
                anchor="e", font=("Segoe UI Semibold", 10),
            )
            return
        self._text(
            canvas, layout.status_x, 31, "TRANSCRIBING",
            fill=CYAN, font=("Segoe UI Semibold", 9),
        )
        self._text(
            canvas, layout.status_x, 55, strategy_label,
            font=("Segoe UI Semibold", 10),
        )
        self._text(
            canvas, layout.elapsed_x, 31, state.elapsed_label,
            anchor="e", font=("Segoe UI Semibold", 11),
        )
        self._text(
            canvas, layout.elapsed_x, 55, "Cancel from tray",
            anchor="e", fill=TEXT_MUTED, font=("Segoe UI", 8),
        )

    @staticmethod
    def _processing_strategy_label(state: UiSnapshot) -> str:
        """Describe configured routing without claiming a selected provider."""

        del state  # The live phase copy is intentionally not a provider claim.
        try:
            from utils.transcription_backend import get_runtime_backend_config

            strategy_id = get_runtime_backend_config().provider_strategy
        except Exception:
            strategy_id = "server-default"
        labels = {
            "server-default": "Gateway default",
            "ubuntu-gpu-preferred": "GPU preferred",
            "openai-preferred": "OpenAI preferred",
            "ubuntu-gpu-only": "Ubuntu GPU only",
            "openai-only": "OpenAI only",
            "gateway-tiny-only": "Emergency tiny only",
        }
        strategy = labels.get(strategy_id, "Preferred route")
        return strategy

    def _draw_result(
        self, canvas: tk.Canvas, width: int, height: int, state: UiSnapshot, *, success: bool
    ) -> None:
        layout = result_capsule_layout(width, height)
        colour = MINT if success else CORAL
        cx, cy = layout.icon_x, height / 2
        radius = layout.icon_radius
        canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            fill=SURFACE_RAISED, outline=colour, width=2,
        )
        mark_scale = 0.72 if layout.compact else 1.0
        if success:
            canvas.create_line(
                cx - 9 * mark_scale, cy, cx - 2 * mark_scale,
                cy + 7 * mark_scale, cx + 11 * mark_scale,
                cy - 9 * mark_scale, fill=colour, width=3,
                capstyle=tk.ROUND, joinstyle=tk.ROUND,
            )
        else:
            delta = 8 * mark_scale
            canvas.create_line(cx - delta, cy - delta, cx + delta, cy + delta, fill=colour, width=3, capstyle=tk.ROUND)
            canvas.create_line(cx + delta, cy - delta, cx - delta, cy + delta, fill=colour, width=3, capstyle=tk.ROUND)
        headline_y = height * (0.35 if layout.compact else 31.0 / 88.0)
        detail_y = height * (0.68 if layout.compact else 55.0 / 88.0)
        headline = fit_overlay_text(
            state.headline.upper(), pixel_width=layout.detail_width
        )
        self._text(
            canvas, layout.text_x, headline_y, headline, fill=colour,
            font=("Segoe UI Semibold", 9), width=layout.detail_width,
        )
        detail = state.detail
        if success and state.provider:
            timing = state.provider.latency_label
            detail = f"Inserted · {state.provider.display_name} · {timing}"
            if state.degraded:
                detail += " · fallback"
        visible_detail = fit_overlay_text(detail, pixel_width=layout.detail_width)
        self._text(
            canvas, layout.text_x, detail_y, visible_detail,
            font=("Segoe UI Semibold", 10 if layout.compact else 12),
            width=layout.detail_width,
            tags=("result-detail",),
        )
        self._text(
            canvas, layout.elapsed_x, height / 2, state.elapsed_label,
            anchor="e", fill=TEXT_MUTED, font=("Segoe UI", 9),
            width=max(1.0, layout.elapsed_x - layout.elapsed_left),
        )

    def _draw_cancelled(
        self, canvas: tk.Canvas, width: int, height: int, state: UiSnapshot
    ) -> None:
        layout = result_capsule_layout(width, height)
        cx, cy, radius = layout.icon_x, height / 2, layout.icon_radius
        canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius,
            fill=SURFACE_RAISED, outline=AMBER, width=2,
        )
        canvas.create_line(
            cx - radius * 0.45, cy, cx + radius * 0.45, cy,
            fill=AMBER, width=3, capstyle=tk.ROUND,
        )
        headline_y = height * (0.35 if layout.compact else 34.0 / 88.0)
        detail_y = height * (0.68 if layout.compact else 58.0 / 88.0)
        self._text(
            canvas, layout.text_x, headline_y, "CANCELLED", fill=AMBER,
            font=("Segoe UI Semibold", 10), width=layout.detail_width,
        )
        self._text(
            canvas, layout.text_x, detail_y,
            fit_overlay_text(state.detail, pixel_width=layout.detail_width),
            font=("Segoe UI", 10 if layout.compact else 11),
            width=layout.detail_width, tags=("result-detail",),
        )
        self._text(
            canvas, layout.elapsed_x, height / 2, state.elapsed_label,
            anchor="e", fill=TEXT_MUTED, font=("Segoe UI", 9),
            width=max(1.0, layout.elapsed_x - layout.elapsed_left),
        )

    def _draw_idle(self, canvas: tk.Canvas, width: int, height: int) -> None:
        self._text(canvas, 28, height / 2, "CtrlSpeak is ready", font=("Segoe UI Semibold", 12))


__all__ = [
    "MidnightSignalOverlay",
    "MonitorBounds",
    "OverlayGeometry",
    "ProcessingCapsuleLayout",
    "RecordingCapsuleLayout",
    "ResultCapsuleLayout",
    "WAVEFORM_BAR_COUNT",
    "active_monitor_bounds",
    "display_scale",
    "fit_overlay_text",
    "flyout_geometry",
    "overlay_geometry",
    "overlay_geometry_spec",
    "processing_capsule_layout",
    "recording_capsule_layout",
    "result_capsule_layout",
    "waveform_bar_levels",
]
