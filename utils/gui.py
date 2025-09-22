# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import threading
import time
from typing import Callable, Optional
from queue import Empty

import tkinter as tk
from tkinter import ttk, messagebox
import pystray
import numpy as np


# ---- System-layer pieces (lifecycle, discovery, status) ----
from utils.system import (
    APP_VERSION,
    CLIENT_ONLY_BUILD,
    settings, settings_lock, load_settings,
    detect_client_only_build, save_settings, notify,
    start_client_listener, stop_client_listener,
    manual_discovery_refresh,
    schedule_management_refresh, enqueue_management_task,
    shutdown_server, initiate_self_uninstall,
)
# IMPORTANT: import the module so we always see the *current* values
from utils import system as sysmod

# ---- Model/CUDA helpers kept in utils.models to avoid GUI bloat ----
from utils.models import (
    AVAILABLE_MODELS,
    cuda_runtime_ready,
    install_cuda_runtime_with_progress,   # opens progress window and installs nvidia wheels
    get_device_preference, set_device_preference,
    get_current_model_name, set_current_model_name,
    model_store_path_for, model_files_present,
    download_model_with_gui,              # opens progress window and downloads model
    is_model_loaded,
    format_bytes,                         # optional pretty bytes
)

from utils.ui_theme import (
    apply_modern_theme,
    ELEVATED_SURFACE,
    ACCENT,
    BACKGROUND,
)

from utils.config_paths import asset_path, get_logger

# Shared UI thread root + instance ref (imported by utils.system.schedule_management_refresh)
tk_root: Optional[tk.Tk] = None
management_window: Optional["ManagementWindow"] = None
_management_thread_ident: Optional[int] = None
_management_thread_lock = threading.Lock()
_management_thread_ready = threading.Event()
_management_queue_job: Optional[str] = None

logger = get_logger(__name__)

# -------- Notification helpers --------
_busy_popup_win: Optional[tk.Toplevel] = None
_busy_popup_label: Optional[ttk.Label] = None
_busy_popup_progress: Optional[ttk.Progressbar] = None
_busy_popup_status_label: Optional[ttk.Label] = None
_busy_popup_close_job: Optional[str] = None
_busy_popup_progress_mode: str = "indeterminate"


def show_notification_popup(title: str, message: str) -> None:
    """Display a simple informational dialog."""
    if tk_root is None or not tk_root.winfo_exists():
        return
    try:
        messagebox.showinfo(title, message, parent=tk_root)
    except Exception:
        logger.exception("Failed to display notification popup: %s", title)


def _cancel_busy_popup_close() -> None:
    global _busy_popup_close_job
    if _busy_popup_win is not None and _busy_popup_close_job is not None:
        try:
            _busy_popup_win.after_cancel(_busy_popup_close_job)
        except Exception:
            logger.exception("Failed to cancel activation popup close timer")
    _busy_popup_close_job = None


def _destroy_busy_popup() -> None:
    global _busy_popup_win, _busy_popup_label, _busy_popup_progress
    global _busy_popup_status_label, _busy_popup_close_job, _busy_popup_progress_mode
    if _busy_popup_win is not None and _busy_popup_win.winfo_exists():
        try:
            _busy_popup_win.destroy()
        except Exception:
            logger.exception("Failed to destroy activation popup window")
    _busy_popup_win = None
    _busy_popup_label = None
    _busy_popup_progress = None
    _busy_popup_status_label = None
    _busy_popup_close_job = None
    _busy_popup_progress_mode = "indeterminate"


def _call_on_management_ui(callback: Callable[[], None], *, log_message: str) -> None:
    """Execute *callback* on the management UI thread."""

    root = tk_root
    if root is None or not root.winfo_exists():
        return

    try:
        if is_management_ui_thread():
            callback()
        else:
            root.after(0, callback)
    except Exception:
        logger.exception(log_message)


def _set_busy_popup_text(message: str) -> None:
    """Update the activation popup message safely from any thread."""

    def _apply() -> None:
        if _busy_popup_label is not None and _busy_popup_label.winfo_exists():
            _busy_popup_label.configure(text=message)

    _call_on_management_ui(_apply, log_message="Failed to update activation popup message text")


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    try:
        total = max(float(seconds), 0.0)
    except (TypeError, ValueError):
        return "—"
    minutes, secs = divmod(int(total + 0.5), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def show_activation_popup(message: str) -> None:
    """Create or update the activation progress popup."""
    global _busy_popup_win, _busy_popup_label, _busy_popup_progress
    global _busy_popup_status_label, _busy_popup_progress_mode
    if tk_root is None or not tk_root.winfo_exists():
        return

    _cancel_busy_popup_close()

    if _busy_popup_win is None or not _busy_popup_win.winfo_exists():
        _busy_popup_win = tk.Toplevel(tk_root)
        _busy_popup_win.title("CtrlSpeak is preparing")
        _busy_popup_win.geometry("420x220")
        _busy_popup_win.resizable(False, False)
        _busy_popup_win.attributes("-topmost", True)
        try:
            _busy_popup_win.attributes("-toolwindow", True)
        except Exception:
            # Not all platforms support this hint
            pass
        apply_modern_theme(_busy_popup_win)
        _busy_popup_win.protocol("WM_DELETE_WINDOW", lambda: None)

        container = ttk.Frame(_busy_popup_win, style="Modern.TFrame", padding=(24, 22))
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="Preparing CtrlSpeak", style="Title.TLabel").pack(anchor=tk.W)

        _busy_popup_label = ttk.Label(
            container,
            text=message,
            style="Body.TLabel",
            wraplength=360,
            justify=tk.LEFT,
        )
        _busy_popup_label.pack(anchor=tk.W, pady=(12, 20))

        _busy_popup_progress = ttk.Progressbar(
            container,
            mode="indeterminate",
            length=340,
            style="Modern.Horizontal.TProgressbar",
        )
        _busy_popup_progress.pack(fill=tk.X)
        _busy_popup_progress_mode = "indeterminate"
        _busy_popup_status_label = ttk.Label(
            container,
            text="",
            style="Caption.TLabel",
            justify=tk.LEFT,
            wraplength=360,
        )
        _busy_popup_status_label.pack(anchor=tk.W, pady=(12, 0))
        try:
            _busy_popup_progress.start(12)
        except Exception:
            logger.exception("Failed to start activation popup progress bar")
    else:
        try:
            _busy_popup_win.deiconify()
            _busy_popup_win.lift()
        except Exception:
            logger.exception("Failed to raise activation popup window")

    _set_busy_popup_text(message)
    if _busy_popup_status_label is not None and _busy_popup_status_label.winfo_exists():
        try:
            _busy_popup_status_label.configure(text="")
        except Exception:
            logger.exception("Failed to reset activation popup status text")
    if _busy_popup_progress is not None and _busy_popup_progress_mode == "indeterminate":
        try:
            _busy_popup_progress.config(mode="indeterminate")
            _busy_popup_progress.start(12)
        except Exception:
            logger.exception("Failed to restart activation popup progress bar")

    try:
        _busy_popup_win.lift()
        _busy_popup_win.focus_force()
    except Exception:
        # Focus hints may fail in some environments
        pass


def update_activation_progress(
    progress: Optional[float],
    elapsed: Optional[float],
    eta: Optional[float],
) -> None:
    """Update the activation popup progress bar and status text."""
    global _busy_popup_progress_mode

    def _apply() -> None:
        if _busy_popup_progress is None or not _busy_popup_progress.winfo_exists():
            return

        clamped_progress: Optional[float]
        if progress is None:
            clamped_progress = None
            if _busy_popup_progress_mode != "indeterminate":
                try:
                    _busy_popup_progress.stop()
                except Exception:
                    pass
                _busy_popup_progress.config(mode="indeterminate")
                _busy_popup_progress_mode = "indeterminate"
            try:
                _busy_popup_progress.start(12)
            except Exception:
                logger.exception("Failed to start activation popup spinner")
        else:
            clamped_progress = max(0.0, min(float(progress), 100.0))
            if _busy_popup_progress_mode != "determinate":
                try:
                    _busy_popup_progress.stop()
                except Exception:
                    pass
                _busy_popup_progress.config(mode="determinate", maximum=100.0)
                _busy_popup_progress_mode = "determinate"
            else:
                try:
                    current_max = float(_busy_popup_progress.cget("maximum"))
                except Exception:
                    current_max = 0.0
                if current_max != 100.0:
                    _busy_popup_progress.config(maximum=100.0)
            _busy_popup_progress["value"] = clamped_progress

        if _busy_popup_status_label is not None and _busy_popup_status_label.winfo_exists():
            parts: list[str] = []
            if clamped_progress is not None:
                parts.append(f"Progress: {clamped_progress:.0f}%")
            elif progress is None:
                parts.append("Progress: calculating")
            if elapsed is not None:
                parts.append(f"Elapsed: {_format_duration(elapsed)}")
            if eta is not None:
                parts.append(f"ETA: {_format_duration(eta)}")
            elif clamped_progress is not None and clamped_progress < 100.0:
                parts.append("ETA: calculating")
            status_text = "  •  ".join(parts)
            try:
                _busy_popup_status_label.configure(text=status_text)
            except Exception:
                logger.exception("Failed to update activation popup status text")

    _call_on_management_ui(_apply, log_message="Failed to update activation popup progress")


def focus_activation_popup(message: Optional[str] = None) -> None:
    """Bring the activation popup to the foreground."""
    if message:
        show_activation_popup(message)
        return

    if _busy_popup_win is None or not _busy_popup_win.winfo_exists():
        return

    _cancel_busy_popup_close()
    try:
        _busy_popup_win.deiconify()
        _busy_popup_win.lift()
        _busy_popup_win.focus_force()
    except Exception:
        pass


def close_activation_popup(message: Optional[str] = None) -> None:
    """Stop the busy popup and close it, optionally after showing a final message."""
    global _busy_popup_close_job, _busy_popup_progress_mode
    if _busy_popup_win is None or not _busy_popup_win.winfo_exists():
        if message:
            show_notification_popup("CtrlSpeak", message)
        return

    _cancel_busy_popup_close()

    if _busy_popup_progress is not None:
        try:
            _busy_popup_progress.stop()
            _busy_popup_progress.config(mode="determinate", maximum=100, value=100)
            _busy_popup_progress_mode = "determinate"
        except Exception:
            logger.exception("Failed to update activation popup progress state")
    if _busy_popup_status_label is not None and _busy_popup_status_label.winfo_exists():
        try:
            _busy_popup_status_label.configure(text="")
        except Exception:
            logger.exception("Failed to clear activation popup status text")

    if message:
        _set_busy_popup_text(message)

    if message:
        try:
            _busy_popup_win.lift()
        except Exception:
            pass
        # Leave the completion message visible briefly before closing
        try:
            _busy_popup_close_job = _busy_popup_win.after(2400, _destroy_busy_popup)
        except Exception:
            logger.exception("Failed to schedule activation popup dismissal")
            _destroy_busy_popup()
    else:
        _destroy_busy_popup()
# -------- Voice Waveform Overlay --------
_waveform_win: Optional[tk.Toplevel] = None
_waveform_canvas: Optional[tk.Canvas] = None
_waveform_job: Optional[str] = None
_waveform_provider: Optional[Callable[[], "np.ndarray"]] = None

# NEW: simple state for live vs processing
_waveform_mode: str = "live"         # "live" | "processing"
_waveform_msg: str = "Processing…"
_pulse_phase: float = 0.0            # animation phase
_waveform_closing: bool = False


def show_waveform_overlay(provider: Callable[[], "np.ndarray"]) -> None:
    global _waveform_win, _waveform_canvas, _waveform_job, _waveform_provider, _waveform_mode, _waveform_closing
    _waveform_provider = provider
    _waveform_mode = "live"
    try:
        if tk_root is None or not tk_root.winfo_exists():
            return
        if _waveform_win and _waveform_win.winfo_exists():
            return
        _waveform_win = tk.Toplevel(tk_root)
        _waveform_win.overrideredirect(True)
        _waveform_win.attributes("-topmost", True)
        try:
            _waveform_win.attributes("-alpha", 0.92)  # translucent
        except Exception:
            logger.exception("Failed to set waveform window transparency")

        # Position: top half center
        try:
            sw, sh = tk_root.winfo_screenwidth(), tk_root.winfo_screenheight()
        except Exception:
            logger.exception("Failed to query screen dimensions for waveform overlay")
            sw, sh = 1200, 800
        target_w = int(sw * 0.4)
        target_h = int(sh * 0.25)
        x = (sw - target_w) // 2
        y = int(sh * 0.05)
        _waveform_win.geometry(f"{target_w}x{target_h}+{x}+{y}")

        _waveform_canvas = tk.Canvas(_waveform_win, bg="#141414", highlightthickness=0)
        _waveform_canvas.pack(fill=tk.BOTH, expand=True)
        _waveform_closing = False

        processing_error_logged = False

        def _tick():
            nonlocal processing_error_logged
            """Redraw the overlay every ~33 ms.
            - LIVE mode: polyline waveform from recent audio samples.
            - PROCESSING mode: pulsing circular ring with 'Processing…' label.
            """
            global _pulse_phase
            if _waveform_win is None or not _waveform_win.winfo_exists():
                return
            if _waveform_canvas is None:
                return

            try:
                w = max(1, _waveform_canvas.winfo_width())
                h = max(1, _waveform_canvas.winfo_height())

                _waveform_canvas.delete("all")
                # background panel
                _waveform_canvas.create_rectangle(12, 12, w - 12, h - 12, fill="#202020", outline="#333333")

                if _waveform_mode == "live":
                    if _waveform_provider is not None:
                        data = _waveform_provider()
                        if data is not None and getattr(data, "size", 0) > 0:
                            # DYNAMIC SCALE WITH UPPER CAP
                            # Aim to keep the current frame’s peak around ~0.9, but never amplify above 8x.
                            TARGET_PEAK = 0.90
                            MAX_GAIN = 8.0

                            m = float(np.max(np.abs(data))) if getattr(data, "size", 0) else 0.0
                            if m > 1e-6:
                                dyn_gain = min(MAX_GAIN, TARGET_PEAK / m)
                            else:
                                dyn_gain = 1.0  # silence; no crazy boost

                            arr = np.clip(data * dyn_gain, -1.0, 1.0)

                            # downsample to ~canvas width
                            count = max(2, w - 40)
                            idxs = np.linspace(0, arr.size - 1, num=count).astype(int)
                            ys = arr[idxs]
                            mid = h / 2.0
                            amp = (h - 48) / 2.0

                            last_x, last_y = None, None
                            for i, v in enumerate(ys):
                                X = 20 + i
                                Y = int(mid - v * amp)
                                if last_x is not None:
                                    _waveform_canvas.create_line(last_x, last_y, X, Y, fill="#6ee7ff", width=2)
                                last_x, last_y = X, Y

                else:
                    # PROCESSING — audio-driven radius + circular wiggle
                    _pulse_phase = (_pulse_phase + 0.06) % (2 * np.pi)  # subtle motion only
                    # 1) Read the current amplitude + recent waveform of loading.wav

                    try:
                        level = float(sysmod.get_processing_level())
                        proc_wave = sysmod.get_processing_waveform(720)  # resolution around the ring
                    except Exception:
                        if not processing_error_logged:
                            logger.exception("Failed to obtain processing waveform data")
                            processing_error_logged = True
                        level = 0.0
                        proc_wave = np.zeros(720, dtype=np.float32)

                    # Normalize/soft-clip the ring waveform
                    if proc_wave.size:
                        m = float(np.max(np.abs(proc_wave))) or 1.0
                        ring_wave = np.clip(proc_wave / m, -1.0, 1.0)
                    else:
                        ring_wave = np.zeros(720, dtype=np.float32)

                    # 2) Map amplitude to base radius scaling (more “one-to-one” feel)
                    #    Increase LEVEL_GAIN to make size swings stronger (try 0.6–1.0)
                    LEVEL_GAIN = 0.75
                    pulse = 1.0 + LEVEL_GAIN * level

                    # 3) Wiggle strength around the ring (how spiky the line looks)
                    #    Try 0.20–0.40 for pronounced wiggle
                    WIGGLE_GAIN = 0.30
                    w = max(1, _waveform_canvas.winfo_width())
                    h = max(1, _waveform_canvas.winfo_height())
                    cx, cy = w // 2, h // 2
                    base_r = int(min(w, h) * 0.20)
                    ring_thickness = 8

                    # Base radius from amplitude
                    R = int(base_r * pulse * 1.06)
                    # 4) Build a closed polyline around the circle with radius modulation
                    #    by the audio waveform (and a tiny phase spin so it feels alive)
                    N = ring_wave.size

                    points = []

                    for i in range(N):
                        a = (2 * np.pi * i) / N + _pulse_phase * 0.5
                        # radius wiggle from waveform
                        r = R + int(WIGGLE_GAIN * base_r * ring_wave[i])
                        x = cx + int(np.cos(a) * r)
                        y = cy + int(np.sin(a) * r)
                        points.append((x, y))

                    # Draw the wiggly ring
                    for i in range(1, len(points)):
                        x0, y0 = points[i - 1]
                        x1, y1 = points[i]
                        _waveform_canvas.create_line(x0, y0, x1, y1, fill="#6ee7ff", width=2)

                    # close the loop
                    if len(points) > 2:
                        _waveform_canvas.create_line(points[-1][0], points[-1][1], points[0][0], points[0][1],
                                                     fill="#6ee7ff", width=2)

                    # (Optional) inner glow ring following the same R for body

                    r_inner = max(4, R - ring_thickness)
                    _waveform_canvas.create_oval(cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner,
                                                 outline="#6ee7ff", width=1)
                    # label
                    _waveform_canvas.create_text(cx, cy, text=_waveform_msg, fill="#d9fbff",
                                                 font=("Segoe UI", 16, "bold"))

                if not _waveform_closing and _waveform_win and _waveform_win.winfo_exists():
                    _waveform_job = _waveform_canvas.after(33, _tick)

            except Exception:
                logger.exception("Waveform overlay tick failed")
                if not _waveform_closing and _waveform_win and _waveform_win.winfo_exists():
                    _waveform_job = _waveform_canvas.after(33, _tick)

        _tick()
    except Exception:
        logger.exception("Failed to open waveform overlay window")

def set_waveform_processing(message: str = "Processing…") -> None:
    global _waveform_mode, _waveform_msg, _pulse_phase
    _waveform_mode = "processing"
    _waveform_msg = message
    _pulse_phase = 0.0   # ← new

def hide_waveform_overlay() -> None:
    global _waveform_win, _waveform_canvas, _waveform_job, _waveform_provider
    global _waveform_mode, _waveform_msg, _waveform_closing
    try:
        _waveform_closing = True
        if _waveform_canvas and _waveform_job:
            try:
                _waveform_canvas.after_cancel(_waveform_job)
            except Exception:
                logger.exception("Failed to cancel waveform overlay job")
        if _waveform_win and _waveform_win.winfo_exists():
            try:
                _waveform_win.withdraw()
            except Exception:
                logger.exception("Failed to withdraw waveform window")
            # destroy shortly after to let any in-flight callbacks finish
            _waveform_win.after(10, _waveform_win.destroy)
    finally:
        _waveform_win = None
        _waveform_canvas = None
        _waveform_job = None
        _waveform_provider = None
        _waveform_mode = "live"
        _waveform_msg = "Processing…"

# ---------------- Splash (1s) ----------------
def show_splash_screen(duration_ms: int) -> None:
    try:
        root = tk.Tk()
    except tk.TclError:
        return
    apply_modern_theme(root)
    root.overrideredirect(True)
    root.attributes("-topmost", True)

    width, height = 320, 340
    try:
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
    except Exception:
        logger.exception("Failed to query screen dimensions for splash screen")
        screen_w, screen_h = 800, 600
    pos_x = int((screen_w - width) / 2); pos_y = int((screen_h - height) / 2)
    root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")

    shell = tk.Frame(root, bg=BACKGROUND, bd=0, highlightthickness=0)
    shell.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

    container = tk.Frame(shell, bg=ELEVATED_SURFACE, bd=0,
                         highlightbackground=ACCENT, highlightcolor=ACCENT, highlightthickness=1)
    container.pack(fill=tk.BOTH, expand=True)

    accent_bar = tk.Frame(container, bg=ACCENT, height=4, bd=0, highlightthickness=0)
    accent_bar.pack(fill=tk.X, side=tk.TOP)

    content = ttk.Frame(container, style="ModernCard.TFrame", padding=(28, 30))
    content.pack(fill=tk.BOTH, expand=True)
    content.pack_propagate(False)

    icon_added = False
    try:
        from PIL import Image, ImageTk
        icon_path = asset_path("icon.ico")
        image = Image.open(icon_path)
        image.thumbnail((160, 160), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(content, image=photo, background=ELEVATED_SURFACE, bd=0)
        label.image = photo
        label.pack(pady=(18, 12))
        icon_added = True
    except Exception:
        logger.exception("Failed to load splash icon")

    title_pad = (12, 4) if icon_added else (28, 8)
    ttk.Label(content, text="CtrlSpeak", style="Title.TLabel").pack(pady=title_pad)
    accent = ttk.Frame(content, style="AccentLine.TFrame")
    accent.configure(height=2)
    accent.pack(fill=tk.X, pady=(8, 16))
    ttk.Label(content, text="Initializing voice systems…", style="Subtitle.TLabel",
              wraplength=240, justify=tk.CENTER).pack(pady=(0, 16))

    progress = ttk.Progressbar(content, mode="indeterminate", length=220,
                               style="Modern.Horizontal.TProgressbar")
    progress.pack(fill=tk.X)
    try:
        progress.start(10)
    except Exception:
        logger.exception("Failed to start splash progress animation")

    root.after(duration_ms, root.destroy)
    root.mainloop()

# ---------------- First-run mode ----------------
def prompt_initial_mode() -> Optional[str]:
    from utils.system import AUTO_MODE_PROFILE
    if AUTO_MODE_PROFILE:
        return AUTO_MODE_PROFILE
    result = {"mode": None}

    def choose(mode: str) -> None:
        result["mode"] = mode
        root.destroy()

    root = tk.Tk()
    apply_modern_theme(root)
    root.title("Welcome to CtrlSpeak")
    root.geometry("560x420")
    root.minsize(520, 360)
    root.resizable(True, True)
    root.attributes("-topmost", True)

    container = ttk.Frame(root, style="Modern.TFrame", padding=(28, 26))
    container.pack(fill=tk.BOTH, expand=True)

    intro = ttk.Frame(container, style="ModernCard.TFrame", padding=(24, 22))
    intro.pack(fill=tk.X)
    ttk.Label(intro, text="Welcome to CtrlSpeak", style="Title.TLabel").pack(anchor=tk.W)
    message = (
        "It looks like this is the first time CtrlSpeak is running on this computer. "
        "Choose how you would like to use it:"
    )
    ttk.Label(intro, text=message, style="Body.TLabel", wraplength=460,
              justify=tk.LEFT).pack(anchor=tk.W, pady=(12, 0))
    accent = ttk.Frame(intro, style="AccentLine.TFrame")
    accent.configure(height=2)
    accent.pack(fill=tk.X, pady=(18, 8))

    cards = ttk.Frame(container, style="Modern.TFrame")
    cards.pack(fill=tk.BOTH, expand=True, pady=(18, 12))

    def make_card(title: str, desc: str, mode_value: str, primary: bool) -> None:
        card = ttk.Frame(cards, style="ModernCard.TFrame", padding=(22, 20))
        card.pack(fill=tk.X, pady=8)
        label_text = f"MODE · {mode_value.replace('_', ' ').upper()}"
        ttk.Label(card, text=label_text, style="PillMuted.TLabel").pack(anchor=tk.W)
        accent_inner = ttk.Frame(card, style="AccentLine.TFrame")
        accent_inner.configure(height=2)
        accent_inner.pack(fill=tk.X, pady=(10, 14))
        ttk.Label(card, text=title, style="SectionHeading.TLabel").pack(anchor=tk.W)
        ttk.Label(card, text=desc, style="Body.TLabel", wraplength=460,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 16))
        btn_style = "Accent.TButton" if primary else "Subtle.TButton"
        ttk.Button(card, text="Use this mode", style=btn_style,
                   command=lambda: choose(mode_value)).pack(anchor=tk.E)

    make_card(
        "Client + Server",
        "Use this computer for local transcription and optionally share it with other CtrlSpeak clients on "
        "your network.",
        "client_server",
        True,
    )
    make_card(
        "Client Only",
        "Connect to another CtrlSpeak server on your network and send recordings there for transcription.",
        "client",
        False,
    )

    def cancel() -> None:
        root.destroy()

    ttk.Button(container, text="Quit Setup", style="Subtle.TButton",
               command=cancel).pack(fill=tk.X, pady=(8, 0))
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.update_idletasks()
    req_w = root.winfo_reqwidth()
    req_h = root.winfo_reqheight()
    root.geometry(f"{req_w}x{req_h}")
    root.minsize(req_w, req_h)
    root.after(200, lambda: root.attributes("-topmost", False))
    root.mainloop()
    return result["mode"]


def ensure_mode_selected() -> None:
    # 1) Load settings from disk first
    load_settings()

    # 2) If mode already set, don't prompt again
    with settings_lock:
        current_mode = settings.get("mode")

    if current_mode in {"client", "client_server"}:
        return  # nothing to do

    # 3) Client-only builds force 'client' once and persist
    if detect_client_only_build():
        with settings_lock:
            settings["mode"] = "client"
        save_settings()
        return

    # 4) First-run prompt only if still not set
    choice = prompt_initial_mode()
    if not choice:
        notify("CtrlSpeak cannot continue without selecting a mode.")
        sys.exit(0)
    with settings_lock:
        settings["mode"] = choice
    save_settings()

# ---------------- UI loop pump (for async updates) ----------------
def _process_management_queue() -> None:
    """Process any pending management UI tasks."""

    global _management_queue_job

    while True:
        try:
            func, args, kwargs = sysmod.management_ui_queue.get_nowait()
        except Empty:
            break
        try:
            func(*args, **kwargs)
        except Exception:
            logger.exception("Management UI task failed")

    root = tk_root
    if root is None or not root.winfo_exists():
        _management_queue_job = None
        return

    try:
        _management_queue_job = root.after(120, _process_management_queue)
    except Exception:
        logger.exception("Failed to reschedule management UI queue processing")
        _management_queue_job = None


def _initialize_management_ui_on_main_thread() -> None:
    """Create the hidden Tk root on the main thread."""

    global tk_root, _management_thread_ident, _management_queue_job

    if tk_root is not None and tk_root.winfo_exists():
        return

    sysmod.management_ui_thread = None

    try:
        root = tk.Tk()
    except Exception:
        logger.exception("Failed to initialize management UI root")
        _management_thread_ready.set()
        return

    tk_root = root
    apply_modern_theme(root)
    try:
        root.withdraw()
    except Exception:
        logger.exception("Failed to withdraw management UI root window")

    _management_thread_ident = threading.get_ident()
    _management_thread_ready.set()
    sysmod.management_ui_thread = threading.current_thread()

    try:
        _management_queue_job = root.after(80, _process_management_queue)
    except Exception:
        logger.exception("Failed to schedule initial management UI queue processing")
        _management_queue_job = None


def ensure_management_ui_thread() -> None:
    """Ensure the management UI root exists on the main thread."""

    if tk_root is not None and tk_root.winfo_exists():
        return

    with _management_thread_lock:
        if tk_root is not None and tk_root.winfo_exists():
            return

        if threading.current_thread() is threading.main_thread():
            _initialize_management_ui_on_main_thread()
        else:
            if not _management_thread_ready.wait(timeout=5.0):
                logger.error("Management UI root was not initialized on the main thread within timeout")


def run_management_ui_loop() -> None:
    """Run the Tk mainloop on the main thread."""

    ensure_management_ui_thread()

    root = tk_root
    if root is None or not root.winfo_exists():
        return

    try:
        root.mainloop()
    finally:
        _teardown_management_ui()


def pump_management_events_once() -> None:
    """Process pending management UI work without entering the Tk mainloop."""

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Management UI events must be pumped on the main thread")

    root = tk_root
    if root is None or not root.winfo_exists():
        return

    # Process any queued management tasks before flushing Tk events so that
    # worker threads can update the UI while the main thread is busy.
    _process_management_queue()

    try:
        root.update_idletasks()
    except tk.TclError:
        return

    try:
        root.update()
    except tk.TclError:
        pass


def _teardown_management_ui() -> None:
    """Reset management UI globals after the loop exits."""

    global tk_root, management_window, _management_thread_ident, _management_queue_job

    root = tk_root
    if root is not None and _management_queue_job is not None:
        try:
            root.after_cancel(_management_queue_job)
        except Exception:
            logger.exception("Failed to cancel management UI queue job during teardown")
    _management_queue_job = None

    if root is not None:
        try:
            root.destroy()
        except Exception:
            logger.exception("Failed to destroy management UI root during teardown")

    tk_root = None
    management_window = None
    _management_thread_ident = None
    _management_thread_ready.clear()
    sysmod.management_ui_thread = None


def request_management_ui_shutdown() -> None:
    """Request the Tk mainloop to exit."""

    root = tk_root
    if root is None or not root.winfo_exists():
        return

    def _quit() -> None:
        try:
            root.quit()
        except Exception:
            logger.exception("Failed to quit management UI mainloop")

    _call_on_management_ui(_quit, log_message="Failed to schedule management UI shutdown")

def is_management_ui_thread() -> bool:
    return _management_thread_ident == threading.get_ident()

# ---------------- Status helper ----------------
def describe_server_status() -> str:
    with settings_lock:
        mode = settings.get("mode")
        port = int(settings.get("server_port", 65432))

    server = sysmod.get_last_connected_server()

    if mode == "client_server" and sysmod.server_thread and sysmod.server_thread.is_alive():
        host = server.host if server else sysmod.get_advertised_host_ip()
        return f"Serving: {host}:{port}"

    if server:
        host = server.host
        prt = server.port
        label = "local CPU" if host == "local-cpu" else ("local" if host == "local" else f"{host}:{prt}")
        return f"Connected: {label}"

    return "Not connected"

# ---------------- Management window (NEW UI) ----------------
def open_management_dialog(icon, item):
    ensure_management_ui_thread()
    enqueue_management_task(_show_management_window, icon)

def _show_management_window(icon: pystray.Icon) -> None:
    global management_window
    if management_window and management_window.is_open():
        management_window.bring_to_front()
        management_window.refresh_status()
        return
    management_window = ManagementWindow(icon)

class ManagementWindow:
    def __init__(self, icon: pystray.Icon):
        self._icon = icon
        self.window = tk.Toplevel(tk_root)
        self.window.title(f"CtrlSpeak Control v{APP_VERSION}")
        self.window.geometry("640x620")
        self.window.minsize(580, 560)
        self.window.resizable(True, True)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _e: self.close())
        apply_modern_theme(self.window)
        try:
            self.window.iconbitmap(str(asset_path("icon.ico")))
        except Exception:
            logger.exception("Failed to set management window icon")

        # Track asynchronous activation to keep status text stable while the worker runs.
        self._activation_busy = False
        self._activation_status_var: Optional[tk.StringVar] = None

        container = ttk.Frame(self.window, style="Modern.TFrame", padding=(30, 26))
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(container, style="ModernCard.TFrame", padding=(26, 24))
        header.pack(fill=tk.X)
        ttk.Label(header, text="CtrlSpeak Control", style="Title.TLabel").pack(anchor=tk.W)
        build_label = "Client Only" if CLIENT_ONLY_BUILD else "Client + Server"
        ttk.Label(header, text=f"Build {APP_VERSION} · {build_label}", style="Subtitle.TLabel").pack(anchor=tk.W, pady=(10, 0))
        header_accent = ttk.Frame(header, style="AccentLine.TFrame")
        header_accent.configure(height=2)
        header_accent.pack(fill=tk.X, pady=(18, 0))

        # Status overview
        status_card = ttk.Frame(container, style="ModernCard.TFrame", padding=(24, 22))
        status_card.pack(fill=tk.X, pady=(18, 12))
        ttk.Label(status_card, text="System status", style="SectionHeading.TLabel").pack(anchor=tk.W)
        status_accent = ttk.Frame(status_card, style="AccentLine.TFrame")
        status_accent.configure(height=2)
        status_accent.pack(fill=tk.X, pady=(12, 12))
        badges = ttk.Frame(status_card, style="ModernCardInner.TFrame")
        badges.pack(fill=tk.X)
        self.mode_badge = ttk.Label(badges, text="", style="PillMuted.TLabel")
        self.mode_badge.pack(side=tk.LEFT)
        self.network_badge = ttk.Label(badges, text="", style="PillMuted.TLabel")
        self.network_badge.pack(side=tk.LEFT, padx=(12, 0))
        self.status_var = tk.StringVar()
        self.server_status_var = tk.StringVar()
        ttk.Label(status_card, textvariable=self.status_var, style="Body.TLabel",
                  justify=tk.LEFT).pack(fill=tk.X, pady=(12, 10))
        ttk.Label(status_card, textvariable=self.server_status_var, style="Caption.TLabel",
                  justify=tk.LEFT).pack(anchor=tk.W)

        # Device preferences
        device_card = ttk.Frame(container, style="ModernCard.TFrame", padding=(24, 22))
        device_card.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(device_card, text="Device preference", style="SectionHeading.TLabel").pack(anchor=tk.W)
        device_accent = ttk.Frame(device_card, style="AccentLine.TFrame")
        device_accent.configure(height=2)
        device_accent.pack(fill=tk.X, pady=(10, 12))
        self.device_var = tk.StringVar(value=get_device_preference())
        device_row = ttk.Frame(device_card, style="ModernCardInner.TFrame")
        device_row.pack(fill=tk.X, pady=(14, 10))
        ttk.Radiobutton(device_row, text="CPU", variable=self.device_var, value="cpu",
                        style="Modern.TRadiobutton").pack(side=tk.LEFT, padx=(0, 18))
        ttk.Radiobutton(device_row, text="GPU (CUDA)", variable=self.device_var, value="cuda",
                        style="Modern.TRadiobutton").pack(side=tk.LEFT)
        self.cuda_status = tk.StringVar()
        ttk.Label(device_card, textvariable=self.cuda_status, style="Caption.TLabel").pack(anchor=tk.W, pady=(4, 0))
        device_buttons = ttk.Frame(device_card, style="ModernCardInner.TFrame")
        device_buttons.pack(fill=tk.X, pady=(16, 0))
        self.apply_device_btn = ttk.Button(device_buttons, text="Apply device", style="Accent.TButton",
                                           command=self._apply_device)
        self.apply_device_btn.pack(side=tk.LEFT)
        self.install_cuda_btn = ttk.Button(device_buttons, text="Install or repair CUDA", style="Subtle.TButton",
                                           command=self._install_cuda)
        self.install_cuda_btn.pack(side=tk.LEFT, padx=(12, 0))

        # Model selection
        model_card = ttk.Frame(container, style="ModernCard.TFrame", padding=(24, 22))
        model_card.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(model_card, text="Speech model", style="SectionHeading.TLabel").pack(anchor=tk.W)
        model_accent = ttk.Frame(model_card, style="AccentLine.TFrame")
        model_accent.configure(height=2)
        model_accent.pack(fill=tk.X, pady=(10, 12))
        self.model_var = tk.StringVar(value=get_current_model_name())
        model_row = ttk.Frame(model_card, style="ModernCardInner.TFrame")
        model_row.pack(fill=tk.X, pady=(14, 12))
        ttk.Label(model_row, text="Whisper model", style="Body.TLabel").pack(side=tk.LEFT)
        ttk.Combobox(
            model_row,
            textvariable=self.model_var,
            values=["small", "large-v3"],
            state="readonly",
            width=18,
            style="Modern.TCombobox",
        ).pack(side=tk.LEFT, padx=(12, 0))

        model_badges = ttk.Frame(model_row, style="ModernCardInner.TFrame")
        model_badges.pack(side=tk.LEFT, padx=(16, 0))
        self.model_small_badge = ttk.Label(model_badges, text="", style="PillMuted.TLabel")
        self.model_small_badge.pack(anchor=tk.W)
        self.model_large_badge = ttk.Label(model_badges, text="", style="PillMuted.TLabel")
        self.model_large_badge.pack(anchor=tk.W, pady=(8, 0))
        model_buttons = ttk.Frame(model_card, style="ModernCardInner.TFrame")
        model_buttons.pack(fill=tk.X, pady=(4, 0))
        self.apply_model_btn = ttk.Button(model_buttons, text="Activate model", style="Accent.TButton",
                                          command=self._apply_model)
        self.apply_model_btn.pack(side=tk.LEFT)
        self.download_model_btn = ttk.Button(model_buttons, text="Download or update", style="Subtle.TButton",
                                             command=self._download_model)
        self.download_model_btn.pack(side=tk.LEFT, padx=(12, 0))
        self.model_status = tk.StringVar()
        ttk.Label(model_card, textvariable=self.model_status, style="Caption.TLabel").pack(anchor=tk.W, pady=(10, 0))

        # Client/server controls
        control_card = ttk.Frame(container, style="ModernCard.TFrame", padding=(24, 22))
        control_card.pack(fill=tk.BOTH, expand=True)
        ttk.Label(control_card, text="Client & server", style="SectionHeading.TLabel").pack(anchor=tk.W)
        control_accent = ttk.Frame(control_card, style="AccentLine.TFrame")
        control_accent.configure(height=2)
        control_accent.pack(fill=tk.X, pady=(10, 12))
        controls = ttk.Frame(control_card, style="ModernCardInner.TFrame")
        controls.pack(fill=tk.X, pady=(14, 0))
        self.start_button = ttk.Button(controls, text="Start client", style="Accent.TButton",
                                       command=self.start_client)
        self.start_button.pack(fill=tk.X, pady=4)
        self.stop_button = ttk.Button(controls, text="Stop client", style="Danger.TButton",
                                      command=self.stop_client)
        self.stop_button.pack(fill=tk.X, pady=4)
        self.refresh_button = ttk.Button(controls, text="Refresh servers", style="Subtle.TButton",
                                         command=self.refresh_servers)
        self.refresh_button.pack(fill=tk.X, pady=4)
        exit_label = "Exit CtrlSpeak" if CLIENT_ONLY_BUILD else "Stop everything"
        self.stop_all_button = ttk.Button(controls, text=exit_label, style="Danger.TButton",
                                          command=self.stop_everything)
        self.stop_all_button.pack(fill=tk.X, pady=(12, 4))

        # Footer actions
        footer = ttk.Frame(container, style="Modern.TFrame")
        footer.pack(fill=tk.X, pady=(18, 0))
        ttk.Button(footer, text="Close control center", style="Accent.TButton",
                   command=self.close).pack(fill=tk.X)
        ttk.Button(footer, text="Delete CtrlSpeak", style="Danger.TButton",
                   command=self.delete_ctrlspeak).pack(fill=tk.X, pady=(10, 0))

        self.window.after(120, self.refresh_status)

        # --- Auto-size window to fit content once everything is laid out ---
        self.window.update_idletasks()
        req_w = max(self.window.winfo_reqwidth(), 580)
        req_h = max(self.window.winfo_reqheight(), 560)
        self.window.minsize(req_w, req_h)

        self.bring_to_front()

    # --- window helpers ---
    def is_open(self) -> bool:
        return bool(self.window and self.window.winfo_exists())

    def bring_to_front(self) -> None:
        if not self.is_open(): return
        self.window.deiconify(); self.window.lift(); self.window.focus_force()
        self.window.attributes("-topmost", True)
        self.window.after(150, lambda: self.window.attributes("-topmost", False))

    # --- state refresh ---
    def refresh_status(self) -> None:
        with settings_lock:
            mode = settings.get("mode") or "unknown"
        device_pref = get_device_preference()
        cuda_available = cuda_runtime_ready(ignore_preference=True)
        model_name = get_current_model_name()
        self.mode_badge.configure(text=f"MODE · {mode.upper()}", style="PillAccent.TLabel")
        network_label = describe_server_status()
        if "Not connected" in network_label:
            badge_style = "PillDanger.TLabel"
            badge_text = "NETWORK · OFFLINE"
        elif network_label.startswith("Serving"):
            badge_style = "PillAccent.TLabel"
            badge_text = "SERVER · ONLINE"
        elif network_label.startswith("Connected") or network_label.startswith("Discovered"):
            badge_style = "PillAccent.TLabel"
            badge_text = "NETWORK · LINKED"
        else:
            badge_style = "PillMuted.TLabel"
            badge_text = "NETWORK · READY"
        self.network_badge.configure(text=badge_text, style=badge_style)
        status_parts = [
            f"• Mode: {mode}",
            f"• Client: {'active' if sysmod.client_enabled else 'stopped'}",
            f"• Server thread: {'running' if sysmod.server_thread and sysmod.server_thread.is_alive() else 'not running'}",
        ]
        self.status_var.set("\n".join(status_parts))
        self.server_status_var.set(f"Network: {describe_server_status()}")
        if not (self._activation_busy and self._activation_status_var is self.cuda_status):
            if not cuda_available:
                cuda_text = "CUDA runtime not detected."
            elif device_pref == "cuda":
                cuda_text = "CUDA runtime active."
            else:
                cuda_text = "CUDA runtime available."
            self.cuda_status.set(cuda_text)

        model_statuses: dict[str, dict[str, object]] = {}
        for candidate, display in (("small", "Small"), ("large-v3", "Large V3")):
            candidate_present = model_files_present(model_store_path_for(candidate))
            if not candidate_present:
                state = "Not downloaded"
                style = "PillDanger.TLabel"
            elif model_name == candidate:
                state = "Active"
                style = "PillAccent.TLabel"
            else:
                state = "Available"
                style = "PillMuted.TLabel"
            model_statuses[candidate] = {
                "display": display,
                "state": state,
                "style": style,
                "present": candidate_present,
            }

        def format_state_text(state: str) -> str:
            return "not downloaded" if state == "Not downloaded" else state.lower()

        small_info = model_statuses.get("small")
        if small_info:
            self.model_small_badge.configure(
                text=f"{small_info['display']}: {format_state_text(small_info['state'])}",
                style=str(small_info["style"]),
            )

        large_info = model_statuses.get("large-v3")
        if large_info:
            self.model_large_badge.configure(
                text=f"{large_info['display']}: {format_state_text(large_info['state'])}",
                style=str(large_info["style"]),
            )

        if not (self._activation_busy and self._activation_status_var is self.model_status):
            active_info = model_statuses.get(model_name)
            if active_info:
                state = str(active_info["state"])
                display = str(active_info["display"])
                if state == "Not downloaded":
                    message = f"{display} model is not downloaded."
                elif state == "Active":
                    message = f"{display} model is active."
                else:
                    message = f"{display} model is available locally."
            else:
                present = model_files_present(model_store_path_for(model_name))
                message = (
                    f"{model_name} model is ready." if present else f"{model_name} model status unknown."
                )
            self.model_status.set(message)

        self.device_var.set(device_pref)
        if model_name in {"small", "large-v3"}:
            self.model_var.set(model_name)
        else:
            self.model_var.set("large-v3")

        if sysmod.client_enabled:
            self.start_button.state(["disabled"]); self.stop_button.state(["!disabled"])
        else:
            self.start_button.state(["!disabled"]); self.stop_button.state(["disabled"])

    def _reload_transcriber_async(
        self,
        *,
        progress_message: str,
        status_var: Optional[tk.StringVar],
        notify_context: str,
        success_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        if not self.is_open():
            return

        from utils.models import unload_transcriber, initialize_transcriber

        buttons = [
            getattr(self, "apply_model_btn", None),
            getattr(self, "download_model_btn", None),
            getattr(self, "apply_device_btn", None),
            getattr(self, "install_cuda_btn", None),
        ]

        for button in buttons:
            if button is None:
                continue
            try:
                button.state(["disabled"])
            except Exception:
                logger.exception("Failed to disable button during %s", notify_context)

        previous_text = status_var.get() if status_var is not None else ""
        if status_var is not None:
            try:
                status_var.set(progress_message)
            except Exception:
                logger.exception("Failed to update status text for %s", notify_context)

        self._activation_busy = True
        self._activation_status_var = status_var

        def worker() -> None:
            success = False
            error: Optional[Exception] = None
            try:
                unload_transcriber()
                success = initialize_transcriber(force=True, allow_client=True) is not None
            except Exception as exc:
                error = exc
                logger.exception("Transcriber initialization failed while handling %s", notify_context)

            def finish() -> None:
                self._activation_busy = False
                self._activation_status_var = None

                if not self.is_open():
                    return

                for button in buttons:
                    if button is None:
                        continue
                    try:
                        button.state(["!disabled"])
                    except Exception:
                        logger.exception("Failed to re-enable button after %s", notify_context)

                if success:
                    if success_callback is not None:
                        try:
                            success_callback()
                        except Exception:
                            logger.exception("Activation success handler failed for %s", notify_context)
                else:
                    if status_var is not None:
                        try:
                            status_var.set(previous_text)
                        except Exception:
                            logger.exception("Failed to restore status text after %s", notify_context)
                    details = sysmod.format_exception_details(error) if error else "Initialization returned no model"
                    sysmod.notify_error(notify_context, details)
                    messagebox.showerror(notify_context, "See the CtrlSpeak log folder for details.", parent=self.window)

                self.refresh_status()

            if self.window and self.window.winfo_exists():
                self.window.after(0, finish)
            else:
                finish()

        threading.Thread(target=worker, daemon=True).start()

    # --- device actions ---
    def _apply_device(self):
        from utils.models import set_device_preference
        choice = self.device_var.get()
        if choice not in {"cpu", "cuda", "auto"}:
            choice = "auto"
        set_device_preference(choice)

        # Offer to install CUDA if they chose GPU and it isn't ready
        if choice == "cuda" and not cuda_runtime_ready():
            if messagebox.askyesno(
                    "CUDA not ready",
                    "CUDA runtime not detected.\n\nInstall now for GPU acceleration?",
                    parent=self.window
            ):
                if install_cuda_runtime_with_progress(self.window):
                    messagebox.showinfo("CUDA", "CUDA runtime installed successfully.", parent=self.window)
                else:
                    messagebox.showwarning("CUDA", "Failed to prepare CUDA. Staying on CPU.", parent=self.window)

        # Reload the model with the new device without blocking the UI
        self._reload_transcriber_async(
            progress_message="Applying device preference…",
            status_var=self.cuda_status,
            notify_context="Device activation failed",
        )

    def _install_cuda(self):
        if install_cuda_runtime_with_progress(self.window):
            messagebox.showinfo("CUDA", "CUDA runtime installed successfully.", parent=self.window)
        else:
            messagebox.showwarning("CUDA", "Failed to prepare CUDA. You can try again.", parent=self.window)
        self.refresh_status()

    # --- model actions ---
    def _apply_model(self):
        from utils.models import set_current_model_name
        name = self.model_var.get()
        if name not in {"small", "large-v3"}:
            name = "large-v3"
        set_current_model_name(name)

        if not model_files_present(model_store_path_for(name)):
            messagebox.showinfo(
                "Model",
                "Model files are not installed yet. Use Download/Update… to fetch them before switching.",
                parent=self.window,
            )
            self.refresh_status()
            return

        def on_success() -> None:
            messagebox.showinfo("Model", f"Active model set to {name}.", parent=self.window)

        self._reload_transcriber_async(
            progress_message="Activating model…",
            status_var=self.model_status,
            notify_context="Model activation failed",
            success_callback=on_success,
        )

    def _download_model(self):
        name = self.model_var.get()
        if name not in {"small", "large-v3"}:
            name = "large-v3"
        model_already_loaded = is_model_loaded()
        if download_model_with_gui(
            name,
            block_during_download=not model_already_loaded,
            activate_after=not model_already_loaded,
        ):
            (model_store_path_for(name) / ".installed").touch(exist_ok=True)
            if model_already_loaded:
                message = "Model downloaded successfully."
            else:
                message = "Model downloaded and activated successfully."
            messagebox.showinfo("Model", message, parent=self.window)
        else:
            messagebox.showwarning("Model", "Model download did not complete.", parent=self.window)
        self.refresh_status()

    # --- client/server actions ---
    def start_client(self) -> None:
        start_client_listener(); self.refresh_status()

    def stop_client(self) -> None:
        stop_client_listener(); self.refresh_status()

    def refresh_servers(self) -> None:
        self.refresh_button.state(["disabled"]); self.refresh_button.config(text="Scanning…")
        self.server_status_var.set("Scanning for servers…")

        def worker() -> None:
            try:
                manual_discovery_refresh()
            finally:
                enqueue_management_task(self._on_refresh_finished)

        threading.Thread(target=worker, daemon=True).start()

    def _on_refresh_finished(self) -> None:
        if not self.is_open(): return
        self.refresh_button.state(["!disabled"]); self.refresh_button.config(text="Refresh Servers")
        self.refresh_status()

    def stop_everything(self) -> None:
        stop_client_listener(); shutdown_server()
        self.refresh_status()
        self.window.after(200, self._icon.stop)
        self.close()

    def delete_ctrlspeak(self) -> None:
        if not messagebox.askyesno("Delete CtrlSpeak",
                                   "This will remove CtrlSpeak and all local data. Continue?",
                                   parent=self.window):
            return
        self.window.after(100, lambda: initiate_self_uninstall(self._icon))

    def close(self) -> None:
        global management_window
        if self.is_open(): self.window.destroy()
        management_window = None
