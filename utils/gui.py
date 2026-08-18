# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING
from queue import Empty

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

if TYPE_CHECKING:
    import pystray

import utils.net_discovery as net_discovery


# ---- System-layer pieces (lifecycle, discovery, status) ----
from utils.system import (
    APP_VERSION,
    CLIENT_ONLY_BUILD,
    settings, settings_lock, load_settings,
    detect_client_only_build, save_settings, notify,
    start_client_listener, stop_client_listener,
    manual_discovery_refresh,
    schedule_management_refresh, enqueue_management_task,
    start_server, shutdown_server, initiate_self_uninstall,
    list_input_audio_devices, get_input_device_preference, set_input_device_preference,
)
# IMPORTANT: import the module so we always see the *current* values
from utils import system as sysmod

# ---- Model/CUDA helpers kept in utils.models to avoid GUI bloat ----
from utils.models import (
    AVAILABLE_MODELS,
    cuda_runtime_ready,
    install_cuda_runtime_with_progress,
    ensure_cuda_runtime_from_existing,
    cuda_runtime_files_present,
    cuda_driver_available,
    get_device_preference, set_device_preference,
    get_current_model_name, set_current_model_name,
    model_store_path_for, model_files_present,
    download_model_with_gui,              # orchestrates welcome window and downloads model
    ensure_model_ready_for_local_server,
    trace_model_download_step,
)

from utils.ui_theme import (
    apply_modern_theme,
    ELEVATED_SURFACE,
    ACCENT,
    BACKGROUND,
    DANGER,
    TEXT_PRIMARY,
    OUTLINE,
)

from utils.config_paths import app_icon_path, asset_path, get_logger
from utils.cuda_probe import automatic_runtime_install_supported
from utils.transcription_backend import (
    BACKEND_DISPLAY_NAMES,
    ApiBackendError,
    ApiTranscriptionClient,
    BackendConfig,
    BackendPersistenceError,
    CredentialStorageError,
    backend_display_name,
    backend_from_display_name,
    forget_openai_api_key,
    get_backend_config,
    get_backend_status,
    get_runtime_backend_config,
    get_session_openai_api_key,
    persist_openai_api_key,
    save_backend_config,
    secure_storage_available,
)
from utils.languages import language_choices
from utils.update_helper import prepare_update_handoff
from utils.update_manager import (
    UpdateError,
    UpdateEvent,
    classify_runtime,
    get_update_coordinator,
    sanitize_release_notes,
    utc_now_iso,
)

# Shared UI thread root + instance ref (imported by utils.system.schedule_management_refresh)
tk_root: Optional[tk.Tk] = None
management_window: Optional["ManagementWindow"] = None
correction_submission_dialog: Optional["CorrectionSubmissionDialog"] = None
tray_flyout = None
_management_thread_ident: Optional[int] = None
_management_thread_lock = threading.Lock()
_management_thread_ready = threading.Event()
_management_queue_job: Optional[str] = None

logger = get_logger(__name__)
_dpi_awareness_configured = False


def configure_process_dpi_awareness() -> bool:
    """Enable per-monitor DPI awareness before creating the first Tk root."""

    global _dpi_awareness_configured
    if _dpi_awareness_configured or not sys.platform.startswith("win"):
        return _dpi_awareness_configured
    try:
        import ctypes

        user32 = ctypes.windll.user32
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        if bool(user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))):
            _dpi_awareness_configured = True
            return True
    except Exception:
        logger.debug("Per-monitor-v2 DPI awareness was unavailable", exc_info=True)
    try:
        import ctypes

        # PROCESS_PER_MONITOR_DPI_AWARE
        result = int(ctypes.windll.shcore.SetProcessDpiAwareness(2))
        _dpi_awareness_configured = result in {0, -2147024891}
    except Exception:
        logger.debug("Fallback process DPI awareness was unavailable", exc_info=True)
    return _dpi_awareness_configured


def _set_window_icon(window: tk.Misc) -> None:
    """Set an ICO on Windows and a PNG icon on Linux/Tk."""
    if sys.platform.startswith("win"):
        window.iconbitmap(str(asset_path("icon.ico")))
        return
    photo = tk.PhotoImage(file=str(app_icon_path()))
    window.iconphoto(True, photo)
    setattr(window, "_ctrlspeak_icon_photo", photo)

# -------- Lockout window state --------
_lockout_win: Optional[tk.Toplevel] = None
_lockout_message_var: Optional[tk.StringVar] = None
_lockout_close_job: Optional[str] = None
_lockout_progress: Optional[ttk.Progressbar] = None
_lockout_cancel_button: Optional[ttk.Button] = None
_lockout_cancel_callback: Optional[Callable[[], None]] = None

# -------- Notification helpers --------


def show_startup_error(title: str, message: str) -> None:
    """Show a blocking error for the windowed build, which has no stderr UI."""
    root = None
    try:
        root = tk.Tk(className="CtrlSpeak")
        root.withdraw()
        messagebox.showerror(title, message, parent=root)
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                logger.debug("Failed to destroy startup error root", exc_info=True)


def show_post_update_notice(release_metadata: dict[str, object]) -> None:
    """Show a bounded one-time What's New message on the existing Tk thread."""

    version = str(release_metadata.get("version") or APP_VERSION)
    notes = sanitize_release_notes(release_metadata.get("notes"), limit=3000)

    def _show() -> None:
        with settings_lock:
            enabled = bool(settings.get("show_whats_new_on_update", True))
            last_seen = str(settings.get("whats_new_last_seen_version") or "")
        if not enabled or last_seen == version:
            return
        message = f"CtrlSpeak {version} was installed successfully."
        if notes:
            message += f"\n\nWhat's new:\n{notes}"
        try:
            messagebox.showinfo("CtrlSpeak updated", message, parent=tk_root)
        except Exception:
            logger.exception("Failed to display post-update release summary")
            return
        with settings_lock:
            settings["whats_new_last_seen_version"] = version
        if not save_settings():
            logger.error("Unable to save the last-seen What's New version")

    _call_on_management_ui(
        _show,
        log_message="Failed to schedule the post-update release summary",
    )


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


def _cancel_lockout_close() -> None:
    global _lockout_close_job
    if _lockout_win is not None and _lockout_close_job is not None:
        try:
            _lockout_win.after_cancel(_lockout_close_job)
        except Exception:
            logger.exception("Failed to cancel lockout window dismissal")
    _lockout_close_job = None


def _destroy_lockout_window() -> None:
    global _lockout_win, _lockout_message_var, _lockout_close_job, _lockout_progress
    if _lockout_win is not None and _lockout_win.winfo_exists():
        try:
            _lockout_win.destroy()
        except Exception:
            logger.exception("Failed to destroy lockout window")
    _lockout_win = None
    _lockout_message_var = None
    _lockout_close_job = None
    if _lockout_progress is not None:
        try:
            _lockout_progress.stop()
        except Exception:
            pass
    _lockout_progress = None


def show_notification_popup(title: str, message: str) -> None:
    """Display a transient notification window on the management UI thread."""

    root = tk_root
    if root is None or not root.winfo_exists():
        try:
            print(f"{title}: {message}")
        except Exception:
            logger.exception("Failed to print fallback notification '%s'", title)
        return

    if not is_management_ui_thread():
        _call_on_management_ui(
            lambda: show_notification_popup(title, message),
            log_message="Failed to schedule notification popup",
        )
        return

    try:
        popup = tk.Toplevel(root)
        popup.title(title)
        popup.transient(root)
        popup.resizable(False, False)
        try:
            popup.attributes("-topmost", True)
        except Exception:
            logger.debug("Topmost attribute not available for notification window")
        frame = ttk.Frame(popup, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=message, wraplength=420, justify="left").pack(anchor="w", fill="x")
        ttk.Button(frame, text="Dismiss", command=popup.destroy).pack(anchor="e", pady=(12, 0))
        popup.bind("<Escape>", lambda event: popup.destroy())
        popup.after(12000, popup.destroy)
        popup.after(120, popup.lift)
    except Exception:
        logger.exception("Failed to create notification popup window for '%s'", title)
        try:
            print(f"{title}: {message}")
        except Exception:
            logger.exception("Failed to print fallback notification '%s'", title)


def show_lockout_window(message: str, cancel_callback: Optional[Callable[[], None]] = None) -> None:
    global _lockout_win, _lockout_message_var, _lockout_progress, _lockout_cancel_button, _lockout_cancel_callback
    if tk_root is None or not tk_root.winfo_exists():
        return

    _cancel_lockout_close()

    if _lockout_win is None or not _lockout_win.winfo_exists():
        _lockout_win = tk.Toplevel(tk_root)
        _lockout_win.title("CtrlSpeak · Preparing CtrlSpeak")
        width, height = 420, 240
        geometry = f"{width}x{height}+32+32"
        try:
            screen_w = _lockout_win.winfo_screenwidth()
            screen_h = _lockout_win.winfo_screenheight()
            offset_x = max(min(32, screen_w - width), 0)
            offset_y = max(min(32, screen_h - height), 0)
            geometry = f"{width}x{height}+{offset_x}+{offset_y}"
        except Exception:
            logger.exception("Failed to query screen dimensions for lockout window")
        _lockout_win.geometry(geometry)
        _lockout_win.minsize(width, height)
        _lockout_win.resizable(False, False)
        _lockout_win.attributes("-topmost", True)
        try:
            _lockout_win.attributes("-toolwindow", True)
        except Exception:
            pass
        apply_modern_theme(_lockout_win)
        _lockout_win.protocol("WM_DELETE_WINDOW", lambda: None)

        container = ttk.Frame(_lockout_win, style="Modern.TFrame", padding=(18, 16))
        container.pack(fill=tk.BOTH, expand=True)

        card = ttk.Frame(container, style="ModernCard.TFrame", padding=(18, 18))
        card.pack(fill=tk.BOTH, expand=True)

        ttk.Label(card, text="Preparing CtrlSpeak", style="Title.TLabel").pack(anchor=tk.W)
        accent = ttk.Frame(card, style="AccentLine.TFrame")
        accent.configure(height=2)
        accent.pack(fill=tk.X, pady=(8, 12))

        _lockout_message_var = tk.StringVar(master=_lockout_win, value=message)
        message_label = ttk.Label(
            card,
            textvariable=_lockout_message_var,
            style="Body.TLabel",
            wraplength=320,
            justify=tk.LEFT,
        )
        message_label.pack(anchor=tk.W, fill=tk.X, expand=True, pady=(0, 8))

        _lockout_progress = ttk.Progressbar(
            card,
            mode="indeterminate",
            length=280,
            style="Modern.Horizontal.TProgressbar",
        )
        _lockout_progress.pack(fill=tk.X)
        try:
            _lockout_progress.start(12)
        except Exception:
            logger.exception("Failed to start lockout spinner")

        def _invoke_cancel() -> None:
            callback = _lockout_cancel_callback
            if callback is None:
                return
            try:
                callback()
            except Exception:
                logger.exception("Lockout cancel callback raised an exception")

        actions = ttk.Frame(card, style="ModernCardInner.TFrame")
        actions.pack(fill=tk.X, pady=(12, 0))
        _lockout_cancel_button = ttk.Button(
            actions,
            text="Cancel download",
            style="Danger.TButton",
            command=_invoke_cancel,
        )
        _lockout_cancel_button.pack(anchor=tk.E)
    else:
        try:
            _lockout_win.deiconify()
            _lockout_win.lift()
        except Exception:
            logger.exception("Failed to raise lockout window")

    if cancel_callback is not None:
        _lockout_cancel_callback = cancel_callback

    if _lockout_message_var is not None:
        try:
            _lockout_message_var.set(message)
        except Exception:
            logger.exception("Failed to set lockout message text")

    if _lockout_progress is not None:
        try:
            _lockout_progress.start(12)
        except Exception:
            pass

    if _lockout_cancel_button is not None:
        try:
            if _lockout_cancel_callback is None:
                _lockout_cancel_button.configure(state=tk.DISABLED)
            else:
                _lockout_cancel_button.configure(state=tk.NORMAL)
        except Exception:
            logger.exception("Failed to update lockout cancel button state")

    try:
        _lockout_win.lift()
        _lockout_win.focus_force()
    except Exception:
        pass


def update_lockout_message(message: str) -> None:
    def _update() -> None:
        if _lockout_win is None or not _lockout_win.winfo_exists() or _lockout_message_var is None:
            show_lockout_window(message, cancel_callback=_lockout_cancel_callback)
            return
        try:
            _lockout_message_var.set(message)
        except Exception:
            logger.exception("Failed to set lockout message text")

    _call_on_management_ui(
        _update,
        log_message="Failed to update lockout message text",
    )


def close_lockout_window(message: Optional[str] = None) -> None:
    global _lockout_close_job, _lockout_cancel_callback
    if _lockout_win is None or not _lockout_win.winfo_exists():
        return

    _cancel_lockout_close()

    if _lockout_message_var is not None and message:
        try:
            _lockout_message_var.set(message)
        except Exception:
            logger.exception("Failed to set lockout completion message")

    if _lockout_progress is not None:
        try:
            _lockout_progress.stop()
        except Exception:
            pass

    if _lockout_cancel_button is not None:
        try:
            _lockout_cancel_button.configure(state=tk.DISABLED)
        except Exception:
            logger.exception("Failed to disable lockout cancel button")
    _lockout_cancel_callback = None

    if message:
        try:
            _lockout_win.lift()
        except Exception:
            pass
        try:
            _lockout_close_job = _lockout_win.after(2400, _destroy_lockout_window)
        except Exception:
            logger.exception("Failed to schedule lockout window dismissal")
            _destroy_lockout_window()
    else:
        _destroy_lockout_window()
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


# v0.7 Midnight Signal overlay.  The legacy renderer above remains only as a
# compact rollback/reference path for older source checkouts; these later
# definitions are the runtime API imported by utils.system.
_midnight_overlay = None


def show_waveform_overlay(provider: Callable[[], "np.ndarray"]) -> None:
    global _midnight_overlay
    with settings_lock:
        enabled = bool(settings.get("overlay_enabled", True))
        reduced_motion = bool(settings.get("reduced_motion", False))
    if not enabled or tk_root is None or not tk_root.winfo_exists():
        return
    try:
        from utils.midnight_overlay import MidnightSignalOverlay

        if _midnight_overlay is not None and _midnight_overlay.is_open():
            _midnight_overlay.set_waveform_provider(provider)
            return
        _midnight_overlay = MidnightSignalOverlay(
            tk_root,
            snapshot_provider=sysmod.transcription_ui_session.snapshot,
            waveform_provider=provider,
            device_label_provider=sysmod.get_input_device_preference,
            reduced_motion=reduced_motion,
        )
        _midnight_overlay.show()
    except Exception:
        logger.exception("Failed to open the Midnight Signal overlay")


def set_waveform_processing(message: str = "Transcribing…") -> None:
    del message  # copy is state-driven and cannot falsely claim a live provider
    global _midnight_overlay
    if _midnight_overlay is None or not _midnight_overlay.is_open():
        show_waveform_overlay(lambda: np.zeros(32, dtype=np.float32))


def hide_waveform_overlay(delay_ms: int = 0) -> None:
    global _midnight_overlay
    overlay = _midnight_overlay
    if overlay is None:
        return
    try:
        overlay.close(delay_ms=max(0, int(delay_ms)))
    except Exception:
        logger.exception("Failed to close the Midnight Signal overlay")
    if delay_ms <= 0:
        _midnight_overlay = None

# ---------------- Splash (1s) ----------------
def show_splash_screen(duration_ms: int) -> None:
    configure_process_dpi_awareness()
    try:
        root = tk.Tk(className="CtrlSpeak")
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
        icon_path = app_icon_path()
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
def prompt_initial_mode(parent: Optional[tk.Misc] = None) -> Optional[str]:
    from utils.system import AUTO_MODE_PROFILE

    if AUTO_MODE_PROFILE:
        return AUTO_MODE_PROFILE

    result = {"mode": None}
    manual_server_var: Optional[tk.StringVar] = None
    manual_server_error: Optional[tk.StringVar] = None
    server_status_var: Optional[tk.StringVar] = None
    server_list_var: Optional[tk.StringVar] = None
    refresh_button: Optional[ttk.Button] = None
    server_listbox: Optional[tk.Listbox] = None
    discovery_listener_local: Optional[net_discovery.DiscoveryListener] = None
    available_servers: list[str] = []
    refresh_in_progress = threading.Event()

    def _cleanup_listener() -> None:
        nonlocal discovery_listener_local
        listener = discovery_listener_local
        discovery_listener_local = None
        if listener is None:
            return
        try:
            listener.stop()
        except Exception:
            logger.exception("Failed to stop temporary discovery listener")
        try:
            listener.join(timeout=1.5)
        except Exception:
            logger.exception("Failed to join temporary discovery listener")

    def _release_grab() -> None:
        if window is None:
            return
        try:
            window.grab_release()
        except Exception:
            pass

    def choose(mode: str) -> None:
        nonlocal available_servers
        if mode == "client" and manual_server_var is not None:
            target = manual_server_var.get().strip()
            if target:
                try:
                    host, port = net_discovery.parse_server_target(target)
                except ValueError as exc:
                    if manual_server_error is not None:
                        manual_server_error.set(str(exc))
                    return
                if manual_server_error is not None:
                    manual_server_error.set("")
                try:
                    net_discovery.set_preferred_server(host, port)
                except Exception:
                    logger.exception("Failed to persist preferred server from mode picker")
            else:
                if manual_server_error is not None:
                    manual_server_error.set("")
        result["mode"] = mode
        _cleanup_listener()
        if not owns_root:
            _release_grab()
        window.destroy()

    owns_root = parent is None
    window: Optional[tk.Misc] = None
    try:
        if owns_root:
            window = tk.Tk()
        else:
            window = tk.Toplevel(parent)
            try:
                window.transient(parent)
            except Exception:
                pass
            try:
                window.grab_set()
            except Exception:
                logger.exception("Failed to grab mode selection dialog")
    except Exception:
        logger.exception("Failed to create mode selection window")
        trace_model_download_step(
            "prompt_initial_mode: failed to initialize UI", "default to client_server"
        )
        return "client_server"

    if window is None:
        trace_model_download_step(
            "prompt_initial_mode: window creation returned None", "default to client_server"
        )
        return "client_server"

    try:
        apply_modern_theme(window)
    except Exception:
        logger.exception("Failed to apply theme to mode selection window")
        try:
            window.destroy()
        except Exception:
            pass
        trace_model_download_step(
            "prompt_initial_mode: theme initialization failed", "default to client_server"
        )
        return "client_server"

    manual_server_var = tk.StringVar(master=window)
    manual_server_error = tk.StringVar(master=window, value="")
    server_status_var = tk.StringVar(
        master=window,
        value="Enter the server address manually or scan for available hosts.",
    )
    server_list_var = tk.StringVar(master=window, value=())

    preferred_host, preferred_port = net_discovery.get_preferred_server_settings()
    if preferred_host and preferred_port:
        manual_server_var.set(f"{preferred_host}:{preferred_port}")

    def _ensure_discovery_listener() -> Optional[net_discovery.DiscoveryListener]:
        nonlocal discovery_listener_local
        if discovery_listener_local is not None:
            return discovery_listener_local
        try:
            port = net_discovery.get_discovery_port()
            listener = net_discovery.DiscoveryListener(port)
            listener.start()
            discovery_listener_local = listener
            return listener
        except Exception:
            logger.exception("Failed to start temporary discovery listener")
            discovery_listener_local = None
            return None

    def _on_server_select(event: tk.Event) -> None:  # type: ignore[valid-type]
        if manual_server_var is None:
            return
        widget = event.widget
        if not isinstance(widget, tk.Listbox):
            return
        selection = widget.curselection()
        if not selection:
            return
        index = selection[0]
        if 0 <= index < len(available_servers):
            manual_server_var.set(available_servers[index])
            if manual_server_error is not None:
                manual_server_error.set("")

    def _trigger_refresh(initial: bool = False) -> None:
        if refresh_in_progress.is_set():
            return
        refresh_in_progress.set()
        if refresh_button is not None:
            try:
                refresh_button.state(["disabled"])
                refresh_button.config(text="Scanning…")
            except Exception:
                logger.exception("Failed to disable refresh button in mode picker")
        if server_status_var is not None:
            server_status_var.set("Scanning for servers…")

        def worker() -> None:
            entries: list[str] = []
            error_message: Optional[str] = None
            listener = _ensure_discovery_listener()
            if listener is None:
                error_message = "Unable to start discovery. Enter the server address manually."
            else:
                try:
                    net_discovery.manual_discovery_refresh(listener, wait_time=1.5)
                    infos = sorted(
                        list(listener.registry.values()),
                        key=lambda info: info.last_seen,
                        reverse=True,
                    )
                    entries = [f"{info.host}:{info.port}" for info in infos]
                except Exception:
                    logger.exception("Failed to refresh servers from mode picker")
                    error_message = "Server scan failed. Enter the server address manually."

            def on_complete() -> None:
                nonlocal available_servers
                if server_list_var is not None:
                    try:
                        server_list_var.set(tuple(entries))
                    except Exception:
                        logger.exception("Failed to update server list in mode picker")
                available_servers = entries
                if server_status_var is not None:
                    if entries:
                        server_status_var.set("Select a server or enter an address manually.")
                    else:
                        server_status_var.set(
                            error_message
                            or "No servers found. Enter the server address manually or refresh again."
                        )
                if manual_server_error is not None and manual_server_error.get():
                    # Do not clear parse errors automatically; only clear when the user edits the field.
                    pass
                if manual_server_var is not None and entries and not manual_server_var.get().strip():
                    manual_server_var.set(entries[0])
                if refresh_button is not None:
                    try:
                        refresh_button.state(["!disabled"])
                        refresh_button.config(text="Refresh servers")
                    except Exception:
                        logger.exception("Failed to reset refresh button in mode picker")
                refresh_in_progress.clear()

            try:
                if window is not None and window.winfo_exists():
                    window.after(0, on_complete)
                else:
                    refresh_in_progress.clear()
            except Exception:
                logger.exception("Failed to schedule refresh completion in mode picker")
                refresh_in_progress.clear()

        threading.Thread(target=worker, daemon=True).start()

    window.title("Welcome to CtrlSpeak")
    window.geometry("600x480")
    window.minsize(560, 380)
    window.resizable(True, True)
    window.attributes("-topmost", True)

    container = ttk.Frame(window, style="Modern.TFrame", padding=(28, 26))
    container.pack(fill=tk.BOTH, expand=True)

    intro = ttk.Frame(container, style="ModernCard.TFrame", padding=(24, 22))
    intro.pack(fill=tk.X)
    ttk.Label(intro, text="Welcome to CtrlSpeak", style="Title.TLabel").pack(anchor=tk.W)
    message = (
        "It looks like this is the first time CtrlSpeak is running on this computer. "
        "Choose how you would like to use it:"
    )
    ttk.Label(intro, text=message, style="Body.TLabel", wraplength=500,
              justify=tk.LEFT).pack(anchor=tk.W, pady=(12, 0))
    accent = ttk.Frame(intro, style="AccentLine.TFrame")
    accent.configure(height=2)
    accent.pack(fill=tk.X, pady=(18, 8))

    cards = ttk.Frame(container, style="Modern.TFrame")
    cards.pack(fill=tk.BOTH, expand=True, pady=(18, 12))

    def make_card(title: str, desc: str, mode_value: str, primary: bool,
                  extra: Optional[Callable[[ttk.Frame], None]] = None) -> None:
        card = ttk.Frame(cards, style="ModernCard.TFrame", padding=(22, 20))
        card.pack(fill=tk.X, pady=8)
        label_text = f"MODE · {mode_value.replace('_', ' ').upper()}"
        ttk.Label(card, text=label_text, style="PillMuted.TLabel").pack(anchor=tk.W)
        accent_inner = ttk.Frame(card, style="AccentLine.TFrame")
        accent_inner.configure(height=2)
        accent_inner.pack(fill=tk.X, pady=(10, 14))
        ttk.Label(card, text=title, style="SectionHeading.TLabel").pack(anchor=tk.W)
        ttk.Label(card, text=desc, style="Body.TLabel", wraplength=500,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 12))
        if extra is not None:
            extra(card)
        btn_style = "Accent.TButton" if primary else "Subtle.TButton"
        ttk.Button(card, text="Use this mode", style=btn_style,
                   command=lambda: choose(mode_value)).pack(anchor=tk.E, pady=(12, 0))

    make_card(
        "Client + Server",
        "Use this computer for local transcription and optionally share it with other CtrlSpeak clients on "
        "your network.",
        "client_server",
        True,
    )

    def _build_client_only_controls(card: ttk.Frame) -> None:
        nonlocal refresh_button, server_listbox
        inner = ttk.Frame(card, style="ModernCardInner.TFrame")
        inner.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        ttk.Label(inner, text="Preferred server", style="Subtitle.TLabel").pack(anchor=tk.W)
        ttk.Label(
            inner,
            text="Enter the IP address and port of a CtrlSpeak server or select one from the list below.",
            style="Caption.TLabel",
            wraplength=500,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(4, 6))

        entry = ttk.Entry(inner, textvariable=manual_server_var, style="Modern.TEntry")
        entry.pack(fill=tk.X)

        def _clear_error(_: object) -> None:
            if manual_server_error is not None:
                manual_server_error.set("")

        entry.bind("<KeyRelease>", _clear_error)

        error_label = tk.Label(
            inner,
            textvariable=manual_server_error,
            anchor="w",
            fg=DANGER,
            bg=ELEVATED_SURFACE,
            font=("{Segoe UI}", 9),
            padx=2,
        )
        error_label.pack(fill=tk.X, pady=(4, 0))

        ttk.Label(inner, textvariable=server_status_var, style="Caption.TLabel",
                  wraplength=500, justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 4))

        server_listbox = tk.Listbox(
            inner,
            listvariable=server_list_var,
            height=4,
            bg=ELEVATED_SURFACE,
            fg="#e7f2ff",
            highlightthickness=0,
            relief=tk.FLAT,
            selectbackground=ACCENT,
            activestyle="dotbox",
            exportselection=False,
        )
        server_listbox.pack(fill=tk.BOTH, expand=False, pady=(0, 8))
        server_listbox.bind("<<ListboxSelect>>", _on_server_select)

        refresh_button = ttk.Button(
            inner,
            text="Refresh servers",
            style="Subtle.TButton",
            command=_trigger_refresh,
        )
        refresh_button.pack(fill=tk.X)

    make_card(
        "Client Only",
        "Connect to another CtrlSpeak server on your network and send recordings there for transcription.",
        "client",
        False,
        extra=_build_client_only_controls,
    )

    def cancel() -> None:
        _cleanup_listener()
        if not owns_root:
            _release_grab()
        window.destroy()

    ttk.Button(container, text="Quit Setup", style="Subtle.TButton",
               command=cancel).pack(fill=tk.X, pady=(8, 0))
    window.protocol("WM_DELETE_WINDOW", cancel)

    if refresh_button is not None:
        window.after(200, _trigger_refresh)

    window.update_idletasks()
    req_w = window.winfo_reqwidth()
    req_h = window.winfo_reqheight()
    window.geometry(f"{req_w}x{req_h}")
    window.minsize(req_w, req_h)
    window.after(200, lambda: window.attributes("-topmost", False))

    if owns_root:
        try:
            window.wait_window()
        finally:
            try:
                window.quit()
            except Exception:
                pass
    else:
        try:
            window.wait_window()
        finally:
            _release_grab()
    _cleanup_listener()
    return result["mode"]


def _ensure_server_mode_model_ready() -> bool:
    """Ensure the active Whisper model is present before running the local server."""

    trace_model_download_step(
        "_ensure_server_mode_model_ready(gui): start",
        "check if model exists",
    )
    name = get_current_model_name()
    if model_files_present(model_store_path_for(name)):
        trace_model_download_step(
            "_ensure_server_mode_model_ready(gui): model already present",
            "return True",
        )
        return True

    trace_model_download_step(
        "_ensure_server_mode_model_ready(gui): model missing",
        "invoke download_model_with_gui",
    )
    if not download_model_with_gui(name, block_during_download=True):
        trace_model_download_step(
            "_ensure_server_mode_model_ready(gui): download failed or cancelled",
            "return False",
        )
        return False

    has_files = model_files_present(model_store_path_for(name))
    trace_model_download_step(
        "_ensure_server_mode_model_ready(gui): verification",
        "return True" if has_files else "report missing files",
    )
    return has_files


def ensure_mode_selected() -> None:
    # 1) Load settings from disk first
    trace_model_download_step(
        "ensure_mode_selected: begin",
        "load_settings",
    )
    load_settings()

    # 2) If mode already set, don't prompt again
    with settings_lock:
        current_mode = settings.get("mode")

    trace_model_download_step(
        "ensure_mode_selected: mode read from settings",
        "skip prompt if already configured",
    )
    if current_mode in {"client", "client_server"}:
        trace_model_download_step(
            "ensure_mode_selected: existing mode detected",
            "verify model if client_server",
        )
        if current_mode == "client_server" and not _ensure_server_mode_model_ready():
            notify("CtrlSpeak will exit because the Whisper model download was cancelled.")
            sys.exit(0)
        return  # nothing to do

    # 3) Client-only builds force 'client' once and persist
    trace_model_download_step(
        "ensure_mode_selected: prompting for mode",
        "handle client-only build or prompt UI",
    )
    if detect_client_only_build():
        with settings_lock:
            settings["mode"] = "client"
        save_settings()
        trace_model_download_step(
            "ensure_mode_selected: client-only build",
            "return",
        )
        return

    # 4) Prompt the user for their preferred mode on first run
    trace_model_download_step(
        "ensure_mode_selected: launching mode picker",
        "await user selection",
    )
    choice = prompt_initial_mode()
    if choice not in {"client", "client_server"}:
        trace_model_download_step(
            "ensure_mode_selected: selection cancelled",
            "exit application",
        )
        notify("CtrlSpeak setup was cancelled. Start CtrlSpeak again to continue setup.")
        sys.exit(0)

    with settings_lock:
        settings["mode"] = choice
    save_settings()

    trace_model_download_step(
        "ensure_mode_selected: selection saved",
        "ensure download",
    )
    if choice == "client_server" and not _ensure_server_mode_model_ready():
        notify("CtrlSpeak will exit because the Whisper model download was cancelled.")
        sys.exit(0)

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

    configure_process_dpi_awareness()
    try:
        root = tk.Tk(className="CtrlSpeak")
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

    global tk_root, management_window, tray_flyout, _management_thread_ident, _management_queue_job

    root = tk_root
    if root is not None and _management_queue_job is not None:
        try:
            root.after_cancel(_management_queue_job)
        except Exception:
            logger.exception("Failed to cancel management UI queue job during teardown")
    _management_queue_job = None

    if root is not None:
        try:
            if tray_flyout is not None:
                tray_flyout.close()
        except Exception:
            logger.debug("Failed to close tray flyout during UI teardown", exc_info=True)
        try:
            root.destroy()
        except Exception:
            logger.exception("Failed to destroy management UI root during teardown")

    tk_root = None
    management_window = None
    tray_flyout = None
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


def _show_management_page(icon: pystray.Icon, page_name: str) -> None:
    _show_management_window(icon)
    active = management_window
    if active is None or not active.is_open():
        return
    notebook = getattr(active, "ms_notebook", None)
    pages = getattr(active, "ms_pages", {})
    page = pages.get(page_name) if isinstance(pages, dict) else None
    if notebook is not None and page is not None:
        notebook.select(page)


def correction_dialog_geometry(
    bounds: object,
    scale: float,
) -> tuple[int, int, int, int, int, int]:
    """Return a centred, work-area-safe correction dialog geometry.

    Values are ``x, y, width, height, minimum_width, minimum_height`` in
    physical pixels.  On constrained displays the minimum collapses to the
    available work area; the dialog's scrollable body then yields space to its
    pinned footer.
    """

    scale = max(0.75, min(3.0, float(scale)))
    bounds_width = max(1, int(getattr(bounds, "width")))
    bounds_height = max(1, int(getattr(bounds, "height")))
    bounds_left = int(getattr(bounds, "left"))
    bounds_top = int(getattr(bounds, "top"))
    horizontal_margin = min(
        max(10, int(round(26 * scale))),
        max(0, (bounds_width - 1) // 2),
    )
    vertical_margin = min(
        max(10, int(round(20 * scale))),
        max(0, (bounds_height - 1) // 2),
    )
    available_width = max(1, bounds_width - (2 * horizontal_margin))
    available_height = max(1, bounds_height - (2 * vertical_margin))
    width = min(max(1, int(round(680 * scale))), available_width)
    height = min(max(1, int(round(600 * scale))), available_height)
    x = bounds_left + max(0, (bounds_width - width) // 2)
    y = bounds_top + max(0, (bounds_height - height) // 2)
    minimum_width = min(width, max(1, int(round(460 * scale))))
    minimum_height = min(height, max(1, int(round(390 * scale))))
    return x, y, width, height, minimum_width, minimum_height


def correction_dialog_geometry_spec(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    reference_right: int,
    reference_bottom: int,
) -> str:
    """Format absolute desktop coordinates for Tk's signed-offset grammar.

    Tk interprets ``-100`` as 100 pixels from its reference screen's right or
    bottom edge, not as the absolute coordinate -100. Convert negative
    coordinates to that edge-relative representation. On Windows, a final
    ``SetWindowPos`` corrects for the native non-client frame after realization.
    """

    x = int(x)
    y = int(y)
    width = max(1, int(width))
    height = max(1, int(height))
    if x < 0:
        horizontal = f"-{max(0, int(reference_right) - (x + width))}"
    else:
        horizontal = f"+{x}"
    if y < 0:
        vertical = f"-{max(0, int(reference_bottom) - (y + height))}"
    else:
        vertical = f"+{y}"
    return f"{width}x{height}{horizontal}{vertical}"


def _tk_geometry_reference_edges(window: tk.Misc) -> tuple[int, int]:
    """Return the right/bottom edges Tk uses for negative geometry offsets."""

    return int(window.winfo_screenwidth()), int(window.winfo_screenheight())


def _place_windows_toplevel_absolute(window: tk.Toplevel, x: int, y: int) -> bool:
    """Place the realized outer Win32 frame at an exact virtual-desktop point."""

    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        from ctypes import wintypes

        from utils.midnight_overlay import _resolve_windows_toplevel_hwnd

        user32 = ctypes.windll.user32
        set_window_pos = user32.SetWindowPos
        set_window_pos.argtypes = (
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        )
        set_window_pos.restype = wintypes.BOOL
        window.update_idletasks()
        hwnd = _resolve_windows_toplevel_hwnd(window)
        # SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE
        if not set_window_pos(hwnd, 0, int(x), int(y), 0, 0, 0x0015):
            raise ctypes.WinError()
        return True
    except Exception:
        logger.debug("Unable to place the correction dialog through Win32", exc_info=True)
    return False


def _show_tray_flyout(icon: pystray.Icon) -> None:
    global tray_flyout
    from utils.midnight_signal_ui import MidnightTrayFlyout

    if tray_flyout is None:
        tray_flyout = MidnightTrayFlyout(
            tk_root,
            icon,
            open_control=lambda: _show_management_window(icon),
            open_corrections=lambda: _show_management_page(icon, "Corrections"),
            check_updates=lambda: _open_and_check_updates(icon),
            quit_app=lambda: sysmod.on_exit(icon, None),
        )
    tray_flyout.toggle()


def _open_and_check_updates(icon: pystray.Icon) -> None:
    _show_management_page(icon, "Updates")
    if management_window is not None:
        management_window.check_for_updates()


def _show_correction_submission_dialog(icon: pystray.Icon) -> None:
    """Open, or focus, the gateway correction submission dialog."""
    global correction_submission_dialog

    config = get_runtime_backend_config()
    if config.backend != "api":
        messagebox.showinfo(
            "Remote gateway required",
            "Known-word corrections are submitted to a CtrlSpeak gateway. "
            "Select Remote API in Open control centre and restart the application first.",
            parent=tk_root,
        )
        return

    if correction_submission_dialog and correction_submission_dialog.is_open():
        correction_submission_dialog.set_config(config)
        correction_submission_dialog.bring_to_front()
        return

    correction_submission_dialog = CorrectionSubmissionDialog(icon, config)


class CorrectionSubmissionDialog:
    """Tray-launched Midnight Signal form for one authenticated correction rule.

    The body is allowed to scroll, but the live status and actions are outside
    that viewport.  That split keeps Submit and Hide reachable on short or
    high-DPI work areas instead of relying on a fixed dialog height.
    """

    def __init__(self, icon: pystray.Icon, config: BackendConfig) -> None:
        self._icon = icon
        self._config = config
        self._pending_config: BackendConfig | None = None
        self._submitting = False
        self._submission_generation = 0
        self._status_kind = "ready"
        parent = (
            management_window.window
            if management_window is not None and management_window.is_open()
            else tk_root
        )
        self.window = tk.Toplevel(parent)
        self.window.title(f"CtrlSpeak {APP_VERSION} · Submit correction")
        self.window.protocol("WM_DELETE_WINDOW", self.hide)
        self.window.bind("<Escape>", self._hide_from_event)
        self.window.bind("<Control-Return>", self._submit_from_event)
        self.window.bind("<Alt-s>", self._submit_from_event)
        self.window.bind("<Alt-h>", self._focus_source)
        self.window.bind("<Alt-r>", self._focus_replacement)
        self.window.bind("<Alt-g>", self._toggle_global_scope)
        self.window.bind("<Prior>", lambda _event: self._scroll_body(-1, pages=True))
        self.window.bind("<Next>", lambda _event: self._scroll_body(1, pages=True))
        if parent is not None:
            try:
                # A transient whose owner is CtrlSpeak's intentionally hidden
                # root is itself forced into the withdrawn state by Tk/Win32.
                # Use modality only when the visible control centre owns us.
                if str(parent.state()) not in {"withdrawn", "iconic"}:
                    self.window.transient(parent)
            except Exception:
                logger.debug("Unable to make the correction dialog transient", exc_info=True)

        from utils.midnight_overlay import active_monitor_bounds, display_scale
        from utils.midnight_signal_ui import (
            AMBER as MS_AMBER,
            CORAL as MS_CORAL,
            CARD_ALT as MS_CARD_ALT,
            CYAN as MS_CYAN,
            INK as MS_INK,
            MINT as MS_MINT,
            MUTED as MS_MUTED,
            SURFACE as MS_SURFACE,
            TEXT as MS_TEXT,
            apply_midnight_signal_theme,
        )

        apply_midnight_signal_theme(self.window)
        style = ttk.Style(self.window)
        style.configure(
            "MS.DialogEyebrow.TLabel",
            background=MS_INK,
            foreground=MS_CYAN,
            font=("Segoe UI Semibold", 9),
        )
        style.configure(
            "MS.DialogSubtitle.TLabel",
            background=MS_INK,
            foreground=MS_MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "MS.DialogSurfaceMuted.TLabel",
            background=MS_SURFACE,
            foreground=MS_MUTED,
            font=("Segoe UI", 9),
        )
        for name, colour in (
            ("Ready", MS_MUTED),
            ("Busy", MS_AMBER),
            ("Success", MS_MINT),
            ("Error", MS_CORAL),
        ):
            style.configure(
                f"MS.DialogStatus{name}.TLabel",
                background=MS_INK,
                foreground=colour,
                font=("Segoe UI Semibold", 9),
            )

        scale = display_scale(self.window)
        bounds = active_monitor_bounds(self.window)
        (
            x,
            y,
            width,
            height,
            minimum_width,
            minimum_height,
        ) = correction_dialog_geometry(bounds, scale)
        self._compact_layout = (
            width < int(round(680 * scale))
            or height < int(round(600 * scale))
        )
        self._status_character_limit = 42 if self._compact_layout else 72
        reference_right, reference_bottom = _tk_geometry_reference_edges(self.window)
        self.window.geometry(
            correction_dialog_geometry_spec(
                x,
                y,
                width,
                height,
                reference_right=reference_right,
                reference_bottom=reference_bottom,
            )
        )
        self.window.minsize(minimum_width, minimum_height)
        self.window.maxsize(width, height)
        self.window.resizable(True, True)
        try:
            _set_window_icon(self.window)
        except Exception:
            logger.exception("Failed to set the correction dialog icon")

        self.source_var = tk.StringVar(master=self.window)
        self.replacement_var = tk.StringVar(master=self.window)
        self.global_scope_var = tk.BooleanVar(master=self.window, value=False)
        self.status_var = tk.StringVar(
            master=self.window,
            value="Ready to submit a correction.",
        )

        shell = ttk.Frame(self.window, style="MS.Root.TFrame")
        shell.pack(fill=tk.BOTH, expand=True)
        shell.rowconfigure(1, weight=1)
        shell.columnconfigure(0, weight=1)

        header_padding = (20, 11, 20, 9) if self._compact_layout else (26, 20, 26, 14)
        header = ttk.Frame(shell, style="MS.Root.TFrame", padding=header_padding)
        header.grid(row=0, column=0, sticky="ew")
        eyebrow = ttk.Label(
            header,
            text="CTRLSPEAK  ·  CORRECTION",
            style="MS.DialogEyebrow.TLabel",
        )
        if not self._compact_layout:
            eyebrow.pack(anchor=tk.W)
        ttk.Label(
            header,
            text=(
                "Submit correction"
                if self._compact_layout
                else "Teach CtrlSpeak what you meant"
            ),
            style="MS.Title.TLabel",
        ).pack(anchor=tk.W, pady=((0, 0) if self._compact_layout else (5, 3)))

        viewport = ttk.Frame(shell, style="MS.Surface.TFrame")
        viewport.grid(row=1, column=0, sticky="nsew")
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)
        self._body_canvas = tk.Canvas(
            viewport,
            background=MS_SURFACE,
            borderwidth=0,
            highlightthickness=0,
        )
        body_scrollbar = ttk.Scrollbar(
            viewport,
            orient=tk.VERTICAL,
            command=self._body_canvas.yview,
        )
        self._body_canvas.configure(yscrollcommand=body_scrollbar.set)
        self._body_canvas.grid(row=0, column=0, sticky="nsew")
        body_scrollbar.grid(row=0, column=1, sticky="ns")

        body = ttk.Frame(
            self._body_canvas,
            style="MS.Surface.TFrame",
            padding=((18, 12) if self._compact_layout else (26, 20)),
        )
        self._body_window_id = self._body_canvas.create_window(
            (0, 0),
            window=body,
            anchor="nw",
        )
        body.bind("<Configure>", self._sync_body_scroll_region)
        self._body_canvas.bind("<Configure>", self._sync_body_width)
        self.window.bind("<MouseWheel>", self._scroll_body_wheel, add="+")
        self.window.bind("<Button-4>", lambda _event: self._scroll_body(-3), add="+")
        self.window.bind("<Button-5>", lambda _event: self._scroll_body(3), add="+")

        self._subtitle_label = ttk.Label(
            body,
            text=(
                "Add one exact phrase replacement. The gateway applies it after "
                "transcription and makes it active immediately."
            ),
            style="MS.DialogSurfaceMuted.TLabel",
            justify=tk.LEFT,
        )
        if not self._compact_layout:
            self._subtitle_label.pack(anchor=tk.W, fill=tk.X, pady=(0, 12))

        card = ttk.Frame(
            body,
            style="MS.OutlinedCard.TFrame",
            padding=((18, 15) if self._compact_layout else (24, 22)),
        )
        card.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            card,
            text="01  ·  WHEN CTRLSPEAK HEARS",
            style="MS.CompactMetric.TLabel",
            underline=22,
        ).pack(anchor=tk.W)
        self._source_help_label = ttk.Label(
            card,
            text="The phrase currently returned by transcription",
            style="MS.CardMuted.TLabel",
            justify=tk.LEFT,
        )
        self._source_help_label.pack(anchor=tk.W, fill=tk.X, pady=(3, 8))
        self.source_entry = ttk.Entry(
            card,
            textvariable=self.source_var,
            style="MS.TEntry",
            takefocus=True,
        )
        self.source_entry.pack(fill=tk.X)
        self.source_entry.bind("<Return>", self._focus_replacement)

        ttk.Label(
            card,
            text="02  ·  REPLACE IT WITH",
            style="MS.CompactMetric.TLabel",
            underline=7,
        ).pack(anchor=tk.W, pady=(22, 0))
        self._replacement_help_label = ttk.Label(
            card,
            text="The exact text CtrlSpeak should return instead",
            style="MS.CardMuted.TLabel",
            justify=tk.LEFT,
        )
        self._replacement_help_label.pack(anchor=tk.W, fill=tk.X, pady=(3, 8))
        self.replacement_entry = ttk.Entry(
            card,
            textvariable=self.replacement_var,
            style="MS.TEntry",
            takefocus=True,
        )
        self.replacement_entry.pack(fill=tk.X)
        self.replacement_entry.bind("<Return>", self._submit_from_event)

        ttk.Separator(card, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(23, 18))
        self.global_scope_check = ttk.Checkbutton(
            card,
            text="Global rule (administrator only)",
            variable=self.global_scope_var,
            style="MS.TCheckbutton",
            underline=0,
            takefocus=True,
        )
        self.global_scope_check.pack(anchor=tk.W)
        self._scope_help_label = ttk.Label(
            card,
            text=(
                "Leave this off to save the rule only for your authenticated gateway "
                "identity. Global rules require gateway administrator permission."
            ),
            style="MS.CardMuted.TLabel",
            justify=tk.LEFT,
        )
        self._scope_help_label.pack(anchor=tk.W, fill=tk.X, pady=(7, 18))

        gateway = ttk.Frame(card, style="MS.CardAlt.TFrame", padding=(14, 11))
        gateway.pack(fill=tk.X)
        ttk.Label(
            gateway,
            text="GATEWAY",
            background=MS_CARD_ALT,
            foreground=MS_MUTED,
            font=("Segoe UI Semibold", 8),
        ).pack(anchor=tk.W)
        self.gateway_var = tk.StringVar(master=self.window)
        self._gateway_label = ttk.Label(
            gateway,
            textvariable=self.gateway_var,
            background=MS_CARD_ALT,
            foreground=MS_TEXT,
            font=("Segoe UI", 9),
            justify=tk.LEFT,
        )
        self._gateway_label.pack(anchor=tk.W, fill=tk.X, pady=(2, 0))
        self._keyboard_help_label = ttk.Label(
            card,
            text="Esc hides  ·  Ctrl+Enter submits",
            style="MS.CardMuted.TLabel",
            justify=tk.LEFT,
        )
        if not self._compact_layout:
            self._keyboard_help_label.pack(anchor=tk.W, fill=tk.X, pady=(15, 0))

        footer_padding = (20, 9, 20, 12) if self._compact_layout else (26, 14, 26, 20)
        footer = ttk.Frame(shell, style="MS.Root.TFrame", padding=footer_padding)
        footer.grid(row=2, column=0, sticky="ew")
        status_row = ttk.Frame(footer, style="MS.Root.TFrame")
        status_row.pack(fill=tk.X)
        self._status_label = ttk.Label(
            status_row,
            textvariable=self.status_var,
            style="MS.DialogStatusReady.TLabel",
            justify=tk.LEFT,
        )
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._progress = ttk.Progressbar(
            status_row,
            mode="indeterminate",
            length=max(80, int(round(110 * scale))),
            style="MS.Horizontal.TProgressbar",
        )

        buttons = ttk.Frame(footer, style="MS.Root.TFrame")
        buttons.pack(fill=tk.X, pady=((8, 0) if self._compact_layout else (13, 0)))
        self.hide_button = ttk.Button(
            buttons,
            text="Hide window",
            style="MS.TButton",
            command=self.hide,
            takefocus=True,
        )
        self.submit_button = ttk.Button(
            buttons,
            text="Submit correction",
            style="MS.Primary.TButton",
            command=self.submit,
            takefocus=True,
            underline=0,
        )
        self.submit_button.pack(side=tk.RIGHT)
        self.hide_button.pack(side=tk.RIGHT, padx=(0, 10))

        self._form_controls = (
            self.source_entry,
            self.replacement_entry,
            self.global_scope_check,
        )
        self.set_config(config)
        self.source_var.trace_add("write", self._form_changed)
        self.replacement_var.trace_add("write", self._form_changed)
        self.window.bind("<Configure>", self._resize_wrapped_copy, add="+")

        self.bring_to_front()
        _place_windows_toplevel_absolute(self.window, x, y)
        self.window.after_idle(self._focus_initial)

    def is_open(self) -> bool:
        try:
            return bool(self.window.winfo_exists())
        except Exception:
            return False

    def is_visible(self) -> bool:
        try:
            return self.is_open() and self.window.state() not in {"withdrawn", "iconic"}
        except tk.TclError:
            return False

    def _focus_initial(self) -> None:
        if self.is_visible() and not self._submitting:
            self.source_entry.focus_set()

    def bring_to_front(self) -> None:
        if not self.is_open():
            return
        try:
            self.window.deiconify()
            self.window.lift()
            self.window.focus_force()
        except Exception:
            logger.exception("Failed to focus the correction dialog")

    def set_config(self, config: BackendConfig) -> None:
        """Refresh the safe endpoint snapshot whenever the hidden form is reopened."""

        if self._submitting:
            # Keep only the latest observed settings. If the user changes A to
            # B and back to A during the request, do not apply stale B after it
            # completes.
            self._pending_config = None if config == self._config else config
            return
        self._apply_config(config)

    def _apply_config(self, config: BackendConfig) -> None:
        self._config = config
        self.gateway_var.set(str(config.api_url or "Configured CtrlSpeak gateway"))

    def _resize_wrapped_copy(self, event) -> None:
        if event.widget is not self.window:
            return
        wrap = max(220, int(event.width) - 90)
        self._subtitle_label.configure(wraplength=wrap)
        self._source_help_label.configure(wraplength=max(180, wrap - 48))
        self._replacement_help_label.configure(wraplength=max(180, wrap - 48))
        self._scope_help_label.configure(wraplength=max(180, wrap - 48))
        self._gateway_label.configure(wraplength=max(180, wrap - 48))
        self._keyboard_help_label.configure(wraplength=max(180, wrap - 48))
        self._status_label.configure(wraplength=wrap)

    def _sync_body_scroll_region(self, _event=None) -> None:
        if self.is_open():
            self._body_canvas.configure(scrollregion=self._body_canvas.bbox("all"))

    def _sync_body_width(self, event) -> None:
        if self.is_open():
            self._body_canvas.itemconfigure(
                self._body_window_id,
                width=max(1, int(event.width)),
            )

    def _scroll_body_wheel(self, event) -> str | None:
        try:
            delta = int(event.delta)
        except (AttributeError, TypeError, ValueError):
            return None
        if delta:
            self._body_canvas.yview_scroll(-3 if delta > 0 else 3, "units")
            return "break"
        return None

    def _scroll_body(self, amount: int, *, pages: bool = False) -> str:
        self._body_canvas.yview_scroll(int(amount), "pages" if pages else "units")
        return "break"

    def _focus_source(self, _event=None) -> str:
        if not self._submitting:
            self.source_entry.focus_set()
        return "break"

    def _focus_replacement(self, _event=None) -> str:
        if not self._submitting:
            self.replacement_entry.focus_set()
        return "break"

    def _toggle_global_scope(self, _event=None) -> str:
        if not self._submitting:
            self.global_scope_check.invoke()
        return "break"

    def _submit_from_event(self, _event=None) -> str:
        self.submit()
        return "break"

    def _hide_from_event(self, _event=None) -> str:
        self.hide()
        return "break"

    def _form_changed(self, *_args) -> None:
        if not self._submitting and self._status_kind in {"error", "success"}:
            self._set_status("ready", "Ready to submit a correction.")

    def _set_form_enabled(self, enabled: bool) -> None:
        state = "!disabled" if enabled else "disabled"
        for widget in self._form_controls:
            try:
                widget.state([state])
            except tk.TclError:
                logger.debug("Correction form control disappeared during state update")
        self.submit_button.state([state])

    def _set_status(self, kind: str, text: str) -> None:
        self._status_kind = kind if kind in {"ready", "busy", "success", "error"} else "ready"
        title = self._status_kind.title()
        normalized = " ".join(str(text or "").split())
        limit = max(24, int(getattr(self, "_status_character_limit", 72)))
        if len(normalized) > limit:
            normalized = normalized[: limit - 1].rstrip() + "…"
        marker = {
            "ready": "●",
            "busy": "◌",
            "success": "✓",
            "error": "!",
        }[self._status_kind]
        self.status_var.set(f"{marker}  {normalized}")
        self._status_label.configure(style=f"MS.DialogStatus{title}.TLabel")
        if self._status_kind == "busy":
            if not self._progress.winfo_ismapped():
                self._progress.pack(side="right", padx=(14, 0))
            self._progress.start(12)
            self.window.configure(cursor="watch")
        else:
            self._progress.stop()
            self._progress.pack_forget()
            self.window.configure(cursor="")

    @staticmethod
    def _safe_error_summary(error: object) -> str:
        """Map untrusted transport details to fixed, non-sensitive UI copy."""

        detail = " ".join(str(error or "").casefold().split())
        if detail in {
            "gateway authentication failed. check the saved gateway token.",
            "gateway rejected the correction. shorten both phrases and try again.",
            "the gateway rejected a duplicate or conflicting correction.",
            "could not reach the gateway. check its connection and try again.",
            "the gateway could not save this correction. see the log for its error category.",
        }:
            return str(error)
        status_match = re.search(r"\bhttp\s+(\d{3})\b", detail[:120])
        if status_match:
            status = int(status_match.group(1))
            if status == 422:
                return "Gateway rejected the correction. Shorten both phrases and try again."
            if status in {401, 403}:
                return "Gateway authentication failed. Check the saved gateway token."
            if status == 409:
                return "The gateway rejected a duplicate or conflicting correction."
            return "The gateway could not save this correction. See the log for its error category."
        if detail.startswith(("validationerror", "valueerror", "validation failed")):
            return "Gateway rejected the correction. Shorten both phrases and try again."
        if detail.startswith(("authentication failed", "authorization failed")):
            return "Gateway authentication failed. Check the saved gateway token."
        if detail.startswith(("duplicate correction", "conflicting correction")):
            return "The gateway rejected a duplicate or conflicting correction."
        if detail.startswith(("could not reach", "connectionerror", "timeouterror")):
            return "Could not reach the gateway. Check its connection and try again."
        return "The gateway could not save this correction. See the log for its error category."

    def _error_status_text(self, error: object) -> str:
        safe = self._safe_error_summary(error)
        if not getattr(self, "_compact_layout", False):
            return safe
        compact = {
            "Gateway authentication failed. Check the saved gateway token.": (
                "Not saved · Check gateway token"
            ),
            "Gateway rejected the correction. Shorten both phrases and try again.": (
                "Not saved · Shorten both phrases"
            ),
            "The gateway rejected a duplicate or conflicting correction.": (
                "Not saved · Duplicate or conflict"
            ),
            "Could not reach the gateway. Check its connection and try again.": (
                "Not saved · Gateway unreachable"
            ),
        }
        return compact.get(safe, "Not saved · See CtrlSpeak log")

    def submit(self) -> None:
        if self._submitting or not self.is_open():
            return
        source = self.source_var.get().strip()
        replacement = self.replacement_var.get().strip()
        if not source:
            self._set_status("error", "Enter the phrase CtrlSpeak currently produces.")
            self.window.bell()
            self.source_entry.focus_set()
            return
        if not replacement:
            self._set_status("error", "Enter the phrase CtrlSpeak should return instead.")
            self.window.bell()
            self.replacement_entry.focus_set()
            return
        if source == replacement:
            self._set_status("error", "The replacement must differ from the phrase being corrected.")
            self.window.bell()
            self.replacement_entry.focus_set()
            return

        scope = "global" if self.global_scope_var.get() else "user"
        self._submitting = True
        self._submission_generation += 1
        request_generation = self._submission_generation
        request_config = self._config
        self._set_form_enabled(False)
        self._set_status("busy", "Submitting securely to the CtrlSpeak gateway…")

        def worker() -> None:
            error_category: str | None = None
            safe_error: str | None = None
            try:
                rule = ApiTranscriptionClient(
                    request_config,
                    timeout_seconds=20.0,
                ).create_correction(source, replacement, scope=scope)
            except Exception as exc:
                # Reduce untrusted HTTP details while the exception is in
                # scope. Logging outside this except block also prevents a
                # failing log handler from printing the raw exception context.
                error_category = exc.__class__.__name__
                safe_error = self._safe_error_summary(exc)
            else:
                enqueue_management_task(
                    self._finish_submission,
                    request_generation,
                    rule,
                    None,
                )
                return
            logger.warning(
                "CtrlSpeak correction submission failed category=%s",
                error_category or "unexpected",
            )
            enqueue_management_task(
                self._finish_submission,
                request_generation,
                None,
                safe_error or self._safe_error_summary(None),
            )

        start_failure: tuple[str, str] | None = None
        try:
            thread = threading.Thread(
                target=worker,
                name="ctrlspeak-correction-submit",
                daemon=True,
            )
            thread.start()
        except Exception as exc:
            start_failure = (exc.__class__.__name__, self._safe_error_summary(exc))
        if start_failure is not None:
            logger.error(
                "Failed to start correction submission worker category=%s",
                start_failure[0],
            )
            self._finish_submission(
                request_generation,
                None,
                start_failure[1],
            )

    def _finish_submission(
        self,
        request_generation: int,
        rule: dict[str, object] | None,
        error: str | None,
    ) -> None:
        if (
            not self.is_open()
            or request_generation != self._submission_generation
        ):
            return
        self._submitting = False
        self._set_form_enabled(True)
        pending_config = self._pending_config
        self._pending_config = None
        if error:
            logger.warning("CtrlSpeak correction submission did not complete")
            if pending_config is not None:
                self._apply_config(pending_config)
            self._set_status(
                "error",
                (
                    (
                        "Previous failed · New gateway ready"
                        if self._compact_layout
                        else "Previous gateway request failed · New gateway ready for the next submission."
                    )
                    if pending_config is not None
                    else self._error_status_text(error)
                ),
            )
            if self.is_visible():
                self.window.bell()
                self.replacement_entry.focus_set()
            return

        rule_id = str((rule or {}).get("id") or "")
        logger.info("Submitted CtrlSpeak correction rule id=%s", rule_id)
        self.source_var.set("")
        self.replacement_var.set("")
        self.global_scope_var.set(False)
        if pending_config is not None:
            self._apply_config(pending_config)
        self._set_status(
            "success",
            (
                (
                    "Saved previously · New gateway ready"
                    if self._compact_layout
                    else "Correction saved on the previous gateway · New gateway ready."
                )
                if pending_config is not None
                else "Correction saved and active on the gateway."
            ),
        )
        if self.is_visible():
            self.source_entry.focus_set()

    def hide(self) -> None:
        """Hide without cancelling or duplicating an in-flight submission."""

        if not self.is_open():
            return
        try:
            self.window.withdraw()
        except tk.TclError:
            logger.debug("Correction dialog was already unavailable while hiding")

    # Compatibility for callers that treated the former window as disposable.
    close = hide

class _LegacyManagementWindow:
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
            _set_window_icon(self.window)
        except Exception:
            logger.exception("Failed to set management window icon")

        body = ttk.Frame(self.window, style="Modern.TFrame")
        body.pack(fill=tk.BOTH, expand=True)

        self._scroll_canvas = tk.Canvas(
            body,
            highlightthickness=0,
            background=BACKGROUND,
        )
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._scrollbar = ttk.Scrollbar(body, orient=tk.VERTICAL, command=self._scroll_canvas.yview)
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas.configure(yscrollcommand=self._scrollbar.set)

        self._scroll_content = ttk.Frame(
            self._scroll_canvas,
            style="Modern.TFrame",
            padding=(30, 26),
        )
        self._scroll_window = self._scroll_canvas.create_window(
            (0, 0),
            window=self._scroll_content,
            anchor="nw",
        )

        self._scroll_content.bind("<Configure>", self._update_scroll_region)
        self._scroll_canvas.bind("<Configure>", self._sync_scroll_width)

        self._mousewheel_bound = False
        self._scroll_content.bind("<Enter>", self._bind_mousewheel)
        self._scroll_content.bind("<Leave>", self._unbind_mousewheel)

        container = self._scroll_content

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

        # Signed application updates
        self._build_update_card(container)

        # Transcription backend selection
        backend_card = ttk.Frame(container, style="ModernCard.TFrame", padding=(24, 22))
        backend_card.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(backend_card, text="Transcription backend", style="SectionHeading.TLabel").pack(anchor=tk.W)
        backend_accent = ttk.Frame(backend_card, style="AccentLine.TFrame")
        backend_accent.configure(height=2)
        backend_accent.pack(fill=tk.X, pady=(10, 12))
        active_backend = get_backend_config()
        self.backend_var = tk.StringVar(value=backend_display_name(active_backend.backend))
        self.api_url_var = tk.StringVar(value=active_backend.api_url)
        with settings_lock:
            saved_token = settings.get("api_token")
        self.api_token_var = tk.StringVar(value=str(saved_token) if saved_token else "")
        self.feedback_capture_var = tk.StringVar(value=active_backend.feedback_capture_method)
        self.provider_strategy_var = tk.StringVar(value=active_backend.provider_strategy)
        self.openai_api_key_var = tk.StringVar(value=get_session_openai_api_key() or "")
        self._secure_key_storage_available = secure_storage_available()
        self.remember_openai_key_var = tk.BooleanVar(
            value=self._secure_key_storage_available
        )

        backend_row = ttk.Frame(backend_card, style="ModernCardInner.TFrame")
        backend_row.pack(fill=tk.X, pady=(8, 6))
        ttk.Label(backend_row, text="Backend", style="Body.TLabel", width=18).pack(side=tk.LEFT)
        ttk.Combobox(
            backend_row,
            textvariable=self.backend_var,
            values=list(BACKEND_DISPLAY_NAMES.values()),
            state="readonly", width=24, style="Modern.TCombobox",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        language_frame = ttk.Frame(backend_card, style="ModernCardInner.TFrame")
        language_frame.pack(fill=tk.X, pady=(10, 4))
        ttk.Label(
            language_frame,
            text="Allowed output languages",
            style="Body.TLabel",
            width=18,
        ).pack(side=tk.LEFT, anchor=tk.N)
        language_list_frame = ttk.Frame(language_frame, style="ModernCardInner.TFrame")
        language_list_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        choices = language_choices()
        self._output_language_codes = tuple(code for code, _name in choices)
        self.output_language_list = tk.Listbox(
            language_list_frame,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            height=7,
            background=ELEVATED_SURFACE,
            foreground=TEXT_PRIMARY,
            selectbackground=ACCENT,
            selectforeground=BACKGROUND,
            highlightbackground=OUTLINE,
            highlightcolor=ACCENT,
            relief=tk.FLAT,
            borderwidth=1,
        )
        language_scroll = ttk.Scrollbar(
            language_list_frame,
            orient=tk.VERTICAL,
            command=self.output_language_list.yview,
        )
        self.output_language_list.configure(yscrollcommand=language_scroll.set)
        self.output_language_list.pack(side=tk.LEFT, fill=tk.X, expand=True)
        language_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        for code, name in choices:
            self.output_language_list.insert(tk.END, f"{name} ({code})")
        self._set_output_language_selection(active_backend.allowed_output_languages)

        language_buttons = ttk.Frame(backend_card, style="ModernCardInner.TFrame")
        language_buttons.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(
            language_buttons,
            text="English only",
            style="Subtle.TButton",
            command=lambda: self._set_output_language_selection(("en",)),
        ).pack(side=tk.LEFT)
        ttk.Button(
            language_buttons,
            text="Clear (automatic)",
            style="Subtle.TButton",
            command=lambda: self._set_output_language_selection(()),
        ).pack(side=tk.LEFT, padx=(12, 0))

        api_url_row = ttk.Frame(backend_card, style="ModernCardInner.TFrame")
        api_url_row.pack(fill=tk.X, pady=6)
        ttk.Label(api_url_row, text="API base URL", style="Body.TLabel", width=18).pack(side=tk.LEFT)
        ttk.Entry(api_url_row, textvariable=self.api_url_var).pack(side=tk.LEFT, fill=tk.X, expand=True)

        token_row = ttk.Frame(backend_card, style="ModernCardInner.TFrame")
        token_row.pack(fill=tk.X, pady=6)
        ttk.Label(token_row, text="Bearer token", style="Body.TLabel", width=18).pack(side=tk.LEFT)
        ttk.Entry(token_row, textvariable=self.api_token_var, show="•").pack(side=tk.LEFT, fill=tk.X, expand=True)

        strategy_row = ttk.Frame(backend_card, style="ModernCardInner.TFrame")
        strategy_row.pack(fill=tk.X, pady=6)
        ttk.Label(strategy_row, text="Gateway route", style="Body.TLabel", width=18).pack(side=tk.LEFT)
        self.provider_strategy_combo = ttk.Combobox(
            strategy_row,
            textvariable=self.provider_strategy_var,
            values=tuple(dict.fromkeys(("server-default", active_backend.provider_strategy))),
            state="readonly",
            width=24,
            style="Modern.TCombobox",
        )
        self.provider_strategy_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.capabilities_button = ttk.Button(
            strategy_row,
            text="Check gateway",
            style="Subtle.TButton",
            command=self._refresh_gateway_capabilities,
        )
        self.capabilities_button.pack(side=tk.LEFT, padx=(12, 0))

        openai_row = ttk.Frame(backend_card, style="ModernCardInner.TFrame")
        openai_row.pack(fill=tk.X, pady=6)
        ttk.Label(openai_row, text="OpenAI API key", style="Body.TLabel", width=18).pack(side=tk.LEFT)
        ttk.Entry(openai_row, textvariable=self.openai_api_key_var, show="*").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        self.forget_openai_key_button = ttk.Button(
            openai_row,
            text="Forget key",
            style="Subtle.TButton",
            command=self._forget_openai_key,
        )
        self.forget_openai_key_button.pack(side=tk.LEFT, padx=(12, 0))
        remember_key = ttk.Checkbutton(
            backend_card,
            text="Remember securely on this computer",
            variable=self.remember_openai_key_var,
        )
        remember_key.pack(anchor=tk.W, pady=(0, 4))
        if not self._secure_key_storage_available:
            remember_key.state(["disabled"])
        ttk.Label(
            backend_card,
            text=(
                "On Windows, the key is stored in your user-scoped Credential Manager "
                "vault when remembering is enabled. It is never written to settings.json "
                "or retained by the gateway, and is sent only for transcription requests."
                if self._secure_key_storage_available
                else "Secure native key storage is unavailable here; the OpenAI key "
                     "will remain in this CtrlSpeak process only."
            ),
            style="Caption.TLabel",
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        feedback_row = ttk.Frame(backend_card, style="ModernCardInner.TFrame")
        feedback_row.pack(fill=tk.X, pady=6)
        ttk.Label(feedback_row, text="Edit feedback", style="Body.TLabel", width=18).pack(side=tk.LEFT)
        ttk.Combobox(
            feedback_row,
            textvariable=self.feedback_capture_var,
            values=["active_field_on_enter", "disabled"],
            state="readonly",
            width=24,
            style="Modern.TCombobox",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(
            backend_card,
            text=("Select no languages for automatic detection. Select one to force it, or "
                  "select up to five; CtrlSpeak accepts automatic detection only within that "
                  "list and uses the first selected language as its safe fallback. "
                  "Automatic capture snapshots the active field when you press bare Enter, "
                  "then restores the clipboard and submits only changed text. Saving any "
                  "backend setting requires a CtrlSpeak restart. Environment variables "
                  "override saved API values."),
            style="Caption.TLabel",
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(8, 0))
        ttk.Button(
            backend_card, text="Save backend and languages", style="Accent.TButton", command=self._apply_backend,
        ).pack(anchor=tk.W, pady=(12, 0))
        self.backend_status_var = tk.StringVar(value=get_backend_status(active_backend))
        ttk.Label(
            backend_card, textvariable=self.backend_status_var, style="Caption.TLabel",
            wraplength=520, justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(10, 0))

        # Device preferences
        device_card = ttk.Frame(container, style="ModernCard.TFrame", padding=(24, 22))
        device_card.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(device_card, text="Device preference", style="SectionHeading.TLabel").pack(anchor=tk.W)
        device_accent = ttk.Frame(device_card, style="AccentLine.TFrame")
        device_accent.configure(height=2)
        device_accent.pack(fill=tk.X, pady=(10, 12))
        pref = get_device_preference()
        if pref not in {"cpu", "cuda"}:
            pref = "cpu"
        self.cuda_supported = cuda_driver_available()
        self.cuda_auto_install_supported = automatic_runtime_install_supported()
        if pref == "cuda" and not self.cuda_supported:
            try:
                set_device_preference("cpu")
            except Exception:
                logger.exception("Failed to reset device preference to CPU when CUDA is unavailable")
            pref = "cpu"
        self.device_var = tk.StringVar(value=pref)
        device_row = ttk.Frame(device_card, style="ModernCardInner.TFrame")
        device_row.pack(fill=tk.X, pady=(14, 10))
        ttk.Radiobutton(device_row, text="CPU", variable=self.device_var, value="cpu",
                        style="Modern.TRadiobutton").pack(side=tk.LEFT, padx=(0, 18))
        self.cuda_radio: Optional[ttk.Radiobutton] = None
        if self.cuda_supported:
            self.cuda_radio = ttk.Radiobutton(device_row, text="GPU (CUDA)", variable=self.device_var, value="cuda",
                                             style="Modern.TRadiobutton")
            self.cuda_radio.pack(side=tk.LEFT)
        self.cuda_status = tk.StringVar()
        if not self.cuda_supported:
            self.cuda_status.set("CUDA acceleration is unavailable on this system (no compatible GPU detected).")
        ttk.Label(device_card, textvariable=self.cuda_status, style="Caption.TLabel").pack(anchor=tk.W, pady=(4, 0))
        device_buttons = ttk.Frame(device_card, style="ModernCardInner.TFrame")
        device_buttons.pack(fill=tk.X, pady=(16, 0))
        self.apply_device_btn = ttk.Button(device_buttons, text="Apply device", style="Accent.TButton",
                                           command=self._apply_device)
        self.apply_device_btn.pack(side=tk.LEFT)
        cuda_button_text = (
            "Install or repair CUDA"
            if self.cuda_auto_install_supported
            else "Recheck system CUDA"
        )
        self.install_cuda_btn = ttk.Button(device_buttons, text=cuda_button_text, style="Subtle.TButton",
                                           command=self._install_cuda)
        self.install_cuda_btn.pack(side=tk.LEFT, padx=(12, 0))
        if not self.cuda_supported:
            try:
                self.install_cuda_btn.state(["disabled"])
            except Exception:
                self.install_cuda_btn.configure(state="disabled")

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

        # Input audio device selection
        audio_card = ttk.Frame(container, style="ModernCard.TFrame", padding=(24, 22))
        audio_card.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(audio_card, text="Input audio device", style="SectionHeading.TLabel").pack(anchor=tk.W)
        audio_accent = ttk.Frame(audio_card, style="AccentLine.TFrame")
        audio_accent.configure(height=2)
        audio_accent.pack(fill=tk.X, pady=(10, 12))
        audio_row = ttk.Frame(audio_card, style="ModernCardInner.TFrame")
        audio_row.pack(fill=tk.X, pady=(14, 12))
        self.audio_device_var = tk.StringVar()
        self.audio_device_combo = ttk.Combobox(
            audio_row,
            textvariable=self.audio_device_var,
            state="readonly",
            style="Modern.TCombobox",
            width=40,
        )
        self.audio_device_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            audio_row,
            text="Refresh list",
            style="Subtle.TButton",
            command=self._refresh_audio_devices,
        ).pack(side=tk.LEFT, padx=(12, 0))
        audio_buttons = ttk.Frame(audio_card, style="ModernCardInner.TFrame")
        audio_buttons.pack(fill=tk.X, pady=(4, 0))
        self.apply_audio_device_btn = ttk.Button(
            audio_buttons,
            text="Apply input device",
            style="Accent.TButton",
            command=self._apply_audio_device,
        )
        self.apply_audio_device_btn.pack(side=tk.LEFT)
        self.audio_device_status = tk.StringVar()
        ttk.Label(audio_card, textvariable=self.audio_device_status, style="Caption.TLabel").pack(anchor=tk.W, pady=(10, 0))
        self._audio_device_map: dict[str, Optional[str]] = {}
        self._refresh_audio_devices(initial=True)

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
        self.change_mode_button = ttk.Button(controls, text="Change mode", style="Subtle.TButton",
                                             command=self.change_mode)
        self.change_mode_button.pack(fill=tk.X, pady=4)
        if CLIENT_ONLY_BUILD:
            try:
                self.change_mode_button.state(["disabled"])
            except Exception:
                logger.exception("Failed to disable change mode button for client-only build")
        exit_label = "Exit CtrlSpeak" if CLIENT_ONLY_BUILD else "Stop everything"
        self.stop_all_button = ttk.Button(controls, text=exit_label, style="Danger.TButton",
                                          command=self.stop_everything)
        self.stop_all_button.pack(fill=tk.X, pady=(12, 4))

        # Footer actions
        footer = ttk.Frame(container, style="Modern.TFrame")
        footer.pack(fill=tk.X, pady=(18, 0))
        ttk.Button(footer, text="Close control center", style="Accent.TButton",
                   command=self.close).pack(fill=tk.X)
        uninstall_label = (
            "Delete CtrlSpeak"
            if sys.platform.startswith("win")
            else "Linux uninstall information"
        )
        ttk.Button(footer, text=uninstall_label, style="Danger.TButton",
                   command=self.delete_ctrlspeak).pack(fill=tk.X, pady=(10, 0))

        self.window.after(120, self.refresh_status)

        # --- Auto-size window to fit content once everything is laid out ---
        self.window.update_idletasks()
        req_w = max(self.window.winfo_reqwidth(), 580)
        self.window.minsize(req_w, 560)

        self.bring_to_front()

    def _build_update_card(self, container: ttk.Frame) -> None:
        update_card = ttk.Frame(container, style="ModernCard.TFrame", padding=(24, 22))
        update_card.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(update_card, text="Application updates", style="SectionHeading.TLabel").pack(anchor=tk.W)
        update_accent = ttk.Frame(update_card, style="AccentLine.TFrame")
        update_accent.configure(height=2)
        update_accent.pack(fill=tk.X, pady=(10, 12))

        self._update_runtime = classify_runtime()
        self._update_release = None
        self._update_transaction = None
        self._update_event = UpdateEvent(0, "idle", "Updates have not been checked in this session.")
        self._last_recorded_check_generation = -1
        self.update_status_var = tk.StringVar()
        self.update_last_checked_var = tk.StringVar()
        self.update_progress_var = tk.DoubleVar(value=0.0)

        ttk.Label(
            update_card,
            text=f"CtrlSpeak {APP_VERSION}  ·  Stable channel",
            style="Body.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            update_card,
            textvariable=self.update_status_var,
            style="Caption.TLabel",
            wraplength=520,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(10, 0))
        ttk.Label(
            update_card,
            textvariable=self.update_last_checked_var,
            style="Caption.TLabel",
        ).pack(anchor=tk.W, pady=(6, 0))

        self.update_progress = ttk.Progressbar(
            update_card,
            variable=self.update_progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.update_progress.pack(fill=tk.X, pady=(12, 0))
        self.update_progress_text_var = tk.StringVar(value="")
        ttk.Label(
            update_card,
            textvariable=self.update_progress_text_var,
            style="Caption.TLabel",
        ).pack(anchor=tk.W, pady=(4, 0))

        buttons = ttk.Frame(update_card, style="ModernCardInner.TFrame")
        buttons.pack(fill=tk.X, pady=(14, 0))
        self.check_update_btn = ttk.Button(
            buttons,
            text="Check for updates",
            style="Accent.TButton",
            command=self.check_for_updates,
        )
        self.check_update_btn.pack(side=tk.LEFT)
        self.install_update_btn = ttk.Button(
            buttons,
            text="Download and install",
            style="Subtle.TButton",
            command=self._download_or_install_update,
        )
        self.install_update_btn.pack(side=tk.LEFT, padx=(10, 0))
        self.cancel_update_btn = ttk.Button(
            buttons,
            text="Cancel",
            style="Subtle.TButton",
            command=self._cancel_update,
        )
        self.cancel_update_btn.pack(side=tk.LEFT, padx=(10, 0))

        secondary = ttk.Frame(update_card, style="ModernCardInner.TFrame")
        secondary.pack(fill=tk.X, pady=(10, 0))
        self.view_release_btn = ttk.Button(
            secondary,
            text="View release",
            style="Subtle.TButton",
            command=self._view_update_release,
        )
        self.view_release_btn.pack(side=tk.LEFT)
        self.copy_update_diagnostics_btn = ttk.Button(
            secondary,
            text="Copy diagnostics",
            style="Subtle.TButton",
            command=self._copy_update_diagnostics,
        )
        self.copy_update_diagnostics_btn.pack(side=tk.LEFT, padx=(10, 0))

        self._update_coordinator = get_update_coordinator(APP_VERSION)
        self._update_coordinator.add_listener(self._on_update_event)
        with settings_lock:
            last_checked = settings.get("last_update_check_at")
        self.update_last_checked_var.set(
            f"Last checked: {last_checked}" if last_checked else "Last checked: Never"
        )
        self._apply_update_event(self._update_coordinator.snapshot())

    @staticmethod
    def _set_button_enabled(button: ttk.Button, enabled: bool) -> None:
        try:
            button.state(["!disabled"] if enabled else ["disabled"])
        except Exception:
            button.configure(state="normal" if enabled else "disabled")

    @staticmethod
    def _format_update_bytes(value: int) -> str:
        if value >= 1024 * 1024:
            return f"{value / (1024 * 1024):.1f} MB"
        if value >= 1024:
            return f"{value / 1024:.1f} KB"
        return f"{value} bytes"

    def _on_update_event(self, event: UpdateEvent) -> None:
        _call_on_management_ui(
            lambda: self._apply_update_event(event),
            log_message="Failed to marshal update status to the management window",
        )

    def _apply_update_event(self, event: UpdateEvent) -> None:
        if not self.is_open():
            return
        self._update_event = event
        if event.release is not None:
            self._update_release = event.release
        if event.transaction is not None:
            self._update_transaction = event.transaction

        message = event.message
        if self._update_runtime == "source_checkout" and event.state in {
            "idle", "up_to_date", "available", "newer_than_release"
        }:
            message += " Source checkouts are updated through Git; binary installation is disabled."
        elif self._update_runtime == "packaged_manual_install_required" and event.state == "available":
            message += " This install location is not writable; open the release for manual installation."
        self.update_status_var.set(message)

        if event.total > 0:
            percentage = min(100.0, max(0.0, (event.downloaded / event.total) * 100.0))
            self.update_progress_var.set(percentage)
            self.update_progress_text_var.set(
                f"{self._format_update_bytes(event.downloaded)} of "
                f"{self._format_update_bytes(event.total)} ({percentage:.0f}%)"
            )
        else:
            self.update_progress_var.set(0.0)
            self.update_progress_text_var.set("")

        busy = event.state in {"checking", "downloading", "verifying", "launching_updater"}
        self._set_button_enabled(self.check_update_btn, not busy)
        self._set_button_enabled(self.cancel_update_btn, event.state == "downloading")
        self._set_button_enabled(self.view_release_btn, self._update_release is not None)

        install_eligible = self._update_runtime == "packaged_user_writable"
        if event.state == "ready_to_install" and install_eligible:
            self.install_update_btn.configure(text="Restart and install")
            self._set_button_enabled(self.install_update_btn, True)
        elif event.state in {"available", "failed"} and self._update_release is not None and install_eligible:
            label = "Retry download" if event.state == "failed" else "Download and install"
            self.install_update_btn.configure(text=label)
            self._set_button_enabled(self.install_update_btn, True)
        else:
            self.install_update_btn.configure(text="Download and install")
            self._set_button_enabled(self.install_update_btn, False)

        if (
            event.state in {"up_to_date", "available", "newer_than_release", "manual_install_required"}
            and event.generation != self._last_recorded_check_generation
        ):
            checked_at = utc_now_iso()
            self._last_recorded_check_generation = event.generation
            self.update_last_checked_var.set(f"Last checked: {checked_at}")
            with settings_lock:
                settings["last_update_check_at"] = checked_at
            if not save_settings():
                logger.error("Unable to persist the last update check timestamp")

    def check_for_updates(self) -> None:
        try:
            self._update_coordinator.check_async()
        except Exception:
            logger.exception("Unable to start update check")
            self.update_status_var.set("The update check could not start. See the CtrlSpeak log.")

    def _download_or_install_update(self) -> None:
        if self._update_event.state == "ready_to_install":
            self._restart_to_install_update()
            return
        release = self._update_release
        if release is None:
            return
        if self._update_runtime != "packaged_user_writable":
            messagebox.showinfo(
                "Manual update required",
                "This CtrlSpeak copy cannot install a binary update in place. Open the verified release instead.",
                parent=self.window,
            )
            return
        details = (
            f"Current version: {APP_VERSION}\n"
            f"New version: {release.version}\n"
            f"Artifact: {release.asset.name}\n"
            f"Download: approximately {self._format_update_bytes(release.asset.size)}\n\n"
            "CtrlSpeak will verify the signed download before it closes. Settings, models, "
            "CUDA files, and corrections will be retained. Continue?"
        )
        if not messagebox.askyesno(
            "Download CtrlSpeak update",
            details,
            parent=self.window,
            default=messagebox.NO,
            icon=messagebox.QUESTION,
        ):
            return
        try:
            self._update_coordinator.download_async(installation_path=Path(sys.executable))
        except UpdateError as exc:
            self.update_status_var.set(exc.user_message)
        except Exception:
            logger.exception("Unable to start update download")
            self.update_status_var.set("The update download could not start. See the CtrlSpeak log.")

    def _restart_to_install_update(self) -> None:
        transaction = self._update_transaction
        release = self._update_release
        if transaction is None or release is None:
            self.update_status_var.set("The verified update transaction is unavailable; check again.")
            return
        if sysmod.is_transcription_busy():
            messagebox.showwarning(
                "CtrlSpeak is busy",
                "Wait for recording and transcription to finish before installing the update.",
                parent=self.window,
            )
            return
        if not messagebox.askyesno(
            "Restart and install",
            f"CtrlSpeak will now close, install version {release.version}, and reopen. "
            "If startup health checks fail, the previous executable will be restored automatically. Continue?",
            parent=self.window,
            default=messagebox.NO,
            icon=messagebox.QUESTION,
        ):
            return
        try:
            helper_pid = prepare_update_handoff(transaction)
        except UpdateError as exc:
            logger.warning("Update helper launch failed [%s]: %s", exc.code, exc.user_message)
            self.update_status_var.set(exc.user_message)
            messagebox.showerror("Update could not start", exc.user_message, parent=self.window)
            return
        except Exception:
            logger.exception("Update helper launch failed unexpectedly")
            self.update_status_var.set("The update helper could not start. See the CtrlSpeak log.")
            messagebox.showerror(
                "Update could not start",
                "The update helper could not start. The current CtrlSpeak executable was not replaced.",
                parent=self.window,
            )
            return
        self.update_status_var.set(f"Update helper {helper_pid} started. CtrlSpeak is closing safely…")
        self._set_button_enabled(self.check_update_btn, False)
        self._set_button_enabled(self.install_update_btn, False)
        self.window.after(100, self._finish_update_shutdown)

    def _finish_update_shutdown(self) -> None:
        logger.info("Gracefully handing off to the external update helper")
        try:
            stop_client_listener()
            shutdown_server()
            sysmod.stop_discovery_listener()
            sysmod.release_single_instance_lock()
        except Exception:
            logger.exception("One or more services failed to stop during update handoff")
        try:
            self._icon.stop()
        except Exception:
            logger.exception("Failed to stop tray during update handoff")
        try:
            request_management_ui_shutdown()
        except Exception:
            logger.exception("Failed to stop management UI during update handoff")

    def _cancel_update(self) -> None:
        self._update_coordinator.cancel()

    def _view_update_release(self) -> None:
        release = self._update_release
        if release is None:
            return
        try:
            webbrowser.open_new_tab(release.release_url)
        except Exception:
            logger.exception("Unable to open the verified GitHub Release URL")
            messagebox.showerror(
                "Could not open release",
                "CtrlSpeak could not open the verified GitHub Release in your browser.",
                parent=self.window,
            )

    def _copy_update_diagnostics(self) -> None:
        event = self._update_event
        release = self._update_release
        transaction = self._update_transaction
        lines = [
            f"CtrlSpeak version: {APP_VERSION}",
            f"Runtime: {self._update_runtime}",
            f"State: {event.state}",
            f"Error category: {event.error_code or 'none'}",
            f"Message: {event.message}",
            f"Target version: {release.version if release else 'none'}",
            f"Release tag: {release.tag if release else 'none'}",
            f"Manifest SHA-256: {release.manifest_sha256 if release else 'none'}",
            f"Transaction: {transaction.transaction_id if transaction else 'none'}",
        ]
        try:
            self.window.clipboard_clear()
            self.window.clipboard_append("\n".join(lines))
            self.window.update_idletasks()
            self.update_status_var.set("Redacted update diagnostics copied to the clipboard.")
        except Exception:
            logger.exception("Unable to copy redacted update diagnostics")

    # --- window helpers ---
    def is_open(self) -> bool:
        return bool(self.window and self.window.winfo_exists())

    def bring_to_front(self) -> None:
        if not self.is_open(): return
        self.window.deiconify(); self.window.lift(); self.window.focus_force()
        self.window.attributes("-topmost", True)
        self.window.after(150, lambda: self.window.attributes("-topmost", False))

    def _update_scroll_region(self, _event: Optional[tk.Event] = None) -> None:
        if not self.is_open():
            return
        try:
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))
        except Exception:
            logger.exception("Failed to update management window scroll region")

    def _sync_scroll_width(self, event: tk.Event) -> None:
        if not self.is_open():
            return
        try:
            self._scroll_canvas.itemconfigure(self._scroll_window, width=event.width)
        except Exception:
            logger.exception("Failed to sync management window scroll width")

    def _on_mousewheel(self, event: tk.Event) -> None:
        if not self.is_open():
            return

        delta = 0
        if hasattr(event, "delta") and event.delta:
            delta = int(event.delta)
        elif getattr(event, "num", None) in (4, 5):
            delta = 120 if event.num == 4 else -120

        if delta == 0:
            return

        direction = -1 if delta > 0 else 1
        self._scroll_canvas.yview_scroll(direction, "units")

    def _bind_mousewheel(self, _event: tk.Event) -> None:
        if self._mousewheel_bound:
            return
        widget = self.window
        widget.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        widget.bind_all("<Button-4>", self._on_mousewheel, add="+")
        widget.bind_all("<Button-5>", self._on_mousewheel, add="+")
        self._mousewheel_bound = True

    def _unbind_mousewheel(self, _event: Optional[tk.Event]) -> None:
        if not self._mousewheel_bound:
            return
        widget = self.window
        widget.unbind_all("<MouseWheel>")
        widget.unbind_all("<Button-4>")
        widget.unbind_all("<Button-5>")
        self._mousewheel_bound = False

    # --- state refresh ---
    def refresh_status(self) -> None:
        with settings_lock:
            mode = settings.get("mode") or "unknown"
        device_pref = get_device_preference()
        if device_pref not in {"cpu", "cuda"}:
            device_pref = "cpu"
        if device_pref == "cuda" and not self.cuda_supported:
            device_pref = "cpu"
            try:
                set_device_preference("cpu")
            except Exception:
                logger.exception("Failed to persist CPU preference after CUDA became unavailable")
        self.device_var.set(device_pref)
        has_cuda_files = cuda_runtime_files_present() if self.cuda_supported else False
        cuda_ready = False
        if self.cuda_supported and (
            not self.cuda_auto_install_supported
            or (device_pref == "cuda" and has_cuda_files)
        ):
            cuda_ready = cuda_runtime_ready(ignore_preference=True, quiet=True)
        model_name = get_current_model_name()
        backend_config = get_runtime_backend_config()
        if backend_config.backend == "api":
            self.mode_badge.configure(text="BACKEND · API", style="PillAccent.TLabel")
            network_label = f"API configured: {backend_config.api_url}"
        else:
            self.mode_badge.configure(text=f"MODE · {mode.upper()}", style="PillAccent.TLabel")
            network_label = describe_server_status()
        if backend_config.backend == "api":
            badge_style = "PillMuted.TLabel"
            badge_text = "API · CONFIGURED"
        elif "Not connected" in network_label:
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
        mode_status = f"{mode}" if backend_config.backend == "bundled" else "not used by API backend"
        status_parts = [
            f"\u2022 Mode: {mode_status}",
            f"\u2022 Client: {'active' if sysmod.client_enabled else 'stopped'}",
            f"\u2022 Server thread: {'running' if sysmod.server_thread and sysmod.server_thread.is_alive() else 'not running'}",
            f"\u2022 Device: {device_pref}",
            f"\u2022 Backend: {backend_config.backend}",
        ]
        self.status_var.set("\n".join(status_parts))
        self.server_status_var.set(f"Network: {network_label}")
        configured_backend = get_backend_config()
        if configured_backend != backend_config:
            self.backend_status_var.set(
                f"Active until restart: {get_backend_status(backend_config)}\n"
                f"Saved for next launch: {get_backend_status(configured_backend)}"
            )
        else:
            self.backend_status_var.set(get_backend_status(backend_config))
        if not self.cuda_supported:
            cuda_text = "CUDA acceleration is unavailable on this system (no compatible GPU detected)."
        elif not self.cuda_auto_install_supported and cuda_ready:
            cuda_text = (
                "System CUDA runtime is available and active."
                if device_pref == "cuda"
                else "System CUDA runtime is available. Switch to GPU to enable it."
            )
        elif not self.cuda_auto_install_supported:
            cuda_text = (
                "System CUDA runtime is not ready. Install a compatible NVIDIA driver, "
                "CUDA/cuDNN libraries, and a CUDA-enabled CTranslate2 build, then recheck."
            )
        elif device_pref == "cuda":
            cuda_text = ("CUDA runtime active." if cuda_ready
                         else "CUDA runtime not ready; using CPU instead.")
        elif has_cuda_files:
            cuda_text = "CUDA runtime staged. Switch to GPU to enable it."
        else:
            cuda_text = "CUDA runtime not staged."
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

    def _apply_backend(self) -> None:
        try:
            backend = backend_from_display_name(self.backend_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid backend settings", str(exc), parent=self.window)
            return
        api_url = self.api_url_var.get().strip()
        token = self.api_token_var.get()
        capture_method = self.feedback_capture_var.get().strip()
        allowed_output_languages = self._selected_output_languages()
        provider_strategy = self.provider_strategy_var.get().strip() or "server-default"

        try:
            saved = save_backend_config(
                backend=backend,
                api_url=api_url,
                api_token=token,
                feedback_capture_method=capture_method,
                allowed_output_languages=allowed_output_languages,
                provider_strategy=provider_strategy,
            )
        except ValueError as exc:
            messagebox.showerror("Invalid backend settings", str(exc), parent=self.window)
            return
        except BackendPersistenceError as exc:
            messagebox.showerror("Backend settings not saved", str(exc), parent=self.window)
            return

        try:
            persist_openai_api_key(
                self.openai_api_key_var.get(),
                remember=self.remember_openai_key_var.get(),
            )
        except CredentialStorageError as exc:
            messagebox.showerror("OpenAI key not saved", str(exc), parent=self.window)
            return

        active = get_runtime_backend_config()
        effective = get_backend_config()
        self.refresh_status()
        restart_required = effective != active
        if restart_required:
            self.backend_status_var.set(
                f"Active until restart: {get_backend_status(active)}\n"
                f"Saved for next launch: {get_backend_status(effective)}"
            )
            message = "Backend settings saved. Restart CtrlSpeak to apply them."
        else:
            self.backend_status_var.set(get_backend_status(active))
            message = "Backend settings are saved and already match the active runtime."
        if effective != saved:
            message += " Environment variables currently override one or more saved values."
        messagebox.showinfo("Backend settings saved", message, parent=self.window)

    def _forget_openai_key(self) -> None:
        if not messagebox.askyesno(
            "Forget OpenAI API key",
            "Remove the saved OpenAI key from this computer and from the running session?",
            parent=self.window,
        ):
            return
        try:
            forget_openai_api_key()
        except CredentialStorageError as exc:
            messagebox.showerror("OpenAI key not removed", str(exc), parent=self.window)
            return
        self.openai_api_key_var.set("")
        self.remember_openai_key_var.set(False)
        messagebox.showinfo(
            "OpenAI key removed",
            "The OpenAI key is no longer stored or active in CtrlSpeak.",
            parent=self.window,
        )

    def _refresh_gateway_capabilities(self) -> None:
        api_url = self.api_url_var.get().strip()
        api_token = self.api_token_var.get().strip() or None
        probe_config = BackendConfig("api", api_url, api_token, "disabled")
        self.capabilities_button.state(["disabled"])

        def finish_error(message: str) -> None:
            self.capabilities_button.state(["!disabled"])
            messagebox.showerror("Gateway check failed", message, parent=self.window)

        def finish_success(capabilities: dict[str, object]) -> None:
            self.capabilities_button.state(["!disabled"])
            raw_strategies = capabilities.get("strategies")
            strategy_ids = [
                str(item["id"])
                for item in raw_strategies
                if isinstance(item, dict) and item.get("id")
            ] if isinstance(raw_strategies, list) else []
            values = tuple(dict.fromkeys(["server-default", *strategy_ids]))
            self.provider_strategy_combo.configure(values=values)
            if self.provider_strategy_var.get() not in values:
                self.provider_strategy_var.set("server-default")
            raw_providers = capabilities.get("providers")
            provider_lines = [
                f"{item.get('id')}: {item.get('status')}"
                for item in raw_providers
                if isinstance(item, dict)
            ] if isinstance(raw_providers, list) else []
            messagebox.showinfo(
                "CtrlSpeak gateway ready",
                f"Gateway version: {capabilities.get('version', 'unknown')}\n\n"
                + "\n".join(provider_lines),
                parent=self.window,
            )

        def probe() -> None:
            try:
                capabilities = ApiTranscriptionClient(
                    probe_config,
                    timeout_seconds=10.0,
                ).get_capabilities()
            except (ApiBackendError, ValueError) as exc:
                error_message = str(exc)
                self.window.after(0, lambda: finish_error(error_message))
                return
            self.window.after(0, lambda: finish_success(capabilities))

        threading.Thread(
            target=probe,
            name="ctrlspeak-capability-probe",
            daemon=True,
        ).start()

    def _selected_output_languages(self) -> tuple[str, ...]:
        return tuple(
            self._output_language_codes[int(index)]
            for index in self.output_language_list.curselection()
        )

    def _set_output_language_selection(self, codes: tuple[str, ...]) -> None:
        selected = set(codes)
        self.output_language_list.selection_clear(0, tk.END)
        for index, code in enumerate(self._output_language_codes):
            if code in selected:
                self.output_language_list.selection_set(index)

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

    def _refresh_audio_devices(self, initial: bool = False) -> None:
        entries: list[tuple[str, Optional[str]]] = [("System default (OS managed)", None)]
        try:
            for name, label in list_input_audio_devices():
                display = label or name
                entries.append((display, name))
        except Exception:
            logger.exception("Failed to refresh audio device list")

        if not entries:
            entries = [("System default (OS managed)", None)]

        self._audio_device_map = {display: value for display, value in entries}
        self.audio_device_combo.configure(values=[display for display, _ in entries])

        preferred = get_input_device_preference()
        selected_display = entries[0][0]
        found = False
        if preferred:
            for display, value in entries:
                if value == preferred:
                    selected_display = display
                    found = True
                    break
        self.audio_device_var.set(selected_display)

        if not any(value is not None for _, value in entries[1:]):
            message = "No microphone devices detected."
        elif preferred and found:
            message = f"Preferred device: {selected_display}"
        elif preferred and not found:
            message = "Preferred device not found. Using system default input device."
        else:
            message = "Using system default input device."

        if initial or not self.audio_device_status.get() or not preferred:
            self.audio_device_status.set(message)
        elif not initial:
            self.audio_device_status.set(message)

    # --- device actions ---
    def _apply_audio_device(self) -> None:
        try:
            selection = self.audio_device_var.get()
        except Exception:
            selection = ""
        device_name = self._audio_device_map.get(selection)
        try:
            set_input_device_preference(device_name)
        except Exception:
            logger.exception("Failed to save preferred audio input device")
            self.audio_device_status.set("Unable to save audio device preference. See logs for details.")
            return

        if device_name:
            self.audio_device_status.set(f"Input device preference saved: {selection}")
        else:
            self.audio_device_status.set("Using system default input device.")

    # --- device actions ---
    def _apply_device(self):
        from utils.models import set_device_preference

        requested = self.device_var.get()
        if requested != "cuda":
            requested = "cpu"

        if requested == "cuda":
            if not self.cuda_supported:
                self.device_var.set("cpu")
                self.cuda_status.set("CUDA acceleration is unavailable on this system (no compatible GPU detected).")
                try:
                    messagebox.showinfo(
                        "CUDA",
                        "This system does not have a CUDA-capable GPU. Staying on CPU instead.",
                        parent=self.window,
                    )
                except Exception:
                    logger.exception("Failed to present CUDA unavailable message box")
                return

            if not self.cuda_auto_install_supported:
                if not cuda_runtime_ready(ignore_preference=True, quiet=True):
                    self.device_var.set("cpu")
                    set_device_preference("cpu")
                    self.cuda_status.set(
                        "System CUDA runtime is not ready; using CPU. Install compatible "
                        "NVIDIA/CUDA/cuDNN and CTranslate2 dependencies, then recheck."
                    )
                    messagebox.showwarning(
                        "CUDA",
                        "CtrlSpeak found an NVIDIA GPU, but CTranslate2 cannot use the system "
                        "CUDA runtime. Linux runtime/driver installation is an operator step; "
                        "see packaging/BUILDING.md.",
                        parent=self.window,
                    )
                    return
                set_device_preference("cuda")
                self._reload_transcriber_async(
                    progress_message="Applying device preference...",
                    status_var=self.cuda_status,
                    notify_context="Device setup failed",
                )
                return

            staged = cuda_runtime_files_present()
            if not staged:
                staged = ensure_cuda_runtime_from_existing()

            if staged and not cuda_runtime_ready(ignore_preference=True, quiet=True):
                logger.warning("CUDA runtime files failed validation; attempting reinstall.")
                staged = False

            if not staged:
                if not install_cuda_runtime_with_progress(self.window):
                    logger.warning("CUDA runtime preparation failed during download attempt; staying on CPU.")
                    self.device_var.set("cpu")
                    set_device_preference("cpu")
                    self.cuda_status.set("CUDA runtime not ready; using CPU instead.")
                    self._reload_transcriber_async(
                        progress_message="Applying device preference...",
                        status_var=self.cuda_status,
                        notify_context="Device setup failed",
                    )
                    return
                staged = cuda_runtime_files_present()

            if not staged or not cuda_runtime_ready(ignore_preference=True, quiet=True):
                logger.warning("CUDA runtime is still unavailable after installation; staying on CPU.")
                self.device_var.set("cpu")
                set_device_preference("cpu")
                self.cuda_status.set("CUDA runtime not ready; using CPU instead.")
                self._reload_transcriber_async(
                    progress_message="Applying device preference...",
                    status_var=self.cuda_status,
                    notify_context="Device setup failed",
                )
                return

        set_device_preference(requested)

        # Reload the model with the new device without blocking the UI
        self._reload_transcriber_async(
            progress_message="Applying device preference...",
            status_var=self.cuda_status,
            notify_context="Device setup failed",
        )


    def _install_cuda(self):
        if not self.cuda_supported:
            try:
                messagebox.showinfo("CUDA", "This system does not have a CUDA-capable GPU.", parent=self.window)
            except Exception:
                logger.exception("Failed to present CUDA unavailable message box")
            return
        if not self.cuda_auto_install_supported:
            if cuda_runtime_ready(ignore_preference=True, quiet=True):
                messagebox.showinfo(
                    "CUDA",
                    "The system CUDA runtime is available to CTranslate2.",
                    parent=self.window,
                )
            else:
                messagebox.showwarning(
                    "CUDA",
                    "The Linux CUDA runtime is not ready. CtrlSpeak does not install system "
                    "GPU drivers or libraries; see packaging/BUILDING.md for prerequisites.",
                    parent=self.window,
                )
        elif ensure_cuda_runtime_from_existing():
            messagebox.showinfo("CUDA", "Reused existing CUDA runtime for CtrlSpeak.", parent=self.window)
        elif install_cuda_runtime_with_progress(self.window):
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
            if not download_model_with_gui(name, block_during_download=True):
                messagebox.showwarning(
                    "Model",
                    "Model download was cancelled before activation could continue.",
                    parent=self.window,
                )
                self.refresh_status()
                return

        def on_success() -> None:
            messagebox.showinfo("Model", f"Active model set to {name}.", parent=self.window)

        self._reload_transcriber_async(
            progress_message="Loading model…",
            status_var=self.model_status,
            notify_context="Model load failed",
            success_callback=on_success,
        )

    def _download_model(self):
        name = self.model_var.get()
        if name not in {"small", "large-v3"}:
            name = "large-v3"
        if download_model_with_gui(
            name,
            block_during_download=True,
        ):
            messagebox.showinfo(
                "Model",
                "Model downloaded successfully.",
                parent=self.window,
            )
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

    def change_mode(self) -> None:
        if CLIENT_ONLY_BUILD:
            messagebox.showinfo(
                "Mode",
                "This build only supports the client mode.",
                parent=self.window,
            )
            return

        if not self.is_open():
            return

        try:
            self.change_mode_button.state(["disabled"])
        except Exception:
            logger.exception("Failed to disable change mode button before dialog")

        with settings_lock:
            current_mode = settings.get("mode")

        try:
            choice = prompt_initial_mode(self.window)
        finally:
            try:
                self.change_mode_button.state(["!disabled"])
            except Exception:
                logger.exception("Failed to re-enable change mode button after dialog")

        if not choice or choice == current_mode:
            return

        if choice == "client_server":
            if not ensure_model_ready_for_local_server():
                messagebox.showwarning(
                    "Mode not changed",
                    "CtrlSpeak could not prepare the local transcription engine.",
                    parent=self.window,
                )
                return
            start_server()
            if not (sysmod.server_thread and sysmod.server_thread.is_alive()):
                messagebox.showerror(
                    "Mode not changed",
                    "CtrlSpeak could not start the local server.",
                    parent=self.window,
                )
                return
        else:
            shutdown_server()
            try:
                manual_discovery_refresh()
            except Exception:
                logger.exception("Failed to refresh discovery after switching to client mode")

        with settings_lock:
            settings["mode"] = choice
        save_settings()

        try:
            self._icon.title = f"CtrlSpeak {APP_VERSION} ({choice})"
        except Exception:
            logger.exception("Failed to update tray icon title after mode change")

        self.refresh_status()

        mode_label = "Client + Server" if choice == "client_server" else "Client Only"
        messagebox.showinfo(
            "Mode updated",
            f"CtrlSpeak will now operate in {mode_label} mode.",
            parent=self.window,
        )

    def stop_everything(self) -> None:
        stop_client_listener(); shutdown_server()
        self.refresh_status()
        self.window.after(200, self._icon.stop)
        self.close()

    def delete_ctrlspeak(self) -> None:
        if not sys.platform.startswith("win"):
            messagebox.showinfo(
                "Linux uninstall",
                "CtrlSpeak does not remove manually installed Linux files. Remove the "
                "executable, desktop entry, icon, and XDG CtrlSpeak data explicitly; see "
                "packaging/BUILDING.md.",
                parent=self.window,
            )
            return
        if not messagebox.askyesno("Delete CtrlSpeak",
                                   "This will remove CtrlSpeak and all local data. Continue?",
                                   parent=self.window):
            return
        self.window.after(100, lambda: initiate_self_uninstall(self._icon))

    def close(self) -> None:
        global management_window
        coordinator = getattr(self, "_update_coordinator", None)
        if coordinator is not None:
            try:
                coordinator.remove_listener(self._on_update_event)
            except Exception:
                logger.exception("Failed to detach management update listener")
        if self.is_open():
            self._unbind_mousewheel(None)
            self.window.destroy()
        management_window = None


# The v0.7 shell retains the mature controllers above behind a hidden rollback
# window and replaces every user-visible surface with Midnight Signal.
from utils.midnight_signal_ui import MidnightSignalManagementMixin


class ManagementWindow(MidnightSignalManagementMixin, _LegacyManagementWindow):
    pass
