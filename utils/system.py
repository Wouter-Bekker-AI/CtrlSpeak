# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime

import atexit
import argparse
from array import array
from contextlib import contextmanager
import http.client
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Queue
from typing import Dict, Optional, Tuple, Callable, List, TYPE_CHECKING
import subprocess
import uuid
import traceback

import numpy as np
import pyaudio
from PIL import Image
import tkinter as tk
from collections import deque

from utils.config_paths import (
    settings, settings_lock, load_settings, save_settings,
    get_config_dir, get_config_file_path, get_temp_dir,
    create_recording_file_path, cleanup_recording_file, cleanup_stale_recordings, resource_path,
    asset_path, app_icon_path, get_logger, get_logs_dir,
)
from utils.hotkeys import (
    DesktopSessionError,
    create_global_listener,
    is_right_control,
    key_name as hotkey_key_name,
)
from utils.version import APP_VERSION
from utils.audio_cues import CueKind, CuePlayer, Pcm16Cue, cue_to_wav_bytes
from utils.ui_state import TranscriptionUiSession, UiPhase


logger = get_logger(__name__)


_TLS_CA_ENVIRONMENT_VARIABLES = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")


def _is_pyinstaller_temporary_path(value: object) -> bool:
    """Recognize a CA path inherited from a one-file PyInstaller extraction."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return any(part.upper().startswith("_MEI") for part in Path(value).parts)
    except (OSError, TypeError, ValueError):
        return False


def _should_replace_runtime_ca(current: str | None, bundled: Path) -> bool:
    if not current:
        return True
    try:
        current_path = Path(current)
        if current_path.resolve() == bundled.resolve():
            return False
        if _is_pyinstaller_temporary_path(current):
            return True
        return not current_path.exists()
    except (OSError, TypeError, ValueError):
        return True


def _bootstrap_runtime_environment() -> None:
    """
    Ensure third-party services can establish HTTPS connections when running
    from a PyInstaller bundle by pointing to the embedded certifi bundle and by
    keeping all Hugging Face caches inside the CtrlSpeak config directory.
    """
    try:
        cfg = get_config_dir()
        hf_root = cfg / "hf-cache"
        hf_root.mkdir(parents=True, exist_ok=True)
        (hf_root / "hub").mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_root))
        os.environ.setdefault("HF_HUB_CACHE", str(hf_root / "hub"))
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    except Exception:
        logger.exception("Failed to prepare Hugging Face cache directories")

    try:
        import certifi  # type: ignore
    except Exception:
        logger.warning("certifi is unavailable; HTTPS certificate bundle not configured")
        return

    cert_path = Path(certifi.where())
    if not cert_path.exists():
        return

    for env_name in _TLS_CA_ENVIRONMENT_VARIABLES:
        current = os.environ.get(env_name)
        if _should_replace_runtime_ca(current, cert_path):
            os.environ[env_name] = str(cert_path)


_bootstrap_runtime_environment()

# recent mono samples of loading.wav for GUI wiggle
_proc_vis_lock = threading.Lock()
_proc_vis_buffers = deque()
_proc_vis_samples = 0
_PROC_VIS_MAX_SAMPLES = 4096  # ~0.1–0.25s depending on rate


def get_processing_waveform(n: int = 512) -> np.ndarray:
    """Return the most recent mono samples of loading.wav in [-1,1]."""
    with _proc_vis_lock:
        if not _proc_vis_buffers:
            return np.zeros(n, dtype=np.float32)
        data = np.concatenate(list(_proc_vis_buffers)) if len(_proc_vis_buffers) > 1 else _proc_vis_buffers[0]
    if data.size <= n:
        out = np.zeros(n, dtype=np.float32); out[-data.size:] = data; return out
    return data[-n:]


# ---------------- Public constants ----------------
SPLASH_DURATION_MS = 1000
ERROR_LOG_FILENAME = "CtrlSpeak-error.log"
LOCK_FILENAME = "CtrlSpeak.lock"
PROCESSING_SAMPLE_RATE = 44100
PROCESSING_SOUND_MAX_PEAK = 0.5  # -6.02 dBFS; keeps the processing loop headphone-safe

INSTANCE_PORT = int(os.environ.get("CTRLSPEAK_SINGLE_INSTANCE_PORT", "54329"))

# ---------------- Build flags ----------------
def detect_client_only_build() -> bool:
    try:
        base_dir = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
        if (base_dir / 'client_only.flag').exists():
            return True
    except Exception:
        logger.exception("Failed to detect client-only build flag")
    return os.environ.get('CTRLSPEAK_CLIENT_ONLY', '0') == '1'

CLIENT_ONLY_BUILD = detect_client_only_build()

# ---------------- Globals ----------------
recording = False
recording_thread: Optional[threading.Thread] = None
listener: Optional[object] = None
recording_file_path: Optional[Path] = None
listener_lock = threading.Lock()
_client_state_lock = threading.RLock()
# Lock ordering, whenever locks must be nested:
#   _client_state_lock -> listener_lock or _transcription_state_lock
# Shutdown releases the client/listener locks before it waits for lifecycle
# workers, so lifecycle code never needs to acquire the client-state lock.
client_enabled = True

instance_lock_handle: Optional[object] = None
processing_sound_thread: Optional[threading.Thread] = None
processing_sound_stop_event = threading.Event()
processing_sound_data: Optional[bytes] = None
processing_sound_settings: Optional[Dict[str, int]] = None
_ready_sound_lock = threading.Lock()
_ready_sound_played = False
# PyAudio calls PortAudio's process-global initialize/terminate functions for
# every ``PyAudio`` instance.  Those lifecycle calls are not safe to overlap:
# the Windows extension can access-violate when microphone capture and a cue or
# device scan initialize concurrently.  Hold this lock for the complete life
# of every remaining PyAudio instance, not merely around its constructor.
_pyaudio_session_lock = threading.Lock()
# Optional PortAudio work and capture publication share this small arbitration
# lock.  Once capture publishes its marker, no later optional operation can
# cross the marker check and construct PyAudio first.  Non-Windows cues never
# use this path; they run through an out-of-process native audio helper.
_pyaudio_priority_lock = threading.Lock()
_pyaudio_capture_pending = threading.Event()
# Device enumeration is optional UI work.  A refresh attempted during capture
# returns this last successful snapshot (or the saved preference) rather than
# claiming that the machine suddenly has no microphones.
_input_device_cache_lock = threading.Lock()
_input_device_cache: tuple[tuple[str, str], ...] = ()
transcription_ui_session = TranscriptionUiSession()
transcription_thread: Optional[threading.Thread] = None
transcription_cancel_event = threading.Event()
_transcription_state_lock = threading.RLock()
_recording_failed_event = threading.Event()
_recording_stop_event = threading.Event()
_transcription_generation = 0
_active_transcription_generation: Optional[int] = None

recording_temp_dir_name = "temp"
AUTO_MODE = False
AUTO_MODE_PROFILE: Optional[str] = None

# Tk management thread (GUI owns windows; system just queues tasks)
management_ui_thread: Optional[threading.Thread] = None
management_ui_queue: "Queue[tuple[Callable[..., None], tuple, dict]]" = Queue()
tk_root: Optional[tk.Tk] = None
management_window: Optional["ManagementWindow"] = None  # created in utils.gui

if TYPE_CHECKING:
    import pystray
    from utils.gui import ManagementWindow

# ---------------- IMPORTS from new split modules (and re-exports) ----------------

# Win32 text insertion / clipboard
from utils.winio import (
    insert_text_into_focus, set_force_sendinput, is_console_window,
    get_clipboard_text, set_clipboard_text, snapshot_active_text_field,
)

# LAN discovery (single source of truth for ServerInfo)
from utils.net_discovery import (
    ServerInfo, DiscoveryListener,
    get_preferred_server_settings, set_preferred_server, clear_preferred_server,
    parse_server_target, probe_server, register_manual_server,
    ensure_preferred_server_registered, send_discovery_query, get_discovery_port,
    manual_discovery_refresh as _nd_manual_discovery_refresh,  # we'll wrap this
    get_best_server as _nd_get_best_server,
    get_advertised_host_ip,
    manage_discovery_broadcast, listen_for_discovery_queries,
)

# Re-export for callers that import from utils.system
__all__ = [
    "APP_VERSION", "SPLASH_DURATION_MS", "CLIENT_ONLY_BUILD",
    "settings", "settings_lock", "load_settings", "save_settings",
    "get_config_dir", "get_config_file_path", "get_temp_dir",
    "create_recording_file_path", "cleanup_recording_file", "resource_path",
    "insert_text_into_focus", "set_force_sendinput", "is_console_window",
    "ServerInfo", "format_exception_details",
]

# Keep a single shared discovery listener and last_connected_server here
discovery_listener: Optional[DiscoveryListener] = None
last_connected_server: Optional[ServerInfo] = None

# Observer-only edit feedback hooks. The coordinator is lazy so importing the
# headless system module never performs clipboard or network work.
_feedback_coordinator = None
_feedback_modifiers: set[str] = set()
_feedback_capture_in_progress = False
_last_transcript_lock = threading.Lock()
_last_transcript: Optional[str] = None
_tray_icon: Optional[object] = None


def _capture_active_feedback_field() -> str | None:
    global _feedback_capture_in_progress
    _feedback_capture_in_progress = True
    try:
        return snapshot_active_text_field()
    finally:
        _feedback_capture_in_progress = False


def _get_feedback_coordinator():
    global _feedback_coordinator
    if _feedback_coordinator is None:
        from utils.feedback_capture import ActiveFieldSnapshotProvider, FeedbackCaptureCoordinator

        _feedback_coordinator = FeedbackCaptureCoordinator(
            snapshot_provider=ActiveFieldSnapshotProvider(_capture_active_feedback_field),
            submit_feedback=_submit_confirmed_feedback,
        )
    return _feedback_coordinator


def _submit_confirmed_feedback(
    transcription_id: str,
    final_text: str,
    capture_method: str,
    metadata: dict[str, object],
    feedback_target,
) -> None:
    from utils.transcription_backend import ApiBackendError, ApiTranscriptionClient, BackendConfig

    client_metadata = {"client": "CtrlSpeak", "version": APP_VERSION, **metadata}
    if feedback_target.backend == "bundled":
        from utils.local_corrections import get_local_correction_library

        try:
            approval = get_local_correction_library().approve_exact_override(
                transcription_id,
                confirmed_text=final_text,
                capture_method=capture_method,
                client_metadata=client_metadata,
            )
            if approval is None:
                notify(
                    "Confirmed edit feedback could not be matched to its local transcription.",
                    title="CtrlSpeak Feedback",
                )
        except Exception as exc:
            logger.exception("Failed to save confirmed edit feedback locally")
            notify(
                f"Confirmed edit feedback could not be saved locally. {exc}",
                title="CtrlSpeak Feedback",
            )
        return

    if feedback_target.backend != "api" or not feedback_target.api_url:
        logger.error("Pending feedback has an invalid backend target: %r", feedback_target.backend)
        return
    config = BackendConfig(
        backend="api",
        api_url=feedback_target.api_url,
        api_token=feedback_target.api_token,
        feedback_capture_method=capture_method,
    )
    try:
        ApiTranscriptionClient(config).submit_feedback(
            transcription_id,
            final_text=final_text,
            capture_method=capture_method,
            client_metadata=client_metadata,
        )
    except ApiBackendError as exc:
        notify(
            f"Confirmed edit feedback could not be sent to {feedback_target.api_url}. {exc}",
            title="CtrlSpeak Feedback",
        )


def observe_feedback_key_event(
    key_name: str,
    action: str,
    modifiers: frozenset[str] = frozenset(),
) -> None:
    """Observe but never consume, replay, or synthesize a key event."""
    from utils.feedback_capture import KeyEvent

    _get_feedback_coordinator().handle_key_event(KeyEvent(key_name, action, modifiers))


def track_feedback_injection(result) -> None:
    from utils.transcription_backend import get_runtime_backend_config

    _get_feedback_coordinator().track_injection(
        result,
        capture_method=get_runtime_backend_config().feedback_capture_method,
    )


def inject_transcription_result(result) -> None:
    """Retain the result, insert once, then make a successful injection eligible for feedback."""
    remember_last_transcript(result.text)
    insert_text_into_focus(result.text)
    track_feedback_injection(result)


def _pynput_key_name(key) -> str:
    return hotkey_key_name(key)


def _observe_pynput_press(key) -> None:
    if _feedback_capture_in_progress:
        return
    name = _pynput_key_name(key)
    modifier_name = (
        "shift" if name in {"shift", "shift_l", "shift_r"}
        else "ctrl" if name in {"ctrl", "ctrl_l", "ctrl_r"}
        else "alt" if name in {"alt", "alt_l", "alt_r"}
        else None
    )
    if modifier_name:
        _feedback_modifiers.add(modifier_name)
    observe_feedback_key_event(name, "press", frozenset(_feedback_modifiers - ({modifier_name} if modifier_name else set())))


def _observe_pynput_release(key) -> None:
    if _feedback_capture_in_progress:
        return
    name = _pynput_key_name(key)
    modifier_name = (
        "shift" if name in {"shift", "shift_l", "shift_r"}
        else "ctrl" if name in {"ctrl", "ctrl_l", "ctrl_r"}
        else "alt" if name in {"alt", "alt_l", "alt_r"}
        else None
    )
    observe_feedback_key_event(name, "release", frozenset(_feedback_modifiers))
    if modifier_name:
        _feedback_modifiers.discard(modifier_name)

# ---------------- Notifications / logging ----------------
def notify(message: str, title: str = "CtrlSpeak") -> None:
    """Display a user-facing notification window (falls back to stdout)."""
    try:
        from utils.gui import ensure_management_ui_thread, show_notification_popup

        ensure_management_ui_thread()
        enqueue_management_task(show_notification_popup, title, message)
    except Exception:
        logger.exception("Failed to display notification '%s': %s", title, message)
        try:
            print(f"{title}: {message}")
        except Exception:
            logger.exception("Failed to print fallback notification '%s'", title)




def ui_show_lockout_window(message: str, cancel_callback: Optional[Callable[[], None]] = None) -> None:
    """Display (or update) the first-run lockout window."""
    try:
        from utils.gui import (
            ensure_management_ui_thread,
            show_lockout_window,
            is_management_ui_thread,
        )

        ensure_management_ui_thread()
        if is_management_ui_thread():
            show_lockout_window(message, cancel_callback=cancel_callback)
        else:
            enqueue_management_task(show_lockout_window, message, cancel_callback=cancel_callback)
    except Exception:
        logger.exception("Failed to show lockout window")
        try:
            print(f"CtrlSpeak: {message}")
        except Exception:
            logger.exception("Failed to print lockout window fallback message")




def ui_update_lockout_message(message: str) -> None:
    """Update the message shown in the lockout window."""
    try:
        from utils.gui import ensure_management_ui_thread, update_lockout_message

        ensure_management_ui_thread()
        enqueue_management_task(update_lockout_message, message)
    except Exception:
        logger.exception("Failed to update lockout message")




def ui_close_lockout_window(message: str | None = None) -> None:
    """Close the lockout window, optionally after showing a completion message."""
    try:
        from utils.gui import (
            ensure_management_ui_thread,
            close_lockout_window,
            is_management_ui_thread,
        )

        ensure_management_ui_thread()
        if is_management_ui_thread():
            close_lockout_window(message)
        else:
            enqueue_management_task(close_lockout_window, message)
    except Exception:
        logger.exception("Failed to close lockout window")


def write_error_log(context: str, snippet: str) -> None:
    try:
        logs_dir = get_logs_dir()
        logs_dir.mkdir(parents=True, exist_ok=True)
        error_path = logs_dir / ERROR_LOG_FILENAME
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with error_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {context}\n{snippet}\n\n")
    except Exception:
        logger.exception("Failed to write error log entry")


def copy_to_clipboard(text: str) -> bool:
    try:
        if not set_clipboard_text(text):
            logger.warning("Failed to stage clipboard text")
            return False
        if get_clipboard_text() != text:
            logger.warning("Clipboard verification failed after staging text")
            return False
        return True
    except Exception:
        logger.exception("Failed to copy text to clipboard")
        return False


def remember_last_transcript(text: str) -> bool:
    """Retain one successful transcript in memory without persisting its content."""
    if not isinstance(text, str) or not text.strip():
        return False
    global _last_transcript
    with _last_transcript_lock:
        _last_transcript = text
    _refresh_tray_menu()
    return True


def get_last_transcript() -> Optional[str]:
    with _last_transcript_lock:
        return _last_transcript


def has_last_transcript(_item=None) -> bool:
    return get_last_transcript() is not None


def _refresh_tray_menu() -> None:
    icon = _tray_icon
    if icon is None:
        return
    try:
        icon.update_menu()
    except Exception:
        logger.exception("Failed to refresh the CtrlSpeak tray menu")


def copy_last_transcript_from_tray(_icon=None, _item=None) -> bool:
    text = get_last_transcript()
    if text is None:
        notify("No successful transcription is available yet.", title="CtrlSpeak")
        return False
    if not copy_to_clipboard(text):
        notify("The last transcript could not be copied to the clipboard.", title="CtrlSpeak")
        return False
    notify("Last transcript copied to the clipboard.", title="CtrlSpeak")
    return True


def request_copy_last_transcript_from_tray(icon=None, item=None) -> None:
    """Dispatch the clipboard write and notification onto CtrlSpeak's UI thread.

    Native tray callbacks run on pystray's worker thread.  Keeping the complete
    action on the management thread gives the temporary Win32 clipboard owner
    a normal UI-thread lifetime and avoids touching Tk while pystray is inside
    the operating-system menu callback.
    """

    enqueue_management_task(copy_last_transcript_from_tray, icon, item)


def notify_error(context: str, details: str) -> None:
    snippet = (details or "").strip() or "Unknown error"
    write_error_log(context, snippet)
    # Full technical information belongs in the rotating log.  Normal failures
    # use short, actionable copy and never overwrite the user's clipboard.
    notify(
        f"{context}. Open CtrlSpeak or the log folder for details.",
        title="CtrlSpeak",
    )


def format_exception_details(exc: BaseException | None) -> str:
    """Return a readable error summary including traceback details when possible."""
    if exc is None:
        return "Unknown error"

    try:
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
    except Exception:
        try:
            return f"{exc.__class__.__name__}: {exc}".strip()
        except Exception:
            return "Unknown error"

# ---------------- Resource helpers ----------------
def create_icon_image():
    return Image.open(app_icon_path())

# ---------------- Management UI pump (GUI thread lives in utils.gui) ----------------
def enqueue_management_task(func: Callable[..., None], *args, **kwargs) -> None:
    try:
        management_ui_queue.put_nowait((func, args, kwargs))
    except Exception:
        logger.exception("Failed to enqueue management task %s", getattr(func, "__name__", str(func)))


def pump_management_events_once() -> None:
    """Process queued management UI work once without blocking."""

    try:
        from utils.gui import ensure_management_ui_thread, pump_management_events_once as _pump_once

        ensure_management_ui_thread()
        _pump_once()
    except RuntimeError:
        raise
    except Exception:
        logger.exception("Failed to pump management UI events")


def schedule_management_refresh(delay_ms: int = 0) -> None:
    # utils.gui sets tk_root and management_window; import FRESH on each call
    from utils.gui import tk_root, management_window
    if tk_root is None:
        return

    def task() -> None:
        # recheck live window object on execution too
        from utils.gui import management_window as _live
        if _live and _live.is_open():
            _live.refresh_status()

    if delay_ms <= 0:
        enqueue_management_task(task)
    else:
        def delayed_task() -> None:
            from utils.gui import tk_root as _root
            if _root is not None:
                _root.after(delay_ms, task)
        enqueue_management_task(delayed_task)


# ---------------- Audio capture + feedback tone ----------------
CHUNKSIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# --- Live waveform ring buffer (for overlay) ---
from collections import deque
_waveform_lock = threading.Lock()
_waveform_buffers = deque()
_waveform_samples = 0
_WAVEFORM_MAX_SECONDS = 2.0  # keep ~2 seconds of recent audio
_WAVEFORM_MAX_SAMPLES = int(RATE * _WAVEFORM_MAX_SECONDS)

# live level of the loading.wav while processing
_processing_level = 0.0
_processing_level_lock = threading.Lock()

def get_processing_level() -> float:
    """Smoothed RMS in [0, ~1], read by GUI for pulsing."""
    with _processing_level_lock:
        return float(_processing_level)


def _push_waveform_bytes(chunk: bytes) -> None:
    try:
        arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        logger.exception("Failed to push waveform chunk of size %d", len(chunk))
        return
    global _waveform_samples
    with _waveform_lock:
        _waveform_buffers.append(arr)
        _waveform_samples += arr.size
        while _waveform_samples > _WAVEFORM_MAX_SAMPLES and _waveform_buffers:
            popped = _waveform_buffers.popleft()
            _waveform_samples -= popped.size


def _clear_waveform_buffers() -> None:
    global _waveform_samples
    with _waveform_lock:
        _waveform_buffers.clear()
        _waveform_samples = 0

def get_recent_waveform(ms: int = 500) -> np.ndarray:
    """Return last ms of audio as float32 [-1,1] for drawing."""
    need = int(RATE * ms / 1000.0)
    with _waveform_lock:
        if not _waveform_buffers:
            return np.zeros(need, dtype=np.float32)
        data = np.concatenate(list(_waveform_buffers)) if len(_waveform_buffers) > 1 else _waveform_buffers[0]
    if data.size <= need:
        out = np.zeros(need, dtype=np.float32)
        out[-data.size:] = data
        return out
    return data[-need:]

def generate_fallback_sound():
    duration = 0.5
    t = np.linspace(0.0, duration, int(PROCESSING_SAMPLE_RATE * duration), endpoint=False)
    envelope = np.exp(-3 * t)
    wave_data = 0.2 * np.sin(2 * np.pi * 440 * t) * envelope
    int_data = np.clip(wave_data * 32767, -32767, 32767).astype(np.int16)
    settings_audio = {"channels": 1, "rate": PROCESSING_SAMPLE_RATE, "width": 2}
    return int_data.tobytes(), settings_audio


def limit_processing_sound_peak(
    frames: bytes,
    settings_audio: Dict[str, int],
    *,
    max_peak: float = PROCESSING_SOUND_MAX_PEAK,
) -> bytes:
    """Apply transparent whole-clip attenuation when 16-bit audio exceeds the peak ceiling."""
    if settings_audio.get("width") != 2 or not frames:
        return frames
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return frames
    peak = max(abs(sample) for sample in samples)
    target = max(1, min(32767, int(round(32767 * max_peak))))
    if peak <= target:
        return frames
    gain = target / peak
    softened = array(
        "h",
        (
            max(-32768, min(32767, int(round(sample * gain))))
            for sample in samples
        ),
    )
    if sys.byteorder == "big":
        softened.byteswap()
    return softened.tobytes()


def load_processing_sound():
    global processing_sound_data, processing_sound_settings
    if processing_sound_data is not None and processing_sound_settings is not None:
        return processing_sound_data, processing_sound_settings
    try:
        sound_path = asset_path("loading.wav")
        with wave.open(str(sound_path), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            settings_audio = {"channels": wav_file.getnchannels(), "rate": wav_file.getframerate(), "width": wav_file.getsampwidth()}
    except Exception:
        logger.exception("Failed to load processing sound from %s; using fallback tone", asset_path("loading.wav"))
        frames, settings_audio = generate_fallback_sound()
    frames = limit_processing_sound_peak(frames, settings_audio)
    processing_sound_data = frames
    processing_sound_settings = settings_audio
    return processing_sound_data, processing_sound_settings


def _set_pyaudio_capture_pending(pending: bool) -> None:
    with _pyaudio_priority_lock:
        if pending:
            _pyaudio_capture_pending.set()
        else:
            _pyaudio_capture_pending.clear()


@contextmanager
def _managed_pyaudio(*, blocking: bool):
    """Yield one exclusively owned PyAudio instance, or ``None`` if busy.

    Capture is the primary operation and waits for ownership.  Optional legacy
    feedback and device enumeration request non-blocking ownership so they
    cannot freeze the UI or delay dictation.
    """

    pa_instance = None
    if blocking:
        acquired = _pyaudio_session_lock.acquire(blocking=True)
    else:
        # Hold arbitration from the marker check through construction.  The
        # hotkey publishes capture under the same lock, closing the last
        # check-then-initialize race without waiting for optional stream work.
        with _pyaudio_priority_lock:
            if _pyaudio_capture_pending.is_set():
                acquired = False
            else:
                acquired = _pyaudio_session_lock.acquire(blocking=False)
                if acquired:
                    try:
                        pa_instance = pyaudio.PyAudio()
                    except Exception:
                        _pyaudio_session_lock.release()
                        raise
    if not acquired:
        yield None
        return
    try:
        if pa_instance is None:
            pa_instance = pyaudio.PyAudio()
        yield pa_instance
    finally:
        if pa_instance is not None:
            try:
                pa_instance.terminate()
            except Exception:
                logger.debug("Failed to terminate the exclusive PyAudio session", exc_info=True)
        _pyaudio_session_lock.release()

def _processing_sound_loop():
    data, settings_audio = load_processing_sound()
    with _managed_pyaudio(blocking=False) as pa_instance:
        if pa_instance is None:
            logger.debug("Skipping legacy processing audio while PortAudio is busy")
            return
        stream = None
        try:
            stream = pa_instance.open(
                format=pyaudio.get_format_from_width(settings_audio["width"]),
                channels=settings_audio["channels"],
                rate=settings_audio["rate"],
                output=True,
            )

            # choose a short hop for snappy visuals (~10 ms)
            bytes_per_sample = settings_audio["width"]
            channels = settings_audio["channels"]
            hop_samples = int(settings_audio["rate"] * 0.010)  # 10 ms
            chunk_bytes = hop_samples * bytes_per_sample * channels

            offset = 0
            nbytes = len(data)

            alpha = 0.35  # smoothing (higher = more responsive)

            while not processing_sound_stop_event.is_set():
                if offset + chunk_bytes > nbytes:
                    offset = 0  # loop the sound

                chunk = data[offset:offset + chunk_bytes]
                offset += chunk_bytes

                stream.write(chunk)

                try:
                    if bytes_per_sample == 2:
                        arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                        if channels > 1:
                            arr = arr.reshape(-1, channels).mean(axis=1)
                        rms = float(np.sqrt(np.mean(arr * arr)))
                        # update smoothed level
                        global _processing_level, _proc_vis_samples
                        with _processing_level_lock:
                            _processing_level = (1.0 - alpha) * _processing_level + alpha * rms
                        # keep recent mono samples for GUI wiggle
                        with _proc_vis_lock:
                            _proc_vis_buffers.append(arr.copy())
                            _proc_vis_samples += arr.size
                            while _proc_vis_samples > _PROC_VIS_MAX_SAMPLES and _proc_vis_buffers:
                                popped = _proc_vis_buffers.popleft()
                                _proc_vis_samples -= popped.size
                except Exception:
                    logger.exception("Failed to update processing waveform metrics")

        except Exception:
            logger.exception("Processing feedback loop crashed")
        finally:
            try:
                if stream is not None:
                    stream.stop_stream(); stream.close()
            except Exception:
                logger.exception("Failed to close processing audio stream cleanly")


def _play_native_non_windows_cue(cue: Pcm16Cue) -> bool:
    """Use an OS audio helper for every non-Windows cue without PortAudio.

    Linux desktop images normally provide PipeWire's ``pw-play`` or ALSA's
    ``aplay``.  Running that helper out of process prevents its audio lifecycle
    from racing the PyAudio extension in CtrlSpeak.  macOS ``afplay`` needs a
    short-lived file rather than standard input.
    """

    payload = cue_to_wav_bytes(cue)
    if sys.platform == "darwin":
        player = shutil.which("afplay")
        if not player:
            logger.warning("Unable to preview UI cue: afplay is unavailable")
            return False
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                handle.write(payload)
                temp_path = handle.name
            subprocess.run(
                [player, temp_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=3.0,
            )
            return True
        except Exception:
            logger.warning("Native macOS UI cue playback failed", exc_info=True)
            return False
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    logger.debug("Failed to remove temporary UI cue", exc_info=True)

    candidates = (
        ("pw-play", ["-"]),
        ("aplay", ["--quiet"]),
    )
    attempted = False
    for executable, arguments in candidates:
        player = shutil.which(executable)
        if not player:
            continue
        attempted = True
        try:
            subprocess.run(
                [player, *arguments],
                input=payload,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=3.0,
            )
            return True
        except subprocess.TimeoutExpired:
            logger.warning(
                "Native UI cue helper %s timed out after %.1f seconds",
                executable,
                3.0,
            )
            return False
        except Exception:
            logger.debug("Native UI cue helper %s failed", executable, exc_info=True)
    if attempted:
        logger.warning("Unable to preview UI cue with the available native audio helpers")
    else:
        logger.warning("Unable to preview UI cue: no native audio helper is available")
    return False


def _play_pcm_cue(cue: Pcm16Cue) -> None:
    """Play one prepared cue; output-device failures never affect transcription."""

    if sys.platform.startswith("win"):
        # PlaySound owns no PortAudio state.  SND_MEMORY is synchronous, while
        # CuePlayer already invokes this sink on its short-lived daemon thread.
        import winsound

        winsound.PlaySound(
            cue_to_wav_bytes(cue),
            winsound.SND_MEMORY | winsound.SND_NODEFAULT,
        )
        return

    _play_native_non_windows_cue(cue)


def play_ui_cue(kind: CueKind, *, background: bool = True) -> bool:
    """Play a short, headphone-safe Midnight Signal cue using saved controls."""

    with settings_lock:
        enabled = bool(settings.get("audio_cues_enabled", True))
        raw_volume = settings.get("audio_cue_volume", 30)
    try:
        volume = max(0.0, min(1.0, int(raw_volume) / 100.0))
    except (TypeError, ValueError):
        volume = 0.30
    return CuePlayer(
        _play_pcm_cue,
        enabled=enabled and volume > 0,
        volume=volume,
        peak_ceiling=0.25,
    ).play(kind, background=background)


def start_processing_feedback():
    """Compatibility entry point: play once; v0.7 deliberately has no loop."""

    play_ui_cue(CueKind.PROCESSING_STARTED)


def stop_processing_feedback():
    """The v0.7 processing cue is finite, so there is nothing to stop."""

    processing_sound_stop_event.set()


def play_model_ready_sound_once() -> None:
    global _ready_sound_played
    with _ready_sound_lock:
        if _ready_sound_played:
            return
        _ready_sound_played = True

    play_ui_cue(CueKind.SUCCESS)


def _cached_input_audio_devices() -> List[Tuple[str, str]]:
    with _input_device_cache_lock:
        cached = list(_input_device_cache)
    if cached:
        return cached
    preferred = get_input_device_preference()
    if preferred:
        return [(preferred, f"{preferred} · saved preference (scan deferred)")]
    return []


def list_input_audio_devices() -> List[Tuple[str, str]]:
    """Return input devices without falsifying a busy refresh as device loss.

    Enumeration must never initialize PortAudio alongside capture.  When the
    runtime is busy, callers receive the last successful snapshot.  Before the
    first successful scan, an existing saved preference is retained as a
    provisional entry so the management UI cannot replace it with the system
    default merely because the microphone is currently in use.
    """

    global _input_device_cache
    devices: List[Tuple[str, str]] = []
    with _managed_pyaudio(blocking=False) as pa_instance:
        if pa_instance is None:
            logger.debug("Returning cached input devices while PortAudio is busy")
            return _cached_input_audio_devices()
        try:
            host_names: Dict[int, str] = {}
            try:
                for host_index in range(pa_instance.get_host_api_count()):
                    host_info = pa_instance.get_host_api_info_by_index(host_index)
                    host_names[host_index] = str(host_info.get("name", ""))
            except Exception:
                logger.exception("Failed to enumerate audio host APIs")

            for index in range(pa_instance.get_device_count()):
                try:
                    info = pa_instance.get_device_info_by_index(index)
                except Exception:
                    logger.exception("Failed to read audio device info for index %s", index)
                    continue
                if int(info.get("maxInputChannels", 0)) <= 0:
                    continue
                name = str(info.get("name", f"Device {index}"))
                host_name = host_names.get(info.get("hostApi"), "")
                label = name
                if host_name:
                    label = f"{label} · {host_name}"
                devices.append((name, label))
        except Exception:
            logger.exception("Failed to enumerate input audio devices")
            return _cached_input_audio_devices()
    with _input_device_cache_lock:
        _input_device_cache = tuple(devices)
    return devices


def get_input_device_preference() -> Optional[str]:
    """Return the stored input device name, or None for system default."""
    with settings_lock:
        preferred = settings.get("input_device")
    if isinstance(preferred, str) and preferred.strip():
        return preferred
    return None


def set_input_device_preference(device_name: Optional[str]) -> None:
    """Persist the preferred input device name (None for default)."""
    cleaned = device_name.strip() if isinstance(device_name, str) else None
    with settings_lock:
        settings["input_device"] = cleaned if cleaned else None
    save_settings()


def _resolve_input_device_index(pa_instance: pyaudio.PyAudio, preferred: Optional[str] = None) -> Optional[int]:
    """Return the PyAudio index for the preferred device, if available."""
    target = preferred if preferred is not None else get_input_device_preference()
    if not target:
        return None
    try:
        for index in range(pa_instance.get_device_count()):
            info = pa_instance.get_device_info_by_index(index)
            if int(info.get("maxInputChannels", 0)) <= 0:
                continue
            if str(info.get("name")) == target:
                return index
    except Exception:
        logger.exception("Failed to resolve preferred input audio device index")
    return None


def record_audio(
    target_path: Path,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """Capture one recording until its session-local stop event is set.

    ``stop_event`` is deliberately per dictation.  Falling back to the legacy
    global flag keeps the helper callable by older integrations, while the
    hotkey lifecycle never reuses that mutable flag to control a worker.
    """

    with _managed_pyaudio(blocking=True) as pyaudio_instance:
        assert pyaudio_instance is not None
        stream = None
        sample_width: Optional[int] = None
        stream_kwargs = dict(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNKSIZE,
        )
        preferred_index = _resolve_input_device_index(pyaudio_instance)
        if preferred_index is not None:
            stream_kwargs["input_device_index"] = preferred_index
        frames = []
        try:
            try:
                stream = pyaudio_instance.open(**stream_kwargs)
            except Exception:
                if "input_device_index" in stream_kwargs:
                    logger.exception("Failed to open preferred input device; falling back to system default")
                    stream_kwargs.pop("input_device_index", None)
                    stream = pyaudio_instance.open(**stream_kwargs)
                else:
                    raise
            sample_width = pyaudio_instance.get_sample_size(FORMAT)
            def should_continue() -> bool:
                return not stop_event.is_set() if stop_event is not None else recording

            while should_continue():
                _chunk = stream.read(CHUNKSIZE)
                frames.append(_chunk)
                _push_waveform_bytes(_chunk)
                transcription_ui_session.update_level_pcm16(_chunk)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    logger.debug("Failed to close microphone stream cleanly", exc_info=True)
    if sample_width is None:
        raise RuntimeError("The microphone stream did not provide a sample width")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(sample_width)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))


def _record_audio_worker(
    target_path: Path,
    generation: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
    failure_event: Optional[threading.Event] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Keep microphone/backend failures inside CtrlSpeak's reporting path."""

    global recording_file_path, recording_thread, _active_transcription_generation
    failure = failure_event or _recording_failed_event
    cancel = cancel_event or transcription_cancel_event
    _set_pyaudio_capture_pending(True)
    try:
        if stop_event is None:
            record_audio(target_path)
        else:
            record_audio(target_path, stop_event)
    except Exception as exc:
        with _transcription_state_lock:
            is_current = (
                generation is None
                or generation == _active_transcription_generation
            )
            if is_current:
                failure.set()
        logger.exception("Microphone recording failed")
        if is_current and not cancel.is_set():
            notify_error("Microphone recording failed", format_exception_details(exc))
    finally:
        # ``record_audio`` returns only after the input stream is closed and
        # _managed_pyaudio has terminated its instance.
        _set_pyaudio_capture_pending(False)
        orphaned = False
        with _transcription_state_lock:
            # If key release could not start its coordinator, the recorder is
            # the only remaining worker capable of closing this generation.
            # It is safe to clear its own reference here: capture and WAV flush
            # are already complete and this function is about to return.
            if (
                _session_is_current_locked(generation)
                and not recording
                and transcription_thread is None
                and recording_thread is threading.current_thread()
                and (failure.is_set() or cancel.is_set())
            ):
                recording_thread = None
                if recording_file_path == target_path:
                    recording_file_path = None
                _active_transcription_generation = None
                orphaned = True
        if orphaned:
            cleanup_recording_file(target_path)
            _clear_waveform_buffers()
            _refresh_tray_menu()
            schedule_management_refresh()

# ---------------- Client keyboard listener ----------------
def _client_hotkey_available() -> bool:
    """Return True when the right-Ctrl hotkey may start a recording."""

    from utils.transcription_backend import get_runtime_backend_config

    if get_runtime_backend_config().backend == "api":
        return True

    with settings_lock:
        mode = settings.get("mode")

    if mode != "client":
        return True

    if last_connected_server is not None:
        return True

    logger.info(
        "Recording hotkey blocked: client-only mode is active without a connected server."
    )
    return False


def _set_terminal_overlay_hide(delay_ms: int) -> None:
    try:
        from utils.gui import hide_waveform_overlay

        enqueue_management_task(hide_waveform_overlay, delay_ms)
    except Exception:
        logger.exception("Failed to schedule the Midnight Signal overlay dismissal")


def _show_recording_overlay() -> None:
    try:
        from utils.gui import show_waveform_overlay

        enqueue_management_task(show_waveform_overlay, lambda: get_recent_waveform(500))
    except Exception:
        logger.exception("Failed to show the Midnight Signal recording overlay")


def _show_processing_overlay() -> None:
    try:
        from utils.gui import set_waveform_processing

        enqueue_management_task(set_waveform_processing, "Transcribing…")
    except Exception:
        logger.exception("Failed to switch the Midnight Signal overlay to processing")


def _prepare_ui_session_for_recording() -> None:
    phase = transcription_ui_session.phase
    if phase is not UiPhase.IDLE:
        if phase in {UiPhase.SUCCESS, UiPhase.ERROR, UiPhase.CANCELLED}:
            transcription_ui_session.reset()
        else:
            raise RuntimeError(f"CtrlSpeak is already {phase.value}")
    transcription_ui_session.begin_recording()


def _next_transcription_generation_locked() -> int:
    global _transcription_generation, _active_transcription_generation
    _transcription_generation += 1
    _active_transcription_generation = _transcription_generation
    return _transcription_generation


def _session_is_current_locked(generation: Optional[int]) -> bool:
    """Return whether a worker still owns the active lifecycle globals."""

    return generation is None or generation == _active_transcription_generation


def _safe_error_category(exc: BaseException) -> str:
    for attribute in ("error_code", "category", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold().replace("-", "_")
    message = str(exc).casefold()
    if any(term in message for term in ("quota", "billing", "credit", "insufficient_quota")):
        return "openai_quota_exhausted"
    if any(term in message for term in ("api key", "unauthorized", "authentication", "401")):
        return "openai_invalid_key"
    if "language" in message:
        return "language_policy"
    if any(term in message for term in ("provider", "worker", "gateway", "connection")):
        return "providers_exhausted"
    return "unexpected_error"


def is_transcription_busy(_item=None) -> bool:
    with _transcription_state_lock:
        # A referenced worker is pending even in the tiny interval between
        # Thread construction and ``start()``.  References are cleared only by
        # that generation's finalizer after the worker has really exited.
        worker_pending = transcription_thread is not None
        recorder_pending = recording_thread is not None
        return (
            recording
            or worker_pending
            or recorder_pending
            or _active_transcription_generation is not None
            or transcription_ui_session.phase in {UiPhase.RECORDING, UiPhase.PROCESSING}
        )


def _finish_cancelled_session() -> None:
    phase = transcription_ui_session.phase
    if phase not in {UiPhase.RECORDING, UiPhase.PROCESSING}:
        return
    transcription_ui_session.cancel()
    _clear_waveform_buffers()
    play_ui_cue(CueKind.CANCELLED)
    _set_terminal_overlay_hide(1200)
    _refresh_tray_menu()
    schedule_management_refresh()


def cancel_active_transcription(_icon=None, _item=None) -> bool:
    """Cancel recording immediately or suppress a pending request's insertion."""

    global recording, transcription_thread
    reaper: Optional[threading.Thread] = None
    reaper_start_failed = False
    with _transcription_state_lock:
        phase = transcription_ui_session.phase
        if phase not in {UiPhase.RECORDING, UiPhase.PROCESSING} and not recording:
            return False
        generation = _active_transcription_generation
        cancel_event = transcription_cancel_event
        cancel_event.set()
        _recording_stop_event.set()
        recording = False
        path = recording_file_path
        recorder = recording_thread
        _finish_cancelled_session()

        # A capture cancelled before key release has no coordinator yet.  Give
        # it one, retain both worker references, and clean only after the
        # recorder has genuinely stopped.
        if transcription_thread is None:
            reaper = threading.Thread(
                target=_cancelled_recording_reaper,
                args=(generation, path, recorder),
                name="CtrlSpeakRecorderCancel",
                daemon=True,
            )
            transcription_thread = reaper

    if reaper is not None:
        try:
            reaper.start()
        except Exception:
            logger.exception("Failed to start the cancelled-recording cleanup worker")
            # Never retain an unstarted coordinator: is_transcription_busy()
            # treats every non-None worker reference as pending.  The recorder
            # remains registered when it is still alive and its existing
            # orphan finalizer will close the generation after capture stops.
            with _transcription_state_lock:
                if transcription_thread is reaper:
                    transcription_thread = None
            reaper_start_failed = True
    if reaper_start_failed:
        _finalize_transcription_session(
            generation,
            path,
            recorder=recorder,
        )
        _refresh_tray_menu()
        schedule_management_refresh()
    return True


def _finalize_transcription_session(
    generation: Optional[int],
    path: Path | None,
    *,
    owner_thread: Optional[threading.Thread] = None,
    recorder: Optional[threading.Thread] = None,
) -> None:
    """Clean one generation without ever clearing a newer generation's state."""

    global recording, recording_file_path, recording_thread, transcription_thread
    global _active_transcription_generation

    should_cleanup = True
    clear_waveform = False
    with _transcription_state_lock:
        if not _session_is_current_locked(generation):
            should_cleanup = True
        else:
            clear_waveform = True
            candidate_recorder = recorder or recording_thread
            recorder_stopped = candidate_recorder is None or not candidate_recorder.is_alive()
            should_cleanup = recorder_stopped
            if (
                recording_thread is candidate_recorder
                and candidate_recorder is not None
                and recorder_stopped
            ):
                recording_thread = None
            if generation is None or owner_thread is None or transcription_thread is owner_thread:
                transcription_thread = None
            if recording_file_path == path and recorder_stopped:
                recording_file_path = None
            recording = False
            if recording_thread is None and transcription_thread is None:
                _active_transcription_generation = None
    if should_cleanup:
        cleanup_recording_file(path)
    if clear_waveform:
        _clear_waveform_buffers()


def _cancelled_recording_reaper(
    generation: Optional[int],
    path: Path | None,
    recorder: Optional[threading.Thread],
) -> None:
    if recorder is not None and recorder is not threading.current_thread():
        try:
            recorder.join()
        except RuntimeError:
            # A cancellation can land in the narrow interval before start().
            # The owning hotkey callback will start the recorder before it
            # releases the lifecycle lock; an unstarted legacy test double is
            # simply treated as already stopped.
            logger.debug("Recorder was not started before cancellation cleanup")
    _finalize_transcription_session(
        generation,
        path,
        owner_thread=threading.current_thread(),
        recorder=recorder,
    )
    _refresh_tray_menu()
    schedule_management_refresh()


def _finish_recording_then_transcribe(
    generation: Optional[int],
    path: Path | None,
    recorder: Optional[threading.Thread],
    failure_event: threading.Event,
    cancel_event: threading.Event,
    started_at: float,
) -> None:
    """Wait for the WAV flush off the listener thread, then transcribe it."""

    global recording_thread
    if recorder is not None and recorder is not threading.current_thread():
        try:
            recorder.join()
        except RuntimeError:
            logger.exception("Recorder coordination failed before transcription")
            failure_event.set()

    with _transcription_state_lock:
        if not _session_is_current_locked(generation):
            cleanup_recording_file(path)
            return
        if recording_thread is recorder:
            recording_thread = None
        cancelled = cancel_event.is_set()
        failed = failure_event.is_set()
        if cancelled:
            _finish_cancelled_session()
        elif failed and transcription_ui_session.phase is UiPhase.PROCESSING:
            transcription_ui_session.fail("microphone_failed")

    if cancelled:
        _finalize_transcription_session(
            generation,
            path,
            owner_thread=threading.current_thread(),
            recorder=recorder,
        )
        _refresh_tray_menu()
        schedule_management_refresh()
        return
    if failed:
        play_ui_cue(CueKind.ERROR)
        _set_terminal_overlay_hide(3800)
        _finalize_transcription_session(
            generation,
            path,
            owner_thread=threading.current_thread(),
            recorder=recorder,
        )
        _refresh_tray_menu()
        schedule_management_refresh()
        return

    _transcribe_recording_worker(
        path,
        started_at,
        generation=generation,
        cancel_event=cancel_event,
        owner_thread=threading.current_thread(),
        recorder=recorder,
    )


def _transcribe_recording_worker(
    path: Path | None,
    started_at: float,
    *,
    generation: Optional[int] = None,
    cancel_event: Optional[threading.Event] = None,
    owner_thread: Optional[threading.Thread] = None,
    recorder: Optional[threading.Thread] = None,
) -> None:
    from utils.models import transcribe_audio_result

    cancel = cancel_event or transcription_cancel_event
    owner = owner_thread or threading.current_thread()
    try:
        with _transcription_state_lock:
            if not _session_is_current_locked(generation):
                return
            if cancel.is_set():
                _finish_cancelled_session()
                return
        if path is None or not path.exists() or path.stat().st_size <= 44:
            raise RuntimeError("The recording did not contain usable audio")
        result = transcribe_audio_result(str(path), play_feedback=False)
        if result is None or not result.text:
            raise RuntimeError("No transcription text was returned")

        # Cancellation and insertion form one commit point.  A cancellation
        # that acquires the lifecycle lock first suppresses insertion; once
        # insertion begins, cancellation cannot report a contradictory
        # CANCELLED terminal state.
        with _transcription_state_lock:
            if not _session_is_current_locked(generation):
                return
            if cancel.is_set():
                _finish_cancelled_session()
                return
            if transcription_ui_session.phase is not UiPhase.PROCESSING:
                return
            inject_transcription_result(result)
            elapsed_ms = (time.monotonic() - started_at) * 1000.0
            transcription_ui_session.complete(result.metadata or {}, elapsed_ms=elapsed_ms)
        play_ui_cue(CueKind.SUCCESS)
        _set_terminal_overlay_hide(1600)
    except Exception as exc:
        with _transcription_state_lock:
            if not _session_is_current_locked(generation):
                return
            cancelled = cancel.is_set()
            if cancelled:
                _finish_cancelled_session()
            else:
                category = _safe_error_category(exc)
                if transcription_ui_session.phase is UiPhase.PROCESSING:
                    transcription_ui_session.fail(
                        category,
                        elapsed_ms=(time.monotonic() - started_at) * 1000.0,
                    )
        if not cancelled:
            logger.exception("Transcription or text insertion failed")
            notify_error("Transcription failed", format_exception_details(exc))
            play_ui_cue(CueKind.ERROR)
            _set_terminal_overlay_hide(3800)
    finally:
        _finalize_transcription_session(
            generation,
            path,
            owner_thread=owner,
            recorder=recorder,
        )
        _refresh_tray_menu()
        schedule_management_refresh()


def on_press(key):
    global recording, recording_thread, recording_file_path
    global transcription_cancel_event, _recording_failed_event, _recording_stop_event
    global _active_transcription_generation
    _observe_pynput_press(key)
    with _client_state_lock:
        if not client_enabled:
            return
    if is_right_control(key):
        with _transcription_state_lock:
            if is_transcription_busy():
                return
        if not _client_hotkey_available():
            return
        # `_client_hotkey_available()` can take long enough for shutdown to
        # disable the listener.  The final state check therefore shares the
        # client-state lock with stop_client_listener().  Starting and
        # registering a recorder is one atomic commit from shutdown's point of
        # view: either shutdown wins and this callback returns, or shutdown
        # waits and then cancels the fully registered generation.
        with _client_state_lock:
            if not client_enabled:
                return
            start_error: Exception | None = None
            failed_path: Path | None = None
            with _transcription_state_lock:
                if is_transcription_busy():
                    return
                try:
                    _prepare_ui_session_for_recording()
                except RuntimeError:
                    return
                generation = _next_transcription_generation_locked()
                transcription_cancel_event = threading.Event()
                _recording_stop_event = threading.Event()
                _recording_failed_event = threading.Event()
                _clear_waveform_buffers()
                recording = True
                recording_file_path = create_recording_file_path()
                recording_thread = threading.Thread(
                    target=_record_audio_worker,
                    args=(
                        recording_file_path,
                        generation,
                        _recording_stop_event,
                        _recording_failed_event,
                        transcription_cancel_event,
                    ),
                    name="CtrlSpeakRecorder",
                    daemon=True,
                )
                # Publish capture priority before the worker is scheduled so
                # a non-Windows cue thread cannot win the PortAudio race in
                # the interval between Thread.start() and record_audio().
                _set_pyaudio_capture_pending(True)
                try:
                    recording_thread.start()
                except Exception as exc:
                    logger.exception("Failed to start the microphone recording worker")
                    start_error = exc
                    failed_path = recording_file_path
                    _set_pyaudio_capture_pending(False)
                    _recording_stop_event.set()
                    _recording_failed_event.set()
                    transcription_cancel_event.set()
                    recording = False
                    recording_thread = None
                    recording_file_path = None
                    _active_transcription_generation = None
                    if transcription_ui_session.phase is UiPhase.RECORDING:
                        transcription_ui_session.fail("microphone_failed")
                    _clear_waveform_buffers()
            if start_error is None:
                _show_recording_overlay()
                play_ui_cue(CueKind.RECORDING_STARTED)
            else:
                cleanup_recording_file(failed_path)
                notify_error(
                    "Microphone recording failed to start",
                    format_exception_details(start_error),
                )
                play_ui_cue(CueKind.ERROR)
                _set_terminal_overlay_hide(3800)
            _refresh_tray_menu()
            schedule_management_refresh()


def on_release(key):
    global recording, recording_file_path, recording_thread, transcription_thread
    global _active_transcription_generation
    _observe_pynput_release(key)
    if is_right_control(key):
        start_error: Exception | None = None
        with _transcription_state_lock:
            if not recording:
                return
            generation = _active_transcription_generation
            cancel_event = transcription_cancel_event
            failure_event = _recording_failed_event
            recorder = recording_thread
            path = recording_file_path
            recording = False
            _recording_stop_event.set()
            if cancel_event.is_set() or transcription_ui_session.phase is not UiPhase.RECORDING:
                _clear_waveform_buffers()
                return
            transcription_ui_session.begin_processing()
            _show_processing_overlay()
            _clear_waveform_buffers()
            start_processing_feedback()
            started_at = time.monotonic()
            transcription_thread = threading.Thread(
                target=_finish_recording_then_transcribe,
                args=(
                    generation,
                    path,
                    recorder,
                    failure_event,
                    cancel_event,
                    started_at,
                ),
                name="CtrlSpeakTranscriber",
                daemon=True,
            )
            try:
                transcription_thread.start()
            except Exception as exc:
                logger.exception("Failed to start the transcription coordinator")
                start_error = exc
                transcription_thread = None
                failure_event.set()
                cancel_event.set()
                _recording_stop_event.set()
                if transcription_ui_session.phase is UiPhase.PROCESSING:
                    transcription_ui_session.fail("unexpected_error")
                _clear_waveform_buffers()
        if start_error is not None:
            if recorder is not None and recorder is not threading.current_thread():
                try:
                    recorder.join(timeout=2.5)
                except RuntimeError:
                    logger.debug("Recorder had not started during coordinator rollback")
            recorder_stopped = recorder is None or not recorder.is_alive()
            with _transcription_state_lock:
                if _session_is_current_locked(generation):
                    if recording_thread is recorder and recorder_stopped:
                        recording_thread = None
                    transcription_thread = None
                    if recording_file_path == path and recorder_stopped:
                        recording_file_path = None
                    if recording_thread is None:
                        _active_transcription_generation = None
            cleanup_recording_file(path)
            _clear_waveform_buffers()
            notify_error(
                "Transcription worker failed to start",
                format_exception_details(start_error),
            )
            play_ui_cue(CueKind.ERROR)
            _set_terminal_overlay_hide(3800)
            _refresh_tray_menu()
            schedule_management_refresh()
            return
        _refresh_tray_menu()
        schedule_management_refresh()

# ---- Discovery wrappers to restore original side-effects ----
def _apply_last_connected(server: Optional[ServerInfo]) -> Optional[ServerInfo]:
    """Mirror original behavior: update global and refresh GUI."""
    global last_connected_server
    last_connected_server = server if server else None
    schedule_management_refresh()
    return server

def manual_discovery_refresh(wait_time: float = 1.5) -> Optional[ServerInfo]:
    """Wrap net_discovery refresh but keep original side-effects here."""
    server = _nd_manual_discovery_refresh(discovery_listener, wait_time=wait_time)
    return _apply_last_connected(server)

def get_last_connected_server() -> Optional[ServerInfo]:
    """Return the most recently discovered or connected server without side effects."""
    return last_connected_server

def get_best_server() -> Optional[ServerInfo]:
    server = _nd_get_best_server(discovery_listener)
    return _apply_last_connected(server)

def _refresh_best_server_async():
    """Background refresh used by start_client_listener(), keeps globals updated."""
    try:
        manual_discovery_refresh()
    except Exception:
        pass

def start_client_listener() -> None:
    global listener, client_enabled
    with _client_state_lock:
        with listener_lock:
            if listener is not None:
                return
            client_enabled = True
            try:
                candidate = create_global_listener(
                    on_press=on_press,
                    on_release=on_release,
                )
                candidate.start()
                listener = candidate
            except DesktopSessionError as exc:
                client_enabled = False
                logger.warning("CtrlSpeak hotkey listener is unavailable: %s", exc)
                notify(str(exc), title="CtrlSpeak desktop support")
                return
            except Exception as exc:
                client_enabled = False
                logger.exception("CtrlSpeak hotkey listener failed to start")
                notify(
                    "The global hotkey listener could not start. On Ubuntu, use an Ubuntu on "
                    f"Xorg session and verify pynput dependencies. {exc}",
                    title="CtrlSpeak desktop support",
                )
                return
    from utils.transcription_backend import uses_bundled_runtime
    if uses_bundled_runtime():
        threading.Thread(target=_refresh_best_server_async, daemon=True).start()
    schedule_management_refresh()


def cancel_and_wait_for_active_transcription(timeout_seconds: float = 2.5) -> bool:
    """Cancel active work and wait a bounded time for its workers to exit.

    The normal lifecycle workers retain and clear their own references.  The
    final reconciliation below exists for shutdown-time legacy/stale state and
    runs only after a referenced thread is demonstrably no longer alive.
    """

    global recording, recording_file_path, recording_thread, transcription_thread
    global _active_transcription_generation

    try:
        timeout = max(0.0, float(timeout_seconds))
    except (TypeError, ValueError):
        timeout = 2.5

    cancel_active_transcription()
    with _transcription_state_lock:
        transcription_cancel_event.set()
        _recording_stop_event.set()
        recording = False
        recorder = recording_thread
        transcriber = transcription_thread
        path = recording_file_path

    deadline = time.monotonic() + timeout
    current = threading.current_thread()
    for worker in (recorder, transcriber):
        if worker is None or worker is current:
            continue
        remaining = max(0.0, deadline - time.monotonic())
        try:
            worker.join(timeout=remaining)
        except RuntimeError:
            logger.debug("Shutdown encountered a worker that had not started")

    cleanup_path: Optional[Path] = None
    with _transcription_state_lock:
        if recording_thread is not None and not recording_thread.is_alive():
            recording_thread = None
        if transcription_thread is not None and not transcription_thread.is_alive():
            transcription_thread = None
        stopped = recording_thread is None and transcription_thread is None
        if stopped:
            cleanup_path = recording_file_path or path
            recording_file_path = None
            _active_transcription_generation = None

    if cleanup_path is not None:
        cleanup_recording_file(cleanup_path)
    if not stopped:
        # The process is shutting down and no new request may start.  Attempt
        # exact-path cleanup even when a provider call ignores cancellation;
        # Windows may defer it while the file is open, so the next locked
        # startup also performs bounded stale-recording cleanup.
        cleanup_recording_file(path)
        logger.warning(
            "CtrlSpeak shutdown timed out waiting for active transcription workers"
        )
    _clear_waveform_buffers()
    return stopped


def stop_client_listener() -> None:
    global listener, client_enabled
    with _client_state_lock:
        with listener_lock:
            _feedback_modifiers.clear()
            client_enabled = False
            if listener is not None:
                listener.stop(); listener = None
    had_active_work = is_transcription_busy()
    cancel_and_wait_for_active_transcription()
    if had_active_work:
        try:
            from utils.gui import hide_waveform_overlay
            enqueue_management_task(hide_waveform_overlay)
        except Exception:
            logger.exception("Failed to hide waveform overlay when stopping listener")
    schedule_management_refresh()

# ---------------- Server (HTTP) ----------------
class TranscriptionRequestHandler(BaseHTTPRequestHandler):
    server_version = "CtrlSpeakServer/1.0"
    def do_GET(self):
        if self.path in {"/ping", "/health", "/status"}:
            payload = json.dumps({"status": "ok", "mode": "server"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers(); self.wfile.write(payload); return
        self.send_error(404, "Unknown endpoint")
    def do_POST(self):
        from utils.models import transcribe_local  # lazy import to avoid circulars
        if self.path != "/transcribe":
            self.send_error(404, "Unknown endpoint"); return
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self.send_error(400, "Missing audio payload"); return
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
            dir=get_temp_dir(),
        ) as tmp:
            remaining = content_length
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk: break
                tmp.write(chunk); remaining -= len(chunk)
            temp_path = tmp.name
        if os.path.getsize(temp_path) == 0:
            os.remove(temp_path); self.send_error(400, "Empty audio payload"); return
        start_time = time.time()
        text = transcribe_local(temp_path, play_feedback=False)
        try:
            os.remove(temp_path)
        except Exception:
            logger.exception("Failed to remove temporary transcription upload %s", temp_path)
        if text is None:
            self.send_error(500, "Transcription failed"); return
        payload = json.dumps({"text": text, "elapsed": time.time() - start_time}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers(); self.wfile.write(payload)
    def log_message(self, format, *args): return

# threads/events for server discovery helpers
server_thread: Optional[threading.Thread] = None
server_httpd: Optional[ThreadingHTTPServer] = None
broadcast_stop_event = threading.Event()
discovery_broadcaster: Optional[threading.Thread] = None
discovery_query_listener: Optional[threading.Thread] = None
discovery_query_stop_event = threading.Event()

def start_server() -> None:
    global server_thread, server_httpd, discovery_broadcaster, discovery_query_listener, discovery_query_stop_event, last_connected_server
    if CLIENT_ONLY_BUILD:
        logger.warning("Server start requested, but this build is client-only")
        notify("Server functionality is not available in this build."); return
    if server_thread and server_thread.is_alive():
        logger.debug("Server start requested but server thread is already running")
        return
    with settings_lock:
        port = int(settings.get("server_port", 65432))
        discovery_port = get_discovery_port()
    logger.info("Starting CtrlSpeak server on port %s (discovery %s)", port, discovery_port)
    try:
        server_httpd = ThreadingHTTPServer(("0.0.0.0", port), TranscriptionRequestHandler)
    except OSError as exc:
        logger.error("Server startup failed on port %s: %s", port, exc)
        notify_error("Server startup failed", str(exc)); server_httpd = None; return
    def serve():
        try:
            server_httpd.serve_forever()
        except Exception as exc:
            notify_error("Server stopped", format_exception_details(exc))
    server_thread = threading.Thread(target=serve, daemon=True); server_thread.start()

    broadcast_stop_event.clear(); discovery_query_stop_event.clear()
    discovery_broadcaster = threading.Thread(
        target=manage_discovery_broadcast,
        args=(broadcast_stop_event, discovery_port, port),
        daemon=True,
    )
    discovery_broadcaster.start()
    discovery_query_listener = threading.Thread(
        target=listen_for_discovery_queries,
        args=(discovery_query_stop_event, discovery_port, port),
        daemon=True,
    )
    discovery_query_listener.start()
    last_connected_server = ServerInfo(host=get_advertised_host_ip(), port=port, last_seen=time.time())
    logger.info("CtrlSpeak server listening on port %s", port)
    print(f"CtrlSpeak server listening on port {port}")
    schedule_management_refresh()


def shutdown_server() -> None:
    global server_thread, server_httpd, discovery_broadcaster, discovery_query_listener, broadcast_stop_event, discovery_query_stop_event, last_connected_server
    logger.info("Shutting down CtrlSpeak server")
    broadcast_stop_event.set(); discovery_query_stop_event.set()
    if discovery_broadcaster and discovery_broadcaster.is_alive():
        discovery_broadcaster.join(timeout=1.0)
    discovery_broadcaster = None
    if discovery_query_listener and discovery_query_listener.is_alive():
        discovery_query_listener.join(timeout=1.0)
    discovery_query_listener = None
    if server_httpd is not None:
        try:
            server_httpd.shutdown(); server_httpd.server_close()
        except Exception:
            logger.exception("Failed to shut down server HTTP listener cleanly")
    server_httpd = None
    if server_thread and server_thread.is_alive():
        server_thread.join(timeout=1.0)
    server_thread = None
    broadcast_stop_event = threading.Event(); discovery_query_stop_event = threading.Event()
    last_connected_server = None
    logger.debug("Server resources cleared")
    schedule_management_refresh()


# local wrapper to restore original side-effects
def register_manual_server(host: str, port: int, update_preference: bool = True) -> ServerInfo:
    """Register a server and update discovery registry + last_connected_server + UI, like the original system.py did."""
    global discovery_listener, last_connected_server
    from utils.net_discovery import register_manual_server as _nd_register_manual_server
    server_info = _nd_register_manual_server(host, port, update_preference=update_preference)

    # Original behavior: remember it in our discovery registry if we have a listener
    try:
        if discovery_listener is not None:
            discovery_listener.registry[(server_info.host, server_info.port)] = server_info
    except Exception:
        logger.exception("Failed to update discovery registry with manual server %s:%s", server_info.host, server_info.port)

    # Original behavior: set last_connected_server and refresh UI
    last_connected_server = server_info
    schedule_management_refresh()
    return server_info

# ---------------- Tray ----------------
def on_exit(icon, item):
    global _tray_icon
    _tray_icon = None
    stop_client_listener(); shutdown_server()
    try:
        from utils.gui import request_management_ui_shutdown
        request_management_ui_shutdown()
    except Exception:
        logger.exception("Failed to request management UI shutdown during exit")
    icon.stop()

def open_management_dialog(icon, item):
    from utils.gui import _show_management_window

    enqueue_management_task(_show_management_window, icon)


def open_tray_flyout(icon, item):
    from utils.gui import _show_tray_flyout

    enqueue_management_task(_show_tray_flyout, icon)


def check_for_updates_from_tray(icon, item):
    from utils.gui import _show_management_window

    def _open_and_check() -> None:
        _show_management_window(icon)
        try:
            from utils.gui import management_window as active_window

            if active_window is not None:
                active_window.check_for_updates()
        except Exception:
            logger.exception("Failed to start update check from tray")

    enqueue_management_task(_open_and_check)


def submit_correction_from_tray(icon, item):
    from utils.gui import _show_correction_submission_dialog

    enqueue_management_task(_show_correction_submission_dialog, icon)


def run_tray():
    global _tray_icon
    from utils.gui import ensure_management_ui_thread, run_management_ui_loop, request_management_ui_shutdown
    ensure_management_ui_thread()  # make sure tk_root exists for overlay
    start_client_listener()
    with settings_lock:
        mode = settings.get("mode")
    from utils.transcription_backend import get_runtime_backend_config
    backend = get_runtime_backend_config().backend
    tray_mode = mode if backend == "bundled" else "api"
    try:
        import pystray
    except Exception as exc:
        message = (
            "The tray backend could not start. CtrlSpeak opened its control window instead "
            f"when a desktop was available. Check the X11/AppIndicator prerequisites. {exc}"
        )
        logger.exception("Failed to import the CtrlSpeak tray backend")
        try:
            print(f"CtrlSpeak tray: {message}", file=sys.stderr)
        except Exception:
            logger.debug("Could not print tray backend failure", exc_info=True)
        notify(message, title="CtrlSpeak tray")

        class _ManagementOnlyIcon:
            title = f"CtrlSpeak {APP_VERSION} ({tray_mode})"

            @staticmethod
            def stop() -> None:
                request_management_ui_shutdown()

        fallback_icon = _ManagementOnlyIcon()
        try:
            from utils.gui import _show_management_window

            enqueue_management_task(_show_management_window, fallback_icon)
        except Exception:
            logger.exception("Failed to queue management-only CtrlSpeak window")
        run_management_ui_loop()
        return

    def tray_status_label(_item) -> str:
        phase = transcription_ui_session.phase
        state = "Ready" if phase in {UiPhase.IDLE, UiPhase.SUCCESS} else phase.value.title()
        return f"CtrlSpeak {APP_VERSION} · {state}"

    menu_items = [
        pystray.MenuItem(tray_status_label, lambda _icon, _item: None, enabled=False),
        pystray.MenuItem("Show / hide quick panel", open_tray_flyout, default=True),
        pystray.MenuItem("Open control centre", open_management_dialog),
        pystray.MenuItem("Submit correction…", submit_correction_from_tray),
        pystray.MenuItem(
            "Copy last transcript",
            request_copy_last_transcript_from_tray,
            enabled=has_last_transcript,
        ),
        pystray.MenuItem(
            "Cancel active transcription",
            cancel_active_transcription,
            enabled=is_transcription_busy,
        ),
        pystray.MenuItem("Check for updates", check_for_updates_from_tray),
    ]
    separator = getattr(pystray.Menu, "SEPARATOR", None)
    if separator is not None:
        menu_items.append(separator)
    menu_items.append(pystray.MenuItem("Quit", on_exit))
    icon = pystray.Icon(
        "CtrlSpeak",
        create_icon_image(),
        f"CtrlSpeak {APP_VERSION} ({tray_mode})",
        menu=pystray.Menu(*menu_items),
    )
    _tray_icon = icon
    def _run_icon() -> None:
        tray_failed = False
        try:
            icon.run()
        except Exception as exc:
            tray_failed = True
            logger.exception("Failed to start the CtrlSpeak tray icon")
            if sys.platform.startswith("linux"):
                notify(
                    "The tray icon could not start. CtrlSpeak opened its control window "
                    f"instead. Check the X11/AppIndicator prerequisites. {exc}",
                    title="CtrlSpeak tray",
                )
                try:
                    from utils.gui import _show_management_window

                    enqueue_management_task(_show_management_window, icon)
                except Exception:
                    logger.exception(
                        "Failed to queue management window after tray startup failure"
                    )
        finally:
            if not tray_failed:
                try:
                    request_management_ui_shutdown()
                except Exception:
                    logger.exception("Failed to shut down management UI after tray loop exited")

    thread = threading.Thread(target=_run_icon, name="CtrlSpeakTray", daemon=True)
    thread.start()

    try:
        run_management_ui_loop()
    finally:
        try:
            request_management_ui_shutdown()
        except Exception:
            logger.exception("Failed to request management UI shutdown while leaving tray loop")
        try:
            icon.stop()
        except Exception:
            logger.exception("Failed to stop tray icon")
        thread.join(timeout=2.0)

# ---------------- Startup / single-instance ----------------
def acquire_single_instance_lock() -> bool:
    global instance_lock_handle
    if instance_lock_handle is not None:
        logger.debug("Instance lock is already held by this process")
        return True
    lock_path = get_config_dir() / LOCK_FILENAME
    logger.debug("Attempting to acquire instance lock at %s", lock_path)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Failed to create directory for instance lock at %s", lock_path.parent)
    try:
        handle = open(lock_path, 'a+')
    except OSError as exc:
        logger.warning("Unable to open lock file at %s: %s", lock_path, exc)
        print(f'Unable to open lock file: {exc}'); return False
    try:
        handle.seek(0)
        if sys.platform.startswith('win'):
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                logger.info("Instance lock already held by another process (Windows)")
                handle.close(); return False
        else:
            import fcntl
            try:
                fcntl.lockf(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                logger.info("Instance lock already held by another process (POSIX)")
                handle.close(); return False
        handle.truncate(0); handle.write(str(os.getpid())); handle.flush()
        logger.info("Acquired instance lock with PID %s", os.getpid())
        instance_lock_handle = handle; return True
    except Exception as exc:
        logger.error("Unable to acquire single-instance lock: %s", exc)
        print(f'Unable to acquire single-instance lock: {exc}')
        try:
            handle.close()
        except Exception:
            logger.exception("Failed to close lock file handle after acquisition error")
        return False

def release_single_instance_lock() -> None:
    global instance_lock_handle
    handle = instance_lock_handle
    if handle is None:
        logger.debug("Release instance lock requested but no lock handle is present")
        return
    instance_lock_handle = None
    try:
        if sys.platform.startswith('win'):
            import msvcrt
            try:
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                logger.exception("Failed to release Windows file lock for CtrlSpeak instance")
        else:
            import fcntl
            try:
                fcntl.lockf(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                logger.exception("Failed to release POSIX file lock for CtrlSpeak instance")
    finally:
        try:
            handle.close()
        except Exception:
            logger.exception("Failed to close instance lock file handle")
        try:
            (get_config_dir() / LOCK_FILENAME).unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to remove instance lock file")
        logger.info("Released instance lock")

# ---------------- Uninstall ----------------
def build_uninstall_script(executable: Path, config_dir: Path) -> Path:
    temp_script = Path(tempfile.gettempdir()) / f"ctrlspeak-uninstall-{uuid.uuid4().hex}.cmd"
    commands = [
        "@echo off",
        "timeout /t 2 /nobreak > nul",
        f'if exist "{executable}" del /f /q "{executable}"',
        f'if exist "{config_dir}" rmdir /s /q "{config_dir}"',
        'del "%~f0"'
    ]
    temp_script.write_text("\r\n".join(commands), encoding="utf-8")
    return temp_script

def initiate_self_uninstall(icon: Optional[pystray.Icon]) -> None:
    if not sys.platform.startswith("win"):
        notify("Automatic uninstall is only supported on Windows."); return
    config_dir = get_config_dir()
    exe_path = Path(sys.executable) if getattr(sys, 'frozen', False) else Path(__file__).resolve()
    script_path = build_uninstall_script(exe_path, config_dir)
    try:
        creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        subprocess.Popen(["cmd.exe", "/c", str(script_path)], creationflags=creation_flags)
    except Exception as exc:
        notify_error("Uninstall failed", format_exception_details(exc)); return
    stop_client_listener(); shutdown_server()
    if icon is not None:
        try:
            icon.stop()
        except Exception:
            logger.exception("Failed to stop tray icon during uninstall")
    os._exit(0)

# ---------------- Lifecycle helpers ----------------
def start_discovery_listener() -> None:
    global discovery_listener
    if discovery_listener is not None and discovery_listener.is_alive():
        logger.debug(
            "Discovery listener already active on port %s",
            getattr(discovery_listener, "port", "unknown"),
        )
        return
    with settings_lock:
        port = get_discovery_port()
    logger.info("Starting discovery listener on UDP port %s", port)
    discovery_listener = DiscoveryListener(port); discovery_listener.start()

def stop_discovery_listener() -> None:
    global discovery_listener
    if discovery_listener is None:
        logger.debug("Discovery listener stop requested but no listener was running")
        return
    try:
        discovery_listener.stop()
    except Exception:
        logger.exception("Failed to stop discovery listener")
    finally:
        logger.info("Discovery listener stopped")
        discovery_listener = None

def shutdown_all():
    logger.info("Shutting down all CtrlSpeak services")
    stop_client_listener(); shutdown_server()
    try:
        stop_discovery_listener()
    except Exception:
        logger.exception("Failed to stop discovery listener during shutdown")
    try:
        from utils.gui import request_management_ui_shutdown
        request_management_ui_shutdown()
    except Exception:
        logger.exception("Failed to shut down management UI during shutdown")

# ---------------- CLI ----------------
def parse_cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="CtrlSpeak", add_help=True, description="CtrlSpeak voice control")
    parser.add_argument("--transcribe", metavar="WAV_PATH", help="Transcribe an audio file and print the result")
    parser.add_argument("--uninstall", action="store_true", help="Remove CtrlSpeak and all local data")
    parser.add_argument("--auto-setup", choices=["client", "client_server"], help="Configure CtrlSpeak without prompts")
    parser.add_argument("--force-sendinput", action="store_true", help="Force SendInput-based insertion (debug)")
    parser.add_argument(
        "--backend",
        choices=["bundled", "api"],
        help="Persistently select the bundled model or configured HTTP API backend",
    )
    parser.add_argument(
        "--api-url",
        help="Persist the API base URL (token is intentionally configured via settings or CTRLSPEAK_API_TOKEN)",
    )
    parser.add_argument(
        "--backend-status",
        action="store_true",
        help="Print backend status without disclosing the bearer token and exit",
    )
    parser.add_argument(
        "--download-cuda-only",
        "--setup-cuda",
        action="store_true",
        dest="cuda_only",
        help="Download CUDA runtime assets and exit without launching the UI",
    )
    parser.add_argument("--automation-flow", action="store_true", help="Run the automated end-to-end regression workflow")
    parser.add_argument("--version", action="store_true", dest="show_version", help="Print the CtrlSpeak version and exit")
    parser.add_argument(
        "--health-check-file",
        metavar="PATH",
        help="Write a packaged runtime/version smoke-test result and exit",
    )
    parser.add_argument(
        "--apply-update",
        metavar="TRANSACTION",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--post-update",
        metavar="TRANSACTION_ID",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rollback-notice",
        metavar="TRANSACTION_ID",
        help=argparse.SUPPRESS,
    )
    args, _ = parser.parse_known_args(argv[1:])
    return args


def apply_backend_cli_config(args: argparse.Namespace) -> bool:
    """Apply safe CLI backend overrides and return whether status-only was requested."""
    from utils.transcription_backend import get_backend_status, save_backend_config

    backend_override = getattr(args, "backend", None)
    url_override = getattr(args, "api_url", None)
    if backend_override is not None or url_override is not None:
        with settings_lock:
            backend = backend_override or str(settings.get("transcription_backend") or "bundled")
            api_url = url_override or str(settings.get("api_url") or "http://127.0.0.1:8765")
            token = settings.get("api_token")
            capture_method = str(
                settings.get("feedback_capture_method") or "active_field_on_enter"
            )
        save_backend_config(
            backend=backend,
            api_url=api_url,
            api_token=str(token) if token else None,
            feedback_capture_method=capture_method,
        )
    status_only = bool(getattr(args, "backend_status", False))
    if status_only:
        print(get_backend_status())
    return status_only

def transcribe_cli(target: str) -> int:
    from utils.models import initialize_transcriber, transcribe_audio
    from utils.transcription_backend import ApiBackendError, get_runtime_backend_config
    file_path = Path(target).expanduser()
    if not file_path.is_file():
        print(f"Audio file not found: {file_path}", file=sys.stderr); return 1
    discovery_started = False
    try:
        backend_config = get_runtime_backend_config()
        if backend_config.backend == "bundled":
            with settings_lock:
                mode = settings.get("mode")
            if mode == "client_server":
                if initialize_transcriber() is None:
                    print("Unable to initialize the transcription engine.", file=sys.stderr); return 2
            elif mode == "client":
                start_discovery_listener(); discovery_started = True
                time.sleep(1.0)
                server = get_best_server()
                if server is not None:
                    logger.info(
                        "CLI transcription discovered server %s:%s; skipping local fallback prompt.",
                        server.host,
                        server.port,
                    )
                else:
                    logger.info(
                        "CLI transcription did not discover a server; fallback prompt may still be shown."
                    )
        try:
            text = transcribe_audio(str(file_path), play_feedback=False)
        except ApiBackendError as exc:
            print(f"API transcription failed: {exc}", file=sys.stderr)
            return 4
        if text is None:
            print("Transcription produced no output.", file=sys.stderr); return 3
        print(text)
        return 0
    finally:
        if discovery_started:
            stop_discovery_listener()

def apply_auto_setup(profile: str) -> None:
    global AUTO_MODE, AUTO_MODE_PROFILE
    AUTO_MODE = True; AUTO_MODE_PROFILE = profile
    logger.info("Applying auto-setup profile: %s", profile)
    with settings_lock: settings["mode"] = profile
    save_settings()
