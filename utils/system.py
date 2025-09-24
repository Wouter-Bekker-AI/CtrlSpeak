# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime

import atexit
import argparse
import http.client
import json
import os
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
import pyautogui
import pystray
from PIL import Image
from pynput import keyboard
import tkinter as tk
from collections import deque

from utils.config_paths import (
    settings, settings_lock, load_settings, save_settings,
    get_config_dir, get_config_file_path, get_temp_dir,
    create_recording_file_path, cleanup_recording_file, resource_path,
    asset_path, get_logger, get_logs_dir,
)


logger = get_logger(__name__)


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

    for env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        current = os.environ.get(env_name)
        if not current or not Path(current).exists():
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
APP_VERSION = "0.6.0"
SPLASH_DURATION_MS = 1000
ERROR_LOG_FILENAME = "CtrlSpeak-error.log"
LOCK_FILENAME = "CtrlSpeak.lock"
PROCESSING_SAMPLE_RATE = 44100

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
listener: Optional[keyboard.Listener] = None
recording_file_path: Optional[Path] = None
listener_lock = threading.Lock()
client_enabled = True

instance_lock_handle: Optional[object] = None
processing_sound_thread: Optional[threading.Thread] = None
processing_sound_stop_event = threading.Event()
processing_sound_data: Optional[bytes] = None
processing_sound_settings: Optional[Dict[str, int]] = None
_ready_sound_lock = threading.Lock()
_ready_sound_played = False

recording_temp_dir_name = "temp"
AUTO_MODE = False
AUTO_MODE_PROFILE: Optional[str] = None

# Tk management thread (GUI owns windows; system just queues tasks)
management_ui_thread: Optional[threading.Thread] = None
management_ui_queue: "Queue[tuple[Callable[..., None], tuple, dict]]" = Queue()
tk_root: Optional[tk.Tk] = None
management_window: Optional["ManagementWindow"] = None  # created in utils.gui

if TYPE_CHECKING:
    from utils.gui import ManagementWindow

# ---------------- IMPORTS from new split modules (and re-exports) ----------------

# Win32 text insertion / clipboard
from utils.winio import (
    insert_text_into_focus, set_force_sendinput, is_console_window,
    set_clipboard_text,
)

# LAN discovery (single source of truth for ServerInfo)
from utils.net_discovery import (
    ServerInfo, DiscoveryListener,
    get_preferred_server_settings, set_preferred_server, clear_preferred_server,
    parse_server_target, probe_server, register_manual_server,
    ensure_preferred_server_registered, send_discovery_query,
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




def ui_show_lockout_window(message: str) -> None:
    """Display (or update) the first-run lockout window."""
    try:
        from utils.gui import (
            ensure_management_ui_thread,
            show_lockout_window,
            is_management_ui_thread,
        )

        ensure_management_ui_thread()
        if is_management_ui_thread():
            show_lockout_window(message)
        else:
            enqueue_management_task(show_lockout_window, message)
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


def copy_to_clipboard(text: str) -> None:
    try:
        if not set_clipboard_text(text):
            logger.warning("Failed to stage clipboard text")
    except Exception:
        logger.exception("Failed to copy text to clipboard")

def notify_error(context: str, details: str) -> None:
    snippet = (details or "").strip() or "Unknown error"
    message = f"{context}\n\nDetails:\n{snippet}"
    write_error_log(context, snippet)
    copy_to_clipboard(message)
    notify(message, title="CtrlSpeak Error")


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
    icon_path = asset_path("icon.ico")
    return Image.open(icon_path)

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
    processing_sound_data = frames
    processing_sound_settings = settings_audio
    return processing_sound_data, processing_sound_settings

def _processing_sound_loop():
    data, settings_audio = load_processing_sound()
    pa_instance = pyaudio.PyAudio()
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
        pa_instance.terminate()


def start_processing_feedback():
    global processing_sound_thread
    if processing_sound_thread and processing_sound_thread.is_alive():
        return
    processing_sound_stop_event.clear()
    processing_sound_thread = threading.Thread(target=_processing_sound_loop, daemon=True)
    processing_sound_thread.start()

def stop_processing_feedback():
    global processing_sound_thread
    processing_sound_stop_event.set()
    if processing_sound_thread and processing_sound_thread.is_alive():
        processing_sound_thread.join(timeout=1.0)
    processing_sound_thread = None


def play_model_ready_sound_once() -> None:
    global _ready_sound_played
    with _ready_sound_lock:
        if _ready_sound_played:
            return
        _ready_sound_played = True

    def _worker() -> None:
        pa_instance = None
        stream = None
        try:
            data, settings_audio = load_processing_sound()
            pa_instance = pyaudio.PyAudio()
            stream = pa_instance.open(
                format=pyaudio.get_format_from_width(settings_audio["width"]),
                channels=settings_audio["channels"],
                rate=settings_audio["rate"],
                output=True,
            )
            stream.write(data)
        except Exception:
            logger.exception("Failed to play model ready sound")
        finally:
            try:
                if stream is not None:
                    stream.stop_stream(); stream.close()
            except Exception:
                logger.exception("Failed to close ready sound stream cleanly")
            if pa_instance is not None:
                try:
                    pa_instance.terminate()
                except Exception:
                    logger.exception("Failed to terminate PyAudio after ready sound")

    threading.Thread(target=_worker, name="CtrlSpeakReadySound", daemon=True).start()


def list_input_audio_devices() -> List[Tuple[str, str]]:
    """Return a list of (device_name, display_label) for input-capable devices."""
    devices: List[Tuple[str, str]] = []
    pa_instance: Optional[pyaudio.PyAudio] = None
    try:
        pa_instance = pyaudio.PyAudio()
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
        return []
    finally:
        if pa_instance is not None:
            try:
                pa_instance.terminate()
            except Exception:
                logger.exception("Failed to terminate PyAudio after device enumeration")
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


def record_audio(target_path: Path) -> None:
    pyaudio_instance = pyaudio.PyAudio()
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
    try:
        stream = pyaudio_instance.open(**stream_kwargs)
    except Exception:
        if "input_device_index" in stream_kwargs:
            logger.exception("Failed to open preferred input device; falling back to system default")
            stream_kwargs.pop("input_device_index", None)
            stream = pyaudio_instance.open(**stream_kwargs)
        else:
            raise
    frames = []
    try:
        while recording:
            _chunk = stream.read(CHUNKSIZE)
            frames.append(_chunk)
            _push_waveform_bytes(_chunk)

    finally:
        stream.stop_stream(); stream.close(); pyaudio_instance.terminate()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

# ---------------- Client keyboard listener ----------------
def on_press(key):
    global recording, recording_thread, recording_file_path
    if not client_enabled:
        return
    if key == keyboard.Key.ctrl_r and not recording:
        recording = True
        recording_file_path = create_recording_file_path()
        recording_thread = threading.Thread(target=record_audio, args=(recording_file_path,), daemon=True)
        recording_thread.start()
        try:
            from utils.gui import show_waveform_overlay
            enqueue_management_task(show_waveform_overlay, lambda: get_recent_waveform(500))
        except Exception:
            logger.exception("Failed to show waveform overlay while recording")


def on_release(key):
    from utils.models import transcribe_audio
    global recording, recording_thread, recording_file_path
    if key == keyboard.Key.ctrl_r:
        if recording:
            recording = False
            if recording_thread:
                recording_thread.join(); recording_thread = None
            path = recording_file_path
            # switch overlay into “processing” mode
            try:
                from utils.gui import set_waveform_processing
                enqueue_management_task(set_waveform_processing, "Processing…")
            except Exception:
                logger.exception("Failed to switch waveform overlay to processing mode")
            # START the loading sound so GUI gets live levels + waveform
            try:
                start_processing_feedback()
            except Exception:
                logger.exception("Failed to start processing feedback loop")
            text = None
            try:
                if path and path.exists() and path.stat().st_size > 0:
                    text = transcribe_audio(str(path))
            except Exception as exc:
                notify_error("Transcription failed", format_exception_details(exc)); text = None
            if text:
                try: insert_text_into_focus(text)
                except Exception as exc: notify_error("Text insertion failed", format_exception_details(exc))
            try:
                stop_processing_feedback()
            except Exception:
                logger.exception("Failed to stop processing feedback loop")
            try:
                from utils.gui import hide_waveform_overlay
                enqueue_management_task(hide_waveform_overlay)
            except Exception:
                logger.exception("Failed to hide waveform overlay")
            cleanup_recording_file(path)
            recording_file_path = None

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
    with listener_lock:
        if listener is not None: return
        client_enabled = True
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
    threading.Thread(target=_refresh_best_server_async, daemon=True).start()
    schedule_management_refresh()

def stop_client_listener() -> None:
    global listener, client_enabled, recording, recording_thread, recording_file_path
    thread_to_join: Optional[threading.Thread] = None
    cleanup_path: Optional[Path] = None
    should_hide_waveform = False
    with listener_lock:
        client_enabled = False
        if listener is not None:
            listener.stop(); listener = None
        if recording:
            recording = False
            should_hide_waveform = True
        if recording_thread is not None:
            thread_to_join = recording_thread
            recording_thread = None
            should_hide_waveform = True
        if recording_file_path is not None:
            cleanup_path = recording_file_path
            recording_file_path = None
            should_hide_waveform = True
    if thread_to_join:
        thread_to_join.join()
    if should_hide_waveform:
        try:
            from utils.gui import hide_waveform_overlay
            enqueue_management_task(hide_waveform_overlay)
        except Exception:
            logger.exception("Failed to hide waveform overlay when stopping listener")
    if cleanup_path is not None:
        cleanup_recording_file(cleanup_path)
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
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
        discovery_port = int(settings.get("discovery_port", 54330))
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
    stop_client_listener(); shutdown_server()
    try:
        from utils.gui import request_management_ui_shutdown
        request_management_ui_shutdown()
    except Exception:
        logger.exception("Failed to request management UI shutdown during exit")
    icon.stop()

def open_management_dialog(icon, item):
    from utils.gui import ensure_management_ui_thread, _show_management_window
    ensure_management_ui_thread()
    enqueue_management_task(_show_management_window, icon)

def run_tray():
    from utils.gui import ensure_management_ui_thread, run_management_ui_loop, request_management_ui_shutdown
    ensure_management_ui_thread()  # make sure tk_root exists for overlay
    start_client_listener()
    with settings_lock:
        mode = settings.get("mode")
    menu_items = [
        pystray.MenuItem("Manage CtrlSpeak", open_management_dialog),
        pystray.MenuItem("Quit", on_exit),
    ]
    icon = pystray.Icon("CtrlSpeak", create_icon_image(), f"CtrlSpeak ({mode})", menu=pystray.Menu(*menu_items))
    def _run_icon() -> None:
        try:
            icon.run()
        finally:
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
        port = int(settings.get("discovery_port", 54330))
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
    parser.add_argument("--setup-cuda", action="store_true", help="Prepare CUDA runtime support without launching the UI")
    parser.add_argument("--automation-flow", action="store_true", help="Run the automated end-to-end regression workflow")
    args, _ = parser.parse_known_args(argv[1:])
    return args

def transcribe_cli(target: str) -> int:
    from utils.models import initialize_transcriber, transcribe_audio
    file_path = Path(target).expanduser()
    if not file_path.is_file():
        print(f"Audio file not found: {file_path}", file=sys.stderr); return 1
    discovery_started = False
    try:
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
        text = transcribe_audio(str(file_path), play_feedback=False)
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

