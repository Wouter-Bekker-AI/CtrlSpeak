# -*- coding: utf-8 -*-
from __future__ import annotations

import atexit
import argparse
import ctypes
from ctypes import wintypes
import http.client
import json
import os
import socket
import sys
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Optional, Tuple, Callable, List
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
from tkinter import ttk, messagebox

# ---------------- Core constants / defaults ----------------
CHUNKSIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

APP_VERSION = "0.6.0"
SPLASH_DURATION_MS = 1000
ERROR_LOG_FILENAME = "CtrlSpeak-error.log"
LOCK_FILENAME = "CtrlSpeak.lock"
PROCESSING_SAMPLE_RATE = 44100

CONFIG_FILENAME = "settings.json"
DEFAULT_SETTINGS = {
    "mode": None,                    # "client" | "client_server"
    "server_port": 65432,
    "discovery_port": 54330,
    "preferred_server_host": None,
    "preferred_server_port": None,
}

INSTANCE_PORT = int(os.environ.get("CTRLSPEAK_SINGLE_INSTANCE_PORT", "54329"))

# ---------------- Build flags ----------------
def detect_client_only_build() -> bool:
    try:
        base_dir = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
        if (base_dir / 'client_only.flag').exists():
            return True
    except Exception:
        pass
    return os.environ.get('CTRLSPEAK_CLIENT_ONLY', '0') == '1'

CLIENT_ONLY_BUILD = detect_client_only_build()

# ---------------- Discovery constants ----------------
DISCOVERY_INTERVAL_SECONDS = 5.0
DISCOVERY_ENTRY_TTL = 15.0
SERVER_BROADCAST_SIGNATURE = "CTRLSPEAK_SERVER"

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

recording_temp_dir_name = "temp"
AUTO_MODE = False
AUTO_MODE_PROFILE: Optional[str] = None
_FORCE_SENDINPUT = False

settings_lock = threading.Lock()
settings: Dict[str, object] = {}
server_thread: Optional[threading.Thread] = None
server_httpd: Optional[ThreadingHTTPServer] = None
broadcast_stop_event = threading.Event()
discovery_broadcaster: Optional[threading.Thread] = None
discovery_listener: Optional["DiscoveryListener"] = None
discovery_query_listener: Optional[threading.Thread] = None
discovery_query_stop_event = threading.Event()
local_hostname = socket.gethostname()

# Tk management thread (GUI owns windows; system just queues tasks)
management_ui_thread: Optional[threading.Thread] = None
management_ui_queue: "Queue[tuple[Callable[..., None], tuple, dict]]" = Queue()
tk_root: Optional[tk.Tk] = None

# Windows GUI helpers
user32 = ctypes.windll.user32 if sys.platform.startswith("win") else None
kernel32 = ctypes.windll.kernel32 if sys.platform.startswith("win") else None
psapi = ctypes.windll.psapi if sys.platform.startswith("win") else None

CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
EM_SETSEL = 0x00B1
EM_REPLACESEL = 0x00C2

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_V = 0x56
KEYEVENTF_UNICODE = 0x0004

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# ---------------- Dataclasses ----------------
@dataclass
class ServerInfo:
    host: str
    port: int
    last_seen: float

last_connected_server: Optional[ServerInfo] = None
management_window: Optional["ManagementWindow"] = None  # created in utils.gui

# ---------------- Paths (APPDATA) ----------------
def get_config_dir() -> Path:
    """
    All persistent data goes here:
      %APPDATA%/CtrlSpeak (Windows)
      $XDG_CONFIG_HOME/CtrlSpeak or ~/.config/CtrlSpeak (others)
    Subfolders used:
      models/  cuda/  temp/  (plus settings.json + logs)
    """
    if sys.platform.startswith("win"):
        base_dir = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    config_dir = base_dir / "CtrlSpeak"
    config_dir.mkdir(parents=True, exist_ok=True)
    # Ensure core subfolders exist
    for sub in ("models", "cuda", "temp"):
        (config_dir / sub).mkdir(parents=True, exist_ok=True)
    return config_dir

def get_config_file_path() -> Path:
    return get_config_dir() / CONFIG_FILENAME

def get_temp_dir() -> Path:
    temp_dir = get_config_dir() / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

def create_recording_file_path() -> Path:
    return get_temp_dir() / f"recording-{uuid.uuid4().hex}.wav"

def cleanup_recording_file(path: Optional[Path]) -> None:
    if not path:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass

# ---------------- Settings ----------------
def load_settings() -> Dict[str, object]:
    path = get_config_file_path()
    loaded = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text("utf-8-sig"))
        except Exception as exc:
            print(f"Failed to read settings: {exc}")

    # Merge defaults + loaded into the existing dict object
    with settings_lock:
        settings.clear()
        settings.update(DEFAULT_SETTINGS)
        settings.update(loaded)

    return settings


def save_settings() -> None:
    path = get_config_file_path()
    with settings_lock:
        snapshot = settings.copy()
    try:
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Failed to save settings: {exc}")

# ---------------- Notifications / logging ----------------
def notify(message: str, title: str = "CtrlSpeak") -> None:
    try:
        print(f"{title}: {message}")
    except Exception:
        pass

def format_exception_details(exc: Exception) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()

def write_error_log(context: str, details: str) -> None:
    try:
        log_path = get_config_dir() / ERROR_LOG_FILENAME
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cleaned = details.strip() or "Unknown error"
        entry = "[{}] {}\n{}\n{}\n".format(timestamp, context, cleaned, "-" * 60)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
    except Exception:
        pass

def copy_to_clipboard(text: str) -> None:
    try:
        root = tk.Tk(); root.withdraw()
        root.clipboard_clear(); root.clipboard_append(text); root.update(); root.destroy()
    except Exception:
        pass

def notify_error(context: str, details: str) -> None:
    snippet = details.strip() or "Unknown error"
    message = f"{context}\n\nDetails:\n{snippet}"
    write_error_log(context, snippet)
    copy_to_clipboard(message)
    notify(message, title="CtrlSpeak Error")

# ---------------- Resource helpers ----------------
def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # pyinstaller
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def create_icon_image():
    icon_path = resource_path("icon.ico")
    return Image.open(icon_path)

# ---------------- Management UI pump (GUI thread lives in utils.gui) ----------------
def enqueue_management_task(func: Callable[..., None], *args, **kwargs) -> None:
    try:
        management_ui_queue.put_nowait((func, args, kwargs))
    except Exception:
        pass

def schedule_management_refresh(delay_ms: int = 0) -> None:
    # utils.gui sets tk_root and management_window
    from utils.gui import tk_root, management_window
    if tk_root is None:
        return

    def task() -> None:
        if management_window and management_window.is_open():
            management_window.refresh_status()

    if delay_ms <= 0:
        enqueue_management_task(task)
    else:
        def delayed_task() -> None:
            from utils.gui import tk_root as _root
            if _root is not None:
                _root.after(delay_ms, task)
        enqueue_management_task(delayed_task)

# ---------------- Discovery ----------------
class DiscoveryListener(threading.Thread):
    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port = port
        self.stop_event = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(("", port))
        except OSError:
            self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(1.0)
        self.registry: Dict[Tuple[str, int], ServerInfo] = {}

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                data, addr = self.sock.recvfrom(4096)
            except socket.timeout:
                self._prune(); continue
            except OSError:
                break
            try:
                message = data.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            parts = message.split("|")
            if len(parts) != 3 or parts[0] != SERVER_BROADCAST_SIGNATURE:
                continue
            try:
                host = parts[1]; port = int(parts[2])
            except ValueError:
                continue
            key = (host, port)
            self.registry[key] = ServerInfo(host=host, port=port, last_seen=time.time())

    def _prune(self) -> None:
        now = time.time()
        expired = [key for key, info in self.registry.items() if now - info.last_seen > DISCOVERY_ENTRY_TTL]
        for key in expired:
            self.registry.pop(key, None)

    def get_best_server(self) -> Optional[ServerInfo]:
        self._prune()
        if not self.registry: return None
        return max(self.registry.values(), key=lambda entry: entry.last_seen)

    def clear_registry(self) -> None:
        self.registry.clear()

    def stop(self) -> None:
        self.stop_event.set()
        try: self.sock.close()
        except Exception: pass

def get_preferred_server_settings() -> tuple[Optional[str], Optional[int]]:
    with settings_lock:
        host = settings.get("preferred_server_host")
        port = settings.get("preferred_server_port")
    if isinstance(host, str): host = host.strip() or None
    if isinstance(port, str):
        try: port = int(port)
        except ValueError: port = None
    if host is None or port is None:
        return None, None
    return host, int(port)

def set_preferred_server(host: str, port: int) -> None:
    with settings_lock:
        settings["preferred_server_host"] = host
        settings["preferred_server_port"] = int(port)
    save_settings()

def clear_preferred_server() -> None:
    with settings_lock:
        settings["preferred_server_host"] = None
        settings["preferred_server_port"] = None
    save_settings()

def parse_server_target(value: str) -> tuple[str, int]:
    target = value.strip()
    if not target: raise ValueError("Server address cannot be empty.")
    if target.count(":") == 0:
        host = target; port = int(DEFAULT_SETTINGS["server_port"])
    else:
        host, port_str = target.rsplit(":", 1)
        host = host.strip()
        if not host: raise ValueError("Server host cannot be empty.")
        try: port = int(port_str.strip())
        except ValueError as exc: raise ValueError("Port must be a number.") from exc
    if port <= 0 or port > 65535: raise ValueError("Port must be between 1 and 65535.")
    return host, port

def probe_server(host: str, port: int, timeout: float = 2.0) -> bool:
    conn: Optional[http.client.HTTPConnection] = None
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/ping")
        response = conn.getresponse(); response.read()
        return response.status == 200
    except Exception:
        return False
    finally:
        if conn is not None:
            try: conn.close()
            except Exception: pass

def register_manual_server(host: str, port: int, update_preference: bool = True) -> ServerInfo:
    server_info = ServerInfo(host=host, port=int(port), last_seen=time.time())
    if discovery_listener is not None:
        discovery_listener.registry[(server_info.host, server_info.port)] = server_info
    global last_connected_server
    last_connected_server = server_info
    if update_preference:
        set_preferred_server(server_info.host, server_info.port)
    schedule_management_refresh()
    return server_info

def ensure_preferred_server_registered(probe: bool = False) -> Optional[ServerInfo]:
    host, port = get_preferred_server_settings()
    if host is None or port is None: return None
    if probe and not probe_server(host, port): return None
    return register_manual_server(host, port, update_preference=False)

def send_discovery_query(timeout: float = 1.0) -> None:
    with settings_lock:
        port = int(settings.get("discovery_port", 54330))
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout)
        message = f"{SERVER_BROADCAST_SIGNATURE}_QUERY".encode("utf-8")
        sock.sendto(message, ("255.255.255.255", port))
    except Exception as exc:
        print(f"Discovery query failed: {exc}")
    finally:
        try: sock.close()
        except Exception: pass

def manual_discovery_refresh(wait_time: float = 1.5) -> Optional[ServerInfo]:
    global last_connected_server
    if discovery_listener is None:
        server = ensure_preferred_server_registered(probe=True)
        if server: last_connected_server = server
        schedule_management_refresh()
        return server
    discovery_listener.clear_registry()
    send_discovery_query()
    time.sleep(max(wait_time, 0.5))
    server = discovery_listener.get_best_server()
    if server:
        last_connected_server = server
        if server.host not in {"local", "local-cpu"}:
            set_preferred_server(server.host, server.port)
        schedule_management_refresh()
        return server
    server = ensure_preferred_server_registered(probe=True)
    last_connected_server = server if server else None
    schedule_management_refresh()
    return server

def get_best_server() -> Optional[ServerInfo]:
    if discovery_listener is None:
        return ensure_preferred_server_registered(probe=True)
    server = discovery_listener.get_best_server()
    if server: return server
    return ensure_preferred_server_registered(probe=True)

def get_advertised_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tmp:
            tmp.connect(("8.8.8.8", 80))
            return tmp.getsockname()[0]
    except Exception:
        return "127.0.0.1"

# ---------------- Audio capture + feedback tone ----------------
def generate_fallback_sound() -> Tuple[bytes, Dict[str, int]]:
    duration = 0.5
    t = np.linspace(0.0, duration, int(PROCESSING_SAMPLE_RATE * duration), endpoint=False)
    envelope = np.exp(-3 * t)
    wave_data = 0.2 * np.sin(2 * np.pi * 440 * t) * envelope
    int_data = np.clip(wave_data * 32767, -32767, 32767).astype(np.int16)
    settings_audio = {"channels": 1, "rate": PROCESSING_SAMPLE_RATE, "width": 2}
    return int_data.tobytes(), settings_audio

def load_processing_sound() -> Tuple[bytes, Dict[str, int]]:
    global processing_sound_data, processing_sound_settings
    if processing_sound_data is not None and processing_sound_settings is not None:
        return processing_sound_data, processing_sound_settings
    try:
        # Always look for loading.wav next to main.py (or the frozen exe)
        base_dir = Path(sys.executable if getattr(sys, "frozen", False) else sys.argv[0]).resolve().parent
        sound_path = base_dir / "loading.wav"
        with wave.open(str(sound_path), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            settings_audio = {"channels": wav_file.getnchannels(), "rate": wav_file.getframerate(), "width": wav_file.getsampwidth()}
    except Exception as exc:
        print(f"Failed to load loading.wav: {exc}")
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
        while not processing_sound_stop_event.is_set():
            stream.write(data)
    except Exception as exc:
        print(f"Processing sound playback failed: {exc}")
    finally:
        try:
            if stream is not None:
                stream.stop_stream(); stream.close()
        except Exception:
            pass
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

def record_audio(target_path: Path) -> None:
    pyaudio_instance = pyaudio.PyAudio()
    stream = pyaudio_instance.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)
    frames = []
    try:
        while recording:
            frames.append(stream.read(CHUNKSIZE))
    finally:
        stream.stop_stream(); stream.close(); pyaudio_instance.terminate()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

# ---------------- Text insertion helpers ----------------
class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)] if sys.platform.startswith("win") else []

class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
        ("rcCaret", RECT),
    ] if sys.platform.startswith("win") else []

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", wintypes.LPARAM)] if sys.platform.startswith("win") else []

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", wintypes.LPARAM)] if sys.platform.startswith("win") else []

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)] if sys.platform.startswith("win") else []

class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)] if sys.platform.startswith("win") else []

class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)] if sys.platform.startswith("win") else []

if sys.platform.startswith("win"):
    _send_input = user32.SendInput
    _send_input.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
    _send_input.restype = wintypes.UINT

def set_force_sendinput(flag: bool) -> None:
    global _FORCE_SENDINPUT
    _FORCE_SENDINPUT = bool(flag)

def get_focused_control() -> Optional[int]:
    if not sys.platform.startswith("win"): return None
    try:
        foreground = user32.GetForegroundWindow()
        if not foreground: return None
        thread_id = user32.GetWindowThreadProcessId(foreground, None)
        current_thread_id = kernel32.GetCurrentThreadId()
        attached = False
        if thread_id != current_thread_id:
            if user32.AttachThreadInput(current_thread_id, thread_id, True):
                attached = True
        info = GUITHREADINFO(); info.cbSize = ctypes.sizeof(GUITHREADINFO)
        hwnd_focus = None
        if user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
            hwnd_focus = info.hwndFocus or info.hwndCaret or info.hwndActive
        if attached: user32.AttachThreadInput(current_thread_id, thread_id, False)
        return hwnd_focus or foreground
    except Exception:
        return None

def get_class_name(hwnd: int) -> str:
    if not sys.platform.startswith("win") or not hwnd: return ""
    buffer = ctypes.create_unicode_buffer(256)
    try:
        if user32.GetClassNameW(hwnd, buffer, len(buffer)) == 0: return ""
    except Exception:
        return ""
    return buffer.value.lower()

def get_window_text(hwnd: int) -> str:
    if not sys.platform.startswith("win"): return ""
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    if user32.GetWindowTextW(hwnd, buffer, len(buffer)) == 0: return ""
    return buffer.value

def get_window_process_name(hwnd: int) -> str:
    if not sys.platform.startswith("win"): return ""
    pid = wintypes.DWORD()
    if user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)) == 0: return ""
    access = PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | PROCESS_QUERY_LIMITED_INFORMATION
    process_handle = kernel32.OpenProcess(access, False, pid.value)
    if not process_handle: return ""
    try:
        buffer = ctypes.create_unicode_buffer(260)
        length = psapi.GetModuleFileNameExW(process_handle, None, buffer, len(buffer))
        if length == 0: return ""
        return Path(buffer.value).name.lower()
    except Exception:
        return ""
    finally:
        kernel32.CloseHandle(process_handle)

def is_console_window(hwnd: int) -> bool:
    if not sys.platform.startswith("win") or not hwnd: return False
    class_name = get_class_name(hwnd)
    if class_name in {"consolewindowclass"}: return True
    if "cascadia" in class_name: return True
    process_name = get_window_process_name(hwnd)
    return process_name in {"conhost.exe", "openconsole.exe", "wt.exe", "windowsterminal.exe", "ubuntu.exe", "wsl.exe"}

def open_clipboard() -> bool:
    if not sys.platform.startswith("win"): return False
    for _ in range(5):
        if user32.OpenClipboard(None): return True
        time.sleep(0.01)
    return False

def get_clipboard_text() -> Optional[str]:
    if not sys.platform.startswith("win"): return None
    if not open_clipboard(): return None
    try:
        if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT): return None
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle: return None
        pointer = kernel32.GlobalLock(handle)
        if not pointer: return None
        try: return ctypes.wstring_at(pointer)
        finally: kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()

def set_clipboard_text(text: str) -> bool:
    if not sys.platform.startswith("win"): return False
    if not open_clipboard(): return False
    try:
        user32.EmptyClipboard()
        data = ctypes.create_unicode_buffer(text)
        size = ctypes.sizeof(ctypes.c_wchar) * len(data)
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle: return False
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            kernel32.GlobalFree(handle); return False
        ctypes.memmove(pointer, ctypes.byref(data), size)
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle); return False
        return True
    finally:
        user32.CloseClipboard()

def restore_clipboard_text(previous: Optional[str]) -> None:
    if not sys.platform.startswith("win"): return
    if previous is None:
        if open_clipboard():
            try: user32.EmptyClipboard()
            finally: user32.CloseClipboard()
        return
    set_clipboard_text(previous)

def send_input_key(vk: int, keyup: bool = False) -> bool:
    if not sys.platform.startswith("win"): return False
    flags = KEYEVENTF_KEYUP if keyup else 0
    union = _INPUTUNION()
    union.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
    inp = INPUT(type=INPUT_KEYBOARD, union=union)
    return _send_input(1, ctypes.byref(inp), ctypes.sizeof(inp)) == 1

def ensure_foreground(hwnd: int) -> None:
    if not sys.platform.startswith("win") or not hwnd: return
    try:
        pid = wintypes.DWORD()
        thread_id = user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        current_thread_id = kernel32.GetCurrentThreadId()
        attached = False
        if thread_id and thread_id != current_thread_id:
            if user32.AttachThreadInput(current_thread_id, thread_id, True):
                attached = True
        try:
            user32.SetForegroundWindow(hwnd); user32.SetFocus(hwnd)
        finally:
            if attached: user32.AttachThreadInput(current_thread_id, thread_id, False)
    except Exception:
        pass

def send_unicode_input(text: str) -> bool:
    if not sys.platform.startswith("win"): return False
    hwnd = get_focused_control()
    ensure_foreground(hwnd)
    for ch in text.replace("\n", "\r"):
        code = ord(ch)
        union = _INPUTUNION()
        union.ki = KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE, time=0, dwExtraInfo=0)
        inp = INPUT(type=INPUT_KEYBOARD, union=union)
        if _send_input(1, ctypes.byref(inp), ctypes.sizeof(inp)) != 1: return False
        union = _INPUTUNION()
        union.ki = KEYBDINPUT(wVk=0, wScan=code, dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=0)
        inp = INPUT(type=INPUT_KEYBOARD, union=union)
        if _send_input(1, ctypes.byref(inp), ctypes.sizeof(inp)) != 1: return False
        time.sleep(0.01)
    time.sleep(0.02)
    return True

def try_sendinput_paste(text: str) -> bool:
    if not sys.platform.startswith("win"): return False
    hwnd = get_focused_control(); ensure_foreground(hwnd)
    previous = get_clipboard_text()
    if not set_clipboard_text(text): return False
    try:
        if not (send_input_key(VK_CONTROL) and send_input_key(VK_V) and send_input_key(VK_V, True) and send_input_key(VK_CONTROL, True)):
            return False
        time.sleep(0.15); return True
    finally:
        restore_clipboard_text(previous)

def try_clipboard_paste(text: str) -> bool:
    if not sys.platform.startswith("win"): return False
    previous = get_clipboard_text()
    if not set_clipboard_text(text): return False
    try:
        pyautogui.hotkey('ctrl', 'v'); time.sleep(0.05); return True
    finally:
        restore_clipboard_text(previous)

def try_direct_text_insert(text: str, hwnd: Optional[int] = None) -> bool:
    if not sys.platform.startswith("win"): return False
    if hwnd is None: hwnd = get_focused_control()
    if not hwnd: return False
    class_name = get_class_name(hwnd)
    if not class_name: return False
    try:
        if any(token in class_name for token in ("edit", "richedit", "notepad", "textarea", "tmemo", "scintilla")):
            user32.SendMessageW(hwnd, EM_SETSEL, 0xFFFFFFFF, 0xFFFFFFFF)
            user32.SendMessageW(hwnd, EM_REPLACESEL, True, text)
            return True
    except Exception:
        return False
    return False

def insert_text_into_focus(text: str) -> None:
    if not text: return
    if not sys.platform.startswith("win"):
        pyautogui.write(text); return
    hwnd = get_focused_control()
    console_window = bool(hwnd and is_console_window(hwnd))
    if hwnd and (_FORCE_SENDINPUT or console_window):
        if send_unicode_input(text): return
        if try_sendinput_paste(text): return
    if not console_window and try_direct_text_insert(text, hwnd): return
    if _FORCE_SENDINPUT and not console_window:
        if send_unicode_input(text): return
    if try_sendinput_paste(text): return
    if try_clipboard_paste(text): return
    pyautogui.write(text)

# ---------------- Client keyboard listener ----------------
def on_press(key):
    from utils.models import transcribe_audio  # imported on-demand to avoid circulars
    global recording, recording_thread, recording_file_path
    if not client_enabled: return
    if key == keyboard.Key.ctrl_r and not recording:
        recording = True
        recording_file_path = create_recording_file_path()
        recording_thread = threading.Thread(target=record_audio, args=(recording_file_path,), daemon=True)
        recording_thread.start()

def on_release(key):
    from utils.models import transcribe_audio
    global recording, recording_thread, recording_file_path
    if key == keyboard.Key.ctrl_r:
        if recording:
            recording = False
            if recording_thread:
                recording_thread.join(); recording_thread = None
            path = recording_file_path
            text = None
            try:
                if path and path.exists() and path.stat().st_size > 0:
                    text = transcribe_audio(str(path))
            except Exception as exc:
                notify_error("Transcription failed", format_exception_details(exc)); text = None
            if text:
                try: insert_text_into_focus(text)
                except Exception as exc: notify_error("Text insertion failed", format_exception_details(exc))
            cleanup_recording_file(path)
            recording_file_path = None

def start_client_listener() -> None:
    global listener, client_enabled
    with listener_lock:
        if listener is not None: return
        client_enabled = True
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
    threading.Thread(target=manual_discovery_refresh, daemon=True).start()
    schedule_management_refresh()

def stop_client_listener() -> None:
    global listener, client_enabled, recording_file_path
    with listener_lock:
        client_enabled = False
        if listener is not None:
            listener.stop(); listener = None
    cleanup_recording_file(recording_file_path)
    recording_file_path = None
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
        try: os.remove(temp_path)
        except Exception: pass
        if text is None:
            self.send_error(500, "Transcription failed"); return
        payload = json.dumps({"text": text, "elapsed": time.time() - start_time}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers(); self.wfile.write(payload)
    def log_message(self, format, *args): return

def start_server() -> None:
    global server_thread, server_httpd, discovery_broadcaster, discovery_query_listener, discovery_query_stop_event, last_connected_server
    if CLIENT_ONLY_BUILD:
        notify("Server functionality is not available in this build."); return
    if server_thread and server_thread.is_alive(): return
    with settings_lock:
        port = int(settings.get("server_port", 65432))
        discovery_port = int(settings.get("discovery_port", 54330))
    try:
        server_httpd = ThreadingHTTPServer(("0.0.0.0", port), TranscriptionRequestHandler)
    except OSError as exc:
        notify_error("Server startup failed", str(exc)); server_httpd = None; return
    def serve():
        try: server_httpd.serve_forever()
        except Exception as exc: notify_error("Server stopped", format_exception_details(exc))
    server_thread = threading.Thread(target=serve, daemon=True); server_thread.start()
    broadcast_stop_event.clear(); discovery_query_stop_event.clear()
    discovery_broadcaster = threading.Thread(target=manage_discovery_broadcast,
                                             args=(broadcast_stop_event, discovery_port, port), daemon=True)
    discovery_broadcaster.start()
    discovery_query_listener = threading.Thread(target=listen_for_discovery_queries,
                                                args=(discovery_query_stop_event, discovery_port, port), daemon=True)
    discovery_query_listener.start()
    last_connected_server = ServerInfo(host=get_advertised_host_ip(), port=port, last_seen=time.time())
    print(f"CtrlSpeak server listening on port {port}")
    schedule_management_refresh()

def manage_discovery_broadcast(stop_event: threading.Event, port: int, server_port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    try:
        while not stop_event.is_set():
            host_ip = get_advertised_host_ip()
            message = f"{SERVER_BROADCAST_SIGNATURE}|{host_ip}|{server_port}".encode("utf-8")
            try: sock.sendto(message, ("255.255.255.255", port))
            except Exception as exc: print(f"Discovery broadcast failed: {exc}")
            stop_event.wait(DISCOVERY_INTERVAL_SECONDS)
    finally:
        sock.close()

def listen_for_discovery_queries(stop_event: threading.Event, port: int, server_port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except OSError:
        pass
    try:
        sock.bind(("", port)); sock.settimeout(1.0)
    except OSError as exc:
        print(f"Failed to bind discovery query listener: {exc}"); sock.close(); return
    try:
        while not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                message = data.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            if message != f"{SERVER_BROADCAST_SIGNATURE}_QUERY": continue
            host_ip = get_advertised_host_ip()
            response = f"{SERVER_BROADCAST_SIGNATURE}|{host_ip}|{server_port}".encode("utf-8")
            try: sock.sendto(response, addr)
            except Exception as exc: print(f"Failed to respond to discovery query: {exc}")
    finally:
        try: sock.close()
        except Exception: pass

def shutdown_server() -> None:
    global server_thread, server_httpd, discovery_broadcaster, discovery_query_listener, broadcast_stop_event, discovery_query_stop_event, last_connected_server
    broadcast_stop_event.set(); discovery_query_stop_event.set()
    if discovery_broadcaster and discovery_broadcaster.is_alive(): discovery_broadcaster.join(timeout=1.0)
    discovery_broadcaster = None
    if discovery_query_listener and discovery_query_listener.is_alive(): discovery_query_listener.join(timeout=1.0)
    discovery_query_listener = None
    if server_httpd is not None:
        try: server_httpd.shutdown(); server_httpd.server_close()
        except Exception: pass
    server_httpd = None
    if server_thread and server_thread.is_alive(): server_thread.join(timeout=1.0)
    server_thread = None
    broadcast_stop_event = threading.Event(); discovery_query_stop_event = threading.Event()
    last_connected_server = None
    schedule_management_refresh()

# ---------------- Tray ----------------
def on_exit(icon, item):
    stop_client_listener(); shutdown_server(); icon.stop()

def open_management_dialog(icon, item):
    from utils.gui import ensure_management_ui_thread, _show_management_window
    ensure_management_ui_thread()
    enqueue_management_task(_show_management_window, icon)

def run_tray():
    start_client_listener()
    with settings_lock:
        mode = settings.get("mode")
    menu_items = [
        pystray.MenuItem("Manage CtrlSpeak", open_management_dialog),
        pystray.MenuItem("Quit", on_exit),
    ]
    icon = pystray.Icon("CtrlSpeak", create_icon_image(), f"CtrlSpeak ({mode})", menu=pystray.Menu(*menu_items))
    icon.run()

# ---------------- Startup / single-instance ----------------
def acquire_single_instance_lock() -> bool:
    global instance_lock_handle
    lock_path = get_config_dir() / LOCK_FILENAME
    try: lock_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception: pass
    try:
        handle = open(lock_path, 'a+')
    except OSError as exc:
        print(f'Unable to open lock file: {exc}'); return False
    try:
        handle.seek(0)
        if sys.platform.startswith('win'):
            import msvcrt
            try: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                handle.close(); return False
        else:
            import fcntl
            try: fcntl.lockf(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close(); return False
        handle.truncate(0); handle.write(str(os.getpid())); handle.flush()
        instance_lock_handle = handle; return True
    except Exception as exc:
        print(f'Unable to acquire single-instance lock: {exc}')
        try: handle.close()
        except Exception: pass
        return False

def release_single_instance_lock() -> None:
    global instance_lock_handle
    handle = instance_lock_handle
    if handle is None: return
    instance_lock_handle = None
    try:
        if sys.platform.startswith('win'):
            import msvcrt
            try: handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception: pass
        else:
            import fcntl
            try: fcntl.lockf(handle.fileno(), fcntl.LOCK_UN)
            except Exception: pass
    finally:
        try: handle.close()
        except Exception: pass
        try: (get_config_dir() / LOCK_FILENAME).unlink(missing_ok=True)
        except Exception: pass

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
    shutdown_all()
    if icon is not None:
        try: icon.stop()
        except Exception: pass
    os._exit(0)

# ---------------- Lifecycle helpers ----------------
def start_discovery_listener() -> None:
    global discovery_listener
    with settings_lock:
        port = int(settings.get("discovery_port", 54330))
    discovery_listener = DiscoveryListener(port); discovery_listener.start()

def shutdown_all():
    stop_client_listener(); shutdown_server()
    if discovery_listener: discovery_listener.stop()

# ---------------- CLI ----------------
def parse_cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="CtrlSpeak", add_help=True, description="CtrlSpeak voice control")
    parser.add_argument("--transcribe", metavar="WAV_PATH", help="Transcribe an audio file and print the result")
    parser.add_argument("--uninstall", action="store_true", help="Remove CtrlSpeak and all local data")
    parser.add_argument("--auto-setup", choices=["client", "client_server"], help="Configure CtrlSpeak without prompts")
    parser.add_argument("--force-sendinput", action="store_true", help="Force SendInput-based insertion (debug)")
    args, _ = parser.parse_known_args(argv[1:])
    return args

def transcribe_cli(target: str) -> int:
    from utils.models import initialize_transcriber, transcribe_audio
    file_path = Path(target).expanduser()
    if not file_path.is_file():
        print(f"Audio file not found: {file_path}", file=sys.stderr); return 1
    with settings_lock:
        mode = settings.get("mode")
    if mode == "client_server":
        if initialize_transcriber() is None:
            print("Unable to initialize the transcription engine.", file=sys.stderr); return 2
    elif mode == "client":
        time.sleep(1.0)
    text = transcribe_audio(str(file_path), play_feedback=False)
    if text is None:
        print("Transcription produced no output.", file=sys.stderr); return 3
    print(text); return 0

def apply_auto_setup(profile: str) -> None:
    global AUTO_MODE, AUTO_MODE_PROFILE
    AUTO_MODE = True; AUTO_MODE_PROFILE = profile
    with settings_lock: settings["mode"] = profile
    save_settings()
