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
from typing import Dict, Optional, Tuple, Callable
import subprocess
import uuid

import ctranslate2
import traceback
import numpy as np
import pyaudio
import pyautogui
import pystray
from PIL import Image, ImageTk
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from pynput import keyboard
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, messagebox

CHUNKSIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
MODEL_NAME = os.environ.get("CTRLSPEAK_MODEL", "large-v3")
DEVICE_PREFERENCE = os.environ.get("CTRLSPEAK_DEVICE", "auto").lower()
COMPUTE_TYPE_OVERRIDE = os.environ.get("CTRLSPEAK_COMPUTE_TYPE")
MODEL_ROOT_OVERRIDE = os.environ.get("CTRLSPEAK_MODEL_DIR")
MODEL_REPO_OVERRIDE = os.environ.get("CTRLSPEAK_MODEL_REPO")
INSTANCE_PORT = int(os.environ.get("CTRLSPEAK_SINGLE_INSTANCE_PORT", "54329"))
CONFIG_FILENAME = "settings.json"
DEFAULT_SETTINGS = {
    "mode": None,
    "server_port": 65432,
    "discovery_port": 54330,
    "preferred_server_host": None,
    "preferred_server_port": None,
}
APP_VERSION = "0.5.10"
SPLASH_DURATION_MS = 1000
ERROR_LOG_FILENAME = "CtrlSpeak-error.log"
LOCK_FILENAME = "CtrlSpeak.lock"
PROCESSING_SAMPLE_RATE = 44100

lock_file_path: Optional[Path] = None
discovery_query_listener: Optional[threading.Thread] = None
discovery_query_stop_event = threading.Event()


def detect_client_only_build() -> bool:
    try:
        base_dir = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
        if (base_dir / 'client_only.flag').exists():
            return True
    except Exception:
        pass
    return os.environ.get('CTRLSPEAK_CLIENT_ONLY', '0') == '1'


CLIENT_ONLY_BUILD = detect_client_only_build()
DISCOVERY_INTERVAL_SECONDS = 5.0
DISCOVERY_ENTRY_TTL = 15.0
SERVER_BROADCAST_SIGNATURE = "CTRLSPEAK_SERVER"

recording = False
recording_thread: Optional[threading.Thread] = None
listener: Optional[keyboard.Listener] = None
recording_file_path: Optional[Path] = None
listener_lock = threading.Lock()
client_enabled = True
model_lock = threading.Lock()
whisper_model: Optional[WhisperModel] = None
model_ready = threading.Event()
warned_cuda_unavailable = False
instance_lock_handle: Optional[object] = None
processing_sound_thread: Optional[threading.Thread] = None
processing_sound_stop_event = threading.Event()
processing_sound_data: Optional[bytes] = None
processing_sound_settings: Optional[Dict[str, int]] = None
cuda_paths_initialized = False
recording_temp_dir_name = "temp"


AUTO_MODE = False
AUTO_MODE_PROFILE: Optional[str] = None


FORCE_SENDINPUT = False


def get_temp_dir() -> Path:
    temp_dir = get_config_dir() / recording_temp_dir_name
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
        notify("Automatic uninstall is only supported on Windows.")
        return
    config_dir = get_config_dir()
    exe_path = Path(sys.executable) if getattr(sys, 'frozen', False) else Path(__file__).resolve()
    script_path = build_uninstall_script(exe_path, config_dir)
    try:
        creation_flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        subprocess.Popen(["cmd.exe", "/c", str(script_path)], creationflags=creation_flags)
    except Exception as exc:
        notify_error("Uninstall failed", format_exception_details(exc))
        return
    shutdown_all()
    if icon is not None:
        try:
            icon.stop()
        except Exception:
            pass
    os._exit(0)


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

management_ui_thread: Optional[threading.Thread] = None
management_ui_queue: "Queue[tuple[Callable[..., None], tuple, dict]]" = Queue()
tk_root: Optional[tk.Tk] = None
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
EM_SETSEL = 0x00B1
EM_REPLACESEL = 0x00C2
WM_SETTEXT = 0x000C


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", RECT),
    ]


management_window: Optional["ManagementWindow"] = None

def get_preferred_server_settings() -> tuple[Optional[str], Optional[int]]:
    with settings_lock:
        host = settings.get("preferred_server_host")
        port = settings.get("preferred_server_port")
    if isinstance(host, str):
        host = host.strip() or None
    if isinstance(port, str):
        try:
            port = int(port)
        except ValueError:
            port = None
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



def probe_server(host: str, port: int, timeout: float = 2.0) -> bool:
    conn: Optional[http.client.HTTPConnection] = None
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/ping")
        response = conn.getresponse()
        response.read()
        return response.status == 200
    except Exception:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass



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
    if host is None or port is None:
        return None
    if probe and not probe_server(host, port):
        return None
    return register_manual_server(host, port, update_preference=False)



def parse_server_target(value: str) -> tuple[str, int]:
    target = value.strip()
    if not target:
        raise ValueError("Server address cannot be empty.")
    if target.count(":") == 0:
        host = target
        port = int(DEFAULT_SETTINGS["server_port"])
    else:
        host, port_str = target.rsplit(":", 1)
        host = host.strip()
        if not host:
            raise ValueError("Server host cannot be empty.")
        try:
            port = int(port_str.strip())
        except ValueError as exc:
            raise ValueError("Port must be a number.") from exc
    if port <= 0 or port > 65535:
        raise ValueError("Port must be between 1 and 65535.")
    return host, port


def enqueue_management_task(func: Callable[..., None], *args, **kwargs) -> None:
    try:
        management_ui_queue.put_nowait((func, args, kwargs))
    except Exception:
        pass

def schedule_management_refresh(delay_ms: int = 0) -> None:
    if tk_root is None:
        return
    def task() -> None:
        if management_window and management_window.is_open():
            management_window.refresh_status()
    if delay_ms <= 0:
        enqueue_management_task(task)
    else:
        def delayed_task() -> None:
            if tk_root is not None:
                tk_root.after(delay_ms, task)
        enqueue_management_task(delayed_task)


class CancelledDownload(Exception):
    pass


@dataclass
class ServerInfo:
    host: str
    port: int
    last_seen: float


last_connected_server: Optional[ServerInfo] = None


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
                self._prune()
                continue
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
                host = parts[1]
                port = int(parts[2])
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
        if not self.registry:
            return None
        return max(self.registry.values(), key=lambda entry: entry.last_seen)

    def clear_registry(self) -> None:
        self.registry.clear()

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.sock.close()
        except Exception:
            pass

def get_config_dir() -> Path:
    if sys.platform.startswith("win"):
        base_dir = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    config_dir = base_dir / "CtrlSpeak"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_file_path() -> Path:
    return get_config_dir() / CONFIG_FILENAME


def load_settings() -> Dict[str, object]:
    global settings
    config_path = get_config_file_path()
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text("utf-8-sig"))
        except Exception as exc:
            print(f"Failed to read settings: {exc}")
            data = DEFAULT_SETTINGS.copy()
    else:
        data = DEFAULT_SETTINGS.copy()
    for key, value in DEFAULT_SETTINGS.items():
        data.setdefault(key, value)
    with settings_lock:
        settings = data
    return data


def save_settings() -> None:
    config_path = get_config_file_path()
    with settings_lock:
        snapshot = settings.copy()
    try:
        config_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Failed to save settings: {exc}")


def notify(message: str, title: str = "CtrlSpeak") -> None:
    try:
        print(f"{title}: {message}")
    except Exception:
        print(f"{title}: {message}")


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
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
    except Exception:
        pass


def notify_error(context: str, details: str) -> None:
    snippet = details.strip() or "Unknown error"
    message = f"{context}\n\nDetails:\n{snippet}"
    write_error_log(context, snippet)
    copy_to_clipboard(message)
    notify(message, title="CtrlSpeak Error")

def show_splash_screen(duration_ms: int = SPLASH_DURATION_MS) -> None:
    try:
        root = tk.Tk()
    except tk.TclError:
        return
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(background="#1a1a1a")
    width, height = 260, 240
    try:
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
    except Exception:
        screen_w, screen_h = 800, 600
    pos_x = int((screen_w - width) / 2)
    pos_y = int((screen_h - height) / 2)
    root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
    try:
        base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        icon_path = base_dir / "icon.ico"
        image = Image.open(icon_path)
        image.thumbnail((160, 160), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(root, image=photo, background="#1a1a1a")
        label.image = photo
        label.pack(pady=(32, 12))
    except Exception:
        tk.Label(root, text="CtrlSpeak", font=("Segoe UI", 18, "bold"), fg="#ffffff", bg="#1a1a1a").pack(pady=(40, 20))
    tk.Label(root, text="Loading...", font=("Segoe UI", 11), fg="#dddddd", bg="#1a1a1a").pack()
    root.after(duration_ms, root.destroy)
    root.mainloop()


def prompt_initial_mode() -> Optional[str]:
    if AUTO_MODE_PROFILE:
        return AUTO_MODE_PROFILE
    result = {"mode": None}

    def choose(mode: str) -> None:
        result["mode"] = mode
        root.destroy()

    root = tk.Tk()
    root.title("Welcome to CtrlSpeak")
    root.geometry("420x260")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    title = tk.Label(root, text="Welcome to CtrlSpeak", font=("Segoe UI", 14, "bold"))
    title.pack(pady=(18, 8))

    message = (
        "It looks like this is the first time CtrlSpeak is running on this computer.\n\n"
        "Choose how you would like to use it:"
    )
    tk.Message(root, text=message, width=360, font=("Segoe UI", 10)).pack(pady=(0, 12))

    button_frame = tk.Frame(root)
    button_frame.pack(pady=(0, 12))

    def make_button(text: str, desc: str, mode_value: str) -> None:
        frame = tk.Frame(button_frame, relief=tk.RIDGE, borderwidth=1)
        frame.pack(padx=8, pady=6, fill=tk.X)
        tk.Label(frame, text=text, font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, padx=8, pady=(6, 0))
        tk.Message(frame, text=desc, width=320).pack(anchor=tk.W, padx=8)
        tk.Button(frame, text="Select", command=lambda: choose(mode_value)).pack(padx=8, pady=(0, 8), anchor=tk.E)

    make_button(
        "Client + Server",
        "Use this computer for local transcription and share it with other CtrlSpeak clients on the network.",
        "client_server",
    )
    make_button(
        "Client Only",
        "Send recordings to another CtrlSpeak server on your local network for transcription.",
        "client",
    )

    def cancel() -> None:
        root.destroy()

    tk.Button(root, text="Quit", command=cancel).pack(pady=(0, 12))
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()
    return result["mode"]


def ensure_mode_selected() -> None:
    with settings_lock:
        current_mode = settings.get("mode")
    if CLIENT_ONLY_BUILD:
        if current_mode != "client":
            with settings_lock:
                settings["mode"] = "client"
            save_settings()
        return
    if current_mode == "client":
        with settings_lock:
            settings["mode"] = "client_server"
        save_settings()
        current_mode = "client_server"
    if current_mode == "client_server":
        return
    choice = prompt_initial_mode()
    if not choice:
        notify("CtrlSpeak cannot continue without selecting a mode.")
        sys.exit(0)
    with settings_lock:
        settings["mode"] = choice
    save_settings()

def prompt_enable_local_server() -> bool:
    message = ("CtrlSpeak cannot reach a server on this network.\n\n"
               "Would you like to enable the server on this computer?")
    try:
        choice = pyautogui.confirm(text=message, title="CtrlSpeak", buttons=["Install Server", "Cancel"])
    except Exception:
        print(message)
        choice = None
    return choice == "Install Server"


def handle_missing_server(file_path: str, play_feedback: bool) -> Optional[str]:
    if CLIENT_ONLY_BUILD:
        notify("No CtrlSpeak server found. Using local CPU transcription.")
        return transcribe_local(file_path, play_feedback=play_feedback, allow_client=True, preferred_device="cpu")
    if not prompt_enable_local_server():
        notify("No CtrlSpeak server is available, and the recording was not transcribed.")
        return None
    with settings_lock:
        settings["mode"] = "client_server"
    save_settings()
    model = initialize_transcriber(force=True)
    if model is None:
        return None
    start_server()
    return transcribe_local(file_path, play_feedback=play_feedback)




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
        try:
            sock.close()
        except Exception:
            pass


def manual_discovery_refresh(wait_time: float = 1.5) -> Optional[ServerInfo]:
    global last_connected_server
    if discovery_listener is None:
        server = ensure_preferred_server_registered(probe=True)
        if server:
            last_connected_server = server
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
    if server:
        last_connected_server = server
    else:
        last_connected_server = None
    schedule_management_refresh()
    return server


def get_cuda_dll_dirs() -> list[Path]:
    base_dirs = [
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
    ]
    bundle_base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    base_dirs.append(bundle_base / "nvidia" / "cudnn" / "bin")
    base_dirs.append(bundle_base / "nvidia" / "cublas" / "bin")
    return base_dirs


def configure_cuda_paths() -> None:
    global cuda_paths_initialized
    if cuda_paths_initialized:
        return
    path_var = os.environ.get("PATH", "")
    for dll_dir in get_cuda_dll_dirs():
        if dll_dir.exists():
            dir_str = str(dll_dir)
            try:
                os.add_dll_directory(dir_str)
            except (AttributeError, FileNotFoundError, OSError):
                pass
            if dir_str not in path_var:
                path_var = dir_str + os.pathsep + path_var
    os.environ["PATH"] = path_var
    cuda_paths_initialized = True


def acquire_single_instance_lock() -> bool:
    global instance_lock_handle, lock_file_path
    with settings_lock:
        lock_path = get_config_dir() / LOCK_FILENAME
    lock_file_path = lock_path
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        handle = open(lock_path, 'a+')
    except OSError as exc:
        print(f'Unable to open lock file: {exc}')
        return False
    try:
        handle.seek(0)
        if sys.platform.startswith('win'):
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                handle.close()
                return False
        else:
            import fcntl
            try:
                fcntl.lockf(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                return False
        handle.truncate(0)
        handle.write(str(os.getpid()))
        handle.flush()
        instance_lock_handle = handle
        return True
    except Exception as exc:
        print(f'Unable to acquire single-instance lock: {exc}')
        try:
            handle.close()
        except Exception:
            pass
        return False


def release_single_instance_lock() -> None:
    global instance_lock_handle, lock_file_path
    handle = instance_lock_handle
    if handle is None:
        return
    instance_lock_handle = None
    try:
        if sys.platform.startswith('win'):
            import msvcrt
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
        else:
            import fcntl
            try:
                fcntl.lockf(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
    finally:
        try:
            handle.close()
        except Exception:
            pass
        if lock_file_path is not None:
            try:
                lock_file_path.unlink(missing_ok=True)
            except Exception:
                pass
            lock_file_path = None


def model_repo_id() -> str:
    if MODEL_REPO_OVERRIDE:
        return MODEL_REPO_OVERRIDE
    return f"Systran/faster-whisper-{MODEL_NAME}"


def model_files_present(model_path: Path) -> bool:
    if not model_path.exists():
        return False
    return any(model_path.rglob("*.bin"))


def get_model_root() -> Path:
    if MODEL_ROOT_OVERRIDE:
        return Path(MODEL_ROOT_OVERRIDE).expanduser()
    if sys.platform.startswith("win"):
        base_dir = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    root = base_dir / "CtrlSpeak" / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


MODEL_ROOT_PATH = get_model_root()
MODEL_STORE_PATH = MODEL_ROOT_PATH / MODEL_NAME
MODEL_MARKER_FILE = MODEL_STORE_PATH / ".installed"


def cuda_runtime_ready() -> bool:
    configure_cuda_paths()
    try:
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception:
        return False
    if sys.platform.startswith("win"):
        for dll in ("cudnn_ops64_9.dll", "cublas64_12.dll"):
            try:
                ctypes.windll.LoadLibrary(dll)
            except OSError:
                return False
    return True


def resolve_device() -> str:
    if CLIENT_ONLY_BUILD:
        return "cpu"
    preference = DEVICE_PREFERENCE
    if preference not in {"auto", "cuda", "cpu"}:
        preference = "auto"
    if preference == "cpu":
        return "cpu"
    if cuda_runtime_ready():
        return "cuda"
    return "cpu"


def resolve_compute_type(device: str) -> str:
    if COMPUTE_TYPE_OVERRIDE:
        return COMPUTE_TYPE_OVERRIDE
    if device == "cuda":
        return "float16"
    if device == "cpu":
        return "int8"
    return "int8_float16"


class DownloadDialog:
    def __init__(self, model_name: str, progress_queue: Queue, cancel_event: threading.Event):
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.result: Optional[str] = None

        self.root = tk.Tk()
        self.root.title("CtrlSpeak Setup")
        self.root.geometry("420x220")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)

        title = tk.Label(self.root, text="Install Speech Model", font=("Segoe UI", 12, "bold"))
        title.pack(pady=(12, 4))

        info_text = ("CtrlSpeak needs to download the Whisper model '"
                     f"{model_name}' the first time it runs on this PC.")
        tk.Message(self.root, text=info_text, width=380).pack(pady=(0, 6))

        self.stage_var = tk.StringVar(value="Preparing download...")
        tk.Label(self.root, textvariable=self.stage_var, font=("Segoe UI", 10)).pack(pady=(0, 4))

        self.progress = ttk.Progressbar(self.root, length=360, mode="determinate", maximum=100)
        self.progress.pack(pady=(0, 4))

        self.status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 9)).pack(pady=(0, 8))

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=(0, 8))

        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel)
        self.cancel_button.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

    def cancel(self) -> None:
        if self.result is None:
            self.result = "cancelled"
            self.cancel_event.set()
            self.status_var.set("Cancelling...")

    def _update_progress(self, desc: str, current: float, total: float) -> None:
        if desc:
            self.stage_var.set(desc)
        if total and total > 0:
            if self.progress["mode"] != "determinate":
                self.progress.config(mode="determinate")
                self.progress.stop()
            self.progress.config(maximum=total)
            self.progress["value"] = current
            percent = min(max(int((current / total) * 100), 0), 100)
            self.status_var.set(f"{percent}%  ({format_bytes(current)} / {format_bytes(total)})")
        else:
            if self.progress["mode"] != "indeterminate":
                self.progress.config(mode="indeterminate")
                self.progress.start(10)
            self.status_var.set(f"{format_bytes(current)} downloaded")

    def _process_queue(self) -> None:
        try:
            while True:
                message = self.progress_queue.get_nowait()
                message_type = message[0]
                if message_type == "progress":
                    _, desc, current, total = message
                    self._update_progress(desc, current, total)
                elif message_type == "stage":
                    _, desc = message
                    self.stage_var.set(desc)
                elif message_type == "error":
                    _, error_text = message
                    self.result = "error"
                    self.status_var.set(error_text)
                elif message_type == "done":
                    self.result = "success"
                    self.status_var.set("Download completed.")
                elif message_type == "cancelled":
                    self.result = "cancelled"
                    self.status_var.set("Download cancelled.")
        except Empty:
            pass

        if self.result is None:
            self.root.after(100, self._process_queue)
        else:
            if self.progress["mode"] == "indeterminate":
                self.progress.stop()
            self.cancel_button.config(state=tk.DISABLED)
            self.root.after(400, self.root.destroy)

    def run(self) -> str:
        self.root.after(100, self._process_queue)
        self.root.mainloop()
        return self.result or "cancelled"


def format_bytes(value: float) -> str:
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    amount = float(value)
    for unit in units:
        if amount < step:
            return f"{amount:.1f} {unit}"
        amount /= step
    return f"{amount:.1f} PB"

def make_tqdm_class(callback, cancel_event):
    class GuiTqdm(tqdm):
        def update(self_inner, n=1):
            super().update(n)
            try:
                callback(self_inner.desc or "", float(self_inner.n), float(self_inner.total or 0))
            except Exception:
                pass
            if cancel_event.is_set():
                raise CancelledDownload()

    return GuiTqdm


def download_model_with_gui() -> bool:
    progress_queue: Queue = Queue()
    cancel_event = threading.Event()
    dialog = DownloadDialog(MODEL_NAME, progress_queue, cancel_event)

    def progress_callback(desc: str, current: float, total: float) -> None:
        progress_queue.put(("progress", desc, current, total))

    def worker() -> None:
        try:
            repo_id = model_repo_id()
            progress_queue.put(("stage", f"Connecting to Hugging Face ({repo_id})"))
            MODEL_STORE_PATH.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(MODEL_STORE_PATH),
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=4,
                tqdm_class=make_tqdm_class(progress_callback, cancel_event),
            )
            if cancel_event.is_set():
                progress_queue.put(("cancelled",))
                return
            MODEL_MARKER_FILE.touch(exist_ok=True)
            progress_queue.put(("done",))
        except CancelledDownload:
            progress_queue.put(("cancelled",))
        except Exception as exc:
            progress_queue.put(("error", f"Failed: {exc}"))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    result = dialog.run()
    thread.join(timeout=0.5)
    return result == "success"


def prompt_for_model_install() -> bool:
    if AUTO_MODE:
        return True
    prompt = ("CtrlSpeak needs to download the Whisper speech model (~3 GB) before it can transcribe on this PC. "
              "Do you want to start the download now?")
    try:
        choice = pyautogui.confirm(text=prompt, title="CtrlSpeak Setup", buttons=["Install Now", "Quit"])
    except Exception:
        print(prompt)
        choice = "Install Now"
    return choice == "Install Now"


def ensure_model_available() -> bool:
    if model_ready.is_set():
        return True
    if not model_files_present(MODEL_STORE_PATH) or not MODEL_MARKER_FILE.exists():
        if not prompt_for_model_install():
            notify("CtrlSpeak cannot transcribe without the Whisper model. You can install it next time you start the app.")
            return False
        if not download_model_with_gui():
            notify("The Whisper model was not installed. Try again later or check your internet connection.")
            return False
    if model_files_present(MODEL_STORE_PATH):
        model_ready.set()
        return True
    notify("Model download completed, but the files were not found. Please try again.")
    return False


def initialize_transcriber(force: bool = False, allow_client: bool = False, preferred_device: Optional[str] = None) -> Optional[WhisperModel]:
    global whisper_model, warned_cuda_unavailable
    with model_lock:
        if whisper_model is not None and not force:
            return whisper_model
        with settings_lock:
            mode = settings.get("mode")
        if not allow_client and mode != "client_server":
            return None
        if not ensure_model_available():
            return None
        device = preferred_device or resolve_device()
        if device == "cpu" and DEVICE_PREFERENCE in {"auto", "cuda"} and not warned_cuda_unavailable and preferred_device is None:
            notify("CUDA dependencies were not detected. CtrlSpeak will run Whisper on the CPU instead.")
            warned_cuda_unavailable = True
        compute_type = resolve_compute_type(device)
        try:
            whisper_model = WhisperModel(
                MODEL_NAME,
                device=device,
                compute_type=compute_type,
                download_root=str(MODEL_ROOT_PATH),
            )
            print(f"Whisper model ready on {device} ({compute_type})")
            return whisper_model
        except Exception as exc:
            print(f"Failed to load model on {device}: {exc}")
            if device != "cpu":
                try:
                    whisper_model = WhisperModel(
                        MODEL_NAME,
                        device="cpu",
                        compute_type="int8",
                        download_root=str(MODEL_ROOT_PATH),
                    )
                    notify("Running CtrlSpeak transcription on CPU fallback. Set CTRLSPEAK_DEVICE=cpu to suppress this message.")
                    warned_cuda_unavailable = True
                    return whisper_model
                except Exception as cpu_exc:
                    print(f"CPU fallback failed: {cpu_exc}")
            notify("Unable to initialize the transcription model. Please check your installation and try again.")
            whisper_model = None
            return None

def record_audio(target_path: Path) -> None:
    pyaudio_instance = pyaudio.PyAudio()
    stream = pyaudio_instance.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNKSIZE,
    )
    frames = []
    try:
        while recording:
            frames.append(stream.read(CHUNKSIZE))
    finally:
        stream.stop_stream()
        stream.close()
        pyaudio_instance.terminate()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))


def collect_text_from_segments(segments) -> str:
    pieces = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            pieces.append(text)
    return " ".join(pieces)


def generate_fallback_sound() -> Tuple[bytes, Dict[str, int]]:
    duration = 0.5
    t = np.linspace(0.0, duration, int(PROCESSING_SAMPLE_RATE * duration), endpoint=False)
    envelope = np.exp(-3 * t)
    wave_data = 0.2 * np.sin(2 * np.pi * 440 * t) * envelope
    int_data = np.clip(wave_data * 32767, -32767, 32767).astype(np.int16)
    settings = {"channels": 1, "rate": PROCESSING_SAMPLE_RATE, "width": 2}
    return int_data.tobytes(), settings


def load_processing_sound() -> Tuple[bytes, Dict[str, int]]:
    global processing_sound_data, processing_sound_settings
    if processing_sound_data is not None and processing_sound_settings is not None:
        return processing_sound_data, processing_sound_settings
    try:
        base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        sound_path = base_dir / "loading.wav"
        with wave.open(str(sound_path), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            settings = {
                "channels": wav_file.getnchannels(),
                "rate": wav_file.getframerate(),
                "width": wav_file.getsampwidth(),
            }
    except Exception as exc:
        print(f"Failed to load loading.wav: {exc}")
        frames, settings = generate_fallback_sound()
    processing_sound_data = frames
    processing_sound_settings = settings
    return processing_sound_data, processing_sound_settings


def _processing_sound_loop():
    data, settings = load_processing_sound()
    pa_instance = pyaudio.PyAudio()
    stream = None
    try:
        stream = pa_instance.open(
            format=pyaudio.get_format_from_width(settings["width"]),
            channels=settings["channels"],
            rate=settings["rate"],
            output=True,
        )
        while not processing_sound_stop_event.is_set():
            stream.write(data)
    except Exception as exc:
        print(f"Processing sound playback failed: {exc}")
    finally:
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
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

def transcribe_local(file_path: str, play_feedback: bool = True, allow_client: bool = False, preferred_device: Optional[str] = None) -> Optional[str]:
    global last_connected_server
    model = initialize_transcriber(allow_client=allow_client, preferred_device=preferred_device)
    if model is None:
        return None
    if play_feedback:
        start_processing_feedback()
    try:
        segments, _ = model.transcribe(
            file_path,
            beam_size=5,
            vad_filter=True,
            temperature=0.2,
        )
        text = collect_text_from_segments(segments)
        if not text:
            print("No speech detected in the recording.")
        if text:
            with settings_lock:
                port = int(settings.get("server_port", 65432))
            if allow_client and preferred_device == "cpu":
                last_connected_server = ServerInfo(host="local-cpu", port=-1, last_seen=time.time())
            else:
                last_connected_server = ServerInfo(host="local", port=port, last_seen=time.time())
        return text or None
    except Exception as exc:
        print(f"Transcription failed: {exc}")
        notify_error("Transcription failed", format_exception_details(exc))
        return None
    finally:
        if play_feedback:
            stop_processing_feedback()


def get_best_server() -> Optional[ServerInfo]:
    if discovery_listener is None:
        return ensure_preferred_server_registered(probe=True)
    server = discovery_listener.get_best_server()
    if server:
        return server
    return ensure_preferred_server_registered(probe=True)


def transcribe_remote(file_path: str, play_feedback: bool = True) -> Optional[str]:
    server = get_best_server()
    if server is None:
        return handle_missing_server(file_path, play_feedback=play_feedback)
    if play_feedback:
        start_processing_feedback()
    conn = None
    try:
        with open(file_path, "rb") as handle:
            data = handle.read()
        conn = http.client.HTTPConnection(server.host, server.port, timeout=180)
        headers = {
            "Content-Type": "audio/wav",
            "Content-Length": str(len(data)),
            "X-CtrlSpeak-Client": local_hostname,
        }
        conn.request("POST", "/transcribe", body=data, headers=headers)
        response = conn.getresponse()
        payload = response.read()
        if response.status != 200:
            snippet = payload[:400].decode("utf-8", errors="replace")
            detail = f"Server {server.host}:{server.port} responded with HTTP {response.status} {response.reason}\n{snippet}"
            notify_error("Remote server error", detail)
            return None
        try:
            data = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            notify_error("Remote response parsing failed", format_exception_details(exc))
            return None
        return data.get("text")
    except Exception as exc:
        notify_error("Remote transcription failed", format_exception_details(exc))
        return None
    finally:
        if play_feedback:
            stop_processing_feedback()
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass





def transcribe_audio(file_path: str, play_feedback: bool = True) -> Optional[str]:
    with settings_lock:
        mode = settings.get("mode")
    if mode == "client_server":
        return transcribe_local(file_path, play_feedback=play_feedback)
    elif mode == "client":
        return transcribe_remote(file_path, play_feedback=play_feedback)
    else:
        notify("CtrlSpeak mode is not configured. Restart the application.")
        return None

def on_press(key):
    global recording, recording_thread, recording_file_path
    if not client_enabled:
        return
    if key == keyboard.Key.ctrl_r and not recording:
        recording = True
        recording_file_path = create_recording_file_path()
        recording_thread = threading.Thread(target=record_audio, args=(recording_file_path,), daemon=True)
        recording_thread.start()


def on_release(key):
    global recording, recording_thread, recording_file_path
    if key == keyboard.Key.ctrl_r:
        if recording:
            recording = False
            if recording_thread:
                recording_thread.join()
                recording_thread = None
            path = recording_file_path
            text = None
            try:
                if path and path.exists() and path.stat().st_size > 0:
                    text = transcribe_audio(str(path))
            except Exception as exc:
                notify_error("Transcription failed", format_exception_details(exc))
                text = None
            if text:
                try:
                    insert_text_into_focus(text)
                except Exception as exc:
                    notify_error("Text insertion failed", format_exception_details(exc))
            cleanup_recording_file(path)
            recording_file_path = None


def start_client_listener() -> None:
    global listener, client_enabled
    with listener_lock:
        if listener is not None:
            return
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
            listener.stop()
            listener = None
    cleanup_recording_file(recording_file_path)
    recording_file_path = None
    schedule_management_refresh()


def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def create_icon_image():
    icon_path = resource_path("icon.ico")
    return Image.open(icon_path)


def open_management_dialog(icon, item):
    ensure_management_ui_thread()
    enqueue_management_task(_show_management_window, icon)



def ensure_management_ui_thread() -> None:
    global management_ui_thread
    if management_ui_thread and management_ui_thread.is_alive():
        return
    management_ui_thread = threading.Thread(
        target=_management_ui_loop,
        name="CtrlSpeakManagementUI",
        daemon=True,
    )
    management_ui_thread.start()



def _management_ui_loop() -> None:
    global tk_root
    tk_root = tk.Tk()
    tk_root.withdraw()

    def process_queue() -> None:
        while True:
            try:
                func, args, kwargs = management_ui_queue.get_nowait()
            except Empty:
                break
            try:
                func(*args, **kwargs)
            except Exception:
                traceback.print_exc()
        if tk_root is not None:
            tk_root.after(120, process_queue)

    tk_root.after(80, process_queue)
    tk_root.mainloop()



def _show_management_window(icon: pystray.Icon) -> None:
    global management_window
    if management_window and management_window.is_open():
        management_window.bring_to_front()
        management_window.refresh_status()
        return
    management_window = ManagementWindow(icon)



def get_focused_control() -> Optional[int]:
    if not sys.platform.startswith("win"):
        return None
    try:
        foreground = user32.GetForegroundWindow()
        if not foreground:
            return None
        thread_id = user32.GetWindowThreadProcessId(foreground, None)
        current_thread_id = kernel32.GetCurrentThreadId()
        attached = False
        if thread_id != current_thread_id:
            if user32.AttachThreadInput(current_thread_id, thread_id, True):
                attached = True
        info = GUITHREADINFO()
        info.cbSize = ctypes.sizeof(GUITHREADINFO)
        hwnd_focus = None
        if user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
            hwnd_focus = info.hwndFocus or info.hwndCaret or info.hwndActive
        if attached:
            user32.AttachThreadInput(current_thread_id, thread_id, False)
        return hwnd_focus or foreground
    except Exception:
        return None


def get_class_name(hwnd: int) -> str:
    if not sys.platform.startswith("win"):
        return ""
    if not hwnd:
        return ""
    buffer = ctypes.create_unicode_buffer(256)
    try:
        if user32.GetClassNameW(hwnd, buffer, len(buffer)) == 0:
            return ""
    except Exception:
        return ""
    return buffer.value.lower()


def get_window_text(hwnd: int) -> str:
    if not sys.platform.startswith("win"):
        return ""
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    if user32.GetWindowTextW(hwnd, buffer, len(buffer)) == 0:
        return ""
    return buffer.value


def get_window_process_name(hwnd: int) -> str:
    if not sys.platform.startswith("win"):
        return ""
    pid = wintypes.DWORD()
    if user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)) == 0:
        return ""
    access = PROCESS_QUERY_INFORMATION | PROCESS_VM_READ | PROCESS_QUERY_LIMITED_INFORMATION
    process_handle = kernel32.OpenProcess(access, False, pid.value)
    if not process_handle:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(260)
        length = psapi.GetModuleFileNameExW(process_handle, None, buffer, len(buffer))
        if length == 0:
            return ""
        return Path(buffer.value).name.lower()
    except Exception:
        return ""
    finally:
        kernel32.CloseHandle(process_handle)


def window_matches_anydesk(hwnd: int) -> bool:
    class_name = get_class_name(hwnd)
    if 'anydesk' in class_name:
        return True
    process_name = get_window_process_name(hwnd)
    if 'anydesk' in process_name:
        return True
    title = get_window_text(hwnd).lower()
    if 'anydesk' in title:
        return True
    return False


def is_anydesk_window(hwnd: int) -> bool:
    if not sys.platform.startswith("win"):
        return False
    current = hwnd
    visited = set()
    for _ in range(6):
        if not current or current in visited:
            break
        visited.add(current)
        if window_matches_anydesk(current):
            return True
        parent = user32.GetParent(current)
        if not parent:
            break
        current = parent
    return False


def is_console_window(hwnd: int) -> bool:
    if not sys.platform.startswith("win") or not hwnd:
        return False
    class_name = get_class_name(hwnd)
    if class_name in {"consolewindowclass"}:
        return True
    if "cascadia" in class_name:
        return True
    process_name = get_window_process_name(hwnd)
    return process_name in {"conhost.exe", "openconsole.exe", "wt.exe", "windowsterminal.exe", "ubuntu.exe", "wsl.exe"}


def try_direct_text_insert(text: str, hwnd: Optional[int] = None) -> bool:
    if not sys.platform.startswith("win"):
        return False
    if hwnd is None:
        hwnd = get_focused_control()
    if not hwnd:
        return False
    if is_anydesk_window(hwnd):
        return False
    class_name = get_class_name(hwnd)
    if not class_name:
        return False
    try:
        if any(token in class_name for token in ("edit", "richedit", "notepad", "textarea", "tmemo", "scintilla")):
            user32.SendMessageW(hwnd, EM_SETSEL, 0xFFFFFFFF, 0xFFFFFFFF)
            user32.SendMessageW(hwnd, EM_REPLACESEL, True, text)
            return True
    except Exception:
        return False
    return False


def open_clipboard() -> bool:
    if not sys.platform.startswith("win"):
        return False
    for _ in range(5):
        if user32.OpenClipboard(None):
            return True
        time.sleep(0.01)
    return False


def get_clipboard_text() -> Optional[str]:
    if not sys.platform.startswith("win"):
        return None
    if not open_clipboard():
        return None
    try:
        if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
            return None
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return None
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            return None
        try:
            return ctypes.wstring_at(pointer)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def set_clipboard_text(text: str) -> bool:
    if not sys.platform.startswith("win"):
        return False
    if not open_clipboard():
        return False
    try:
        user32.EmptyClipboard()
        data = ctypes.create_unicode_buffer(text)
        size = ctypes.sizeof(ctypes.c_wchar) * len(data)
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            return False
        pointer = kernel32.GlobalLock(handle)
        if not pointer:
            kernel32.GlobalFree(handle)
            return False
        ctypes.memmove(pointer, ctypes.byref(data), size)
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)
            return False
        return True
    finally:
        user32.CloseClipboard()


def restore_clipboard_text(previous: Optional[str]) -> None:
    if not sys.platform.startswith("win"):
        return
    if previous is None:
        if open_clipboard():
            try:
                user32.EmptyClipboard()
            finally:
                user32.CloseClipboard()
        return
    set_clipboard_text(previous)


INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_V = 0x56


PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

KEYEVENTF_UNICODE = 0x0004

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.LPARAM),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.LPARAM),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _INPUTUNION),
    ]


_send_input = user32.SendInput
_send_input.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
_send_input.restype = wintypes.UINT


def send_input_key(vk: int, keyup: bool = False) -> bool:
    flags = KEYEVENTF_KEYUP if keyup else 0
    union = _INPUTUNION()
    union.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
    inp = INPUT(type=INPUT_KEYBOARD, union=union)
    return _send_input(1, ctypes.byref(inp), ctypes.sizeof(inp)) == 1


def ensure_foreground(hwnd: int) -> None:
    if not sys.platform.startswith("win") or not hwnd:
        return
    try:
        pid = wintypes.DWORD()
        thread_id = user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        current_thread_id = kernel32.GetCurrentThreadId()
        attached = False
        if thread_id and thread_id != current_thread_id:
            if user32.AttachThreadInput(current_thread_id, thread_id, True):
                attached = True
        try:
            user32.SetForegroundWindow(hwnd)
            user32.SetFocus(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(current_thread_id, thread_id, False)
    except Exception:
        pass




def send_unicode_input(text: str) -> bool:
    if not sys.platform.startswith("win"):
        return False
    hwnd = get_focused_control()
    ensure_foreground(hwnd)
    for char in text:
        ch = ord(char)
        if ch == 10:
            ch = 13
        union = _INPUTUNION()
        union.ki = KEYBDINPUT(wVk=0, wScan=ch, dwFlags=KEYEVENTF_UNICODE, time=0, dwExtraInfo=0)
        inp = INPUT(type=INPUT_KEYBOARD, union=union)
        if _send_input(1, ctypes.byref(inp), ctypes.sizeof(inp)) != 1:
            return False
        union = _INPUTUNION()
        union.ki = KEYBDINPUT(wVk=0, wScan=ch, dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=0)
        inp = INPUT(type=INPUT_KEYBOARD, union=union)
        if _send_input(1, ctypes.byref(inp), ctypes.sizeof(inp)) != 1:
            return False
        time.sleep(0.01)
    time.sleep(0.02)
    return True


def try_sendinput_paste(text: str) -> bool:
    if not sys.platform.startswith("win"):
        return False
    hwnd = get_focused_control()
    ensure_foreground(hwnd)
    previous = get_clipboard_text()
    if not set_clipboard_text(text):
        return False
    try:
        if not (send_input_key(VK_CONTROL) and send_input_key(VK_V) and send_input_key(VK_V, True) and send_input_key(VK_CONTROL, True)):
            return False
        time.sleep(0.15)
        return True
    finally:
        restore_clipboard_text(previous)


def try_clipboard_paste(text: str) -> bool:
    if not sys.platform.startswith("win"):
        return False
    previous = get_clipboard_text()
    if not set_clipboard_text(text):
        return False
    try:
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(0.05)
        return True
    finally:
        restore_clipboard_text(previous)


def insert_text_into_focus(text: str) -> None:
    if not text:
        return
    if not sys.platform.startswith("win"):
        pyautogui.write(text)
        return
    hwnd = get_focused_control()
    console_window = False
    if hwnd:
        console_window = is_console_window(hwnd)
    if hwnd and (FORCE_SENDINPUT or is_anydesk_window(hwnd) or console_window):
        if send_unicode_input(text):
            return
        if try_sendinput_paste(text):
            return
    if not console_window and try_direct_text_insert(text, hwnd):
        return
    if FORCE_SENDINPUT and not console_window:
        if send_unicode_input(text):
            return
    if try_sendinput_paste(text):
        return
    if try_clipboard_paste(text):
        return
    pyautogui.write(text)


def describe_server_status() -> str:
    with settings_lock:
        mode = settings.get("mode")
        port = int(settings.get("server_port", 65432))
    if mode == "client_server" and server_thread and server_thread.is_alive():
        host = last_connected_server.host if last_connected_server else get_advertised_host_ip()
        label = f"{host}:{port}"
        return f"Serving: {label}"
    if last_connected_server:
        host = last_connected_server.host
        port = last_connected_server.port
        if host == "local-cpu":
            label = "local CPU"
        elif host == "local":
            label = "local"
        else:
            label = f"{host}:{port}"
        return f"Connected: {label}"
    server = get_best_server()
    if server:
        return f"Discovered: {server.host}:{server.port}"
    return "Not connected"



class ManagementWindow:
    def __init__(self, icon: pystray.Icon):
        self._icon = icon
        self.window = tk.Toplevel(tk_root)
        self.window.title(f"CtrlSpeak Control v{APP_VERSION}")
        self.window.geometry("420x360")
        self.window.minsize(380, 320)
        self.window.resizable(True, True)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _event: self.close())
        try:
            self.window.iconbitmap(resource_path("icon.ico"))
        except Exception:
            pass

        self.status_var = tk.StringVar()
        self.server_status_var = tk.StringVar()

        frame = ttk.Frame(self.window, padding=(16, 14))
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="CtrlSpeak Control", font=("Segoe UI", 14, "bold")).pack()
        build_label = "Client Only" if CLIENT_ONLY_BUILD else "Client + Server"
        ttk.Label(frame, text=f"Build {APP_VERSION} - {build_label}", font=("Segoe UI", 9)).pack(pady=(2, 10))

        ttk.Label(frame, textvariable=self.status_var, justify=tk.LEFT, anchor="w").pack(fill=tk.X, pady=(0, 6))
        ttk.Label(
            frame,
            textvariable=self.server_status_var,
            font=("Segoe UI", 10, "italic"),
            justify=tk.LEFT,
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 6))

        if CLIENT_ONLY_BUILD:
            manual_frame = ttk.LabelFrame(frame, text="Server connection")
            manual_frame.pack(fill=tk.X, pady=(0, 10))
            ttk.Label(manual_frame, text="Host (host[:port])").pack(anchor="w", padx=4, pady=(6, 2))
            self.server_entry_var = tk.StringVar(value=self._default_server_target())
            ttk.Entry(manual_frame, textvariable=self.server_entry_var).pack(fill=tk.X, padx=4, pady=(0, 6))
            self.connect_button = ttk.Button(manual_frame, text="Connect", command=self.connect_manual_server)
            self.connect_button.pack(anchor="e", padx=4, pady=(0, 6))

        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X)

        self.stop_button = ttk.Button(button_frame, text="Stop Client", command=self.stop_client)
        self.stop_button.pack(fill=tk.X, pady=2)
        self.start_button = ttk.Button(button_frame, text="Start Client", command=self.start_client)
        self.start_button.pack(fill=tk.X, pady=2)

        self.refresh_button = ttk.Button(button_frame, text="Refresh Servers", command=self.refresh_servers)
        self.refresh_button.pack(fill=tk.X, pady=2)

        exit_label = "Exit CtrlSpeak" if CLIENT_ONLY_BUILD else "Stop Everything"
        self.stop_all_button = ttk.Button(button_frame, text=exit_label, command=self.stop_everything)
        self.stop_all_button.pack(fill=tk.X, pady=(10, 6))

        ttk.Separator(frame).pack(fill=tk.X, pady=(6, 10))
        ttk.Button(frame, text="Close", command=self.close).pack(fill=tk.X, pady=(0, 4))
        delete_button = tk.Button(frame, text="Delete CtrlSpeak", command=self.delete_ctrlspeak, bg="#d32f2f", fg="white", activebackground="#b71c1c", activeforeground="white")
        delete_button.pack(fill=tk.X, pady=(0, 10))

        self.window.after(120, self.refresh_status)
        self.bring_to_front()

    def _default_server_target(self) -> str:
        host, port = get_preferred_server_settings()
        if host and port:
            return f"{host}:{port}"
        if last_connected_server and last_connected_server.host not in {"local-cpu", "local"}:
            return f"{last_connected_server.host}:{last_connected_server.port}"
        return ""

    def is_open(self) -> bool:
        return bool(self.window and self.window.winfo_exists())

    def bring_to_front(self) -> None:
        if not self.is_open():
            return
        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()
        self.window.attributes("-topmost", True)
        self.window.after(150, lambda: self.window.attributes("-topmost", False))

    def refresh_status(self) -> None:
        with settings_lock:
            mode = settings.get("mode") or "unknown"
        status_parts = [
            f"Mode: {mode}",
            "Client Only build" if CLIENT_ONLY_BUILD else "Full build",
            f"Client: {'Active' if client_enabled else 'Stopped'}",
        ]
        server_running = server_thread and server_thread.is_alive()
        status_parts.append(f"Server thread: {'Running' if server_running else 'Not running'}")
        self.status_var.set("\n".join(status_parts))
        self.server_status_var.set(describe_server_status())
        if hasattr(self, "server_entry_var"):
            default_target = self._default_server_target()
            if default_target and not self.server_entry_var.get().strip():
                self.server_entry_var.set(default_target)
        if client_enabled:
            self.start_button.state(["disabled"])
            self.stop_button.state(["!disabled"])
        else:
            self.start_button.state(["!disabled"])
            self.stop_button.state(["disabled"])

    def start_client(self) -> None:
        start_client_listener()
        self.refresh_status()

    def stop_client(self) -> None:
        stop_client_listener()
        self.refresh_status()

    def refresh_servers(self) -> None:
        self.refresh_button.state(["disabled"])
        self.refresh_button.config(text="Scanning...")
        self.server_status_var.set("Scanning for servers...")

        def worker() -> None:
            try:
                manual_discovery_refresh()
            finally:
                enqueue_management_task(self._on_refresh_finished)

        threading.Thread(target=worker, daemon=True).start()

    def _on_refresh_finished(self) -> None:
        if not self.is_open():
            return
        self.refresh_button.state(["!disabled"])
        self.refresh_button.config(text="Refresh Servers")
        self.refresh_status()

    def connect_manual_server(self) -> None:
        target = self.server_entry_var.get().strip() if hasattr(self, "server_entry_var") else ""
        try:
            host, port = parse_server_target(target)
        except ValueError as exc:
            messagebox.showerror("CtrlSpeak", str(exc), parent=self.window)
            return
        self.connect_button.state(["disabled"])
        self.connect_button.config(text="Connecting...")
        self.server_status_var.set(f"Connecting to {host}:{port}...")

        def worker() -> None:
            success = probe_server(host, port, timeout=3.0)
            def finish() -> None:
                if not self.is_open():
                    return
                self.connect_button.state(["!disabled"])
                self.connect_button.config(text="Connect")
                if success:
                    register_manual_server(host, port)
                    self.server_entry_var.set(f"{host}:{port}")
                    self.server_status_var.set(f"Connected: {host}:{port}")
                    messagebox.showinfo("CtrlSpeak", f"Connected to {host}:{port}", parent=self.window)
                else:
                    messagebox.showerror("CtrlSpeak", f"Unable to reach {host}:{port}.", parent=self.window)
                self.refresh_status()
            enqueue_management_task(finish)

        threading.Thread(target=worker, daemon=True).start()

    def delete_ctrlspeak(self) -> None:
        if not messagebox.askyesno("Delete CtrlSpeak", "This will remove CtrlSpeak and all local data. Continue?", parent=self.window):
            return
        self.window.after(100, lambda: initiate_self_uninstall(self._icon))

    def stop_everything(self) -> None:
        stop_client_listener()
        shutdown_server()
        self.refresh_status()
        self.window.after(200, self._icon.stop)
        self.close()

    def close(self) -> None:
        global management_window
        if self.is_open():
            self.window.destroy()
        management_window = None


def on_exit(icon, item):
    stop_client_listener()
    shutdown_server()
    icon.stop()


def get_advertised_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tmp:
            tmp.connect(("8.8.8.8", 80))
            return tmp.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def manage_discovery_broadcast(stop_event: threading.Event, port: int, server_port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    try:
        while not stop_event.is_set():
            host_ip = get_advertised_host_ip()
            message = f"{SERVER_BROADCAST_SIGNATURE}|{host_ip}|{server_port}".encode("utf-8")
            try:
                sock.sendto(message, ("255.255.255.255", port))
            except Exception as exc:
                print(f"Discovery broadcast failed: {exc}")
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
        sock.bind(("", port))
        sock.settimeout(1.0)
    except OSError as exc:
        print(f"Failed to bind discovery query listener: {exc}")
        sock.close()
        return
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
            if message != f"{SERVER_BROADCAST_SIGNATURE}_QUERY":
                continue
            host_ip = get_advertised_host_ip()
            response = f"{SERVER_BROADCAST_SIGNATURE}|{host_ip}|{server_port}".encode("utf-8")
            try:
                sock.sendto(response, addr)
            except Exception as exc:
                print(f"Failed to respond to discovery query: {exc}")
    finally:
        try:
            sock.close()
        except Exception:
            pass


class TranscriptionRequestHandler(BaseHTTPRequestHandler):
    server_version = "CtrlSpeakServer/1.0"

    def do_GET(self):
        if self.path in {"/ping", "/health", "/status"}:
            payload = json.dumps({"status": "ok", "mode": "server"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404, "Unknown endpoint")

    def do_POST(self):
        if self.path != "/transcribe":
            self.send_error(404, "Unknown endpoint")
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            self.send_error(400, "Missing audio payload")
            return
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            remaining = content_length
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                tmp.write(chunk)
                remaining -= len(chunk)
            temp_path = tmp.name
        if os.path.getsize(temp_path) == 0:
            os.remove(temp_path)
            self.send_error(400, "Empty audio payload")
            return
        start_time = time.time()
        text = transcribe_local(temp_path, play_feedback=False)
        try:
            os.remove(temp_path)
        except Exception:
            pass
        if text is None:
            self.send_error(500, "Transcription failed")
            return
        payload = json.dumps({"text": text, "elapsed": time.time() - start_time}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        return


def start_server() -> None:
    global server_thread, server_httpd, discovery_broadcaster, discovery_query_listener, discovery_query_stop_event, last_connected_server
    if CLIENT_ONLY_BUILD:
        notify("Server functionality is not available in this build.")
        return
    if server_thread and server_thread.is_alive():
        return
    with settings_lock:
        port = int(settings.get("server_port", 65432))
        discovery_port = int(settings.get("discovery_port", 54330))
    try:
        server_httpd = ThreadingHTTPServer(("0.0.0.0", port), TranscriptionRequestHandler)
    except OSError as exc:
        notify_error("Server startup failed", str(exc))
        server_httpd = None
        return

    def serve():
        try:
            server_httpd.serve_forever()
        except Exception as exc:
            notify_error("Server stopped", format_exception_details(exc))

    server_thread = threading.Thread(target=serve, daemon=True)
    server_thread.start()
    broadcast_stop_event.clear()
    discovery_query_stop_event.clear()
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
    print(f"CtrlSpeak server listening on port {port}")
    schedule_management_refresh()


def shutdown_server() -> None:
    global server_thread, server_httpd, discovery_broadcaster, discovery_query_listener, broadcast_stop_event, discovery_query_stop_event, last_connected_server
    broadcast_stop_event.set()
    discovery_query_stop_event.set()
    if discovery_broadcaster and discovery_broadcaster.is_alive():
        discovery_broadcaster.join(timeout=1.0)
    discovery_broadcaster = None
    if discovery_query_listener and discovery_query_listener.is_alive():
        discovery_query_listener.join(timeout=1.0)
    discovery_query_listener = None
    if server_httpd is not None:
        try:
            server_httpd.shutdown()
            server_httpd.server_close()
        except Exception:
            pass
    server_httpd = None
    if server_thread and server_thread.is_alive():
        server_thread.join(timeout=1.0)
    server_thread = None
    broadcast_stop_event = threading.Event()
    discovery_query_stop_event = threading.Event()
    last_connected_server = None
    schedule_management_refresh()

def run_tray():
    start_client_listener()
    with settings_lock:
        mode = settings.get("mode")
    menu_items = [
        pystray.MenuItem("Manage CtrlSpeak", open_management_dialog),
        pystray.MenuItem("Quit", on_exit),
    ]
    icon = pystray.Icon(
        "CtrlSpeak",
        create_icon_image(),
        f"CtrlSpeak ({mode})",
        menu=pystray.Menu(*menu_items),
    )
    icon.run()


def start_discovery_listener() -> None:
    global discovery_listener
    with settings_lock:
        port = int(settings.get("discovery_port", 54330))
    discovery_listener = DiscoveryListener(port)
    discovery_listener.start()


def transcribe_cli(target: str) -> int:
    file_path = Path(target).expanduser()
    if not file_path.is_file():
        print(f"Audio file not found: {file_path}", file=sys.stderr)
        return 1
    with settings_lock:
        mode = settings.get("mode")
    if mode == "client_server":
        if initialize_transcriber() is None:
            print("Unable to initialize the transcription engine.", file=sys.stderr)
            return 2
    elif mode == "client":
        time.sleep(1.0)
    text = transcribe_audio(str(file_path), play_feedback=False)
    if text is None:
        print("Transcription produced no output.", file=sys.stderr)
        return 3
    print(text)
    return 0


def shutdown_all():
    stop_client_listener()
    shutdown_server()
    if discovery_listener:
        discovery_listener.stop()


def parse_cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="CtrlSpeak", add_help=True, description="CtrlSpeak voice control")
    parser.add_argument("--transcribe", metavar="WAV_PATH", help="Transcribe an audio file and print the result")
    parser.add_argument("--uninstall", action="store_true", help="Remove CtrlSpeak and all local data")
    parser.add_argument("--auto-setup", choices=["client", "client_server"], help="Configure CtrlSpeak without prompts")
    parser.add_argument("--force-sendinput", action="store_true", help="Force SendInput-based insertion (debug)")
    args, _ = parser.parse_known_args(argv[1:])
    return args


def apply_auto_setup(profile: str) -> None:
    global AUTO_MODE, AUTO_MODE_PROFILE
    AUTO_MODE = True
    AUTO_MODE_PROFILE = profile
    with settings_lock:
        settings["mode"] = profile
    save_settings()


def main(argv: list[str]) -> int:
    args = parse_cli_args(argv)
    if args.force_sendinput:
        global FORCE_SENDINPUT
        FORCE_SENDINPUT = True
    if args.uninstall:
        initiate_self_uninstall(None)
        return 0
    load_settings()
    if args.auto_setup:
        apply_auto_setup(args.auto_setup)
    cli_mode = args.transcribe is not None
    if not acquire_single_instance_lock():
        message = "CtrlSpeak is already running. Please close the existing instance before starting a new one."
        if cli_mode:
            print(message, file=sys.stderr)
        else:
            notify(message)
        return 0
    if not cli_mode and not AUTO_MODE:
        show_splash_screen(SPLASH_DURATION_MS)
    ensure_mode_selected()
    if cli_mode:
        start_discovery_listener()
        with settings_lock:
            mode = settings.get("mode")
        if mode == "client_server":
            initialize_transcriber()
        elif mode == "client":
            time.sleep(1.0)
        return transcribe_cli(args.transcribe)

    start_discovery_listener()
    with settings_lock:
        mode = settings.get("mode")
    if mode == "client_server":
        initialize_transcriber()
        start_server()
    elif mode == "client":
        time.sleep(1.0)
    run_tray()
    return 0


atexit.register(release_single_instance_lock)
atexit.register(shutdown_all)


if __name__ == "__main__":
    sys.exit(main(sys.argv))









