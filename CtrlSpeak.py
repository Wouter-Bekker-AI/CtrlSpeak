# -*- coding: utf-8 -*-
# CtrlSpeak — tray-based voice control with resizable management, GPU opt-in and model picker
# Startup behavior unchanged: 1s splash, tray only. Management opens from tray.
# Version kept fixed at 0.6.0 as requested.

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
import math
import shutil
import traceback

import ctranslate2
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

# ---------------- Core constants / defaults ----------------
CHUNKSIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# ENV defaults (kept, but now overridable from Settings UI)
ENV_DEFAULT_MODEL = os.environ.get("CTRLSPEAK_MODEL", "large-v3")
ENV_DEVICE_PREF = os.environ.get("CTRLSPEAK_DEVICE", "auto").lower()
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

    # NEW: device + model choices persisted here (UI manages these)
    "device_preference": None,      # "cpu" | "cuda" | "auto"
    "model_name": None,             # "small" | "large-v3"
}

APP_VERSION = "0.6.0"           # fixed per request
SPLASH_DURATION_MS = 1000       # exactly 1 second per request
ERROR_LOG_FILENAME = "CtrlSpeak-error.log"
LOCK_FILENAME = "CtrlSpeak.lock"
PROCESSING_SAMPLE_RATE = 44100

# ---------------- Singletons / state ----------------
lock_file_path: Optional[Path] = None
discovery_query_listener: Optional[threading.Thread] = None
discovery_query_stop_event = threading.Event()

CLIENT_ONLY_BUILD = False  # preserved logic below will set properly
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

# Windows GUI helpers
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002
EM_SETSEL = 0x00B1
EM_REPLACESEL = 0x00C2
WM_SETTEXT = 0x000C

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
KEYEVENTF_UNICODE = 0x0004

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_V = 0x56

# Optional icon path
def resource_path(relative_path: str) -> str:
    try:
        base_path = sys._MEIPASS  # pyinstaller
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ---------------- Small helpers kept from your original ----------------
class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]

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

@dataclass
class ServerInfo:
    host: str
    port: int
    last_seen: float

last_connected_server: Optional[ServerInfo] = None
management_window: Optional["ManagementWindow"] = None

# ---------------- Settings / config ----------------
def detect_client_only_build() -> bool:
    try:
        base_dir = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent))
        if (base_dir / 'client_only.flag').exists():
            return True
    except Exception:
        pass
    return os.environ.get('CTRLSPEAK_CLIENT_ONLY', '0') == '1'

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
    path = get_config_file_path()
    data = DEFAULT_SETTINGS.copy()
    if path.exists():
        try:
            data.update(json.loads(path.read_text("utf-8-sig")))
        except Exception as exc:
            print(f"Failed to read settings: {exc}")
    with settings_lock:
        settings = data
    return data

def save_settings() -> None:
    path = get_config_file_path()
    with settings_lock:
        snapshot = settings.copy()
    try:
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Failed to save settings: {exc}")

# ---------------- Splash (unchanged behavior, 1s) ----------------
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
    pos_x = int((screen_w - width) / 2); pos_y = int((screen_h - height) / 2)
    root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
    try:
        icon_path = resource_path("icon.ico")
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

# ---------------- Model / device helpers (NEW: driven by Settings) ----------------
def get_current_model_name() -> str:
    with settings_lock:
        name = settings.get("model_name")
    if isinstance(name, str) and name in {"small", "large-v3"}:
        return name
    return ENV_DEFAULT_MODEL  # fallback to env default for first run

def set_current_model_name(name: str) -> None:
    with settings_lock:
        settings["model_name"] = name
    save_settings()
    # reset readiness to ensure re-check
    model_ready.clear()

def get_device_preference() -> str:
    with settings_lock:
        pref = settings.get("device_preference")
    if pref in {"cpu", "cuda", "auto"}:
        return pref  # user-chosen
    return ENV_DEVICE_PREF if ENV_DEVICE_PREF in {"cpu", "cuda", "auto"} else "auto"

def set_device_preference(pref: str) -> None:
    with settings_lock:
        settings["device_preference"] = pref
    save_settings()

def model_repo_id() -> str:
    if MODEL_REPO_OVERRIDE:
        return MODEL_REPO_OVERRIDE
    name = get_current_model_name()
    # Using Systran's faster-whisper format as in your original
    return f"Systran/faster-whisper-{name}"

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

def model_store_path_for(name: str) -> Path:
    return get_model_root() / name

def model_files_present(model_path: Path) -> bool:
    if not model_path.exists():
        return False
    return any(model_path.rglob("*.bin"))

def format_bytes(value: float) -> str:
    step = 1024.0; units = ["B", "KB", "MB", "GB", "TB"]
    amount = float(value)
    for u in units:
        if amount < step:
            return f"{amount:.1f} {u}"
        amount /= step
    return f"{amount:.1f} PB"

# ---------------- CUDA readiness (kept from your original, plus helpers) ----------------
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

def _popen_stream(cmd: List[str], timeout: int | None = None) -> int:
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
            start = time.time()
            while True:
                if timeout and (time.time() - start) > timeout:
                    p.kill()
                    return 124
                line = p.stdout.readline() if p.stdout else ""
                if not line and p.poll() is not None:
                    break
            return p.returncode or 0
    except FileNotFoundError:
        return 127
    except Exception:
        return 1

# ---------------- Progress window (reusable, resizable) ----------------
class ProgressWindow:
    def __init__(self, parent: tk.Misc, title: str = "Working…"):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.resizable(True, True)
        self.top.transient(parent)
        self.top.grab_set()

        frm = ttk.Frame(self.top, padding=14)
        frm.grid(row=0, column=0, sticky="nsew")
        self.top.columnconfigure(0, weight=1)
        self.top.rowconfigure(0, weight=1)

        self.action_var = tk.StringVar(value="Starting…")
        self.dest_var = tk.StringVar(value="")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.percent_var = tk.StringVar(value="")
        self.detail_var = tk.StringVar(value="")

        ttk.Label(frm, textvariable=self.action_var, font=("", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(frm, textvariable=self.dest_var, foreground="#666").grid(row=1, column=0, sticky="w", pady=(2,8))

        self.pb = ttk.Progressbar(frm, orient="horizontal", mode="indeterminate",
                                  variable=self.progress_var, maximum=100.0, length=420)
        self.pb.grid(row=2, column=0, sticky="ew")
        frm.columnconfigure(0, weight=1)

        ttk.Label(frm, textvariable=self.percent_var).grid(row=3, column=0, sticky="e", pady=(4,0))
        ttk.Label(frm, textvariable=self.detail_var, foreground="#666").grid(row=4, column=0, sticky="w", pady=(6,0))

        self._start_time = time.time()
        self._total_bytes: Optional[int] = None
        self._done_bytes: int = 0
        self._marquee = True
        self._lock = threading.Lock()

        self.pb.start(24)
        self._tick()

    def _tick(self):
        with self._lock:
            elapsed = max(0.1, time.time() - self._start_time)
            if self._total_bytes and self._total_bytes > 0:
                speed = self._done_bytes / elapsed
                remain = max(0, self._total_bytes - self._done_bytes)
                eta = remain / speed if speed > 0 else math.inf
                self.detail_var.set(f"{format_bytes(self._done_bytes)} / {format_bytes(self._total_bytes)}"
                                    f"   •   {format_bytes(speed)}/s   •   ETA {self._fmt_time(eta)}")
            else:
                self.detail_var.set(f"Elapsed: {self._fmt_time(elapsed)}")
        if self.top.winfo_exists():
            self.top.after(150, self._tick)

    def set_action(self, text: str, dest: str = ""):
        self.action_var.set(text)
        if dest:
            self.dest_var.set(f"→ {dest}")
        self.top.update_idletasks()

    def set_total(self, total_bytes: Optional[int]):
        with self._lock:
            self._total_bytes = total_bytes
            self._done_bytes = 0
        if total_bytes and total_bytes > 0:
            if self._marquee:
                self.pb.stop()
                self.pb.config(mode="determinate")
                self._marquee = False
            self.progress_var.set(0.0)
            self.percent_var.set("0%")
        else:
            if not self._marquee:
                self.pb.config(mode="indeterminate")
                self.pb.start(24)
                self._marquee = True
            self.percent_var.set("")
        self._start_time = time.time()
        self.top.update_idletasks()

    def bump(self, bytes_inc: int):
        with self._lock:
            self._done_bytes += max(0, bytes_inc)
            if self._total_bytes and self._total_bytes > 0:
                pct = max(0.0, min(100.0, (self._done_bytes / self._total_bytes) * 100.0))
                self.progress_var.set(pct)
                self.percent_var.set(f"{pct:0.1f}%")
        self.top.update_idletasks()

    def set_done(self):
        with self._lock:
            if self._total_bytes and self._total_bytes > 0:
                self.progress_var.set(100.0); self.percent_var.set("100%")
        self.top.update_idletasks()

    def close(self):
        try:
            self.top.grab_release()
        except Exception:
            pass
        try:
            if self._marquee:
                self.pb.stop()
        except Exception:
            pass
        self.top.destroy()

    @staticmethod
    def _fmt_time(sec: float) -> str:
        if math.isinf(sec) or sec > 9e6:
            return "—"
        sec = int(sec)
        h, r = divmod(sec, 3600); m, s = divmod(r, 60)
        if h: return f"{h:d}h {m:02d}m {s:02d}s"
        if m: return f"{m:d}m {s:02d}s"
        return f"{s:d}s"

# ---------------- CUDA installer (optional, when user wants GPU) ----------------
def install_cuda_runtime_with_progress(parent: tk.Misc) -> bool:
    """
    Installs NVIDIA runtime wheels from NVIDIA's PyPI (cudart, cublas, cudnn).
    Shows indeterminate progress during pip, then re-checks readiness.
    """
    prog = ProgressWindow(parent, title="CUDA Setup")
    ok = {"val": False}
    def worker():
        try:
            pkgs = ["nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"]
            cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                   "--extra-index-url", "https://pypi.nvidia.com", *pkgs]
            prog.set_action("Installing NVIDIA CUDA components…")
            prog.set_total(None)
            rc = _popen_stream(cmd, timeout=600)
            # After pip, update PATH/DLL search and verify:
            if rc == 0:
                # Give file system a moment then re-check
                time.sleep(0.5)
                if cuda_runtime_ready():
                    ok["val"] = True
        finally:
            prog.set_done()
            prog.close()
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    prog.top.wait_window(prog.top)
    return ok["val"]

# ---------------- Model download (progress) ----------------
class CancelledDownload(Exception): pass

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

class DownloadDialog:
    def __init__(self, title: str, progress_queue: Queue, cancel_event: threading.Event):
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.result: Optional[str] = None

        self.root = tk.Toplevel(tk_root or tk.Tk())
        self.root.title(title)
        self.root.geometry("520x220")
        self.root.minsize(420, 200)
        self.root.resizable(True, True)
        self.root.attributes("-topmost", True)

        title_lbl = tk.Label(self.root, text=title, font=("Segoe UI", 12, "bold"))
        title_lbl.pack(pady=(12, 4))

        self.stage_var = tk.StringVar(value="Preparing download…")
        tk.Label(self.root, textvariable=self.stage_var, font=("Segoe UI", 10)).pack(pady=(0, 4))

        self.progress = ttk.Progressbar(self.root, length=460, mode="determinate", maximum=100)
        self.progress.pack(pady=(0, 4), fill=tk.X, padx=16)

        self.status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 9)).pack(pady=(0, 8))

        btns = ttk.Frame(self.root); btns.pack(pady=(0, 10))
        self.cancel_button = ttk.Button(btns, text="Cancel", command=self.cancel)
        self.cancel_button.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.cancel)

    def cancel(self) -> None:
        if self.result is None:
            self.result = "cancelled"
            self.cancel_event.set()
            self.status_var.set("Cancelling…")

    def _update_progress(self, desc: str, current: float, total: float) -> None:
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
                typ = message[0]
                if typ == "progress":
                    _, desc, cur, tot = message
                    self._update_progress(desc, cur, tot)
                elif typ == "stage":
                    _, desc = message
                    self.stage_var.set(desc)
                elif typ == "error":
                    _, error_text = message
                    self.result = "error"
                    self.status_var.set(error_text)
                elif typ == "done":
                    self.result = "success"
                    self.status_var.set("Download completed.")
                elif typ == "cancelled":
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

def download_model_with_gui(model_short: str) -> bool:
    progress_queue: Queue = Queue()
    cancel_event = threading.Event()
    dialog = DownloadDialog(f"Download Whisper model ({model_short})", progress_queue, cancel_event)

    def progress_callback(desc: str, current: float, total: float) -> None:
        progress_queue.put(("progress", desc, current, total))

    def worker() -> None:
        try:
            repo_id = model_repo_id()  # uses current model name
            progress_queue.put(("stage", f"Connecting to Hugging Face ({repo_id})"))
            target_dir = model_store_path_for(model_short)
            target_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=4,
                tqdm_class=make_tqdm_class(progress_callback, cancel_event),
            )
            if cancel_event.is_set():
                progress_queue.put(("cancelled",))
                return
            (target_dir / ".installed").touch(exist_ok=True)
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

def ensure_model_available() -> bool:
    """Checks/installs model selected in settings."""
    name = get_current_model_name()
    store = model_store_path_for(name)
    marker = store / ".installed"
    if model_ready.is_set() and model_files_present(store):
        return True
    if not model_files_present(store) or not marker.exists():
        # Ask user through a simple confirm box (non-blocking style preserved)
        prompt = (f"CtrlSpeak needs to download the Whisper model '{name}'.\n"
                  "This is a one-time download. Continue?")
        try:
            choice = pyautogui.confirm(text=prompt, title="CtrlSpeak Setup",
                                       buttons=["Install Now", "Quit"])
        except Exception:
            choice = "Install Now"
        if choice != "Install Now":
            return False
        if not download_model_with_gui(name):
            return False
    if model_files_present(store):
        model_ready.set()
        return True
    return False

# ---------------- Device resolution (now respects settings UI) ----------------
def resolve_device() -> str:
    if CLIENT_ONLY_BUILD:
        return "cpu"
    preference = get_device_preference()
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        return "cuda" if cuda_runtime_ready() else "cpu"
    # auto
    return "cuda" if cuda_runtime_ready() else "cpu"

def resolve_compute_type(device: str) -> str:
    if COMPUTE_TYPE_OVERRIDE:
        return COMPUTE_TYPE_OVERRIDE
    if device == "cuda":
        return "float16"
    if device == "cpu":
        return "int8"
    return "int8_float16"

# ---------------- Transcriber init (minimal changes) ----------------
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

        device_pref = preferred_device or resolve_device()
        if device_pref == "cpu" and get_device_preference() in {"auto", "cuda"} and not warned_cuda_unavailable and preferred_device is None:
            notify("CUDA dependencies were not detected. CtrlSpeak will run Whisper on the CPU instead.")
            warned_cuda_unavailable = True

        compute_type = resolve_compute_type(device_pref)

        try:
            # IMPORTANT: use selected model name and shared cache root
            model_name = get_current_model_name()
            whisper_model = WhisperModel(
                model_name,
                device=device_pref,
                compute_type=compute_type,
                download_root=str(get_model_root()),
            )
            print(f"Whisper model ready on {device_pref} ({compute_type})")
            return whisper_model
        except Exception as exc:
            print(f"Failed to load model on {device_pref}: {exc}")
            if device_pref != "cpu":
                try:
                    whisper_model = WhisperModel(
                        get_current_model_name(),
                        device="cpu",
                        compute_type="int8",
                        download_root=str(get_model_root()),
                    )
                    notify("Running CtrlSpeak transcription on CPU fallback.")
                    warned_cuda_unavailable = True
                    return whisper_model
                except Exception as cpu_exc:
                    print(f"CPU fallback failed: {cpu_exc}")
            notify("Unable to initialize the transcription model. Please check your installation and try again.")
            whisper_model = None
            return None

# ---------------- Audio capture + playback (unchanged) ----------------
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
        base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
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

def collect_text_from_segments(segments) -> str:
    return " ".join([s.text.strip() for s in segments if s.text.strip()])

# ---------------- Tray + discovery (unchanged behavior) ----------------
def create_icon_image():
    icon_path = resource_path("icon.ico")
    return Image.open(icon_path)

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

# -------- Management UI thread pump (unchanged design; windows now resizable) --------
def enqueue_management_task(func: Callable[..., None], *args, **kwargs) -> None:
    try:
        management_ui_queue.put_nowait((func, args, kwargs))
    except Exception:
        pass

def ensure_management_ui_thread() -> None:
    global management_ui_thread
    if management_ui_thread and management_ui_thread.is_alive():
        return
    management_ui_thread = threading.Thread(target=_management_ui_loop, name="CtrlSpeakManagementUI", daemon=True)
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

# ---------------- Server discovery (unchanged) ----------------
@dataclass
class ServerInfo:
    host: str
    port: int
    last_seen: float

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
        for key in expired: self.registry.pop(key, None)

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

def ensure_preferred_server_registered(probe: bool = False) -> Optional[ServerInfo]:
    host, port = get_preferred_server_settings()
    if host is None or port is None: return None
    if probe and not probe_server(host, port): return None
    server_info = ServerInfo(host=host, port=int(port), last_seen=time.time())
    if discovery_listener is not None:
        discovery_listener.registry[(server_info.host, server_info.port)] = server_info
    global last_connected_server
    last_connected_server = server_info
    schedule_management_refresh()
    return server_info

# ---------------- Keyboard + transcription flow (unchanged) ----------------
def on_press(key):
    global recording, recording_thread, recording_file_path
    if not client_enabled: return
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

# ---------------- Insert text (unchanged) ----------------
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

def window_matches_anydesk(hwnd: int) -> bool:
    class_name = get_class_name(hwnd)
    if 'anydesk' in class_name: return True
    process_name = get_window_process_name(hwnd)
    if 'anydesk' in process_name: return True
    title = get_window_text(hwnd).lower()
    if 'anydesk' in title: return True
    return False

def is_console_window(hwnd: int) -> bool:
    if not sys.platform.startswith("win") or not hwnd: return False
    class_name = get_class_name(hwnd)
    if class_name in {"consolewindowclass"}: return True
    if "cascadia" in class_name: return True
    process_name = get_window_process_name(hwnd)
    return process_name in {"conhost.exe", "openconsole.exe", "wt.exe", "windowsterminal.exe", "ubuntu.exe", "wsl.exe"}

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", wintypes.LPARAM)]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", wintypes.LPARAM)]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]

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
    console_window = bool(hwnd and get_class_name(hwnd) in {"consolewindowclass"} or (hwnd and "cascadia" in get_class_name(hwnd)))
    if hwnd and (FORCE_SENDINPUT or window_matches_anydesk(hwnd) or console_window):
        if send_unicode_input(text): return
        if try_sendinput_paste(text): return
    if not console_window and try_direct_text_insert(text, hwnd): return
    if FORCE_SENDINPUT and not console_window:
        if send_unicode_input(text): return
    if try_sendinput_paste(text): return
    if try_clipboard_paste(text): return
    pyautogui.write(text)

# ---------------- Transcription routing (unchanged) ----------------
def get_best_server() -> Optional[ServerInfo]:
    if discovery_listener is None:
        return ensure_preferred_server_registered(probe=True)
    server = discovery_listener.get_best_server()
    if server: return server
    return ensure_preferred_server_registered(probe=True)

def transcribe_local(file_path: str, play_feedback: bool = True, allow_client: bool = False, preferred_device: Optional[str] = None) -> Optional[str]:
    global last_connected_server
    model = initialize_transcriber(allow_client=allow_client, preferred_device=preferred_device)
    if model is None: return None
    if play_feedback: start_processing_feedback()
    try:
        segments, _ = model.transcribe(file_path, beam_size=5, vad_filter=True, temperature=0.2)
        text = collect_text_from_segments(segments)
        if text:
            with settings_lock:
                port = int(settings.get("server_port", 65432))
            if allow_client and preferred_device == "cpu":
                last_connected_server = ServerInfo(host="local-cpu", port=-1, last_seen=time.time())
            else:
                last_connected_server = ServerInfo(host="local", port=port, last_seen=time.time())
        return text or None
    except Exception as exc:
        notify_error("Transcription failed", format_exception_details(exc)); return None
    finally:
        if play_feedback: stop_processing_feedback()

def get_advertised_host_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tmp:
            tmp.connect(("8.8.8.8", 80))
            return tmp.getsockname()[0]
    except Exception:
        return "127.0.0.1"

def describe_server_status() -> str:
    with settings_lock:
        mode = settings.get("mode")
        port = int(settings.get("server_port", 65432))
    if mode == "client_server" and server_thread and server_thread.is_alive():
        host = last_connected_server.host if last_connected_server else get_advertised_host_ip()
        return f"Serving: {host}:{port}"
    if last_connected_server:
        host = last_connected_server.host; port = last_connected_server.port
        label = "local CPU" if host == "local-cpu" else ("local" if host == "local" else f"{host}:{port}")
        return f"Connected: {label}"
    server = get_best_server()
    if server: return f"Discovered: {server.host}:{server.port}"
    return "Not connected"

def transcribe_remote(file_path: str, play_feedback: bool = True) -> Optional[str]:
    server = get_best_server()
    if server is None:
        # fall back like your original
        return handle_missing_server(file_path, play_feedback=play_feedback)
    if play_feedback: start_processing_feedback()
    conn = None
    try:
        with open(file_path, "rb") as handle:
            data = handle.read()
        conn = http.client.HTTPConnection(server.host, server.port, timeout=180)
        headers = {"Content-Type": "audio/wav", "Content-Length": str(len(data)), "X-CtrlSpeak-Client": local_hostname}
        conn.request("POST", "/transcribe", body=data, headers=headers)
        response = conn.getresponse(); payload = response.read()
        if response.status != 200:
            snippet = payload[:400].decode("utf-8", errors="replace")
            detail = f"Server {server.host}:{server.port} responded with HTTP {response.status} {response.reason}\n{snippet}"
            notify_error("Remote server error", detail); return None
        try:
            data_json = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            notify_error("Remote response parsing failed", format_exception_details(exc)); return None
        return data_json.get("text")
    except Exception as exc:
        notify_error("Remote transcription failed", format_exception_details(exc)); return None
    finally:
        if play_feedback: stop_processing_feedback()
        if conn is not None:
            try: conn.close()
            except Exception: pass

def handle_missing_server(file_path: str, play_feedback: bool) -> Optional[str]:
    # unchanged fallback behavior from your original
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
    if model is None: return None
    start_server()
    return transcribe_local(file_path, play_feedback=play_feedback)

def prompt_enable_local_server() -> bool:
    message = ("CtrlSpeak cannot reach a server on this network.\n\n"
               "Would you like to enable the server on this computer?")
    try:
        choice = pyautogui.confirm(text=message, title="CtrlSpeak", buttons=["Install Server", "Cancel"])
    except Exception:
        print(message); choice = None
    return choice == "Install Server"

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

# ---------------- Management window (RESIZABLE + device/model controls) ----------------
def schedule_management_refresh(delay_ms: int = 0) -> None:
    if tk_root is None: return
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

def open_management_dialog(icon, item):
    ensure_management_ui_thread()
    enqueue_management_task(_show_management_window, icon)

def _show_management_window(icon: pystray.Icon) -> None:
    global management_window
    if management_window and management_window.is_open():
        management_window.bring_to_front(); management_window.refresh_status(); return
    management_window = ManagementWindow(icon)

class ManagementWindow:
    def __init__(self, icon: pystray.Icon):
        self._icon = icon
        self.window = tk.Toplevel(tk_root)
        self.window.title(f"CtrlSpeak Control v{APP_VERSION}")
        self.window.geometry("560x520")
        self.window.minsize(520, 480)
        self.window.resizable(True, True)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda _event: self.close())
        try:
            self.window.iconbitmap(resource_path("icon.ico"))
        except Exception:
            pass

        wrap = ttk.Frame(self.window, padding=(16, 14)); wrap.pack(fill=tk.BOTH, expand=True)
        ttk.Label(wrap, text="CtrlSpeak Control", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        build_label = "Client Only" if CLIENT_ONLY_BUILD else "Client + Server"
        ttk.Label(wrap, text=f"Build {APP_VERSION} - {build_label}", font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 10))

        # Status
        self.status_var = tk.StringVar()
        self.server_status_var = tk.StringVar()
        ttk.Label(wrap, textvariable=self.status_var, justify=tk.LEFT, anchor="w").pack(fill=tk.X, pady=(0, 6))
        ttk.Label(wrap, textvariable=self.server_status_var, font=("Segoe UI", 10, "italic"),
                  justify=tk.LEFT, anchor="w").pack(fill=tk.X, pady=(0, 10))

        # --- Device selection (CPU/GPU) ---
        device_frame = ttk.LabelFrame(wrap, text="Device")
        device_frame.pack(fill=tk.X, pady=(0, 10))
        self.device_var = tk.StringVar(value=get_device_preference())
        row = ttk.Frame(device_frame); row.pack(fill=tk.X, padx=6, pady=6)
        ttk.Radiobutton(row, text="CPU", variable=self.device_var, value="cpu").pack(side=tk.LEFT, padx=(0,12))
        ttk.Radiobutton(row, text="GPU (CUDA)", variable=self.device_var, value="cuda").pack(side=tk.LEFT)

        self.cuda_status = tk.StringVar()
        ttk.Label(device_frame, textvariable=self.cuda_status).pack(anchor="w", padx=6, pady=(2,8))
        btns_dev = ttk.Frame(device_frame); btns_dev.pack(fill=tk.X, padx=6, pady=(0,10))
        self.apply_device_btn = ttk.Button(btns_dev, text="Apply Device", command=self._apply_device)
        self.apply_device_btn.pack(side=tk.LEFT)
        self.install_cuda_btn = ttk.Button(btns_dev, text="Install/Repair CUDA…", command=self._install_cuda)
        self.install_cuda_btn.pack(side=tk.LEFT, padx=(8,0))

        # --- Model selection ---
        model_frame = ttk.LabelFrame(wrap, text="Model")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        self.model_var = tk.StringVar(value=get_current_model_name())
        rowm = ttk.Frame(model_frame); rowm.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(rowm, text="Whisper model:").pack(side=tk.LEFT)
        ttk.Combobox(rowm, textvariable=self.model_var, values=["small", "large-v3"],
                     state="readonly", width=18).pack(side=tk.LEFT, padx=(8,0))
        btns_m = ttk.Frame(model_frame); btns_m.pack(fill=tk.X, padx=6, pady=(6,10))
        self.apply_model_btn = ttk.Button(btns_m, text="Set Model", command=self._apply_model)
        self.apply_model_btn.pack(side=tk.LEFT)
        self.download_model_btn = ttk.Button(btns_m, text="Download/Update…", command=self._download_model)
        self.download_model_btn.pack(side=tk.LEFT, padx=(8,0))
        self.model_status = tk.StringVar()
        ttk.Label(model_frame, textvariable=self.model_status).pack(anchor="w", padx=6, pady=(4,0))

        # Client/Server controls (unchanged)
        ctrl = ttk.LabelFrame(wrap, text="Client & Server")
        ctrl.pack(fill=tk.X, pady=(0,10))
        self.stop_button = ttk.Button(ctrl, text="Stop Client", command=self.stop_client); self.stop_button.pack(fill=tk.X, pady=2, padx=6)
        self.start_button = ttk.Button(ctrl, text="Start Client", command=self.start_client); self.start_button.pack(fill=tk.X, pady=2, padx=6)
        self.refresh_button = ttk.Button(ctrl, text="Refresh Servers", command=self.refresh_servers); self.refresh_button.pack(fill=tk.X, pady=2, padx=6)
        exit_label = "Exit CtrlSpeak" if CLIENT_ONLY_BUILD else "Stop Everything"
        self.stop_all_button = ttk.Button(ctrl, text=exit_label, command=self.stop_everything)
        self.stop_all_button.pack(fill=tk.X, pady=(8, 6), padx=6)

        ttk.Separator(wrap).pack(fill=tk.X, pady=(8, 10))
        ttk.Button(wrap, text="Close", command=self.close).pack(fill=tk.X)
        delete_btn = tk.Button(wrap, text="Delete CtrlSpeak", command=self.delete_ctrlspeak,
                               bg="#d32f2f", fg="white", activebackground="#b71c1c", activeforeground="white")
        delete_btn.pack(fill=tk.X, pady=(6,0))

        self.window.after(120, self.refresh_status)
        self.bring_to_front()

    def is_open(self) -> bool:
        return bool(self.window and self.window.winfo_exists())

    def bring_to_front(self) -> None:
        if not self.is_open(): return
        self.window.deiconify(); self.window.lift(); self.window.focus_force()
        self.window.attributes("-topmost", True)
        self.window.after(150, lambda: self.window.attributes("-topmost", False))

    def refresh_status(self) -> None:
        with settings_lock:
            mode = settings.get("mode") or "unknown"
        device_pref = get_device_preference()
        cuda_ok = cuda_runtime_ready()
        model_name = get_current_model_name()
        present = model_files_present(model_store_path_for(model_name))
        status_parts = [
            f"Mode: {mode}",
            "Client Only build" if CLIENT_ONLY_BUILD else "Full build",
            f"Client: {'Active' if client_enabled else 'Stopped'}",
            f"Device preference: {device_pref.upper()}  •  CUDA ready: {'Yes' if cuda_ok else 'No'}",
            f"Model: {model_name}  •  Installed: {'Yes' if present else 'No'}",
        ]
        server_running = server_thread and server_thread.is_alive()
        status_parts.append(f"Server thread: {'Running' if server_running else 'Not running'}")
        self.status_var.set("\n".join(status_parts))
        self.server_status_var.set(describe_server_status())
        self.cuda_status.set("CUDA runtime is available." if cuda_ok else "CUDA runtime not detected.")
        self.model_status.set("Model is installed." if present else "Model not installed.")
        if client_enabled:
            self.start_button.state(["disabled"]); self.stop_button.state(["!disabled"])
        else:
            self.start_button.state(["!disabled"]); self.stop_button.state(["disabled"])

    # Device actions
    def _apply_device(self):
        choice = self.device_var.get()
        if choice not in {"cpu", "cuda", "auto"}: choice = "auto"
        set_device_preference(choice)
        if choice == "cuda" and not cuda_runtime_ready():
            if messagebox.askyesno("CUDA not ready",
                                   "CUDA runtime not detected.\n\nInstall now for GPU acceleration?",
                                   parent=self.window):
                if install_cuda_runtime_with_progress(self.window):
                    messagebox.showinfo("CUDA", "CUDA runtime installed successfully.", parent=self.window)
                else:
                    messagebox.showwarning("CUDA", "Failed to prepare CUDA. Staying on CPU.", parent=self.window)
        self.refresh_status()

    def _install_cuda(self):
        if install_cuda_runtime_with_progress(self.window):
            messagebox.showinfo("CUDA", "CUDA runtime installed successfully.", parent=self.window)
        else:
            messagebox.showwarning("CUDA", "Failed to prepare CUDA. You can try again.", parent=self.window)
        self.refresh_status()

    # Model actions
    def _apply_model(self):
        name = self.model_var.get()
        if name not in {"small", "large-v3"}: name = "large-v3"
        set_current_model_name(name)
        messagebox.showinfo("Model", f"Active model set to {name}.", parent=self.window)
        self.refresh_status()

    def _download_model(self):
        name = self.model_var.get()
        if name not in {"small", "large-v3"}: name = "large-v3"
        if download_model_with_gui(name):
            (model_store_path_for(name) / ".installed").touch(exist_ok=True)
            messagebox.showinfo("Model", "Model downloaded successfully.", parent=self.window)
        else:
            messagebox.showwarning("Model", "Model download did not complete.", parent=self.window)
        self.refresh_status()

    # Client/server actions (kept)
    def start_client(self) -> None:
        start_client_listener(); self.refresh_status()

    def stop_client(self) -> None:
        stop_client_listener(); self.refresh_status()

    def refresh_servers(self) -> None:
        self.refresh_button.state(["disabled"]); self.refresh_button.config(text="Scanning…")
        self.server_status_var.set("Scanning for servers…")
        def worker() -> None:
            try: manual_discovery_refresh()
            finally: enqueue_management_task(self._on_refresh_finished)
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
        if not messagebox.askyesno("Delete CtrlSpeak", "This will remove CtrlSpeak and all local data. Continue?", parent=self.window):
            return
        self.window.after(100, lambda: initiate_self_uninstall(self._icon))

    def close(self) -> None:
        global management_window
        if self.is_open(): self.window.destroy()
        management_window = None

# ---------------- Server (unchanged) ----------------
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

# ---------------- Tray integration (unchanged: NO management window at startup) ----------------
def on_exit(icon, item):
    stop_client_listener(); shutdown_server(); icon.stop()

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

# ---------------- First-run mode prompt (unchanged but resizable) ----------------
def prompt_initial_mode() -> Optional[str]:
    if AUTO_MODE_PROFILE:
        return AUTO_MODE_PROFILE
    result = {"mode": None}
    def choose(mode: str) -> None:
        result["mode"] = mode; root.destroy()
    root = tk.Tk()
    root.title("Welcome to CtrlSpeak"); root.geometry("480x300")
    root.minsize(440, 260); root.resizable(True, True); root.attributes("-topmost", True)
    title = tk.Label(root, text="Welcome to CtrlSpeak", font=("Segoe UI", 14, "bold")); title.pack(pady=(18, 8))
    message = ("It looks like this is the first time CtrlSpeak is running on this computer.\n\n"
               "Choose how you would like to use it:")
    tk.Message(root, text=message, width=420, font=("Segoe UI", 10)).pack(pady=(0, 12))
    button_frame = tk.Frame(root); button_frame.pack(pady=(0, 12), fill=tk.X, padx=16)
    def make_button(text: str, desc: str, mode_value: str) -> None:
        frame = tk.Frame(button_frame, relief=tk.RIDGE, borderwidth=1)
        frame.pack(padx=8, pady=6, fill=tk.X)
        tk.Label(frame, text=text, font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, padx=8, pady=(6, 0))
        tk.Message(frame, text=desc, width=360).pack(anchor=tk.W, padx=8)
        tk.Button(frame, text="Select", command=lambda: choose(mode_value)).pack(padx=8, pady=(0, 8), anchor=tk.E)
    make_button("Client + Server",
                "Use this computer for local transcription and share it with other CtrlSpeak clients on the network.",
                "client_server")
    make_button("Client Only",
                "Send recordings to another CtrlSpeak server on your local network for transcription.",
                "client")
    def cancel() -> None: root.destroy()
    tk.Button(root, text="Quit", command=cancel).pack(pady=(0, 12))
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.mainloop()
    return result["mode"]

def ensure_mode_selected() -> None:
    with settings_lock:
        current_mode = settings.get("mode")
    if CLIENT_ONLY_BUILD:
        if current_mode != "client":
            with settings_lock: settings["mode"] = "client"; save_settings()
        return
    if current_mode == "client":
        with settings_lock: settings["mode"] = "client_server"; save_settings(); current_mode = "client_server"
    if current_mode == "client_server": return
    choice = prompt_initial_mode()
    if not choice:
        notify("CtrlSpeak cannot continue without selecting a mode."); sys.exit(0)
    with settings_lock:
        settings["mode"] = choice
    save_settings()

# ---------------- Single instance + CLI (unchanged) ----------------
def acquire_single_instance_lock() -> bool:
    global instance_lock_handle, lock_file_path
    with settings_lock:
        lock_path = get_config_dir() / LOCK_FILENAME
    lock_file_path = lock_path
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
    global instance_lock_handle, lock_file_path
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
        if lock_file_path is not None:
            try: lock_file_path.unlink(missing_ok=True)
            except Exception: pass
            lock_file_path = None

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
    AUTO_MODE = True; AUTO_MODE_PROFILE = profile
    with settings_lock: settings["mode"] = profile
    save_settings()

# ---------------- Uninstall helper (unchanged) ----------------
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

# ---------------- Discovery glue (unchanged) ----------------
def start_discovery_listener() -> None:
    global discovery_listener
    with settings_lock:
        port = int(settings.get("discovery_port", 54330))
    discovery_listener = DiscoveryListener(port); discovery_listener.start()

def shutdown_all():
    stop_client_listener(); shutdown_server()
    if discovery_listener: discovery_listener.stop()

# ---------------- Main ----------------
def transcribe_cli(target: str) -> int:
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

def main(argv: list[str]) -> int:
    global CLIENT_ONLY_BUILD
    CLIENT_ONLY_BUILD = detect_client_only_build()

    args = parse_cli_args(argv)
    if args.force_sendinput:
        global FORCE_SENDINPUT; FORCE_SENDINPUT = True
    if args.uninstall:
        initiate_self_uninstall(None); return 0

    load_settings()
    if args.auto_setup: apply_auto_setup(args.auto_setup)
    cli_mode = args.transcribe is not None

    if not acquire_single_instance_lock():
        message = "CtrlSpeak is already running. Please close the existing instance before starting a new one."
        if cli_mode: print(message, file=sys.stderr)
        else: notify(message)
        return 0

    if not cli_mode and not AUTO_MODE:
        show_splash_screen(SPLASH_DURATION_MS)

    # First-run mode selection (resizable dialog)
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
        # Initialize early so server can use it if needed
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
