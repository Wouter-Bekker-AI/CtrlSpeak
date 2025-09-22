# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
import threading
import subprocess
from contextlib import contextmanager
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, List, Set, Callable, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import ctypes
import ctranslate2
from faster_whisper import WhisperModel
try:  # pragma: no cover - optional dependency rarely installed
    import hf_xet  # type: ignore  # noqa: F401
except ImportError:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi
import tkinter as tk
from tkinter import ttk

from utils.system import (
    settings, settings_lock,
    notify, notify_error, format_exception_details,
    get_config_dir, save_settings,
    start_processing_feedback, stop_processing_feedback,
    ui_show_activation_popup, ui_close_activation_popup,
    ui_show_lockout_window, ui_update_lockout_message, ui_close_lockout_window,
    pump_management_events_once,
)
from utils.system import get_best_server, CLIENT_ONLY_BUILD
from utils.ui_theme import apply_modern_theme
from utils.config_paths import get_logger


# ---------------- Env / defaults ----------------
# These are *defaults*. Active choices come from settings via getters below.
AVAILABLE_MODELS: tuple[str, ...] = ("small", "large-v3")
DEFAULT_MODEL_NAME = AVAILABLE_MODELS[0]
ENV_DEFAULT_MODEL = os.environ.get("CTRLSPEAK_MODEL", DEFAULT_MODEL_NAME)
ENV_DEVICE_PREF   = os.environ.get("CTRLSPEAK_DEVICE", "cpu").lower()
COMPUTE_TYPE_OVERRIDE = os.environ.get("CTRLSPEAK_COMPUTE_TYPE")
MODEL_REPO_OVERRIDE   = os.environ.get("CTRLSPEAK_MODEL_REPO")

# All model content under %APPDATA%/CtrlSpeak/models
MODEL_ROOT_PATH = get_config_dir() / "models"

# Hard-coded download targets (bytes) for progress tracking
EXPECTED_MODEL_BYTES: dict[str, int] = {
    "small": 463 * (1024 ** 2),
    "large-v3": int(2.87 * (1024 ** 3)),
}

CUDA_RUNTIME_EXPECTED_BYTES = int(1.2 * (1024 ** 3))

# CUDA search: also look in %APPDATA%/CtrlSpeak/cuda
cuda_paths_initialized = False

logger = get_logger(__name__)


# ---------------- Settings-backed getters/setters ----------------
def get_device_preference() -> str:
    """Returns 'cpu' | 'cuda' | 'auto' (settings first, env fallback)."""
    with settings_lock:
        pref = settings.get("device_preference")
    if pref in {"cpu", "cuda", "auto"}:
        return str(pref)
    return ENV_DEVICE_PREF if ENV_DEVICE_PREF in {"cpu", "cuda", "auto"} else "auto"


def set_device_preference(pref: str) -> None:
    if pref not in {"cpu", "cuda", "auto"}:
        pref = "auto"
    with settings_lock:
        settings["device_preference"] = pref
    save_settings()


def _normalize_model_name(name: object) -> Optional[str]:
    if not isinstance(name, str):
        return None
    cleaned = name.strip()
    return cleaned or None


def get_current_model_name() -> str:
    """Reads the selected model, falling back to an installed/default option."""

    with settings_lock:
        stored_name = settings.get("model_name")

    preferred = _normalize_model_name(stored_name)
    env_preferred = _normalize_model_name(ENV_DEFAULT_MODEL)

    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    if env_preferred and env_preferred in AVAILABLE_MODELS and env_preferred not in candidates:
        candidates.append(env_preferred)
    for known in AVAILABLE_MODELS:
        if known not in candidates:
            candidates.append(known)

    for candidate in candidates:
        if _is_model_installed(candidate):
            if candidate != preferred:
                set_current_model_name(candidate)
            return candidate

    fallback = AVAILABLE_MODELS[0] if AVAILABLE_MODELS else DEFAULT_MODEL_NAME
    if preferred != fallback:
        set_current_model_name(fallback)
    return fallback


def set_current_model_name(name: str) -> None:
    """Persists selected model (call initialize_transcriber(force=True) to live-switch)."""
    with settings_lock:
        settings["model_name"] = name
    save_settings()


# ---------------- Model storage (AppData/CtrlSpeak/models) ----------------
def model_store_path_for(model_short: str) -> Path:
    return MODEL_ROOT_PATH / model_short


def model_files_present(model_path: Path) -> bool:
    if not model_path.exists():
        return False
    # faster-whisper weights are .bin files
    return any(model_path.rglob("*.bin"))


def _is_model_installed(model_short: str) -> bool:
    store = model_store_path_for(model_short)
    marker = store / ".installed"
    return model_files_present(store) and marker.exists()


def _model_repo_id(model_short: str) -> str:
    if MODEL_REPO_OVERRIDE:
        return MODEL_REPO_OVERRIDE
    # Your builds use Systran/faster-whisper-<name>
    return f"Systran/faster-whisper-{model_short}"


def _model_activation_cache_path(model_short: str) -> Path:
    repo_id = _model_repo_id(model_short)
    safe_repo = repo_id.replace("/", "--")
    return MODEL_ROOT_PATH / f"models--{safe_repo}"


# ---------------- CUDA lookup / readiness ----------------
def get_cuda_dll_dirs() -> list[Path]:
    paths = []
    # user runtime in AppData (preferred first)
    cuda_root = get_config_dir() / "cuda"
    for sub in ("bin", "cudnn/bin", "cublas/bin"):
        paths.append(cuda_root / sub)
    # embedded bundle (if any, e.g., PyInstaller)
    bundle_base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    paths.append(bundle_base / "nvidia" / "cudnn" / "bin")
    paths.append(bundle_base / "nvidia" / "cublas" / "bin")
    # site-packages wheels (if present)
    paths.append(Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin")
    paths.append(Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin")
    return paths


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
            except (AttributeError, FileNotFoundError, OSError) as exc:
                logger.warning("Unable to add CUDA DLL directory %s: %s", dir_str, exc)
            if dir_str not in path_var:
                path_var = dir_str + os.pathsep + path_var
    os.environ["PATH"] = path_var
    cuda_paths_initialized = True


def cuda_runtime_ready(*, ignore_preference: bool = False) -> bool:
    # ⬅️ New: never probe CUDA when the user picked CPU — avoids native crash
    try:
        if not ignore_preference and get_device_preference() == "cpu":
            return False
    except Exception:
        logger.exception("Failed to read device preference while checking CUDA readiness")
        # If settings aren't ready yet, fall through safely.
    configure_cuda_paths()
    try:
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception:
        logger.exception("CUDA device probe failed")
        return False
    if sys.platform.startswith("win"):
        for dll in ("cudnn_ops64_9.dll", "cublas64_12.dll"):
            try:
                ctypes.windll.LoadLibrary(dll)
            except OSError as exc:
                logger.warning("Failed to load CUDA DLL %s: %s", dll, exc)
                return False
    return True



def _run_cmd_stream(cmd: List[str], timeout: int | None = None) -> int:
    """Run a command, stream output, return returncode."""
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
        logger.exception("Command execution failed: %s", " ".join(cmd))
        return 1


def install_cuda_runtime_with_progress(parent=None) -> bool:
    """
    Headless CUDA installer (pip NVIDIA wheels). GUI wrappers can messagebox around it.
    Looks in %APPDATA%\\CtrlSpeak\\cuda as well afterwards.
    """
    pkgs = ["nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"]
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--quiet", "--disable-pip-version-check",
        "--extra-index-url", "https://pypi.nvidia.com",
        *pkgs
    ]
    with activation_guard(
        "installing CUDA runtime components",
        busy_message=(
            "CtrlSpeak is installing CUDA runtime components required for GPU transcription. "
            "Please wait for this to finish."
        ),
        success_message=(
            "CUDA runtime components are ready. CtrlSpeak will continue preparing the Whisper model."
        ),
        failure_message=(
            "CUDA runtime installation did not complete. CtrlSpeak will continue without GPU acceleration."
        ),
    ) as complete:
        rc = _run_cmd_stream(cmd, timeout=600)
        if rc != 0:
            notify("CUDA runtime installation failed (pip returned non-zero).")
            complete(False, "CUDA runtime installation failed. CtrlSpeak will continue using the CPU.")
            return False
        # refresh DLL search path + verify
        time.sleep(0.5)
        ready = cuda_runtime_ready(ignore_preference=True)
        if not ready:
            complete(False, "CtrlSpeak installed the CUDA runtime but could not validate the GPU libraries.")
            return False
        complete(True, "CUDA runtime components were installed successfully. Continuing activation…")
        return True


# ---------------- Download dialog (GUI) ----------------
class CancelledDownload(Exception): pass


def format_bytes(value: float) -> str:
    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    amount = float(value)
    for u in units:
        if amount < step:
            return f"{amount:.1f} {u}"
        amount /= step
    return f"{amount:.1f} PB"


MB_DIVISOR = 1024.0 * 1024.0
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_USER_AGENT = "CtrlSpeak Downloader/1.0"


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, secs = divmod(int(seconds + 0.5), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


_dir_size_command_failures: Set[str] = set()


def _measure_directory_size_python(path: Path) -> int:
    total = 0
    try:
        iterator = path.rglob("*")
    except Exception:
        return 0
    for entry in iterator:
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _measure_directory_size(path: Path) -> int:
    if not path.exists():
        return 0

    command_key = "win" if os.name.startswith("nt") else "posix"
    try:
        if os.name.startswith("nt"):
            path_str = str(path)
            safe_path = path_str.replace("'", "''")
            command = [
                "powershell",
                "-NoLogo",
                "-NoProfile",
                "-Command",
                (
                    "(Get-ChildItem -LiteralPath '"
                    f"{safe_path}"
                    "' -Recurse -Force -File | Measure-Object -Property Length -Sum).Sum"
                ),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            output = result.stdout.strip()
            return int(output) if output else 0

        command = ["du", "-sb", str(path)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        size_str = output.split("\t", 1)[0].strip()
        return int(size_str) if size_str else 0
    except Exception:
        if command_key not in _dir_size_command_failures:
            logger.exception("Failed to measure directory size via system command: %s", path)
            _dir_size_command_failures.add(command_key)
        return _measure_directory_size_python(path)


class DownloadDialog:
    def __init__(self, model_name: str, progress_queue: Queue, cancel_event: threading.Event):
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.result: Optional[str] = None
        self._close_scheduled = False

        self.root = tk.Tk()
        self.root.title("CtrlSpeak Setup")
        self.root.geometry("960x520")
        self.root.minsize(900, 500)
        self.root.resizable(True, True)
        self.root.attributes("-topmost", True)
        apply_modern_theme(self.root)

        container = ttk.Frame(self.root, style="Modern.TFrame", padding=(26, 24))
        container.pack(fill=tk.BOTH, expand=True)

        card = ttk.Frame(container, style="ModernCard.TFrame", padding=(24, 22))
        card.pack(fill=tk.BOTH, expand=True)
        ttk.Label(card, text="Install speech model", style="Title.TLabel").pack(anchor=tk.W)
        model_tag = ttk.Label(card, text=f"MODEL · {model_name.upper()}", style="PillMuted.TLabel")
        model_tag.pack(anchor=tk.W, pady=(10, 0))
        accent = ttk.Frame(card, style="AccentLine.TFrame")
        accent.configure(height=2)
        accent.pack(fill=tk.X, pady=(12, 16))

        info_text = (
            "CtrlSpeak needs to download the Whisper model '"
            f"{model_name}' the first time it runs on this PC."
        )
        ttk.Label(card, text=info_text, style="Body.TLabel", wraplength=380,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(12, 16))

        self.stage_var = tk.StringVar(value="Preparing download…")
        ttk.Label(card, textvariable=self.stage_var, style="SectionHeading.TLabel").pack(anchor=tk.W)

        self.progress = ttk.Progressbar(card, length=360, mode="determinate", maximum=100,
                                        style="Modern.Horizontal.TProgressbar")
        self.progress.pack(fill=tk.X, pady=(12, 8))

        self._start_time = time.time()

        metrics = ttk.Frame(card, style="ModernCardInner.TFrame")
        metrics.pack(fill=tk.X, pady=(8, 0))

        self.metric_vars: dict[str, tk.StringVar] = {
            "progress": tk.StringVar(),
            "file_size": tk.StringVar(),
            "downloaded": tk.StringVar(),
            "speed": tk.StringVar(),
            "eta": tk.StringVar(),
            "elapsed": tk.StringVar(),
        }

        for label_text, key in [
            ("Progress", "progress"),
            ("File size", "file_size"),
            ("Downloaded", "downloaded"),
            ("Average speed", "speed"),
            ("ETA", "eta"),
            ("Elapsed", "elapsed"),
        ]:
            column_frame = ttk.Frame(metrics, style="ModernCardInner.TFrame")
            column_frame.pack(side=tk.LEFT, padx=(0, 24))

            ttk.Label(column_frame, text=f"{label_text}:", style="Caption.TLabel").pack(anchor=tk.W)
            ttk.Label(
                column_frame,
                textvariable=self.metric_vars[key],
                style="Caption.TLabel",
            ).pack(anchor=tk.W)
        self.status_var = tk.StringVar(value="")
        ttk.Label(card, textvariable=self.status_var, style="Caption.TLabel", wraplength=360,
                  justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 0))

        actions = ttk.Frame(card, style="ModernCardInner.TFrame")
        actions.pack(fill=tk.X, pady=(20, 0))
        self.cancel_button = ttk.Button(actions, text="Cancel download", style="Danger.TButton",
                                        command=self.cancel)
        self.cancel_button.pack(side=tk.RIGHT)

        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        self._last_progress_was_bytes = False
        self._progress_updates_started = False
        self._placeholder_after_id: Optional[str] = None
        self._update_placeholder_status()
        self._schedule_placeholder_refresh()

    def cancel(self) -> None:
        if self.result is None:
            self.result = "cancelled"
            self.cancel_event.set()
            self.status_var.set("Cancelling...")
            self._cancel_placeholder_refresh()

    def _schedule_close(self, delay_ms: int = 0) -> None:
        if self._close_scheduled:
            return
        self._close_scheduled = True

        def _close() -> None:
            try:
                self.root.quit()
            except Exception:
                logger.exception("Failed to quit download dialog mainloop")
            try:
                self.root.destroy()
            except Exception:
                logger.exception("Failed to destroy download dialog window")

        try:
            self.root.after(delay_ms, _close)
        except Exception:
            logger.exception("Failed to schedule download dialog close")
            _close()

    def _update_progress(self, desc: str, current: float, total: float, *, is_bytes: bool) -> None:
        if desc:
            desc_clean = desc.strip()
            if desc_clean:
                self.stage_var.set(f"Downloading: {desc_clean}")
        if is_bytes:
            if not self._progress_updates_started:
                self._progress_updates_started = True
                self._cancel_placeholder_refresh()
            else:
                self.status_var.set("")
            if total and total > 0:
                percent_float = max(min((current / total) * 100.0, 100.0), 0.0)
                if self.progress["mode"] != "determinate":
                    self.progress.stop()
                    self.progress.config(mode="determinate", maximum=100.0)
                else:
                    try:
                        current_max = float(self.progress.cget("maximum"))
                    except Exception:
                        current_max = 0.0
                    if current_max != 100.0:
                        self.progress.config(maximum=100.0)
                self.progress["value"] = percent_float
                percent = int(round(percent_float))
            else:
                if self.progress["mode"] != "indeterminate":
                    self.progress.config(mode="indeterminate")
                    self.progress.start(10)
                self.progress["value"] = 0
                percent = 0
            now = time.time()
            elapsed = max(now - self._start_time, 1e-6)
            speed = current / elapsed if elapsed > 0.0 else 0.0
            downloaded_mb = current / MB_DIVISOR
            total_mb = total / MB_DIVISOR if total and total > 0 else None
            remaining = (total - current) if total and total > current else None
            eta_seconds = (remaining / speed) if (remaining is not None and speed > 1e-6) else None
            eta_text = format_duration(eta_seconds) if eta_seconds is not None else "calculating"
            elapsed_text = format_duration(elapsed)
            total_text = f"{total_mb:.2f} MB" if total_mb is not None else "calculating"
            self.metric_vars["progress"].set(f"{percent}%")
            self.metric_vars["file_size"].set(total_text)
            self.metric_vars["downloaded"].set(f"{downloaded_mb:.2f} MB")
            self.metric_vars["speed"].set(
                f"{format_bytes(speed)}/s" if speed > 0 else "calculating"
            )
            self.metric_vars["eta"].set(eta_text)
            self.metric_vars["elapsed"].set(elapsed_text)
            self._last_progress_was_bytes = True
        else:
            if not self._progress_updates_started:
                self._progress_updates_started = True
                self._cancel_placeholder_refresh()
            else:
                self.status_var.set("")
            if self._last_progress_was_bytes:
                # Ignore non-byte updates once byte progress has started to avoid regressions.
                return
            if self.progress["mode"] != "indeterminate":
                self.progress.config(mode="indeterminate")
                self.progress.start(10)
            now = time.time()
            elapsed = max(now - self._start_time, 1e-6)
            percent = 0
            if total and total > 0:
                percent = min(max(int((current / total) * 100), 0), 100)
                progress_text = f"{percent}%"
            else:
                progress_text = f"Items: {int(current)}"
            elapsed_text = format_duration(elapsed)
            self.metric_vars["progress"].set(progress_text)
            self.metric_vars["file_size"].set("calculating")
            self.metric_vars["downloaded"].set("calculating")
            self.metric_vars["speed"].set("calculating")
            self.metric_vars["eta"].set("calculating")
            self.metric_vars["elapsed"].set(elapsed_text)

    def _placeholder_status_text(self) -> dict[str, str]:
        elapsed = max(time.time() - self._start_time, 0.0)
        elapsed_text = format_duration(elapsed)
        return {
            "progress": "0%",
            "file_size": "calculating",
            "downloaded": "0.00 MB",
            "speed": "calculating",
            "eta": "calculating",
            "elapsed": elapsed_text,
        }

    def _update_placeholder_status(self) -> None:
        self.status_var.set("")
        placeholder_values = self._placeholder_status_text()
        for key, value in placeholder_values.items():
            self.metric_vars[key].set(value)

    def _schedule_placeholder_refresh(self) -> None:
        if self._placeholder_after_id is not None or self.result is not None:
            return

        def _tick() -> None:
            self._placeholder_after_id = None
            if self._progress_updates_started or self.result is not None:
                return
            self._update_placeholder_status()
            self._schedule_placeholder_refresh()

        try:
            self._placeholder_after_id = self.root.after(500, _tick)
        except Exception:
            logger.exception("Failed to schedule placeholder status refresh")

    def _cancel_placeholder_refresh(self) -> None:
        if self._placeholder_after_id is None:
            return
        try:
            self.root.after_cancel(self._placeholder_after_id)
        except Exception:
            logger.exception("Failed to cancel placeholder status refresh")
        finally:
            self._placeholder_after_id = None

    def _process_queue(self) -> None:
        while True:
            try:
                message = self.progress_queue.get_nowait()
            except Empty:
                break
            message_type = message[0]
            if message_type == "progress":
                _, desc, current, total, is_bytes = message
                self._update_progress(desc, current, total, is_bytes=is_bytes)
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

        if self.result is None:
            self.root.after(100, self._process_queue)
        else:
            if self.progress["mode"] == "indeterminate":
                self.progress.stop()
            self.cancel_button.config(state=tk.DISABLED)
            delay = 400 if self.result == "success" else 0
            self._schedule_close(delay)

    def run(self) -> str:
        self.root.after(100, self._process_queue)
        self.root.after(80, self._pump_management_events)
        self.root.mainloop()
        return self.result or "cancelled"

    def _pump_management_events(self) -> None:
        if self.result is not None:
            return
        try:
            pump_management_events_once()
        except Exception:
            logger.exception("Failed to pump management UI while downloading model")
        finally:
            try:
                self.root.after(120, self._pump_management_events)
            except Exception:
                logger.exception("Failed to schedule management UI pump during download")

def _prompt_for_model_install(model_name: str) -> bool:
    # Small confirm; default to Install if headless
    prompt = (f"CtrlSpeak needs to download the Whisper speech model '{model_name}' (~GBs). "
              "Do you want to start the download now?")
    try:
        import pyautogui
        choice = pyautogui.confirm(text=prompt, title="CtrlSpeak Setup", buttons=["Install Now", "Quit"])
    except Exception:
        logger.exception("Failed to display model installation prompt via GUI")
        print(prompt)
        choice = "Install Now"
    return choice == "Install Now"


def download_model_with_gui(
    model_short: Optional[str] = None,
    *,
    block_during_download: bool = False,
    activate_after: bool = False,
) -> bool:
    """
    Download the selected model (or the argument provided) with a simple progress dialog.
    Keeps files under %APPDATA%/CtrlSpeak/models/<model>.
    """
    model_name = (model_short or get_current_model_name()).strip()
    progress_queue: Queue = Queue()
    cancel_event = threading.Event()
    dialog = DownloadDialog(model_name, progress_queue, cancel_event)

    def worker() -> None:
        success = False
        try:
            repo_id = _model_repo_id(model_name)
            progress_queue.put(("stage", f"Connecting to Hugging Face ({repo_id})"))
            store_path = model_store_path_for(model_name)
            store_path.mkdir(parents=True, exist_ok=True)

            api = HfApi()
            try:
                try:
                    model_info = api.model_info(repo_id, files_metadata=True)
                except TypeError:
                    # Older huggingface-hub versions do not accept files_metadata
                    model_info = api.model_info(repo_id)
            except Exception as exc:
                progress_queue.put(("error", f"Failed to query repository: {exc}"))
                logger.exception("Unable to query Hugging Face model info for %s", repo_id)
                return

            files: list[dict[str, Optional[int]]] = []
            total_bytes = 0
            for sibling in getattr(model_info, "siblings", []) or []:
                path = getattr(sibling, "rfilename", None)
                if not path:
                    continue
                size = getattr(sibling, "size", None)
                size_value: Optional[int]
                try:
                    size_value = int(size) if size is not None else None
                except (TypeError, ValueError):
                    size_value = None
                if size_value is not None and size_value > 0:
                    total_bytes += size_value
                files.append({"path": path, "size": size_value})

            if not files:
                progress_queue.put(("error", "No files available for download."))
                return

            files.sort(key=lambda item: item["path"])

            if total_bytes <= 0:
                expected = EXPECTED_MODEL_BYTES.get(model_name)
                if expected is not None and expected > 0:
                    total_bytes = expected

            progress_queue.put(("stage", f"Downloading {model_name} model files…"))
            downloaded_bytes = 0
            progress_queue.put(("progress", "", float(downloaded_bytes), float(total_bytes), True))

            encoded_repo = quote(repo_id, safe="/")

            for entry in files:
                if cancel_event.is_set():
                    raise CancelledDownload()
                rel_path = entry["path"]
                file_size = entry["size"]
                dest_path = store_path / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = dest_path.with_name(dest_path.name + ".part")
                encoded_rel = quote(rel_path.replace("\\", "/"))
                url = f"https://huggingface.co/{encoded_repo}/resolve/main/{encoded_rel}?download=true"

                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        logger.debug("Unable to remove stale partial file %s", temp_path)

                if file_size is None:
                    try:
                        head_req = Request(url, method="HEAD", headers={"User-Agent": DOWNLOAD_USER_AGENT})
                        with urlopen(head_req) as head_resp:
                            cl = head_resp.headers.get("Content-Length") or head_resp.getheader("Content-Length")
                            if cl:
                                file_size = int(cl)
                                entry["size"] = file_size
                                total_bytes += file_size
                                progress_queue.put(("progress", "", float(downloaded_bytes), float(total_bytes), True))
                    except Exception:
                        file_size = None

                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except Exception:
                        logger.debug("Unable to remove existing file %s before download", dest_path)

                try:
                    req = Request(url, method="GET", headers={"User-Agent": DOWNLOAD_USER_AGENT})
                    with urlopen(req) as response, open(temp_path, "wb") as output:
                        while True:
                            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                            if not chunk:
                                break
                            if cancel_event.is_set():
                                raise CancelledDownload()
                            output.write(chunk)
                            chunk_len = len(chunk)
                            downloaded_bytes += chunk_len
                            progress_queue.put(("progress", rel_path, float(downloaded_bytes), float(total_bytes), True))
                    temp_path.replace(dest_path)
                except CancelledDownload:
                    try:
                        if temp_path.exists():
                            temp_path.unlink()
                    except Exception:
                        logger.debug("Failed to remove partial download %s", temp_path)
                    raise
                except HTTPError as exc:
                    try:
                        if temp_path.exists():
                            temp_path.unlink()
                    except Exception:
                        logger.debug("Failed to remove partial download %s after HTTP error", temp_path)
                    progress_queue.put(("error", f"HTTP error while downloading {rel_path}: {exc.code} {exc.reason}"))
                    return
                except URLError as exc:
                    try:
                        if temp_path.exists():
                            temp_path.unlink()
                    except Exception:
                        logger.debug("Failed to remove partial download %s after network error", temp_path)
                    progress_queue.put(("error", f"Network error while downloading {rel_path}: {exc.reason}"))
                    return
                except Exception as exc:
                    try:
                        if temp_path.exists():
                            temp_path.unlink()
                    except Exception:
                        logger.debug("Failed to remove partial download %s after error", temp_path)
                    progress_queue.put(("error", f"Failed downloading {rel_path}: {exc}"))
                    logger.exception("Error downloading %s from %s", rel_path, repo_id)
                    return

            if cancel_event.is_set():
                raise CancelledDownload()

            (store_path / ".installed").touch(exist_ok=True)
            success = True
        except CancelledDownload:
            progress_queue.put(("cancelled",))
            return
        except Exception as exc:
            progress_queue.put(("error", f"Failed: {exc}"))
            logger.exception("Model download failed")
            return

        if success:
            progress_queue.put(("done",))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    def _run() -> bool:
        result = dialog.run()
        thread.join(timeout=0.5)
        return result == "success"

    if not block_during_download:
        return _run()

    with activation_guard(
        "downloading the Whisper model",
        busy_message="CtrlSpeak is downloading the Whisper model. Please wait for the download to finish before using CtrlSpeak.",
        success_message="The Whisper model download completed successfully.",
        failure_message="CtrlSpeak could not download the Whisper model. Please try again.",
        lockout=True,
    ) as finalize:
        success = _run()
        if not success:
            finalize(False, "CtrlSpeak could not download the Whisper model. Please try again.")
            return False

        if not activate_after:
            finalize(True, "The Whisper model download completed successfully.")
            return True

        try:
            ui_update_lockout_message(
                "CtrlSpeak is activating the Whisper model. Please wait for activation to complete before using CtrlSpeak."
            )
        except Exception:
            logger.exception("Failed to update lockout window before model activation")

        try:
            ui_show_activation_popup(
                "CtrlSpeak is activating the Whisper model. Please wait while the model starts up…"
            )
        except Exception:
            logger.exception("Failed to update activation popup before model activation")

        activation_done = threading.Event()
        activation_success = {"value": False}

        def _activate() -> None:
            try:
                activated = initialize_transcriber(
                    force=True,
                    allow_client=True,
                    activation_finalizer=finalize,
                )
                activation_success["value"] = activated is not None
            finally:
                activation_done.set()

        activation_thread = threading.Thread(
            target=_activate,
            name="CtrlSpeakActivation",
            daemon=True,
        )
        activation_thread.start()

        # Keep the activation popup responsive while the model initializes.
        try:
            pump_management_events_once()
            while not activation_done.wait(0.05):
                pump_management_events_once()

            # Give the completion message a brief opportunity to display and dismiss.
            end_time = time.time() + 0.6
            while time.time() < end_time:
                pump_management_events_once()
                time.sleep(0.05)
        finally:
            activation_thread.join(timeout=0.5)

        return activation_success["value"]


# ---------------- Transcriber ----------------
model_lock = threading.Lock()
whisper_model: Optional[WhisperModel] = None
model_ready = threading.Event()
warned_cuda_unavailable = False
cuda_activation_disabled = False
_missing_model_notified: Set[str] = set()


def ensure_model_ready_for_local_server() -> bool:
    """Ensure a Whisper model is available and activated for local server mode."""

    with model_lock:
        if whisper_model is not None:
            return True

    model_name = get_current_model_name()
    store = model_store_path_for(model_name)
    marker = store / ".installed"

    if model_files_present(store) and marker.exists():
        return True

    return download_model_with_gui(
        model_name,
        block_during_download=True,
        activate_after=True,
    )

# Track long-running activation/installation work so the hotkey can be gated.
_activation_event = threading.Event()
_activation_lock = threading.Lock()
_activation_reasons: list[Tuple[str, bool]] = []

  
def is_model_loaded() -> bool:
    with model_lock:
        return whisper_model is not None

      
def _activation_busy_message(reason: str) -> str:
    clean = reason.rstrip(". ")
    return (
        "CtrlSpeak is busy "
        f"{clean}. Please wait for this to finish before using CtrlSpeak."
    )


def _activation_success_message(reason: str) -> str:
    clean = reason.rstrip(". ")
    return f"Finished {clean}. CtrlSpeak is ready to use."


def _activation_failure_message(reason: str) -> str:
    clean = reason.rstrip(". ")
    return (
        f"CtrlSpeak could not finish {clean}. "
        "Please review the logs or try again."
    )


def _begin_activation(reason: str, busy_message: str, *, lockout: bool) -> None:
    with _activation_lock:
        _activation_reasons.append((reason, lockout))
        _activation_event.set()
    try:
        if lockout:
            ui_show_lockout_window(busy_message)
        else:
            ui_show_activation_popup(busy_message)
    except Exception:
        logger.exception("Failed to present activation busy popup")


def _end_activation(reason: str, *, success: bool, message: Optional[str]) -> None:
    new_top: Optional[Tuple[str, bool]] = None
    popped_lockout = False
    with _activation_lock:
        if _activation_reasons:
            _reason, popped_lockout = _activation_reasons.pop()
        if _activation_reasons:
            new_top = _activation_reasons[-1]
        else:
            _activation_event.clear()
    if new_top:
        next_reason, is_lockout = new_top
        try:
            if is_lockout:
                ui_show_lockout_window(_activation_busy_message(next_reason))
            else:
                ui_show_activation_popup(_activation_busy_message(next_reason))
        except Exception:
            logger.exception("Failed to refresh activation popup message")
        return

    final_text = message
    if final_text is None:
        final_text = (
            _activation_success_message(reason)
            if success else _activation_failure_message(reason)
        )
    try:
        if popped_lockout:
            ui_close_lockout_window(final_text)
            ui_close_activation_popup(final_text)
        else:
            ui_close_activation_popup(final_text)
    except Exception:
        logger.exception("Failed to close activation popup")


@contextmanager
def activation_guard(
    reason: str,
    *,
    busy_message: Optional[str] = None,
    success_message: Optional[str] = None,
    failure_message: Optional[str] = None,
    lockout: bool = False,
):
    """Track long-running activation work and drive the busy popup lifecycle."""
    busy = busy_message or _activation_busy_message(reason)
    _begin_activation(reason, busy, lockout=lockout)
    outcome = {"success": True, "message": success_message}

    def finalize(success: bool, message: Optional[str] = None) -> None:
        outcome["success"] = bool(success)
        outcome["message"] = message

    try:
        yield finalize
    except Exception:
        _end_activation(reason, success=False, message=failure_message)
        raise
    else:
        final_message = outcome["message"]
        if outcome["success"]:
            if final_message is None:
                final_message = _activation_success_message(reason)
        else:
            if final_message is None:
                final_message = failure_message or _activation_failure_message(reason)
        _end_activation(reason, success=outcome["success"], message=final_message)


def is_activation_in_progress() -> bool:
    return _activation_event.is_set()


def describe_activation_block() -> Optional[str]:
    if not _activation_event.is_set():
        return None
    with _activation_lock:
        entry = _activation_reasons[-1] if _activation_reasons else None
    if entry:
        reason, _ = entry
        return _activation_busy_message(reason)
    return "CtrlSpeak is preparing resources. Please wait before using the app."

def _force_cpu_env() -> None:
    """
    Make absolutely sure CUDA paths/devices are ignored when we want CPU.
    Safe to call multiple times.
    """
    # Hide all CUDA devices from child libs
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Common ctranslate2 toggles (harmless if unknown):
    os.environ["CT2_FORCE_CPU"] = "1"
    os.environ["CT2_USE_CUDA"] = "0"


def unload_transcriber() -> None:
    """
    Explicitly unload the in-memory model so a different device/model can be loaded.
    Guarded by model_lock and followed by GC to release native memory.
    """
    global whisper_model
    with model_lock:
        if whisper_model is not None:
            try:
                del whisper_model
            except Exception:
                logger.exception("Failed to delete whisper model instance")
            whisper_model = None
        model_ready.clear()
    # Encourage fast cleanup for large model memory and CUDA contexts
    try:
        import gc
        gc.collect()
    except Exception:
        logger.exception("Garbage collection after unloading model failed")
    # tiny delay helps native libs tear down cleanly before re-init
    time.sleep(0.15)



def _ensure_model_available_active(interactive: bool = True) -> bool:
    """Checks/installs the model selected in settings."""
    name = get_current_model_name()
    store = model_store_path_for(name)
    marker = store / ".installed"
    if model_ready.is_set() and model_files_present(store):
        return True
    if not model_files_present(store) or not marker.exists():
        if not interactive:
            if name not in _missing_model_notified:
                notify(
                    (
                        f"The Whisper model '{name}' is not installed. "
                        "Right-click the CtrlSpeak tray icon and choose 'Manage CtrlSpeak' "
                        "to download it."
                    )
                )
                _missing_model_notified.add(name)
            return False
        if not _prompt_for_model_install(name):
            notify("CtrlSpeak cannot transcribe without the Whisper model. You can install it next time you start the app.")
            return False
        block_download = whisper_model is None
        if not download_model_with_gui(
            name,
            block_during_download=block_download,
        ):
            notify("The Whisper model was not installed. Try again later or check your internet connection.")
            return False
    if model_files_present(store):
        model_ready.set()
        return True
    notify("Model download completed, but the files were not found. Please try again.")
    return False


def resolve_device() -> str:
    global cuda_activation_disabled
    if CLIENT_ONLY_BUILD:
        return "cpu"
    preference = get_device_preference()
    if preference == "cpu":
        return "cpu"
    if cuda_activation_disabled and preference != "cuda":
        return "cpu"
    if preference == "cuda":
        return "cuda" if cuda_runtime_ready() else "cpu"
    # auto
    if cuda_activation_disabled:
        return "cpu"
    return "cuda" if cuda_runtime_ready() else "cpu"


def resolve_compute_type(device: str) -> str:
    if COMPUTE_TYPE_OVERRIDE:
        return COMPUTE_TYPE_OVERRIDE
    if device == "cuda":
        return "float16"
    if device == "cpu":
        return "int8"
    return "int8_float16"


def initialize_transcriber(
    force: bool = False,
    allow_client: bool = False,
    preferred_device: Optional[str] = None,
    interactive: bool = True,
    activation_finalizer: Optional[Callable[[bool, Optional[str]], None]] = None,
) -> Optional[WhisperModel]:
    """
    Loads the Whisper model selected in settings into faster-whisper with the active device choice.
    """
    global whisper_model, warned_cuda_unavailable, cuda_activation_disabled

    def perform_activation(finalize: Callable[[bool, Optional[str]], None]) -> Optional[WhisperModel]:
        nonlocal device, compute_type, model_name
        global whisper_model, warned_cuda_unavailable, cuda_activation_disabled

        if device == "cpu":
            _force_cpu_env()

        try:
            whisper_model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=str(MODEL_ROOT_PATH),
            )
        except Exception as exc:
            print(f"Failed to load model on {device}: {exc}")
            logger.exception("Failed to load Whisper model on %s", device)
            if device != "cpu":
                try:
                    whisper_model = WhisperModel(
                        model_name,
                        device="cpu",
                        compute_type="int8",
                        download_root=str(MODEL_ROOT_PATH),
                    )
                except Exception as cpu_exc:
                    print(f"CPU fallback failed: {cpu_exc}")
                    logger.exception("CPU fallback model load failed")
                else:
                    notify(
                        "Running CtrlSpeak transcription on CPU fallback. "
                        "CtrlSpeak will continue using the CPU until you explicitly switch the device preference back to GPU."
                    )
                    warned_cuda_unavailable = True
                    cuda_activation_disabled = True
                    logger.warning(
                        "CUDA activation failed; defaulting future activations to CPU until GPU is explicitly requested."
                    )
                    finalize(
                        True,
                        (
                            "Activation complete! You can now use CtrlSpeak. "
                            f"The Whisper model '{model_name}' is running on the CPU fallback."
                        ),
                    )
                    return whisper_model
            notify("Unable to initialize the transcription model. Please check your installation and try again.")
            finalize(
                False,
                "CtrlSpeak was unable to activate the Whisper model. Please check your installation and try again.",
            )
            whisper_model = None
            return None
        else:
            cuda_activation_disabled = False
            print(f"Whisper model '{model_name}' ready on {device} ({compute_type})")
            finalize(
                True,
                (
                    "Activation complete! You can now use CtrlSpeak. "
                    f"The Whisper model '{model_name}' is ready on {device.upper()} ({compute_type})."
                ),
            )
            return whisper_model

    with model_lock:
        if whisper_model is not None and not force:
            if activation_finalizer is not None:
                activation_finalizer(True, None)
            return whisper_model

        with settings_lock:
            mode = settings.get("mode")
        if not allow_client and mode != "client_server":
            if activation_finalizer is not None:
                activation_finalizer(False, "CtrlSpeak is not running in local transcription mode.")
            return None

        if not _ensure_model_available_active(interactive=interactive):
            if activation_finalizer is not None:
                activation_finalizer(False, "CtrlSpeak could not find the Whisper model files.")
            return None

        device = preferred_device or resolve_device()
        if (
            device == "cpu"
            and get_device_preference() in {"auto", "cuda"}
            and not warned_cuda_unavailable
            and preferred_device is None
        ):
            notify(
                "CUDA dependencies were not detected. CtrlSpeak will run Whisper on the CPU instead. "
                "Open the CtrlSpeak management window from the system tray to install GPU support."
            )
            warned_cuda_unavailable = True

        compute_type = resolve_compute_type(device)
        model_name = get_current_model_name()

        if activation_finalizer is None:
            with activation_guard(
                "activating the Whisper model",
                busy_message="CtrlSpeak is activating the Whisper model. Please wait for the model to finish preparing.",
                success_message="The Whisper model is ready. You may now use CtrlSpeak.",
                failure_message="CtrlSpeak could not activate the Whisper model.",
            ) as complete:
                return perform_activation(complete)

        return perform_activation(activation_finalizer)

# ---------------- Transcription API ----------------
def collect_text_from_segments(segments) -> str:
    pieces = []
    for s in segments:
        t = s.text.strip()
        if t:
            pieces.append(t)
    return " ".join(pieces)


def transcribe_local(file_path: str, play_feedback: bool = True, allow_client: bool = False, preferred_device: Optional[str] = None) -> Optional[str]:
    from utils.system import ServerInfo  # local import to avoid cycles
    from utils import system as _sysmod

    model = initialize_transcriber(allow_client=allow_client, preferred_device=preferred_device)
    if model is None:
        return None
    if play_feedback:
        start_processing_feedback()
    try:
        segments, _ = model.transcribe(file_path, beam_size=5, vad_filter=True, temperature=0.2, task="translate")
        text = collect_text_from_segments(segments)
        if text:
            with settings_lock:
                port = int(settings.get("server_port", 65432))
            if allow_client and preferred_device == "cpu":
                server = ServerInfo(host="local-cpu", port=-1, last_seen=time.time())
            else:
                server = ServerInfo(host="local", port=port, last_seen=time.time())
            try:
                apply_last_connected = getattr(_sysmod, "_apply_last_connected", None)
                if callable(apply_last_connected):
                    apply_last_connected(server)
                else:
                    _sysmod.last_connected_server = server
                    _sysmod.schedule_management_refresh()
            except Exception:
                _sysmod.last_connected_server = server
                try:
                    _sysmod.schedule_management_refresh()
                except Exception:
                    pass
        return text or None
    except Exception as exc:
        notify_error("Transcription failed", format_exception_details(exc))
        return None
    finally:
        if play_feedback:
            stop_processing_feedback()


def handle_missing_server(file_path: str, play_feedback: bool) -> Optional[str]:
    from utils.system import notify, start_server
    if CLIENT_ONLY_BUILD:
        notify("No CtrlSpeak server found. Using local CPU transcription.")
        return transcribe_local(file_path, play_feedback=play_feedback, allow_client=True, preferred_device="cpu")
    # Ask to enable local server
    try:
        import pyautogui
        choice = pyautogui.confirm(
            text="CtrlSpeak cannot reach a server on this network.\n\nEnable the server on this computer?",
            title="CtrlSpeak",
            buttons=["Install Server", "Cancel"],
        )
    except Exception:
        logger.exception("Failed to prompt for enabling local server")
        print("Enable local server?"); choice = "Install Server"
    if choice != "Install Server":
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


def transcribe_remote(file_path: str, play_feedback: bool = True) -> Optional[str]:
    server = get_best_server()
    if server is None:
        return handle_missing_server(file_path, play_feedback=play_feedback)
    if play_feedback:
        start_processing_feedback()
    conn = None
    import http.client, json
    try:
        with open(file_path, "rb") as handle:
            data = handle.read()
        conn = http.client.HTTPConnection(server.host, server.port, timeout=180)
        headers = {"Content-Type": "audio/wav", "Content-Length": str(len(data))}
        conn.request("POST", "/transcribe", body=data, headers=headers)
        response = conn.getresponse(); payload = response.read()
        if response.status != 200:
            snippet = payload[:400].decode("utf-8", errors="replace")
            detail = f"Server {server.host}:{server.port} responded with HTTP {response.status} {response.reason}\n{snippet}"
            notify_error("Remote server error", detail)
            return None
        try:
            data_json = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            notify_error("Remote response parsing failed", format_exception_details(exc))
            return None
        return data_json.get("text")
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
                logger.exception("Failed to close HTTP connection after remote transcription")


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
