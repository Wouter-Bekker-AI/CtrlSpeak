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
from typing import Optional, List, Set, Callable

import ctypes
import ctranslate2
from faster_whisper import WhisperModel
try:  # pragma: no cover - optional dependency rarely installed
    import hf_xet  # type: ignore  # noqa: F401
except ImportError:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk

from utils.system import (
    settings, settings_lock,
    notify, notify_error, format_exception_details,
    get_config_dir, save_settings,
    start_processing_feedback, stop_processing_feedback,
    ui_show_activation_popup, ui_close_activation_popup,
)
from utils.system import get_best_server, CLIENT_ONLY_BUILD
from utils.ui_theme import apply_modern_theme
from utils.config_paths import get_logger


# ---------------- Env / defaults ----------------
# These are *defaults*. Active choices come from settings via getters below.
ENV_DEFAULT_MODEL = os.environ.get("CTRLSPEAK_MODEL", "small")
ENV_DEVICE_PREF   = os.environ.get("CTRLSPEAK_DEVICE", "cpu").lower()
COMPUTE_TYPE_OVERRIDE = os.environ.get("CTRLSPEAK_COMPUTE_TYPE")
MODEL_REPO_OVERRIDE   = os.environ.get("CTRLSPEAK_MODEL_REPO")

# All model content under %APPDATA%/CtrlSpeak/models
MODEL_ROOT_PATH = get_config_dir() / "models"

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


def get_current_model_name() -> str:
    """Reads the model selected in the Settings UI; falls back to env default."""
    with settings_lock:
        name = settings.get("model_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return ENV_DEFAULT_MODEL


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


def _model_repo_id(model_short: str) -> str:
    if MODEL_REPO_OVERRIDE:
        return MODEL_REPO_OVERRIDE
    # Your builds use Systran/faster-whisper-<name>
    return f"Systran/faster-whisper-{model_short}"


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


class DownloadDialog:
    def __init__(self, model_name: str, progress_queue: Queue, cancel_event: threading.Event):
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.result: Optional[str] = None

        self.root = tk.Tk()
        self.root.title("CtrlSpeak Setup")
        self.root.geometry("480x260")
        self.root.resizable(False, False)
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

        self.status_var = tk.StringVar(value="")
        ttk.Label(card, textvariable=self.status_var, style="Caption.TLabel").pack(anchor=tk.W)

        actions = ttk.Frame(card, style="ModernCardInner.TFrame")
        actions.pack(fill=tk.X, pady=(20, 0))
        self.cancel_button = ttk.Button(actions, text="Cancel download", style="Danger.TButton",
                                        command=self.cancel)
        self.cancel_button.pack(side=tk.RIGHT)

        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        self._start_time = time.time()
        self._last_bytes = 0.0
        self._last_update = self._start_time

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
            now = time.time()
            elapsed = max(now - self._start_time, 1e-6)
            window_elapsed = max(now - self._last_update, 1e-6)
            avg_speed = current / elapsed
            inst_speed = (current - self._last_bytes) / window_elapsed
            speed = inst_speed if window_elapsed >= 0.5 else avg_speed
            self.status_var.set(
                f"{percent}%  ({format_bytes(current)} / {format_bytes(total)})"
                f"  •  {format_bytes(speed)}/s"
            )
            self._last_bytes = current
            self._last_update = now
        else:
            if self.progress["mode"] != "indeterminate":
                self.progress.config(mode="indeterminate")
                self.progress.start(10)
            now = time.time()
            elapsed = max(now - self._start_time, 1e-6)
            speed = current / elapsed
            self.status_var.set(f"{format_bytes(current)} downloaded  •  {format_bytes(speed)}/s")

    def _process_queue(self) -> None:
        while True:
            try:
                message = self.progress_queue.get_nowait()
            except Empty:
                break
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


def make_tqdm_class(callback, cancel_event):
    class GuiTqdm(tqdm):
        def update(self_inner, n=1):
            super().update(n)
            try:
                callback(self_inner.desc or "", float(self_inner.n), float(self_inner.total or 0))
            except Exception:
                logger.exception("Progress callback failed")
            if cancel_event.is_set():
                raise CancelledDownload()
    return GuiTqdm


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

    def progress_callback(desc: str, current: float, total: float) -> None:
        progress_queue.put(("progress", desc, current, total))

    def worker() -> None:
        try:
            repo_id = _model_repo_id(model_name)
            progress_queue.put(("stage", f"Connecting to Hugging Face ({repo_id})"))
            store_path = model_store_path_for(model_name)
            store_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(store_path),
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=4,
                tqdm_class=make_tqdm_class(progress_callback, cancel_event),
            )
            if cancel_event.is_set():
                progress_queue.put(("cancelled",)); return
            (store_path / ".installed").touch(exist_ok=True)
            progress_queue.put(("done",))
        except CancelledDownload:
            progress_queue.put(("cancelled",))
        except Exception as exc:
            progress_queue.put(("error", f"Failed: {exc}"))
            logger.exception("Model download failed")

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
    ) as finalize:
        success = _run()
        if not success:
            finalize(False, "CtrlSpeak could not download the Whisper model. Please try again.")
            return False

        if not activate_after:
            finalize(True, "The Whisper model download completed successfully.")
            return True

        try:
            ui_show_activation_popup("The Whisper model has been downloaded. Activating it now…")
        except Exception:
            logger.exception("Failed to update activation popup before model activation")

        activated = initialize_transcriber(
            force=True,
            allow_client=True,
            activation_finalizer=finalize,
        )
        return activated is not None


# ---------------- Transcriber ----------------
model_lock = threading.Lock()
whisper_model: Optional[WhisperModel] = None
model_ready = threading.Event()
warned_cuda_unavailable = False
_missing_model_notified: Set[str] = set()

# Track long-running activation/installation work so the hotkey can be gated.
_activation_event = threading.Event()
_activation_lock = threading.Lock()
_activation_reasons: list[str] = []

  
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


def _begin_activation(reason: str, busy_message: str) -> None:
    with _activation_lock:
        _activation_reasons.append(reason)
        _activation_event.set()
    try:
        ui_show_activation_popup(busy_message)
    except Exception:
        logger.exception("Failed to present activation busy popup")


def _end_activation(reason: str, *, success: bool, message: Optional[str]) -> None:
    new_top: Optional[str] = None
    with _activation_lock:
        if _activation_reasons:
            _activation_reasons.pop()
        if _activation_reasons:
            new_top = _activation_reasons[-1]
        else:
            _activation_event.clear()
    if new_top:
        try:
            ui_show_activation_popup(_activation_busy_message(new_top))
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
):
    """Track long-running activation work and drive the busy popup lifecycle."""
    busy = busy_message or _activation_busy_message(reason)
    _begin_activation(reason, busy)
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
        reason = _activation_reasons[-1] if _activation_reasons else None
    if reason:
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
    global whisper_model, warned_cuda_unavailable

    def perform_activation(finalize: Callable[[bool, Optional[str]], None]) -> Optional[WhisperModel]:
        nonlocal device, compute_type, model_name
        global whisper_model, warned_cuda_unavailable

        if device == "cpu":
            _force_cpu_env()

        try:
            whisper_model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=str(MODEL_ROOT_PATH),
            )
            print(f"Whisper model '{model_name}' ready on {device} ({compute_type})")
            finalize(True, f"The Whisper model '{model_name}' is ready on {device.upper()} ({compute_type}).")
            return whisper_model
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
                    notify("Running CtrlSpeak transcription on CPU fallback. Set CTRLSPEAK_DEVICE=cpu to suppress this message.")
                    warned_cuda_unavailable = True
                    finalize(
                        True,
                        f"The Whisper model '{model_name}' is ready on CPU fallback. CtrlSpeak will continue using the CPU.",
                    )
                    return whisper_model
                except Exception as cpu_exc:
                    print(f"CPU fallback failed: {cpu_exc}")
                    logger.exception("CPU fallback model load failed")
            notify("Unable to initialize the transcription model. Please check your installation and try again.")
            finalize(
                False,
                "CtrlSpeak was unable to activate the Whisper model. Please check your installation and try again.",
            )
            whisper_model = None
            return None

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
        segments, _ = model.transcribe(file_path, beam_size=5, vad_filter=True, temperature=0.2)
        text = collect_text_from_segments(segments)
        if text:
            with settings_lock:
                port = int(settings.get("server_port", 65432))
            if allow_client and preferred_device == "cpu":
                _sysmod.last_connected_server = ServerInfo(host="local-cpu", port=-1, last_seen=time.time())
            else:
                _sysmod.last_connected_server = ServerInfo(host="local", port=port, last_seen=time.time())
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
