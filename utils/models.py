# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
import threading
import subprocess
import shutil
import multiprocessing
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Optional, List, Set, Tuple

import ctypes
import ctranslate2
from faster_whisper import WhisperModel
try:  # pragma: no cover - optional dependency rarely installed
    import hf_xet  # type: ignore  # noqa: F401
except ImportError:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

try:  # pragma: no cover - optional dependency not always available
    from ffpyplayer.player import MediaPlayer  # type: ignore
except Exception:  # pragma: no cover - missing runtime pieces should not crash
    MediaPlayer = None  # type: ignore


from utils.system import (
    settings, settings_lock,
    notify, notify_error, format_exception_details,
    get_config_dir, save_settings,
    start_processing_feedback, stop_processing_feedback,
    ui_show_lockout_window, ui_close_lockout_window, ui_update_lockout_message,
    pump_management_events_once,
    play_model_ready_sound_once,
)
from utils.system import get_best_server, CLIENT_ONLY_BUILD
from utils.ui_theme import apply_modern_theme
from utils.config_paths import get_logger, asset_path


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
MODEL_TRACE_PATH = get_config_dir() / "logs" / "model_download_trace.log"

CUDA_RUNTIME_EXPECTED_BYTES = int(1.2 * (1024 ** 3))

# CUDA search: also look in %APPDATA%/CtrlSpeak/cuda
cuda_paths_initialized = False

logger = get_logger(__name__)
_trace_lock = threading.Lock()


def trace_model_download_step(current_step: str, expected_next: Optional[str] = None) -> None:
    """Append a human-readable trace entry for the model download workflow."""

    step = current_step.strip()
    next_step = (expected_next or "").strip()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")

    try:
        with _trace_lock:
            try:
                MODEL_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.exception("Failed to ensure trace log directory exists")
                return

            with MODEL_TRACE_PATH.open("a", encoding="utf-8") as handle:
                if next_step:
                    handle.write(f"{timestamp} | {step} | next: {next_step}\n")
                else:
                    handle.write(f"{timestamp} | {step}\n")
    except Exception:
        logger.exception("Failed to write model download trace entry")


# ---------------- Settings-backed getters/setters ----------------
def get_device_preference() -> str:
    """Return the persisted compute device preference, defaulting to CPU."""

    valid = {"cpu", "cuda"}
    with settings_lock:
        pref = settings.get("device_preference")

    if isinstance(pref, str) and pref in valid:
        return pref

    env_pref = ENV_DEVICE_PREF if ENV_DEVICE_PREF in valid else None
    if env_pref:
        return env_pref

    with settings_lock:
        settings["device_preference"] = "cpu"
    save_settings()
    return "cpu"


def set_device_preference(pref: str) -> None:
    normalized = "cuda" if pref == "cuda" else "cpu"
    with settings_lock:
        settings["device_preference"] = normalized
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
def _legacy_model_store_path(model_short: str) -> Path:
    return MODEL_ROOT_PATH / model_short


def model_store_path_for(model_short: str) -> Path:
    return _model_activation_cache_path(model_short)


def model_files_present(model_path: Path) -> bool:
    if not model_path.exists():
        return False
    # faster-whisper weights are .bin files
    return any(model_path.rglob("*.bin"))


def _iter_model_candidate_paths(model_short: str):
    primary = model_store_path_for(model_short)
    seen: Set[Path] = set()
    if primary not in seen:
        seen.add(primary)
        yield primary
    legacy = _legacy_model_store_path(model_short)
    if legacy not in seen:
        seen.add(legacy)
        yield legacy


def _is_model_installed(model_short: str) -> bool:
    for candidate in _iter_model_candidate_paths(model_short):
        if model_files_present(candidate):
            return True
    return False


def _model_repo_id(model_short: str) -> str:
    if MODEL_REPO_OVERRIDE:
        return MODEL_REPO_OVERRIDE
    # Your builds use Systran/faster-whisper-<name>
    return f"Systran/faster-whisper-{model_short}"



def get_model_repo_id(model_short: str) -> str:
    """Expose the Hugging Face repo id used for the given Whisper model."""
    return _model_repo_id(model_short)


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


def cuda_runtime_ready(*, ignore_preference: bool = False, quiet: bool = False) -> bool:
    """Validate that CUDA DLLs can be loaded for GPU inference."""

    try:
        if not ignore_preference and get_device_preference() == "cpu":
            return False
    except Exception:
        if quiet:
            logger.debug("Failed to read device preference while checking CUDA readiness", exc_info=True)
        else:
            logger.exception("Failed to read device preference while checking CUDA readiness")
        # Fall back to attempting detection in case settings are not ready yet.

    configure_cuda_paths()

    try:
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception:
        if quiet:
            logger.debug("CUDA device probe failed", exc_info=True)
        else:
            logger.exception("CUDA device probe failed")
        return False

    if sys.platform.startswith("win"):
        for dll in ("cudnn_ops64_9.dll", "cublas64_12.dll"):
            try:
                ctypes.windll.LoadLibrary(dll)
            except OSError as exc:
                if quiet:
                    logger.debug("Failed to load CUDA DLL %s: %s", dll, exc)
                else:
                    logger.warning("Failed to load CUDA DLL %s: %s", dll, exc)
                return False

    return True



def _cuda_destination_root() -> Path:
    dest_root = get_config_dir() / "cuda"
    dest_root.mkdir(parents=True, exist_ok=True)
    return dest_root


def cuda_runtime_files_present() -> bool:
    """Return True when essential CUDA runtime files exist under the app data folder."""

    root = get_config_dir() / "cuda"
    if not root.exists():
        return False

    if sys.platform.startswith("win"):
        required = (
            "cudart64_*.dll",
            "cublas64_*.dll",
            "cublasLt64_*.dll",
            "cudnn*.dll",
        )
    else:
        required = (
            "libcudart.so*",
            "libcublas.so*",
            "libcudnn*.so*",
        )

    for pattern in required:
        if not any(root.rglob(pattern)):
            return False
    return True


def _copy_from_nvidia_packages(dest_root: Path) -> bool:
    staged = False
    prefixes = {Path(sys.prefix)}
    base_prefix = getattr(sys, "base_prefix", None)
    if base_prefix:
        prefixes.add(Path(base_prefix))

    for prefix in prefixes:
        package_root = prefix / "Lib" / "site-packages" / "nvidia"
        if not package_root.exists():
            continue

        for component in ("cuda_runtime", "cudnn", "cublas"):
            src = package_root / component
            if not src.exists():
                continue
            dest = dest_root / component
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            staged = True

        runtime_bin = dest_root / "cuda_runtime" / "bin"
        dest_bin = dest_root / "bin"
        if runtime_bin.exists():
            if dest_bin.exists():
                shutil.rmtree(dest_bin)
            shutil.copytree(runtime_bin, dest_bin)
            staged = True

    return staged


def _collect_system_cuda_bin_dirs() -> list[Path]:
    candidates: list[Path] = []

    def _extend(path_value: str | os.PathLike[str] | None) -> None:
        if not path_value:
            return
        base = Path(path_value)
        if not base.exists():
            return
        for sub in ("bin", "lib", "lib64"):
            candidate = base / sub
            if candidate.exists():
                candidates.append(candidate)

    for key, value in os.environ.items():
        upper = key.upper()
        if upper.startswith("CUDA_PATH") or upper in {"CUDA_HOME", "CUDA_ROOT"}:
            _extend(value)

    for env_key in ("ProgramFiles", "ProgramFiles(x86)"):
        base_val = os.environ.get(env_key)
        if not base_val:
            continue
        toolkit_root = Path(base_val) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if not toolkit_root.exists():
            continue
        for child in sorted(toolkit_root.iterdir()):
            if child.is_dir():
                _extend(child)

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _copy_from_system_cuda(dest_root: Path) -> bool:
    candidates = _collect_system_cuda_bin_dirs()
    if not candidates:
        return False

    dest_bin = dest_root / "bin"
    if dest_bin.exists():
        shutil.rmtree(dest_bin)
    dest_bin.mkdir(parents=True, exist_ok=True)

    staged = False
    patterns = (
        "cudart64_*.dll",
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "nvblas64_*.dll",
        "cudnn*.dll",
        "libcudart.so*",
        "libcublas.so*",
        "libcudnn*.so*",
    )
    for source_dir in candidates:
        for pattern in patterns:
            for file in source_dir.glob(pattern):
                try:
                    shutil.copy2(file, dest_bin / file.name)
                    staged = True
                except Exception:
                    logger.debug("Failed to copy CUDA file %s", file, exc_info=True)
    return staged


def stage_cuda_runtime_from_existing() -> bool:
    dest_root = _cuda_destination_root()

    if cuda_runtime_files_present() and cuda_runtime_ready(ignore_preference=True, quiet=True):
        return True

    staged = False
    try:
        staged = _copy_from_nvidia_packages(dest_root) or staged
        if not staged:
            staged = _copy_from_system_cuda(dest_root)
    except Exception:
        logger.exception("Failed to reuse existing CUDA runtime files")
        return False

    if not staged:
        return False

    global cuda_paths_initialized
    cuda_paths_initialized = False
    if not cuda_runtime_ready(ignore_preference=True, quiet=True):
        logger.debug("CUDA runtime validation failed after staging existing assets")
        return False

    return True


def ensure_cuda_runtime_from_existing() -> bool:
    if stage_cuda_runtime_from_existing():
        logger.info("Reused existing CUDA runtime assets for CtrlSpeak.")
        return True
    return False



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
    """Install or repair CUDA runtime support for CtrlSpeak."""

    if ensure_cuda_runtime_from_existing():
        return True

    try:
        import pip  # type: ignore
    except ModuleNotFoundError:
        logger.info("pip module not found; bootstrapping ensurepip before CUDA install")
        ensure_rc = _run_cmd_stream([sys.executable, "-m", "ensurepip", "--upgrade"], timeout=300)
        if ensure_rc != 0:
            notify("CUDA runtime installation failed because pip could not be bootstrapped.")
            try:
                ui_close_lockout_window("CtrlSpeak could not install pip. GPU support remains unavailable; continuing with CPU.")
            except Exception:
                logger.exception("Failed to close lockout window after ensurepip failure")
            return False

    pkgs = ["nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12", "nvidia-cudnn-cu12"]
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--disable-pip-version-check",
        "--extra-index-url",
        "https://pypi.nvidia.com",
        *pkgs,
    ]

    lockout_open = False
    try:
        ui_show_lockout_window(
            "CtrlSpeak is installing CUDA runtime components required for GPU transcription. "
            "Please wait for this to finish."
        )
        lockout_open = True
    except Exception:
        logger.exception("Failed to show lockout window during CUDA installation")

    rc = _run_cmd_stream(cmd, timeout=600)
    if rc != 0:
        notify("CUDA runtime installation failed (pip returned non-zero).")
        if lockout_open:
            try:
                ui_close_lockout_window("CUDA runtime installation failed. CtrlSpeak will continue using the CPU.")
            except Exception:
                logger.exception("Failed to close lockout window after CUDA installation failure")
        return False

    if not ensure_cuda_runtime_from_existing():
        notify("CUDA runtime installation completed, but validation failed. CtrlSpeak will continue using the CPU.")
        if lockout_open:
            try:
                ui_close_lockout_window(
                    "CUDA runtime installation completed, but validation failed. CtrlSpeak will continue using the CPU."
                )
            except Exception:
                logger.exception("Failed to close lockout window after CUDA validation failure")
        return False

    if lockout_open:
        try:
            ui_close_lockout_window(
                "CUDA runtime components were installed successfully. CtrlSpeak will continue preparing the Whisper model."
            )
        except Exception:
            logger.exception("Failed to close lockout window after CUDA installation")

    return True



# ---------------- Download dialog (GUI) ----------------


def format_bytes(value: float) -> str:
    """Return a human-readable string for a byte count."""

    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    amount = float(value)
    for unit in units:
        if amount < step:
            return f"{amount:.1f} {unit}"
        amount /= step
    return f"{amount:.1f} PB"


FUN_FACT_INTERVAL_MS = 5000


class DownloadDialog:
    """Welcome window shown while models are downloading."""

    def __init__(self, model_name: str, progress_queue: Queue, cancel_event: threading.Event):
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.result: Optional[str] = None
        self._close_scheduled = False

        self._video_capture: Optional[cv2.VideoCapture] = None
        self._video_after_id: Optional[str] = None
        self._video_fps: float = 30.0
        self._video_display_size: Tuple[int, int] = (480, 270)
        self._current_frame: Optional[ImageTk.PhotoImage] = None

        self._audio_thread: Optional[threading.Thread] = None
        self._audio_stop_event: Optional[threading.Event] = None
        self._audio_player: Optional[Any] = None
        self._topmost_release_job: Optional[str] = None

        self._fun_facts: List[str] = self._load_fun_facts()
        if MediaPlayer is None:
            logger.warning(
                "Intro audio unavailable: install the optional 'ffpyplayer[full]' package to enable the welcome video soundtrack."
            )
            self._fun_facts.insert(
                0,
                "Install the optional 'ffpyplayer[full]' package to enable the welcome video soundtrack.",
            )
        self._fun_fact_index: int = 0
        self._fun_fact_job: Optional[str] = None

        self.video_path = asset_path("TrueAI_Intro_Video.mp4")
        self.icon_path = asset_path("icon.ico")

        self.root = tk.Tk()
        self.root.title("CtrlSpeak Welcome")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        apply_modern_theme(self.root)

        self.container = ttk.Frame(self.root, style="Modern.TFrame")
        self.container.pack(fill=tk.BOTH, expand=True)

        self.display_frame = ttk.Frame(self.container, style="Modern.TFrame")
        self.display_frame.pack(fill=tk.BOTH, expand=True)

        footer = ttk.Frame(self.container, style="ModernCardInner.TFrame", padding=(18, 16))
        footer.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="Model download in progress…")
        self.status_label = ttk.Label(
            footer,
            textvariable=self.status_var,
            style="Body.TLabel",
            wraplength=440,
            justify=tk.LEFT,
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._icon_image: Optional[ImageTk.PhotoImage] = None
        self._fun_fact_var = tk.StringVar(value="")

        self._configure_window_size()
        self._start_video()
        self._bring_to_front_once()

    def _configure_window_size(self) -> None:
        scale = 0.8
        width, height = max(int(480 * scale), 200), max(int(270 * scale), 120)
        try:
            video_path = Path(self.video_path)
            if video_path.exists():
                capture = cv2.VideoCapture(str(video_path))
                if capture.isOpened():
                    read_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    read_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    if read_width > 0 and read_height > 0:
                        width = max(int(read_width * scale), 200)
                        height = max(int(read_height * scale), 120)
                capture.release()
        except Exception:
            logger.exception("Failed to probe intro video size")

        self._video_display_size = (width, height)

        try:
            self.root.update_idletasks()
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            x = max((screen_w - width) // 2, 0)
            y = max((screen_h - height) // 2, 0)
            self.root.geometry(f"{width}x{height}+{x}+{y}")
            self.root.minsize(width, height)
            self.root.resizable(False, False)
        except Exception:
            logger.exception("Failed to size welcome window")

        try:
            wrap = max(width - 60, 220)
            self.status_label.configure(wraplength=wrap)
        except Exception:
            logger.exception("Failed to update welcome window wrap length")

    def _bring_to_front_once(self) -> None:
        try:
            self.root.lift()
            self.root.attributes("-topmost", True)
        except Exception:
            logger.exception("Failed to raise welcome window")

        try:
            self.root.focus_force()
        except Exception:
            pass

        def _release_topmost() -> None:
            self._topmost_release_job = None
            try:
                self.root.attributes("-topmost", False)
            except Exception:
                pass

        try:
            self._topmost_release_job = self.root.after(600, _release_topmost)
        except Exception:
            logger.exception("Failed to schedule welcome window topmost release")
            _release_topmost()

    def _cancel_topmost_release(self) -> None:
        job = self._topmost_release_job
        if job is None:
            return
        self._topmost_release_job = None
        try:
            self.root.after_cancel(job)
        except Exception:
            logger.exception("Failed to cancel welcome window topmost release")

    def _start_video(self) -> None:
        try:
            video_path = Path(self.video_path)
            if not video_path.exists():
                logger.warning("Intro video not found at %s", video_path)
                self._show_static_content()
                return
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                logger.warning("Intro video could not be opened")
                self._show_static_content()
                return
            self._video_fps = 30.0
            self._video_capture = capture
            self.video_label = ttk.Label(self.display_frame)
            self.video_label.pack(fill=tk.BOTH, expand=True)
            self._start_audio(video_path)
            self._queue_next_frame()
        except Exception:
            logger.exception("Failed to start intro video playback")
            self._show_static_content()

    def _queue_next_frame(self) -> None:
        if self._video_capture is None:
            self._show_static_content()
            return
        try:
            ok, frame = self._video_capture.read()
        except Exception:
            logger.exception("Failed to read intro video frame")
            ok = False
            frame = None
        if not ok or frame is None:
            self._finish_video()
            return

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
        except Exception:
            logger.exception("Failed to convert intro video frame to RGB")
            self._finish_video()
            return

        if self._video_display_size:
            try:
                image = image.resize(self._video_display_size, Image.LANCZOS)
            except Exception:
                logger.exception("Failed to scale intro video frame")

        try:
            tk_image = ImageTk.PhotoImage(image=image, master=self.root)
        except Exception:
            logger.exception("Failed to create PhotoImage for intro video frame")
            self._finish_video()
            return

        self._current_frame = tk_image

        try:
            self.video_label.configure(image=self._current_frame)
            # Keep a direct reference on the widget to prevent Tk from
            # discarding the backing image between frames.
            self.video_label.image = self._current_frame
        except tk.TclError:
            logger.exception("Failed to display intro video frame")
            self._finish_video()
            return

        delay = max(int(1000 / max(self._video_fps, 1.0)), 1)
        try:
            self._video_after_id = self.root.after(delay, self._queue_next_frame)
        except Exception:
            logger.exception("Failed to schedule intro video frame update")
            self._finish_video()

    def _start_audio(self, video_path: Path) -> None:
        if MediaPlayer is None:
            self.status_var.set(
                "Model download in progress… (Install 'ffpyplayer[full]' to enable welcome audio.)"
            )
            return
        try:
            import numpy as _np  # type: ignore
        except Exception:
            logger.exception("Failed to import numpy for intro video audio playback")
            self.status_var.set("Model download in progress… (Intro audio unavailable.)")
            return
        try:
            import pyaudio as _pyaudio  # type: ignore
        except Exception:
            logger.exception("Failed to import PyAudio for intro video audio playback")
            self.status_var.set("Model download in progress… (Intro audio unavailable.)")
            return

        try:
            ff_opts = {"sync": "audio", "paused": False, "vn": "1"}
            player = MediaPlayer(str(video_path), ff_opts=ff_opts)
        except Exception:
            logger.exception("Failed to initialize intro video audio playback")
            self.status_var.set("Model download in progress… (Intro audio unavailable.)")
            return

        stop_event = threading.Event()
        self._audio_stop_event = stop_event
        self._audio_player = player

        try:
            player.set_mute(False)
        except Exception:
            pass

        try:
            player.set_volume(1.0)
        except Exception:
            pass

        np = _np
        pyaudio = _pyaudio

        def _drain_audio() -> None:
            pa_instance: Optional[pyaudio.PyAudio] = None  # type: ignore[attr-defined]
            stream: Optional[Any] = None
            try:
                while not stop_event.is_set():
                    try:
                        frame, val = player.get_frame()
                    except Exception:
                        logger.exception("Failed to read intro video audio frame")
                        break
                    if val == "eof":
                        break
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    if val != "audio":
                        continue
                    audio_frame = frame[0]
                    try:
                        array = audio_frame.to_ndarray()
                    except Exception:
                        logger.exception("Failed to convert intro audio frame to ndarray")
                        continue
                    if array.size == 0:
                        continue
                    dtype = array.dtype
                    if array.ndim > 1:
                        try:
                            array = np.swapaxes(array, 0, 1).reshape(-1)
                        except Exception:
                            array = array.flatten()
                    else:
                        array = array.reshape(-1)

                    format_map = {}
                    try:
                        format_map = {
                            np.uint8: pyaudio.paUInt8,  # type: ignore[attr-defined]
                            np.int16: pyaudio.paInt16,  # type: ignore[attr-defined]
                            np.int32: pyaudio.paInt32,  # type: ignore[attr-defined]
                            np.float32: pyaudio.paFloat32,  # type: ignore[attr-defined]
                        }
                    except Exception:
                        format_map = {}
                    pa_format = format_map.get(dtype.type)
                    if pa_format is None:
                        array = array.astype(np.float32)
                        pa_format = pyaudio.paFloat32  # type: ignore[attr-defined]

                    channels = 1
                    channels_getter = getattr(audio_frame, "get_channels", None)
                    if callable(channels_getter):
                        try:
                            channels = int(channels_getter())
                        except Exception:
                            channels = 1
                    elif hasattr(audio_frame, "channels"):
                        try:
                            channels = max(int(getattr(audio_frame, "channels")), 1)
                        except Exception:
                            channels = 1

                    sample_rate: Optional[int] = None
                    if hasattr(audio_frame, "sample_rate"):
                        try:
                            sample_rate = int(getattr(audio_frame, "sample_rate"))
                        except Exception:
                            sample_rate = None
                    if sample_rate is None:
                        rate_getter = getattr(audio_frame, "get_rate", None)
                        if callable(rate_getter):
                            try:
                                sample_rate = int(rate_getter())
                            except Exception:
                                sample_rate = None
                    if sample_rate is None or sample_rate <= 0:
                        sample_rate = 44100

                    try:
                        buffer_bytes = array.tobytes()
                    except Exception:
                        logger.exception("Failed to serialize intro audio frame")
                        continue

                    if pa_instance is None:
                        try:
                            pa_instance = pyaudio.PyAudio()
                        except Exception:
                            logger.exception("Failed to initialize PyAudio for intro video")
                            break

                    if stream is None:
                        try:
                            stream = pa_instance.open(
                                format=pa_format,
                                channels=max(channels, 1),
                                rate=sample_rate,
                                output=True,
                            )
                        except Exception:
                            logger.exception("Failed to open PyAudio stream for intro video")
                            break

                    try:
                        stream.write(buffer_bytes)
                    except Exception:
                        logger.exception("Failed to play intro audio frame")
                        break
            except Exception:
                logger.exception("Audio playback thread failed for intro video")
            finally:
                if stream is not None:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        logger.exception("Failed to close intro audio stream")
                if pa_instance is not None:
                    try:
                        pa_instance.terminate()
                    except Exception:
                        logger.exception("Failed to terminate PyAudio instance for intro video")
                try:
                    player.close_player()
                except Exception:
                    logger.exception("Failed to close intro video audio player")
                if self._audio_player is player:
                    self._audio_player = None
                if self._audio_thread is threading.current_thread():
                    self._audio_thread = None
                if self._audio_stop_event is stop_event:
                    self._audio_stop_event = None

        audio_thread = threading.Thread(target=_drain_audio, daemon=True)
        self._audio_thread = audio_thread
        try:
            audio_thread.start()
        except Exception:
            logger.exception("Failed to start intro video audio thread")
            self._audio_thread = None
            try:
                player.close_player()
            except Exception:
                logger.exception("Failed to close intro video audio player after thread start failure")
            self._audio_player = None
            self._audio_stop_event = None

    def _finish_video(self) -> None:
        self._cancel_video_playback()
        self._show_static_content()

    def _cancel_video_playback(self) -> None:
        self._cancel_topmost_release()
        if self._video_after_id is not None:
            try:
                self.root.after_cancel(self._video_after_id)
            except Exception:
                logger.exception("Failed to cancel intro video playback timer")
            self._video_after_id = None
        if self._video_capture is not None:
            try:
                self._video_capture.release()
            except Exception:
                logger.exception("Failed to release intro video capture")
        self._video_capture = None
        self._stop_audio()

    def _stop_audio(self) -> None:
        thread = self._audio_thread
        stop_event = self._audio_stop_event
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            try:
                thread.join(timeout=0.5)
            except Exception:
                logger.exception("Failed to join intro video audio thread")
        self._audio_thread = None
        self._audio_stop_event = None
        self._audio_player = None

    def _show_static_content(self) -> None:
        for child in self.display_frame.winfo_children():
            child.destroy()

        card = ttk.Frame(self.display_frame, style="ModernCard.TFrame", padding=(32, 32))
        card.pack(fill=tk.BOTH, expand=True)
        card.columnconfigure(0, weight=1)

        header = ttk.Label(card, text="Welcome to CtrlSpeak", style="Title.TLabel")
        header.pack(pady=(0, 16))

        self._icon_image = self._load_icon_image()
        if self._icon_image is not None:
            icon_label = ttk.Label(card, image=self._icon_image)
            icon_label.image = self._icon_image
            icon_label.pack(pady=(0, 18))

        ttk.Label(
            card,
            text="While we prepare your speech model, enjoy a few fun facts!",
            style="Body.TLabel",
            wraplength=420,
            justify=tk.CENTER,
        ).pack(pady=(0, 20))

        if self._fun_facts:
            fun_fact_heading = ttk.Label(card, text="Did you know?", style="SectionHeading.TLabel")
            fun_fact_heading.pack()
            ttk.Label(
                card,
                textvariable=self._fun_fact_var,
                style="Body.TLabel",
                wraplength=420,
                justify=tk.CENTER,
            ).pack(pady=(12, 0))
            self._start_fun_fact_cycle()

    def _load_icon_image(self) -> Optional[ImageTk.PhotoImage]:
        try:
            icon_path = Path(self.icon_path)
            if not icon_path.exists():
                logger.warning("Welcome icon not found at %s", icon_path)
                return None
            image = Image.open(icon_path)
            image = image.convert("RGBA")
            image.thumbnail((160, 160), Image.LANCZOS)
            return ImageTk.PhotoImage(image=image, master=self.root)
        except Exception:
            logger.exception("Failed to load welcome icon")
            return None

    def _load_fun_facts(self) -> List[str]:
        facts_path = Path(asset_path("fun_facts.txt"))
        facts: List[str] = []
        try:
            if facts_path.exists():
                for line in facts_path.read_text(encoding="utf-8").splitlines():
                    cleaned = line.strip()
                    if cleaned:
                        facts.append(cleaned)
        except Exception:
            logger.exception("Failed to load fun facts")
        if not facts:
            facts = [
                "CtrlSpeak keeps your recordings in temporary storage and cleans them up automatically after transcription.",
                "You can switch between CPU and GPU transcription anytime from the management window.",
            ]
        return facts

    def _start_fun_fact_cycle(self) -> None:
        if not self._fun_facts:
            return
        self._display_next_fun_fact()

    def _display_next_fun_fact(self) -> None:
        if self.result is not None or not self._fun_facts:
            return
        fact = self._fun_facts[self._fun_fact_index]
        self._fun_fact_var.set(fact)
        self._fun_fact_index = (self._fun_fact_index + 1) % len(self._fun_facts)
        try:
            self._fun_fact_job = self.root.after(FUN_FACT_INTERVAL_MS, self._display_next_fun_fact)
        except Exception:
            logger.exception("Failed to schedule fun fact rotation")

    def _stop_fun_fact_cycle(self) -> None:
        if self._fun_fact_job is None:
            return
        try:
            self.root.after_cancel(self._fun_fact_job)
        except Exception:
            logger.exception("Failed to cancel fun fact rotation")
        finally:
            self._fun_fact_job = None

    def cancel(self) -> None:
        if self.result is None:
            self.result = "cancelled"
            self.cancel_event.set()
            self.status_var.set("Closing welcome window…")
            try:
                ui_update_lockout_message("Cancelling the download…")
            except Exception:
                logger.exception("Failed to update lockout message while cancelling download")
            try:
                self.progress_queue.put(("cancelled",))
            except Exception:
                logger.exception("Failed to enqueue cancellation message for welcome window")
            self._stop_fun_fact_cycle()
            self._cancel_video_playback()
            self._schedule_close(0)

    def _schedule_close(self, delay_ms: int = 0) -> None:
        if self._close_scheduled:
            return
        self._close_scheduled = True

        def _close() -> None:
            self._cancel_video_playback()
            self._stop_fun_fact_cycle()
            self._cancel_topmost_release()
            try:
                self.root.quit()
            except Exception:
                logger.exception("Failed to quit welcome window mainloop")
            try:
                self.root.destroy()
            except Exception:
                logger.exception("Failed to destroy welcome window")

        try:
            self.root.after(delay_ms, _close)
        except Exception:
            logger.exception("Failed to schedule welcome window close")
            _close()

    def _process_queue(self) -> None:
        while True:
            try:
                message = self.progress_queue.get_nowait()
            except Empty:
                break
            message_type = message[0]
            if message_type == "stage":
                _, desc = message
                self.status_var.set(desc)
                try:
                    ui_update_lockout_message(desc)
                except Exception:
                    logger.exception("Failed to propagate stage update to lockout window")
            elif message_type == "error":
                _, error_text = message
                self.result = "error"
                self.status_var.set(error_text)
                try:
                    ui_update_lockout_message(error_text)
                except Exception:
                    logger.exception("Failed to propagate error message to lockout window")
            elif message_type == "done":
                self.result = "success"
                self.status_var.set("Download completed.")
                try:
                    ui_update_lockout_message("Download completed.")
                except Exception:
                    logger.exception("Failed to propagate completion message to lockout window")
            elif message_type == "cancelled":
                self.result = "cancelled"
                self.status_var.set("Download cancelled.")
                try:
                    ui_update_lockout_message("Download cancelled.")
                except Exception:
                    logger.exception("Failed to propagate cancellation message to lockout window")

        if self.result is None:
            try:
                self.root.after(120, self._process_queue)
            except Exception:
                logger.exception("Failed to schedule welcome window queue polling")
        else:
            self._stop_fun_fact_cycle()
            self._cancel_video_playback()
            delay = 400 if self.result == "success" else 0
            self._schedule_close(delay)

    def run(self) -> str:
        self.root.after(80, self._pump_management_events)
        self.root.after(120, self._process_queue)
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
                self.root.after(160, self._pump_management_events)
            except Exception:
                logger.exception("Failed to schedule management UI pump during download")
def _prompt_for_model_install(model_name: str, *, auto_accept: bool = False) -> bool:
    """Request user confirmation before downloading a Whisper model."""

    prompt = (
        f"CtrlSpeak needs to download the Whisper speech model '{model_name}' (~GBs). "
        "Do you want to start the download now?"
    )

    if auto_accept:
        logger.info("Auto-approving Whisper model download prompt for '%s'", model_name)
        trace_model_download_step(
            "_prompt_for_model_install: auto-accept triggered",
            "proceed to download",
        )
        return True

    try:
        import pyautogui

        trace_model_download_step(
            "_prompt_for_model_install: showing confirmation dialog",
            "await user selection",
        )
        choice = pyautogui.confirm(
            text=prompt,
            title="CtrlSpeak Setup",
            buttons=["Install Now", "Quit"],
        )
    except Exception:
        logger.exception("Failed to display model installation prompt via GUI")
        print(prompt)
        choice = "Install Now"

    trace_model_download_step(
        "_prompt_for_model_install: user selection processed",
        "return result",
    )
    return choice == "Install Now"


def _run_snapshot_download_subprocess(
    repo_id: str,
    store_path: str,
    status_queue: "multiprocessing.queues.Queue[tuple[str, Optional[str]]]",
) -> None:
    """Run the Hugging Face snapshot download inside an isolated process."""

    try:
        snapshot_download(
            repo_id,
            local_dir=store_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        try:
            status_queue.put(("error", str(exc)))
        except Exception:
            pass
    else:
        try:
            status_queue.put(("success", None))
        except Exception:
            pass


def download_model_with_gui(
    model_short: Optional[str] = None,
    *,
    block_during_download: bool = False,
) -> bool:
    """Download the selected model snapshot with a simple progress dialog."""

    model_name = (model_short or get_current_model_name()).strip()
    progress_queue: Queue = Queue()
    cancel_event = threading.Event()
    dialog = DownloadDialog(model_name, progress_queue, cancel_event)

    class _ModelDownloadCancelled(RuntimeError):
        """Raised when the user cancels the model download."""

    trace_model_download_step(
        "download_model_with_gui: start",
        "spawn worker thread",
    )

    def worker() -> None:
        try:
            if cancel_event.is_set():
                trace_model_download_step(
                    "download_model_with_gui.worker: cancelled before starting",
                    "propagate cancellation",
                )
                progress_queue.put(("cancelled",))
                return
            repo_id = _model_repo_id(model_name)
            trace_model_download_step(
                "download_model_with_gui.worker: prepared repo id",
                "announce stage and ensure directory",
            )
            progress_queue.put((
                "stage",
                f"Model download in progress from Hugging Face ({repo_id})",
            ))
            store_path = model_store_path_for(model_name)
            store_path.mkdir(parents=True, exist_ok=True)
            trace_model_download_step(
                "download_model_with_gui.worker: starting snapshot_download",
                "wait for huggingface client to finish",
            )
            ctx = multiprocessing.get_context("spawn")
            status_queue: "multiprocessing.queues.Queue[tuple[str, Optional[str]]]" = ctx.Queue()
            process = ctx.Process(
                target=_run_snapshot_download_subprocess,
                args=(repo_id, str(store_path), status_queue),
            )
            process.start()

            try:
                while True:
                    if cancel_event.is_set():
                        if process.is_alive():
                            process.terminate()
                            process.join(timeout=5)
                            if process.is_alive():
                                try:
                                    process.kill()
                                except AttributeError:
                                    pass
                                process.join(timeout=5)
                        raise _ModelDownloadCancelled()

                    try:
                        status, payload = status_queue.get(timeout=0.2)
                    except Empty:
                        status = None
                        payload = None
                    except EOFError:
                        status = None
                        payload = None
                        if not process.is_alive():
                            exit_code = process.exitcode
                            if exit_code not in (0, None):
                                raise RuntimeError(
                                    f"Download worker exited unexpectedly (code {exit_code})"
                                )
                            if exit_code == 0:
                                break
                        continue

                    if status == "success":
                        break
                    if status == "error":
                        raise RuntimeError(payload or "Unknown error during download")
                    if status == "cancelled":
                        raise _ModelDownloadCancelled()

                    if not process.is_alive():
                        exit_code = process.exitcode
                        if exit_code not in (0, None):
                            raise RuntimeError(
                                f"Download worker exited unexpectedly (code {exit_code})"
                            )
                        if exit_code == 0:
                            break
            finally:
                try:
                    status_queue.close()
                except Exception:
                    pass
                try:
                    status_queue.join_thread()
                except Exception:
                    pass
                if process.is_alive():
                    process.join(timeout=5)
                    if process.is_alive():
                        try:
                            process.kill()
                        except AttributeError:
                            pass
                        process.join(timeout=5)
                else:
                    process.join(timeout=5)
        except _ModelDownloadCancelled:
            progress_queue.put(("cancelled",))
            trace_model_download_step(
                "download_model_with_gui.worker: cancellation propagated",
                "notify main thread",
            )
            return
        except Exception as exc:
            progress_queue.put(("error", f"Failed: {exc}"))
            logger.exception("Model download failed")
            trace_model_download_step(
                "download_model_with_gui.worker: snapshot_download raised exception",
                "report error to user",
            )
            return

        trace_model_download_step(
            "download_model_with_gui.worker: snapshot_download completed",
            "signal stage completion",
        )
        progress_queue.put(("stage", "Download complete."))
        progress_queue.put(("done",))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def _run() -> str:
        result = dialog.run()
        cancel_event.set()
        thread.join(timeout=0.5)
        return result

    if not block_during_download:
        trace_model_download_step(
            "download_model_with_gui: running dialog without lockout",
            "wait for dialog result",
        )
        return _run() == "success"

    try:
        ui_show_lockout_window(
            "CtrlSpeak is downloading the Whisper model. Please wait for the download to finish before using CtrlSpeak.",
            on_cancel=dialog.cancel,
        )
    except Exception:
        logger.exception("Failed to show lockout window during model download")
        trace_model_download_step(
            "download_model_with_gui: lockout window failed",
            "continue waiting for dialog result",
        )

    result = _run()
    success = result == "success"

    cancellation_requires_exit = False

    try:
        if success:
            ui_close_lockout_window("The Whisper model download completed successfully.")
        elif result == "cancelled":
            has_existing_model = model_files_present(model_store_path_for(model_name))
            if not has_existing_model:
                for candidate in AVAILABLE_MODELS:
                    if candidate == model_name:
                        continue
                    if _is_model_installed(candidate):
                        has_existing_model = True
                        break
            if has_existing_model:
                ui_close_lockout_window(
                    "Download cancelled. Your existing model remains available."
                )
            else:
                ui_close_lockout_window(
                    "Download cancelled. CtrlSpeak will now exit."
                )
                cancellation_requires_exit = True
        else:
            ui_close_lockout_window("CtrlSpeak could not download the Whisper model. Please try again.")
    except Exception:
        logger.exception("Failed to close lockout window after model download")
        trace_model_download_step(
            "download_model_with_gui: closing lockout failed",
            "finish with dialog result",
        )

    if cancellation_requires_exit:
        trace_model_download_step(
            "download_model_with_gui: exiting after cancellation",
            None,
        )
        sys.exit(0)

    trace_model_download_step(
        "download_model_with_gui: completed",
        "return success flag",
    )

    return success


# ---------------- Transcriber ----------------
model_lock = threading.Lock()
whisper_model: Optional[WhisperModel] = None
warned_cuda_unavailable = False
_missing_model_notified: Set[str] = set()


def ensure_model_ready_for_local_server() -> bool:
    """Ensure a Whisper model is available and activated for local server mode."""

    trace_model_download_step(
        "ensure_model_ready_for_local_server: requested",
        "check in-memory model",
    )
    with model_lock:
        if whisper_model is not None:
            trace_model_download_step(
                "ensure_model_ready_for_local_server: model already loaded",
                "return True",
            )
            return True

    trace_model_download_step(
        "ensure_model_ready_for_local_server: loading from disk",
        "call _ensure_model_files",
    )
    return _ensure_model_files(interactive=True)


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
    # Encourage fast cleanup for large model memory and CUDA contexts
    try:
        import gc
        gc.collect()
    except Exception:
        logger.exception("Garbage collection after unloading model failed")
    # tiny delay helps native libs tear down cleanly before re-init
    time.sleep(0.15)



def _ensure_model_files(interactive: bool = True) -> bool:
    """Ensure the selected model snapshot exists locally."""

    trace_model_download_step(
        "_ensure_model_files: entry",
        "determine installed path",
    )
    name = get_current_model_name()

    candidates = list(_iter_model_candidate_paths(name))
    installed_path = next((path for path in candidates if model_files_present(path)), None)

    with settings_lock:
        auto_install_complete = bool(settings.get("model_auto_install_complete"))

    if installed_path is not None:
        trace_model_download_step(
            "_ensure_model_files: files already installed",
            "update auto_install flag if needed",
        )
        if not auto_install_complete:
            with settings_lock:
                settings["model_auto_install_complete"] = True
            save_settings()
        return True

    if auto_install_complete:
        trace_model_download_step(
            "_ensure_model_files: auto-install flag reset",
            "prompt or auto-accept install",
        )
        with settings_lock:
            settings["model_auto_install_complete"] = False
        save_settings()
        auto_install_complete = False

    if not interactive:
        trace_model_download_step(
            "_ensure_model_files: non-interactive missing model",
            "notify user about missing model",
        )
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

    auto_install = not auto_install_complete

    trace_model_download_step(
        "_ensure_model_files: prompting for install",
        "download model" if auto_install else "await user confirmation",
    )
    if not _prompt_for_model_install(name, auto_accept=auto_install):
        trace_model_download_step(
            "_ensure_model_files: user declined install",
            "notify cancellation",
        )
        if not auto_install:
            notify(
                "CtrlSpeak cannot transcribe without the Whisper model. You can install it next time you start the app."
            )
        return False

    block_download = whisper_model is None
    trace_model_download_step(
        "_ensure_model_files: initiating download",
        "wait for download to finish",
    )
    if not download_model_with_gui(
        name,
        block_during_download=block_download,
    ):
        trace_model_download_step(
            "_ensure_model_files: download returned failure",
            "notify user about failure",
        )
        notify("The Whisper model was not installed. Try again later or check your internet connection.")
        return False

    trace_model_download_step(
        "_ensure_model_files: verifying downloaded files",
        "mark install complete if files exist",
    )

    if model_files_present(model_store_path_for(name)):
        trace_model_download_step(
            "_ensure_model_files: verification succeeded",
            "set auto-install flag and return True",
        )
        with settings_lock:
            settings["model_auto_install_complete"] = True
        save_settings()
        return True

    trace_model_download_step(
        "_ensure_model_files: verification failed",
        "notify user about missing files",
    )
    notify("Model download completed, but the files were not found. Please try again.")
    return False


def ensure_initial_model_installation() -> bool:
    """Ensure the default Whisper model is installed so CtrlSpeak is immediately usable."""

    with settings_lock:
        auto_install_needed = not bool(settings.get("model_auto_install_complete"))

    if not auto_install_needed:
        trace_model_download_step(
            "ensure_initial_model_installation: already complete",
            "return True",
        )
        return True

    trace_model_download_step(
        "ensure_initial_model_installation: beginning auto install",
        "call _ensure_model_files",
    )
    set_current_model_name(DEFAULT_MODEL_NAME)
    return _ensure_model_files(interactive=True)


def resolve_device() -> str:
    if CLIENT_ONLY_BUILD:
        return "cpu"

    preference = get_device_preference()
    if preference == "cuda" and cuda_runtime_ready(quiet=True):
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


def initialize_transcriber(
    force: bool = False,
    allow_client: bool = False,
    preferred_device: Optional[str] = None,
    interactive: bool = True,
) -> Optional[WhisperModel]:
    """Load the configured Whisper model into memory if it is not already active."""

    global whisper_model, warned_cuda_unavailable

    trace_model_download_step(
        "initialize_transcriber: entry",
        "check cached whisper model",
    )
    with model_lock:
        if whisper_model is not None and not force:
            trace_model_download_step(
                "initialize_transcriber: model already loaded",
                "return existing instance",
            )
            play_model_ready_sound_once()
            return whisper_model

        with settings_lock:
            mode = settings.get("mode")
        trace_model_download_step(
            "initialize_transcriber: mode resolved",
            "ensure local server if allowed",
        )
        if not allow_client and mode != "client_server":
            trace_model_download_step(
                "initialize_transcriber: not in server mode",
                "return None",
            )
            return None

        if not _ensure_model_files(interactive=interactive):
            trace_model_download_step(
                "initialize_transcriber: model files missing",
                "abort load",
            )
            return None

        device = preferred_device or resolve_device()
        trace_model_download_step(
            "initialize_transcriber: device resolved",
            "notify if cpu fallback needed",
        )
        if (
            device == "cpu"
            and get_device_preference() == "cuda"
            and not warned_cuda_unavailable
            and preferred_device is None
        ):
            logger.info("CUDA runtime not available; continuing on CPU.")
            warned_cuda_unavailable = True

        compute_type = resolve_compute_type(device)
        model_name = get_current_model_name()
        model_path = model_store_path_for(model_name)
        trace_model_download_step(
            "initialize_transcriber: starting WhisperModel load",
            "construct WhisperModel instance",
        )

        if device == "cpu":
            _force_cpu_env()

        try:
            whisper_model = WhisperModel(
                str(model_path),
                device=device,
                compute_type=compute_type,
            )
        except Exception as exc:
            logger.exception("Failed to load Whisper model on %s", device)
            trace_model_download_step(
                "initialize_transcriber: WhisperModel raised exception",
                "attempt cpu fallback" if device != "cpu" else "notify failure",
            )
            if device != "cpu":
                try:
                    whisper_model = WhisperModel(
                        str(model_path),
                        device="cpu",
                        compute_type="int8",
                    )
                except Exception:
                    logger.exception("CPU fallback model load failed")
                    notify(
                        "Unable to initialize the transcription model. Please check your installation and try again."
                    )
                    whisper_model = None
                    return None
                else:
                    logger.info("CUDA runtime not available; using CPU fallback.")
                    warned_cuda_unavailable = True
                    trace_model_download_step(
                        "initialize_transcriber: cpu fallback succeeded",
                        "return fallback model",
                    )
                    play_model_ready_sound_once()
                    return whisper_model

            notify("Unable to initialize the transcription model. Please check your installation and try again.")
            whisper_model = None
            trace_model_download_step(
                "initialize_transcriber: load failed",
                "return None",
            )
            return None

        trace_model_download_step(
            "initialize_transcriber: load succeeded",
            "return whisper model",
        )
        play_model_ready_sound_once()
        return whisper_model

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


