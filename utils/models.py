# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
import threading
import subprocess
import shutil
import platform
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Optional, List, Set, Callable, Tuple, Dict, Any

from multiprocessing import Process, Queue as MPQueue

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

try:  # pragma: no cover - optional dependency for richer playback
    from ffpyplayer.player import MediaPlayer  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    MediaPlayer = None  # type: ignore

from utils.system import (
    settings, settings_lock,
    notify, notify_error, format_exception_details,
    get_config_dir, save_settings,
    start_processing_feedback, stop_processing_feedback,
    ui_show_lockout_window, ui_update_lockout_message, ui_close_lockout_window,
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



_FUN_FACT_ROTATION_MS = 6000
_WELCOME_VIDEO_TARGET_DURATION_MS = 5000
_WELCOME_VIDEO_MIN_FRAME_DELAY_MS = 5
_DEFAULT_FUN_FACTS: List[str] = [
    "CtrlSpeak records while you hold the right Ctrl key and pastes the transcript when you release it.",
    "Whisper models live in %APPDATA%/CtrlSpeak/models so updates never overwrite your cached downloads.",
    "Need GPU acceleration later? Use the management window’s CUDA installer without reinstalling CtrlSpeak.",
    "Fun fact: Whisper small balances accuracy and speed, making it the ideal first model for most PCs.",
    "Speech recognition loves quiet rooms—CtrlSpeak lets you change microphones from the management window anytime.",
    "You can automate install validation with python main.py --automation-flow for headless health checks.",
]


def _choose_audio_driver() -> Optional[str]:
    if platform.system().lower() == "windows":
        return "wasapi"
    return None


def _probe_video_metadata(video_path: Path) -> Dict[str, Any]:
    if MediaPlayer is None:
        return {}
    try:
        player = MediaPlayer(str(video_path), ff_opts={"sync": "video"})
    except Exception:
        logger.exception("Failed to probe welcome video metadata")
        return {}
    try:
        metadata = player.get_metadata() or {}
    except Exception:
        logger.exception("Failed to fetch welcome video metadata")
        metadata = {}
    finally:
        try:
            player.close_player()
        except Exception:
            pass
    return metadata


def _decide_ff_options(meta: Dict[str, Any]) -> Dict[str, str]:
    ff_opts: Dict[str, str] = {"sync": "audio"}
    audio_info = meta.get("audio") or {}
    sample_rate = audio_info.get("sample_rate")
    channels = audio_info.get("channels")

    if isinstance(channels, int) and channels > 2:
        ff_opts["ac"] = "2"
    if not isinstance(sample_rate, int) or sample_rate not in (44100, 48000):
        ff_opts["ar"] = "48000"
    ff_opts["af"] = "aresample=async=1:min_hard_comp=0.100:first_pts=0"
    return ff_opts


def _load_fun_facts() -> List[str]:
    path = asset_path("fun_facts.txt")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Fun facts asset missing; using built-in defaults")
        return list(_DEFAULT_FUN_FACTS)
    except Exception:
        logger.exception("Failed to read fun facts asset; using defaults")
        return list(_DEFAULT_FUN_FACTS)

    facts = [line.strip() for line in text.splitlines() if line.strip()]
    return facts or list(_DEFAULT_FUN_FACTS)


class WelcomeWindow:
    def __init__(self, model_name: str, progress_queue: MPQueue, process: Process):
        self.model_name = model_name
        self.progress_queue = progress_queue
        self.process = process
        self.result: Optional[str] = None
        self.error_message: Optional[str] = None
        self._closing = False

        self.root = tk.Tk()
        self.root.title("CtrlSpeak Setup")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)
        apply_modern_theme(self.root)

        self._video_path = asset_path("TrueAI_Intro_Video.mp4")
        self._video_metadata = _probe_video_metadata(self._video_path)
        self._target_video_duration_ms = self._resolve_target_duration_ms(self._video_metadata)
        width, height = self._configure_geometry(int(1920 * 0.8), int(1080 * 0.8))
        self._window_width = width
        self._window_height = height

        self.container = ttk.Frame(self.root, style="Modern.TFrame")
        self.container.pack(fill=tk.BOTH, expand=True)
        self._video_label = tk.Label(self.container, bd=0)
        self._video_label.pack(fill=tk.BOTH, expand=True)

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._icon_photo: Optional[ImageTk.PhotoImage] = None
        self._video_job: Optional[str] = None
        self._fun_fact_job: Optional[str] = None
        self._facts = _load_fun_facts()
        self._fact_index = 0
        self._fact_label: Optional[ttk.Label] = None
        self._fact_wrap = max(self._window_width - 200, 360)
        self._facts_frame: Optional[tk.Widget] = None
        self._facts_canvas: Optional[tk.Canvas] = None
        self._video_player = self._prepare_video_player()
        self._video_target_size: Optional[Tuple[int, int]] = None
        self._video_deadline: Optional[float] = None

        self.root.after(120, self._pump_management_events)
        self.root.after(150, self._poll_queue)

        if self._video_player is not None:
            self.root.after(10, self._play_next_frame)
        else:
            self.root.after(10, self._show_fun_facts_view)

    def _configure_geometry(self, width: int, height: int) -> Tuple[int, int]:
        try:
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
        except Exception:
            logger.exception("Failed to query screen dimensions for welcome window")
            screen_w, screen_h = 1920, 1080
        width = max(min(width, screen_w - 40), 320)
        height = max(min(height, screen_h - 40), 240)
        pos_x = max(int((screen_w - width) / 2), 0)
        pos_y = max(int((screen_h - height) / 2), 0)
        self.root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        self.root.minsize(width, height)
        self.root.resizable(False, False)
        return width, height

    def _resolve_target_duration_ms(self, metadata: Dict[str, Any]) -> int:
        duration_value = metadata.get("duration")
        duration_ms = 0
        if isinstance(duration_value, (int, float)) and duration_value > 0:
            try:
                duration_ms = int(round(duration_value * 1000))
            except Exception:
                duration_ms = 0
        if duration_ms <= 0:
            duration_ms = _WELCOME_VIDEO_TARGET_DURATION_MS
        return min(duration_ms, _WELCOME_VIDEO_TARGET_DURATION_MS)

    def _calculate_video_target_size(self, frame_width: int, frame_height: int) -> Tuple[int, int]:
        try:
            scale = min(self._window_width / max(frame_width, 1), self._window_height / max(frame_height, 1))
        except Exception:
            return frame_width, frame_height
        target_width = max(int(frame_width * scale), 1)
        target_height = max(int(frame_height * scale), 1)
        return target_width, target_height

    def _prepare_video_player(self):
        if MediaPlayer is None:
            logger.debug("ffpyplayer not available; welcome video disabled")
            return None
        if not self._video_path.exists():
            logger.warning("Welcome video asset missing; showing fun facts immediately")
            return None

        driver = _choose_audio_driver()
        if driver:
            os.environ["SDL_AUDIODRIVER"] = driver

        os.environ.setdefault("FFMPEG_PRINT_LEVEL", "quiet")

        ff_opts = _decide_ff_options(self._video_metadata)
        try:
            return MediaPlayer(str(self._video_path), ff_opts=ff_opts)
        except Exception:
            logger.exception("Failed to initialize welcome video playback")
            return None

    def _play_next_frame(self) -> None:
        if self._closing or self._video_player is None:
            return

        try:
            frame, value = self._video_player.get_frame()
        except Exception:
            logger.exception("Failed to fetch welcome video frame")
            self._show_fun_facts_view()
            return

        if value == "eof":
            self._show_fun_facts_view()
            return

        delay_ms = _WELCOME_VIDEO_MIN_FRAME_DELAY_MS

        if frame is not None:
            img, _pts = frame
            try:
                frame_width, frame_height = img.get_size()
            except Exception:
                frame_width = frame_height = 0

            if frame_width <= 0 or frame_height <= 0:
                self._show_fun_facts_view()
                return

            if not self._video_label.winfo_exists():
                self._show_fun_facts_view()
                return

            if self._video_target_size is None:
                self._video_target_size = self._calculate_video_target_size(frame_width, frame_height)

            target_width, target_height = self._video_target_size

            try:
                rgb_bytes = img.to_bytearray()[0]
                pil_image = Image.frombytes("RGB", (frame_width, frame_height), rgb_bytes)
            except Exception:
                logger.exception("Failed to convert welcome video frame")
                self._show_fun_facts_view()
                return

            if (frame_width, frame_height) != (target_width, target_height):
                try:
                    pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                except Exception:
                    logger.exception("Failed to resize welcome video frame")
                    self._show_fun_facts_view()
                    return

            try:
                self._photo = ImageTk.PhotoImage(image=pil_image, master=self.root)
                self._video_label.configure(image=self._photo, width=target_width, height=target_height)
                self._video_label.image = self._photo
            except Exception:
                logger.exception("Failed to render welcome video frame")
                self._show_fun_facts_view()
                return

            if self._video_deadline is None:
                self._video_deadline = time.monotonic() + (self._target_video_duration_ms / 1000.0)

        if isinstance(value, (int, float)) and value > 0:
            delay_ms = max(int(value * 1000), _WELCOME_VIDEO_MIN_FRAME_DELAY_MS)

        if self._video_deadline is not None and time.monotonic() >= self._video_deadline:
            self._show_fun_facts_view()
            return

        try:
            self._video_job = self.root.after(delay_ms, self._play_next_frame)
        except Exception:
            logger.exception("Failed to schedule welcome video frame advance")
            self._show_fun_facts_view()

    def _stop_fun_facts(self) -> None:
        if self._fun_fact_job is None:
            return
        try:
            self.root.after_cancel(self._fun_fact_job)
        except Exception:
            logger.exception("Failed to cancel fun fact rotation")
        finally:
            self._fun_fact_job = None

    def _finish_video(self) -> None:
        if self._video_job is not None:
            try:
                self.root.after_cancel(self._video_job)
            except Exception:
                logger.exception("Failed to cancel welcome video job")
            finally:
                self._video_job = None
        self._video_target_size = None
        self._video_deadline = None
        player = self._video_player
        self._video_player = None
        if player is not None:
            def _close() -> None:
                try:
                    player.set_pause(True)
                except Exception:
                    pass
                try:
                    player.close_player()
                except Exception:
                    logger.exception("Failed to close welcome video player")

            threading.Thread(target=_close, daemon=True).start()

    def _show_fun_facts_view(self) -> None:
        if self._closing:
            return
        self._finish_video()
        if getattr(self, "_facts_frame", None):
            return
        for child in list(self.container.children.values()):
            try:
                child.destroy()
            except Exception:
                logger.exception("Failed to destroy welcome video widget")
        canvas = tk.Canvas(self.container, highlightthickness=0, bd=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        self._facts_canvas = canvas

        self._draw_fun_facts_gradient(canvas, self._window_width, self._window_height)

        style = ttk.Style(master=self.root)
        style.configure(
            "FunFactsCard.TFrame",
            background="#132543",
            relief="flat",
        )
        style.configure(
            "FunFactsTitle.TLabel",
            background="#132543",
            foreground="#f4fbff",
            font=("{Segoe UI Semibold}", 28),
        )
        style.configure(
            "FunFactsSubtitle.TLabel",
            background="#132543",
            foreground="#b7c9e6",
            font=("{Segoe UI}", 14),
        )
        style.configure(
            "FunFactsHeading.TLabel",
            background="#132543",
            foreground="#f4fbff",
            font=("{Segoe UI Semibold}", 18),
        )
        style.configure(
            "FunFactsBody.TLabel",
            background="#132543",
            foreground="#f4fbff",
            font=("{Segoe UI}", 17),
        )

        card = ttk.Frame(canvas, style="FunFactsCard.TFrame", padding=(40, 40))
        canvas.create_window(
            self._window_width // 2,
            self._window_height // 2,
            window=card,
            anchor="center",
        )
        self._facts_frame = card

        title = ttk.Label(card, text="Welcome to CtrlSpeak", style="FunFactsTitle.TLabel")
        title.pack(pady=(0, 10))

        subtitle = ttk.Label(
            card,
            text="Here are a few fun facts while we prepare your Whisper model.",
            style="FunFactsSubtitle.TLabel",
            wraplength=max(self._fact_wrap - 120, 360),
            justify=tk.CENTER,
        )
        subtitle.pack(pady=(0, 24))

        try:
            if card.winfo_exists():
                icon_image = Image.open(asset_path("icon.ico")).convert("RGBA")
                icon_image.thumbnail((176, 176), Image.Resampling.LANCZOS)
                white_bg = Image.new("RGBA", icon_image.size, (255, 255, 255, 255))
                alpha_channel = icon_image.getchannel("A") if "A" in icon_image.getbands() else None
                white_bg.paste(icon_image, mask=alpha_channel)
                display_image = white_bg.convert("RGB")
                self._icon_photo = ImageTk.PhotoImage(display_image, master=self.root)
                icon_label = tk.Label(
                    card,
                    image=self._icon_photo,
                    bg="#ffffff",
                    bd=1,
                    relief=tk.SOLID,
                    highlightthickness=0,
                    padx=12,
                    pady=12,
                )
                icon_label.image = self._icon_photo
                icon_label.pack(pady=(0, 16))
        except Exception:
            logger.exception("Failed to load welcome icon asset")

        heading = ttk.Label(card, text="Fun facts:", style="FunFactsHeading.TLabel")
        heading.pack(pady=(0, 12))

        fact_label = ttk.Label(
            card,
            text=self._facts[0] if self._facts else "",
            style="FunFactsBody.TLabel",
            wraplength=max(self._fact_wrap - 160, 360),
            justify=tk.CENTER,
        )
        fact_label.pack(pady=(0, 16))
        self._fact_label = fact_label
        self._fact_index = 0

        self._schedule_next_fact(initial_delay=_FUN_FACT_ROTATION_MS)

    def _draw_fun_facts_gradient(self, canvas: tk.Canvas, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return

        def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
            value = value.lstrip("#")
            r = int(value[0:2], 16)
            g = int(value[2:4], 16)
            b = int(value[4:6], 16)
            return (r, g, b)

        top_color = _hex_to_rgb("#1a355d")
        bottom_color = _hex_to_rgb("#040913")
        steps = max(height, 1)

        for i in range(height):
            ratio = i / max(steps - 1, 1)
            r = int(top_color[0] + (bottom_color[0] - top_color[0]) * ratio)
            g = int(top_color[1] + (bottom_color[1] - top_color[1]) * ratio)
            b = int(top_color[2] + (bottom_color[2] - top_color[2]) * ratio)
            color = f"#{r:02x}{g:02x}{b:02x}"
            canvas.create_line(0, i, width, i, fill=color)

    def _schedule_next_fact(self, initial_delay: int = _FUN_FACT_ROTATION_MS) -> None:
        if self._closing or not self._facts or self._fact_label is None:
            return
        self._stop_fun_facts()

        def _advance() -> None:
            if self._closing or not self._facts or self._fact_label is None:
                return
            self._fun_fact_job = None
            self._fact_index = (self._fact_index + 1) % len(self._facts)
            try:
                self._fact_label.configure(text=self._facts[self._fact_index])
            except Exception:
                logger.exception("Failed to update fun fact text")
                return
            self._schedule_next_fact()

        try:
            self._fun_fact_job = self.root.after(initial_delay, _advance)
        except Exception:
            logger.exception("Failed to schedule fun fact rotation")
            self._fun_fact_job = None

    def _poll_queue(self) -> None:
        while True:
            try:
                message = self.progress_queue.get_nowait()
            except Empty:
                break

            message_type = message[0]
            if message_type == "stage":
                _, desc = message
                if desc:
                    ui_update_lockout_message(desc)
            elif message_type == "error":
                _, error_text = message
                self.result = "error"
                self.error_message = error_text
            elif message_type == "done":
                self.result = "success"
            elif message_type == "cancelled":
                self.result = "cancelled"

        self._resolve_process_exit()

        if self.result is not None:
            self._finalize()
            return

        try:
            self.root.after(200, self._poll_queue)
        except Exception:
            logger.exception("Failed to schedule progress queue polling")

    def _resolve_process_exit(self) -> None:
        if self.result is not None:
            return
        if self.process.is_alive():
            return
        exit_code = self.process.exitcode
        if exit_code is None:
            return
        if exit_code == 0:
            self.result = "success"
        else:
            self.result = "error"
            if self.error_message is None:
                self.error_message = "CtrlSpeak could not download the Whisper model. Please try again."

    def _finalize(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._finish_video()
        self._stop_fun_facts()
        try:
            self.root.after(0, self.root.quit)
        except Exception:
            try:
                self.root.quit()
            except Exception:
                logger.exception("Failed to quit welcome window mainloop")

    def _pump_management_events(self) -> None:
        if self._closing:
            return
        try:
            pump_management_events_once()
        except Exception:
            logger.exception("Failed to pump management UI during welcome flow")
        finally:
            try:
                self.root.after(160, self._pump_management_events)
            except Exception:
                logger.exception("Failed to schedule management UI pump during welcome flow")

    def run(self) -> Tuple[str, Optional[str]]:
        try:
            self.root.mainloop()
        finally:
            self._finish_video()
            self._stop_fun_facts()
            try:
                self.root.destroy()
            except Exception:
                logger.exception("Failed to destroy welcome window")
        return self.result or "cancelled", self.error_message


def _model_download_worker(model_name: str, queue: MPQueue) -> None:
    def _put(payload: Tuple[str, ...]) -> None:
        try:
            queue.put_nowait(payload)
        except Exception:
            pass

    try:
        repo_id = _model_repo_id(model_name)
        trace_model_download_step(
            "download_model_with_gui.worker: prepared repo id",
            "announce stage and ensure directory",
        )
        _put(("stage", f"Preparing download from Hugging Face ({repo_id})"))
        store_path = model_store_path_for(model_name)
        store_path.mkdir(parents=True, exist_ok=True)
        trace_model_download_step(
            "download_model_with_gui.worker: starting snapshot_download",
            "wait for huggingface client to finish",
        )
        _put(("stage", "Downloading Whisper model…"))
        snapshot_download(
            repo_id,
            local_dir=str(store_path),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:
        logger.exception("Model download failed")
        trace_model_download_step(
            "download_model_with_gui.worker: snapshot_download raised exception",
            "report error to user",
        )
        _put(("error", f"Model download failed: {exc}"))
        return

    trace_model_download_step(
        "download_model_with_gui.worker: snapshot_download completed",
        "signal stage completion",
    )
    _put(("stage", "Download complete. Verifying files…"))
    _put(("done",))


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




def download_model_with_gui(
    model_short: Optional[str] = None,
    *,
    block_during_download: bool = False,
) -> bool:
    """Download the selected model snapshot while presenting the welcome experience."""

    model_name = (model_short or get_current_model_name()).strip()
    trace_model_download_step(
        "download_model_with_gui: start",
        "spawn worker process",
    )

    progress_queue: MPQueue = MPQueue()
    process = Process(target=_model_download_worker, args=(model_name, progress_queue))
    process.daemon = True
    process.start()

    cancel_requested = threading.Event()

    def _request_cancel() -> None:
        if cancel_requested.is_set():
            return
        cancel_requested.set()
        trace_model_download_step(
            "download_model_with_gui: cancel requested",
            "terminate worker process",
        )
        try:
            ui_update_lockout_message("Cancelling model download…")
        except Exception:
            logger.exception("Failed to update lockout message while cancelling model download")
        if process.is_alive():
            try:
                process.terminate()
            except Exception:
                logger.exception("Failed to terminate model download process")
        try:
            progress_queue.put_nowait(("cancelled",))
        except Exception:
            pass

    initial_message = (
        "CtrlSpeak is downloading the Whisper model. Use Cancel download if you need to stop the download."
    )

    lockout_open = False
    try:
        ui_show_lockout_window(initial_message, cancel_callback=_request_cancel)
        lockout_open = True
    except Exception:
        logger.exception("Failed to show lockout window during model download")

    window = WelcomeWindow(model_name, progress_queue, process)
    status, error_message = window.run()

    trace_model_download_step(
        "download_model_with_gui: welcome window completed",
        "finalize lockout window",
    )

    process.join(timeout=1.0)
    if process.is_alive():
        try:
            process.terminate()
        except Exception:
            logger.exception("Failed to terminate hung model download process")
        process.join(timeout=0.5)

    try:
        progress_queue.close()
    except Exception:
        pass
    try:
        progress_queue.join_thread()
    except Exception:
        pass

    if status == "success":
        try:
            if lockout_open:
                ui_close_lockout_window("The Whisper model download completed successfully.")
        except Exception:
            logger.exception("Failed to close lockout window after successful download")
        trace_model_download_step(
            "download_model_with_gui: completed",
            "return success flag",
        )
        return True

    if status == "cancelled":
        message = "The Whisper model download was cancelled."
        trace_model_download_step(
            "download_model_with_gui: cancelled",
            "return False",
        )
    else:
        message = error_message or "CtrlSpeak could not download the Whisper model. Please try again."
        trace_model_download_step(
            "download_model_with_gui: failed",
            "return False",
        )

    try:
        if lockout_open:
            ui_close_lockout_window(message)
    except Exception:
        logger.exception("Failed to close lockout window after download failure")

    return False



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


