# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import time
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, List, Set

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

from utils.system import (
    settings, settings_lock,
    notify, notify_error, format_exception_details,
    get_config_dir, save_settings,
    start_processing_feedback, stop_processing_feedback,
    ui_show_lockout_window, ui_close_lockout_window,
    pump_management_events_once,
    play_model_ready_sound_once,
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
    try:
        ui_show_lockout_window(
            "CtrlSpeak is installing CUDA runtime components required for GPU transcription. "
            "Please wait for this to finish."
        )
    except Exception:
        logger.exception("Failed to show lockout window during CUDA installation")

    rc = _run_cmd_stream(cmd, timeout=600)
    if rc != 0:
        notify("CUDA runtime installation failed (pip returned non-zero).")
        try:
            ui_close_lockout_window("CUDA runtime installation failed. CtrlSpeak will continue using the CPU.")
        except Exception:
            logger.exception("Failed to close lockout window after CUDA installation failure")
        return False

    # refresh DLL search path + verify
    time.sleep(0.5)
    ready = cuda_runtime_ready(ignore_preference=True)
    if not ready:
        try:
            ui_close_lockout_window(
                "CtrlSpeak installed the CUDA runtime but could not validate the GPU libraries."
            )
        except Exception:
            logger.exception("Failed to close lockout window after CUDA validation failure")
        return False

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


MB_DIVISOR = 1024.0 * 1024.0


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, secs = divmod(int(seconds + 0.5), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


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
        try:
            self.cancel_button.configure(state=tk.DISABLED)
        except Exception:
            pass

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
    """Download the selected model snapshot with a simple progress dialog."""

    model_name = (model_short or get_current_model_name()).strip()
    progress_queue: Queue = Queue()
    cancel_event = threading.Event()
    dialog = DownloadDialog(model_name, progress_queue, cancel_event)

    trace_model_download_step(
        "download_model_with_gui: start",
        "spawn worker thread",
    )

    def worker() -> None:
        try:
            repo_id = _model_repo_id(model_name)
            trace_model_download_step(
                "download_model_with_gui.worker: prepared repo id",
                "announce stage and ensure directory",
            )
            progress_queue.put(("stage", f"Preparing download from Hugging Face ({repo_id})"))
            store_path = model_store_path_for(model_name)
            store_path.mkdir(parents=True, exist_ok=True)
            trace_model_download_step(
                "download_model_with_gui.worker: starting snapshot_download",
                "wait for huggingface client to finish",
            )
            snapshot_download(
                repo_id,
                local_dir=str(store_path),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
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

    def _run() -> bool:
        result = dialog.run()
        thread.join(timeout=0.5)
        return result == "success"

    if not block_during_download:
        trace_model_download_step(
            "download_model_with_gui: running dialog without lockout",
            "wait for dialog result",
        )
        return _run()

    try:
        ui_show_lockout_window(
            "CtrlSpeak is downloading the Whisper model. Please wait for the download to finish before using CtrlSpeak."
        )
    except Exception:
        logger.exception("Failed to show lockout window during model download")
        trace_model_download_step(
            "download_model_with_gui: lockout window failed",
            "continue waiting for dialog result",
        )

    success = _run()

    try:
        if success:
            ui_close_lockout_window("The Whisper model download completed successfully.")
        else:
            ui_close_lockout_window("CtrlSpeak could not download the Whisper model. Please try again.")
    except Exception:
        logger.exception("Failed to close lockout window after model download")
        trace_model_download_step(
            "download_model_with_gui: closing lockout failed",
            "finish with dialog result",
        )

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
    """Auto-install the default model on first run without prompting the user."""

    with settings_lock:
        mode = settings.get("mode")
        auto_install_needed = not bool(settings.get("model_auto_install_complete"))

    if mode == "client":
        trace_model_download_step(
            "ensure_initial_model_installation: client mode",
            "skip auto install",
        )
        return True

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
    return _ensure_model_files(interactive=True)


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
                    notify(
                        "Running CtrlSpeak transcription on CPU fallback. "
                        "Set CTRLSPEAK_DEVICE=cpu to suppress this message."
                    )
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
