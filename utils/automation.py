# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from huggingface_hub import snapshot_download

from utils.config_paths import asset_path, get_config_dir, get_logger
from utils.system import (
    acquire_single_instance_lock,
    load_settings,
    settings,
    settings_lock,
    apply_auto_setup,
    save_settings,
    shutdown_all,
)
from utils.models import (
    AVAILABLE_MODELS,
    set_current_model_name,
    model_store_path_for,
    model_files_present,
    get_model_repo_id,
    set_device_preference,
    initialize_transcriber,
    transcribe_local,
    unload_transcriber,
    cuda_runtime_ready,
    ensure_cuda_runtime_from_existing,
    install_cuda_runtime_with_progress,
)


logger = get_logger(__name__)

def _automation_root() -> Path:
    root = get_config_dir() / "automation"
    root.mkdir(parents=True, exist_ok=True)
    return root


automation_root = _automation_root()
STATE_FILE = automation_root / "state.json"
ARTIFACT_DIR = automation_root / "artifacts"


class AutomationError(RuntimeError):
    """Raised when the automated regression workflow fails."""


def _log(message: str) -> None:
    """Emit a console + log message for automation progress."""

    logger.info("Automation: %s", message)
    print(f"[automation] {message}")


# ---------------------------------------------------------------------------
# Persistent state helpers
# ---------------------------------------------------------------------------


def _default_state() -> Dict[str, Any]:
    return {
        "model": {"done": False, "path": None},
        "cuda": {"done": False, "details": None},
        "cpu": {"done": False, "text": None},
        "gpu": {"done": False, "text": None},
        "export": {"done": False, "paths": []},
    }


def _load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return _default_state()
    try:
        state = _default_state()
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        for key, value in state.items():
            if isinstance(value, dict):
                value.update(data.get(key, {}))
            else:
                state[key] = data.get(key, value)
        return state
    except Exception:
        logger.exception("Failed to read automation state; starting fresh")
        return _default_state()


def _save_state(state: Dict[str, Any]) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        logger.exception("Failed to persist automation state")


def _clear_state() -> None:
    try:
        STATE_FILE.unlink(missing_ok=True)
    except Exception:
        logger.exception("Failed to remove automation state file")


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _config_dir_path() -> Path:
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "CtrlSpeak"


# ---------------------------------------------------------------------------
# Model + transcription helpers
# ---------------------------------------------------------------------------


def _mark_model_installed() -> None:
    with settings_lock:
        settings["model_auto_install_complete"] = True
    save_settings()


def _ensure_model_downloaded(model_name: str) -> Path:
    store_path = model_store_path_for(model_name)
    if model_files_present(store_path):
        _log(f"Model '{model_name}' already present at {store_path}")
        _mark_model_installed()
        return store_path

    repo_id = get_model_repo_id(model_name)
    _log(f"Downloading model '{model_name}' from {repo_id} into {store_path}")
    store_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id,
        local_dir=str(store_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    if not model_files_present(store_path):
        raise AutomationError(
            f"Model '{model_name}' download completed but files were not found at {store_path}"
        )
    _mark_model_installed()
    _log(f"Model '{model_name}' downloaded successfully")
    return store_path


def _ensure_test_audio() -> Path:
    test_wav = asset_path("test.wav")
    if not test_wav.is_file():
        raise AutomationError(f"Test audio not found at {test_wav}")
    return test_wav


def _transcribe_with_device(file_path: Path, device: str) -> str:
    unload_transcriber()
    model = initialize_transcriber(
        force=True,
        allow_client=True,
        preferred_device=device,
        interactive=False,
    )
    if model is None:
        raise AutomationError(f"Unable to initialize the transcriber on '{device}'")

    actual_device = str(getattr(model, "device", getattr(model, "_device", device))).lower()
    if device == "cuda" and "cuda" not in actual_device:
        raise AutomationError(
            f"Requested CUDA inference but model initialized on '{actual_device or 'unknown'}'"
        )

    text = transcribe_local(
        str(file_path),
        play_feedback=False,
        allow_client=True,
        preferred_device=device,
    )
    if not text:
        raise AutomationError(f"Transcription on '{device}' returned no text")

    preview = text if len(text) <= 80 else f"{text[:77]}..."
    _log(f"Transcription on '{device}' succeeded: {preview}")
    return text


# ---------------------------------------------------------------------------
# Stage execution helpers
# ---------------------------------------------------------------------------


def _ensure_settings_loaded() -> None:
    load_settings()
    apply_auto_setup("client_server")


def _stage_model(state: Dict[str, Any]) -> None:
    _ensure_settings_loaded()
    model_name = AVAILABLE_MODELS[0]
    set_current_model_name(model_name)
    path = _ensure_model_downloaded(model_name)
    state["model"]["done"] = True
    state["model"]["path"] = str(path)
    _save_state(state)


def _stage_cuda(state: Dict[str, Any]) -> None:
    _ensure_settings_loaded()
    if ensure_cuda_runtime_from_existing():
        state["cuda"]["details"] = "reused"
    else:
        _log("CUDA runtime not detected; installing NVIDIA CUDA components")
        if not install_cuda_runtime_with_progress(parent=None):
            raise AutomationError("CUDA runtime installation failed")
        if not ensure_cuda_runtime_from_existing():
            raise AutomationError("CUDA runtime validation failed after installation")
        state["cuda"]["details"] = "installed"
    state["cuda"]["done"] = True
    _save_state(state)


def _stage_cpu(state: Dict[str, Any]) -> None:
    _ensure_settings_loaded()
    test_wav = _ensure_test_audio()
    set_device_preference("cpu")
    cpu_text = _transcribe_with_device(test_wav, "cpu")
    state["cpu"]["done"] = True
    state["cpu"]["text"] = cpu_text
    _save_state(state)


def _stage_gpu(state: Dict[str, Any]) -> None:
    _ensure_settings_loaded()
    test_wav = _ensure_test_audio()
    set_device_preference("cuda")
    ensure_cuda_runtime_from_existing()
    if not cuda_runtime_ready(ignore_preference=True):
        raise AutomationError("CUDA runtime is not ready; cannot execute GPU transcription")
    gpu_text = _transcribe_with_device(test_wav, "cuda")
    state["gpu"]["done"] = True
    state["gpu"]["text"] = gpu_text
    _save_state(state)


def _export_transcripts(cpu_text: Optional[str], gpu_text: Optional[str], label: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = ARTIFACT_DIR / f"{label.lower().replace(' ', '_')}_{timestamp}.txt"
    canonical = gpu_text or cpu_text or "<no transcription available>"
    lines = [
        f"CtrlSpeak Automation {label}",
        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=== CPU Transcription ===",
        cpu_text or "<no cpu transcription captured>",
        "",
        "=== GPU Transcription ===",
        gpu_text or "<no gpu transcription captured>",
        "",
        "=== Text Injection Method Outputs ===",
    ]
    for method_name in (
        "Direct insert (simulation)",
        "SendInput paste (simulation)",
        "Clipboard paste (simulation)",
        "PyAutoGUI typing (simulation)",
    ):
        lines.extend([f"--- {method_name} ---", canonical, ""])
    lines.extend([
        "=== Canonical Transcript Used For Injection ===",
        canonical,
        "",
        "Note: Injection methods are simulated in headless automation mode.",
        "      Interactive verification requires a focused target window.",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    _log(f"{label}: transcript written to {output_path}")
    return output_path


def _stage_export(state: Dict[str, Any]) -> None:
    cpu_text = state["cpu"].get("text")
    gpu_text = state["gpu"].get("text")
    if not (cpu_text or gpu_text):
        raise AutomationError("No transcription available to export")
    export_path = _export_transcripts(cpu_text, gpu_text, label="Automation Run")
    paths = list(state["export"].get("paths", []))
    paths.append(str(export_path))
    state["export"]["paths"] = paths
    state["export"]["done"] = True
    _save_state(state)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_automation_flow() -> int:
    if not acquire_single_instance_lock():
        _log("Another CtrlSpeak instance is already running; aborting automation flow")
        return 1

    state = _load_state()
    _log(f"Automation managing app data at {_config_dir_path()}")

    try:
        if not state["model"].get("done"):
            _stage_model(state)

        if not state["cuda"].get("done"):
            _stage_cuda(state)

        if not state["cpu"].get("done"):
            _stage_cpu(state)

        if not state["gpu"].get("done"):
            _stage_gpu(state)

        if not state["export"].get("done"):
            _stage_export(state)

        _log("Automation flow completed successfully.")
        _clear_state()
        return 0
    except AutomationError as exc:
        _log(f"Automation flow halted: {exc}")
        return 2
    except Exception as exc:  # pragma: no cover - unexpected failure path
        logger.exception("Automation flow crashed")
        _log(f"Automation flow crashed: {exc}")
        return 3
    finally:
        shutdown_all()
        unload_transcriber()
        time.sleep(0.2)



