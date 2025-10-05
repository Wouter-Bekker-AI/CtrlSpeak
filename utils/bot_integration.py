# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
import sys
import time
import threading
import subprocess
from multiprocessing import Process, Queue as MPQueue
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import requests

from background_agents.datetime_memory_agent import refresh_datetime_memory
from background_agents.document_memory_agent import refresh_document_memory
from utils.config_paths import get_logger, settings, settings_lock
from utils.system import (
    CLIENT_ONLY_BUILD,
    get_best_server,
    load_settings,
    save_settings,
    start_server,
    ui_close_lockout_window,
    ui_show_lockout_window,
    ui_update_lockout_message,
)

from utils.models import DEFAULT_MODEL_NAME, WelcomeWindow
from utils.memory_lock import IdentityLock, IdentityLockError
from utils.memory_paths import get_bot_memory_dir, get_bot_traces_dir
from utils.metrics import MetricsRecorder

def _ensure_server_defaults() -> None:
    """Make sure settings permit running the embedded server."""
    load_settings()
    should_save = False
    with settings_lock:
        mode = settings.get("mode")
        if mode != "client_server":
            settings["mode"] = "client_server"
            should_save = True
        if not settings.get("model_name"):
            settings["model_name"] = DEFAULT_MODEL_NAME
            should_save = True
    if should_save:
        save_settings()

logger = get_logger(__name__)

_SOCIAL_ROBOT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "social_robot"
_DEFAULT_PERSONAS_ROOT = _SOCIAL_ROBOT_ROOT / "personas"
_DEFAULT_IDENTITY_NAME = "reception"
_DEFAULT_LLM_MODEL = "gemma3:1b"
_DEFAULT_LLM_URL = "http://localhost:11434/api/chat"
_BACKGROUND_AGENTS_ROOT = Path(__file__).resolve().parents[1] / "background_agents"
_TTS_PREPROCESSOR_DIR = _BACKGROUND_AGENTS_ROOT / "tts_preprocessing_agent"

_bot_proc: Optional[subprocess.Popen] = None
_bot_stdin_lock = threading.Lock()
_active_identity: Optional[str] = None
_identity_lock: Optional[IdentityLock] = None

_OLLAMA_HARDWARE_CHOICES = {"cpu_only", "cpu_and_gpu", "gpu_only"}
_DOC_REFRESH_IDENTITIES = {"vision", "reception", "einstein"}
_PRELAUNCH_REFRESHERS = (
    ("documentation memory", refresh_document_memory),
    ("date/time memory", refresh_datetime_memory),
)


def _should_refresh_docs_on_start(identity: str) -> bool:
    return identity.strip().lower() in _DOC_REFRESH_IDENTITIES


def _prepare_identity_memories(identity: str) -> bool:
    for description, helper in _PRELAUNCH_REFRESHERS:
        try:
            if not helper(identity, reason="startup"):
                logger.error(
                    "Aborting launch because %s preparation failed for %s",
                    description,
                    identity,
                )
                return False
        except Exception:
            logger.exception(
                "Unexpected error while preparing %s for %s",
                description,
                identity,
            )
            return False
    return True


def list_available_identities(identities_dir: Optional[str] = None) -> list[str]:
    """Return the sorted list of SocialRobot persona folder names."""

    root = _resolve_identities_root(identities_dir)
    try:
        return sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    except FileNotFoundError:
        logger.warning("Persona directory %s does not exist", root)
    except Exception:
        logger.exception("Failed to enumerate personas under %s", root)
    return []


def _resolve_stt_url() -> Optional[str]:
    """Return a usable /transcribe URL or None if unavailable."""
    try:
        if not CLIENT_ONLY_BUILD:
            _ensure_server_defaults()
            # Ensure local server running; read configured port
            start_server()
            port = int(settings.get("server_port", 65432))
            return f"http://127.0.0.1:{port}/transcribe"
        else:
            # Client-only: try a discovered server
            best = get_best_server()
            if best:
                return f"http://{best.host}:{best.port}/transcribe"
            return None
    except Exception:
        logger.exception("Failed to resolve STT URL")
        return None


def _resolve_identities_root(identities_dir: Optional[str]) -> Path:
    candidate = identities_dir or os.getenv("BOT_IDENTITIES_DIR")
    if candidate:
        try:
            return Path(candidate).expanduser().resolve()
        except Exception:
            logger.exception("Failed to resolve persona directory %s", candidate)
    return _DEFAULT_PERSONAS_ROOT


def _load_identity_config(identity: str, identities_dir: Optional[str]) -> tuple[dict, Path]:
    root = _resolve_identities_root(identities_dir)
    identity_path = (root / identity).expanduser()
    try:
        identity_path = identity_path.resolve()
    except Exception:
        logger.exception("Failed to resolve identity path for %s", identity)

    config: dict = {}
    config_path = identity_path / "identity.json"
    if config_path.exists():
        try:
            config.update(json.loads(config_path.read_text(encoding="utf-8")))
        except Exception:
            logger.exception("Failed to parse identity config at %s", config_path)

    return config, identity_path


def _identity_requires_text_cleaning(
    identity: Optional[str], identities_dir: Optional[str]
) -> bool:
    target = _normalized_identity(identity)
    config, _identity_path = _load_identity_config(target, identities_dir)
    value = config.get("require_text_cleaning")
    if isinstance(value, bool):
        return value
    if value is not None:
        logger.warning(
            "Ignoring require_text_cleaning for identity %s; expected boolean but received %r",
            target,
            value,
        )
    return True


def _load_preprocessor_identity() -> tuple[Optional[Path], Optional[str], Optional[str], Dict[str, Any], Optional[str]]:
    agent_dir = _TTS_PREPROCESSOR_DIR
    if not agent_dir.exists():
        logger.warning("TTS preprocessing agent directory not found at %s", agent_dir)
        return None, None, None, {}, None

    identity_path = agent_dir / "identity.json"
    if not identity_path.exists():
        logger.warning("TTS preprocessing agent identity.json missing at %s", identity_path)
        return agent_dir, None, None, {}, None

    try:
        config = json.loads(identity_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to parse TTS preprocessing agent identity at %s", identity_path)
        return agent_dir, None, None, {}, None

    options, hardware = _extract_ollama_preferences(config)
    llm_url = config.get("llm_url")
    llm_model = config.get("llm_model")
    return agent_dir, llm_url, llm_model, options, hardware


def _extract_ollama_preferences(config: dict) -> tuple[Dict[str, Any], Optional[str]]:
    options: Dict[str, Any] = {}
    raw_options = config.get("ollama_options")
    if isinstance(raw_options, dict):
        options = {k: v for k, v in raw_options.items() if v is not None}
    elif raw_options is not None:
        logger.warning("Ignoring ollama_options in identity config because it is not an object: %r", raw_options)

    hardware_pref = config.get("ollama_hardware")
    if isinstance(hardware_pref, str):
        normalized = hardware_pref.strip().lower()
        if normalized in _OLLAMA_HARDWARE_CHOICES:
            return options, normalized
        if normalized:
            logger.warning(
                "Ignoring ollama_hardware value '%s'; expected one of %s", hardware_pref, sorted(_OLLAMA_HARDWARE_CHOICES)
            )
        return options, None

    if hardware_pref is not None:
        logger.warning("Ignoring ollama_hardware in identity config because it is not a string: %r", hardware_pref)

    return options, None


def _compose_ollama_options(
    base_options: Optional[Dict[str, Any]], hardware_mode: Optional[str]
) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    if base_options:
        options.update({k: v for k, v in base_options.items() if v is not None})
    if hardware_mode:
        mode = hardware_mode.strip().lower()
        if mode == "cpu_only":
            options["gpu_only"] = False
            options["num_gpu"] = 0
        elif mode == "gpu_only":
            options["gpu_only"] = True
        elif mode == "cpu_and_gpu":
            options.setdefault("gpu_only", False)
    return options


def _identity_display_name(identity: str, config: dict) -> str:
    display = str(config.get("name") or "").strip()
    if not display:
        display = identity.replace("_", " ").strip()
    if not display:
        display = "Assistant"
    return display.title()


def _normalize_ollama_base_url(llm_url: str) -> Optional[str]:
    llm_url = (llm_url or "").strip()
    if not llm_url:
        return None

    parsed = urlparse(llm_url)
    if not parsed.scheme or not parsed.netloc:
        return None

    path = parsed.path or ""
    if path.endswith("/chat"):
        path = path[: -len("/chat")]
    path = path or "/api"
    if not path.endswith("/api"):
        path = path.rstrip("/")
        if not path:
            path = "/api"
        else:
            if not path.endswith("/api"):
                path = f"{path}/api"
    if not path.startswith("/"):
        path = f"/{path}"

    normalized = urlunparse((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "", ""))
    return normalized.rstrip("/")


def _ollama_model_state(base_url: str, model_name: str) -> Optional[bool]:
    endpoint = f"{base_url.rstrip('/')}/show"
    payload = {"model": model_name}
    try:
        response = requests.post(endpoint, json=payload, timeout=10)
    except Exception as exc:
        logger.warning(
            "Failed to reach Ollama when checking model %s at %s: %s",
            model_name,
            base_url,
            exc,
        )
        return None

    if response.status_code == 200:
        return True

    error_text = ""
    try:
        data = response.json()
        error_text = str(data.get("error") or "")
    except Exception:
        error_text = response.text or ""

    lowered = error_text.lower()
    if response.status_code in (400, 404) and ("not found" in lowered):
        return False
    if response.status_code == 404:
        return False
    if "not found" in lowered:
        return False

    if response.status_code >= 400:
        logger.warning(
            "Unexpected response while checking Ollama model %s (HTTP %s): %s",
            model_name,
            response.status_code,
            error_text.strip() or response.reason,
        )

    return None


def _ollama_pull_worker(model_name: str, base_url: str, queue: MPQueue) -> None:
    def _put(payload: tuple[str, ...]) -> None:
        try:
            queue.put_nowait(payload)
        except Exception:
            logger.exception("Failed to enqueue Ollama pull progress update")

    session: Optional[requests.Session] = None
    try:
        session = requests.Session()
        pull_url = f"{base_url.rstrip('/')}/pull"
        _put(("stage", f"Contacting Ollama to download {model_name}…"))
        response = session.post(
            pull_url,
            json={"model": model_name, "stream": True},
            stream=True,
            timeout=30,
        )
        response.raise_for_status()

        total_bytes: Optional[int] = None
        completed_bytes = 0

        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line.decode("utf-8"))
            except Exception:
                continue

            status_text = str(payload.get("status") or "").strip()
            if status_text:
                _put(("stage", status_text))

            error_text = str(payload.get("error") or "").strip()
            if error_text:
                raise RuntimeError(error_text)

            if isinstance(payload.get("total"), int):
                total_bytes = payload["total"]
            if isinstance(payload.get("completed"), int):
                completed_bytes = payload["completed"]

            if total_bytes is not None and total_bytes > 0:
                _put(("progress", completed_bytes, total_bytes))
            elif completed_bytes > 0:
                _put(("progress", completed_bytes, 0))

        _put(("done",))
    except Exception as exc:
        logger.exception("Failed to download Ollama model %s", model_name)
        _put(("error", f"Failed to download the assistant model: {exc}"))
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                logger.exception("Failed to close Ollama pull session")


def _download_ollama_model_with_gui(
    identity_display: str,
    model_name: str,
    base_url: str,
) -> bool:
    progress_queue: MPQueue = MPQueue()
    process = Process(target=_ollama_pull_worker, args=(model_name, base_url, progress_queue))
    process.daemon = True
    process.start()

    cancel_requested = threading.Event()
    monitor_stop = threading.Event()

    def _request_cancel() -> None:
        if cancel_requested.is_set():
            return
        cancel_requested.set()
        try:
            ui_update_lockout_message("Cancelling persona model download…")
        except Exception:
            logger.exception("Failed to update lockout message while cancelling Ollama download")
        if process.is_alive():
            try:
                process.terminate()
            except Exception:
                logger.exception("Failed to terminate Ollama download process")
        try:
            progress_queue.put_nowait(("cancelled",))
        except Exception:
            logger.exception("Failed to enqueue Ollama download cancellation notification")

    pretty_name = (identity_display or "Vision").strip() or "Vision"
    if pretty_name.lower().endswith("persona"):
        persona_label = pretty_name
    else:
        persona_label = f"{pretty_name} persona"

    window_label = f"{persona_label} model ({model_name})"
    initial_message = f"We're downloading the {model_name} model for the {persona_label}."

    def _monitor_model_availability() -> None:
        """Poll Ollama so the UI can finish even if streaming never signals EOF."""

        # Give Ollama a brief head start so we don't spam logs before it reacts.
        time.sleep(2.0)

        while not cancel_requested.is_set() and not monitor_stop.is_set():
            try:
                state = _ollama_model_state(base_url, model_name)
            except Exception:
                state = None

            if state is True:
                try:
                    progress_queue.put_nowait(("stage", f"Preparing the {persona_label}…"))
                except Exception:
                    logger.exception("Failed to enqueue Ollama preparation stage update")
                try:
                    progress_queue.put_nowait(("done",))
                except Exception:
                    logger.exception("Failed to enqueue Ollama download completion signal")
                return

            # If Ollama can't confirm the model yet just keep waiting – a
            # successful pull will eventually make the model visible via /show.
            time.sleep(2.5)

    lockout_open = False
    try:
        ui_show_lockout_window(initial_message, cancel_callback=_request_cancel)
        lockout_open = True
    except Exception:
        logger.exception("Failed to show lockout window during Ollama model download")

    monitor_thread = threading.Thread(target=_monitor_model_availability, name="OllamaModelMonitor", daemon=True)
    monitor_thread.start()

    window = WelcomeWindow(window_label, progress_queue, process)
    status, error_message = window.run()

    monitor_stop.set()
    if monitor_thread.is_alive():
        monitor_thread.join(timeout=1.5)

    process.join(timeout=1.0)
    if process.is_alive():
        try:
            process.terminate()
        except Exception:
            logger.exception("Failed to terminate Ollama download worker after window closed")
        process.join(timeout=0.5)

    try:
        progress_queue.close()
    except Exception:
        logger.exception("Failed to close Ollama download progress queue")
    try:
        progress_queue.join_thread()
    except Exception:
        logger.exception("Failed to join Ollama download queue thread")

    if status == "success":
        # Double-check with Ollama so we don't race returning before the model
        # is actually advertised as available.
        deadline = time.time() + 30.0
        while time.time() < deadline and not cancel_requested.is_set():
            state = _ollama_model_state(base_url, model_name)
            if state is True:
                break
            if state is False:
                time.sleep(1.0)
                continue
            # When Ollama is unreachable we should not block indefinitely.
            time.sleep(1.0)

        try:
            if lockout_open:
                ui_close_lockout_window(f"The {persona_label} model is ready.")
        except Exception:
            logger.exception("Failed to close lockout window after Ollama download success")
        return True

    if status == "cancelled":
        message = f"The {persona_label} model download was cancelled."
    else:
        message = error_message or f"CtrlSpeak could not download the {persona_label} model."

    try:
        if lockout_open:
            ui_close_lockout_window(message)
    except Exception:
        logger.exception("Failed to close lockout window after Ollama download failure")

    return False


def _ensure_identity_llm_ready(
    identity: Optional[str],
    identities_dir: Optional[str],
    override_model: Optional[str],
    override_url: Optional[str],
) -> bool:
    identity_name = (identity or os.getenv("BOT_IDENTITY") or _DEFAULT_IDENTITY_NAME).strip() or _DEFAULT_IDENTITY_NAME
    config, _identity_path = _load_identity_config(identity_name, identities_dir)
    display_name = _identity_display_name(identity_name, config)

    resolved_model = (override_model or config.get("llm_model") or os.getenv("BOT_LLM_MODEL") or _DEFAULT_LLM_MODEL)
    resolved_model = (resolved_model or "").strip()
    resolved_url = (override_url or config.get("llm_url") or os.getenv("BOT_LLM_URL") or _DEFAULT_LLM_URL)
    resolved_url = (resolved_url or "").strip()

    base_url = _normalize_ollama_base_url(resolved_url)
    if not resolved_model or not base_url:
        logger.debug(
            "Skipping Ollama model management (model=%s, base_url=%s)",
            resolved_model,
            base_url,
        )
        return True

    state = _ollama_model_state(base_url, resolved_model)
    if state is True:
        logger.info("Ollama model %s already present for identity %s", resolved_model, identity_name)
        return True

    if state is None:
        logger.warning(
            "Unable to confirm Ollama model %s for identity %s; proceeding without managed download.",
            resolved_model,
            identity_name,
        )
        return True

    logger.info(
        "Downloading Ollama model %s for identity %s via welcome workflow", resolved_model, identity_name
    )
    return _download_ollama_model_with_gui(display_name, resolved_model, base_url)


def _ensure_preprocessor_llm_ready(llm_model: Optional[str], llm_url: Optional[str]) -> bool:
    model_name = (llm_model or "").strip()
    resolved_url = (llm_url or "").strip()
    if not model_name or not resolved_url:
        logger.debug(
            "Skipping TTS preprocessing agent model management (model=%s, url=%s)",
            model_name,
            resolved_url,
        )
        return True

    base_url = _normalize_ollama_base_url(resolved_url)
    if not base_url:
        logger.warning(
            "Unable to normalize Ollama URL %s for TTS preprocessing agent", resolved_url
        )
        return True

    state = _ollama_model_state(base_url, model_name)
    if state is True:
        logger.info("Ollama model %s already present for TTS preprocessing agent", model_name)
        return True

    if state is None:
        logger.warning(
            "Unable to confirm Ollama model %s for TTS preprocessing agent; proceeding without managed download.",
            model_name,
        )
        return True

    logger.info(
        "Downloading Ollama model %s for TTS preprocessing agent via welcome workflow",
        model_name,
    )
    return _download_ollama_model_with_gui("TTS preprocessing agent", model_name, base_url)


def _load_identity_llm_config(
    identity: str, identities_root: Path
) -> tuple[Optional[str], Optional[str], Dict[str, Any], Optional[str]]:
    config_path = identities_root / identity / "identity.json"
    if not config_path.exists():
        return None, None, {}, None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to parse identity configuration for '%s'", identity)
        return None, None, {}, None
    options, hardware = _extract_ollama_preferences(data)
    return data.get("llm_url"), data.get("llm_model"), options, hardware


def _resolve_llm_settings(
    *,
    llm_url: Optional[str],
    llm_model: Optional[str],
    identity: Optional[str],
    identities_root: Path,
) -> tuple[Optional[str], Optional[str], Dict[str, Any], Optional[str]]:
    resolved_url = llm_url or os.environ.get("BOT_LLM_URL")
    resolved_model = llm_model or os.environ.get("BOT_LLM_MODEL")
    resolved_options: Dict[str, Any] = {}
    resolved_hardware: Optional[str] = None
    if identity:
        identity_url, identity_model, identity_options, identity_hardware = _load_identity_llm_config(
            identity, identities_root
        )
        if not resolved_url:
            resolved_url = identity_url
        if not resolved_model:
            resolved_model = identity_model
        resolved_options = identity_options
        resolved_hardware = identity_hardware
    return resolved_url, resolved_model, resolved_options, resolved_hardware


def _warm_ollama_model(
    llm_url: Optional[str],
    llm_model: Optional[str],
    identity_options: Optional[Dict[str, Any]] = None,
    hardware_mode: Optional[str] = None,
) -> None:
    if not llm_url or not llm_model:
        return

    warm_url = llm_url.rstrip("/")
    if warm_url.endswith("/chat"):
        warm_url = warm_url[: -len("/chat")]
    warm_url = f"{warm_url}/generate"

    payload = {
        "model": llm_model,
        "prompt": "",
        "stream": False,
        "keep_alive": "5m",
    }
    options = _compose_ollama_options(identity_options, hardware_mode)
    if options:
        payload["options"] = options

    try:
        response = requests.post(warm_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Preloaded Ollama model %s", llm_model)
    except Exception:
        logger.warning("Failed to preload Ollama model %s", llm_model, exc_info=True)


def _normalized_identity(identity: Optional[str]) -> str:
    candidate = (identity or os.getenv("BOT_IDENTITY") or _DEFAULT_IDENTITY_NAME).strip()
    return candidate or _DEFAULT_IDENTITY_NAME


def _monitor_bot_exit(proc: subprocess.Popen) -> None:
    global _active_identity, _identity_lock
    try:
        proc.wait()
    except Exception:
        logger.exception("Bot monitor thread encountered an error")
    finally:
        if _bot_proc is not None and _bot_proc is proc:
            _active_identity = None
            if _identity_lock is not None:
                _identity_lock.release()
                _identity_lock = None


def _release_identity_lock() -> None:
    global _identity_lock
    if _identity_lock is not None:
        _identity_lock.release()
        _identity_lock = None


def start_bot(
    llm_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    voice: Optional[str] = None,
    identity: Optional[str] = None,
    identities_dir: Optional[str] = None,
    prompt_file: Optional[str] = None,
    read_prompt_from_file: Optional[bool] = None,
    system_prompt: Optional[str] = None,
    memory_dir: Optional[str] = None,
) -> bool:
    """Spawn SocialRobot main as a subprocess with CTRLSPEAK_STT_URL.
    Returns True if process started; False otherwise.

    Additional optional arguments allow choosing a predefined identity profile,
    overriding the Kokoro voice or LLM settings, or pointing at alternate identity
    directories and prompt files.
    """
    global _bot_proc, _active_identity, _identity_lock
    if _bot_proc and _bot_proc.poll() is None:
        logger.info("Bot already running")
        return True

    stt_url = _resolve_stt_url()
    if not stt_url:
        logger.error("No CtrlSpeak STT server available; cannot start bot")
        return False

    target_identity = _normalized_identity(identity)

    # Assume SocialRobot vendored under third_party/social_robot
    root = Path(__file__).resolve().parents[1]
    robot_dir = root / "third_party" / "social_robot"
    entry = robot_dir / "main.py"
    if not entry.exists():
        logger.error("SocialRobot entrypoint not found at %s", entry)
        return False

    if identities_dir:
        identities_root = Path(identities_dir).expanduser()
        if not identities_root.is_absolute():
            identities_root = robot_dir / identities_root
    else:
        identities_root = robot_dir / "personas"
    try:
        identities_root = identities_root.resolve()
    except FileNotFoundError:
        logger.exception("Failed to resolve SocialRobot personas root at %s", identities_root)

    (
        resolved_llm_url,
        resolved_llm_model,
        identity_options,
        identity_hardware,
    ) = _resolve_llm_settings(
        llm_url=llm_url,
        llm_model=llm_model,
        identity=identity,
        identities_root=identities_root,
    )
    if not _ensure_identity_llm_ready(identity, identities_dir, llm_model, llm_url):
        return False

    memory_root: Optional[Path]
    if memory_dir:
        try:
            candidate = Path(memory_dir).expanduser()
            candidate.mkdir(parents=True, exist_ok=True)
            memory_root = candidate.resolve()
        except Exception:
            logger.exception("Failed to prepare explicit memory directory %s", memory_dir)
            return False
    else:
        try:
            memory_root = get_bot_memory_dir(target_identity)
        except Exception:
            logger.exception("Failed to prepare memory root for identity %s", target_identity)
            return False

    metrics_path = get_bot_traces_dir(target_identity) / "metrics.csv"
    lock_started = time.perf_counter()
    try:
        lock = IdentityLock(target_identity)
        lock.acquire(timeout=0.0)
    except IdentityLockError:
        logger.error("Identity '%s' is already in use.", target_identity)
        return False
    _identity_lock = lock
    lock_wait_ms = (time.perf_counter() - lock_started) * 1000.0
    try:
        MetricsRecorder(metrics_path).record("bootstrap", {"lock_wait_ms": round(lock_wait_ms, 2)})
    except Exception:
        logger.debug("Failed to record lock wait metric", exc_info=True)

    if _should_refresh_docs_on_start(target_identity):
        if not _prepare_identity_memories(target_identity):
            _release_identity_lock()
            return False

    if _identity_requires_text_cleaning(target_identity, identities_dir):
        if _TTS_PREPROCESSOR_DIR.exists():
            logger.info(
                "Identity %s requires text cleaning; skipping TTS preprocessing agent warm-up",
                target_identity,
            )
        else:
            logger.info(
                "Identity %s requires text cleaning but no preprocessing agent assets were found at %s",
                target_identity,
                _TTS_PREPROCESSOR_DIR,
            )
    else:
        logger.info(
            "Identity %s does not require TTS preprocessing; skipping agent preparation",
            target_identity,
        )

    normalized_base_url: Optional[str] = None
    resolved_model_name: Optional[str] = None
    if resolved_llm_model:
        resolved_model_name = resolved_llm_model.strip() or None
    if resolved_llm_url:
        normalized_base_url = _normalize_ollama_base_url(resolved_llm_url)

    if normalized_base_url and resolved_model_name:
        state = _ollama_model_state(normalized_base_url, resolved_model_name)
        if state is True:
            _warm_ollama_model(
                resolved_llm_url,
                resolved_model_name,
                identity_options,
                identity_hardware,
            )
        else:
            logger.debug(
                "Skipping Ollama warm-up for model %s (state=%s)",
                resolved_model_name,
                state,
            )

    env = os.environ.copy()
    env["CTRLSPEAK_STT_URL"] = stt_url
    env["BOT_STT"] = "remote"
    if llm_url:
        env["BOT_LLM_URL"] = llm_url
    if llm_model:
        env["BOT_LLM_MODEL"] = llm_model
    if voice:
        env["BOT_VOICE"] = voice
    if identity:
        env["BOT_IDENTITY"] = identity
    if identities_dir:
        env["BOT_IDENTITIES_DIR"] = identities_dir

    load_settings()
    forced_langgraph = False
    theme_updated = False
    theme_pref = "dark"
    with settings_lock:
        use_langgraph = bool(settings.get("use_langgraph_memory_orchestrator", False))
        if _should_refresh_docs_on_start(target_identity) and not use_langgraph:
            use_langgraph = True
            settings["use_langgraph_memory_orchestrator"] = True
            forced_langgraph = True
        theme_pref = str(settings.get("chat_theme", "dark") or "dark").lower()
        if theme_pref not in {"light", "dark"}:
            theme_pref = "dark"
            settings["chat_theme"] = theme_pref
            theme_updated = True
    if forced_langgraph:
        try:
            save_settings()
            logger.info(
                "Enabled LangGraph memory orchestrator for %s identity to guarantee documentation retrieval.",
                target_identity,
            )
        except Exception:
            logger.exception(
                "Failed to persist LangGraph orchestrator setting for %s", target_identity
            )
    elif theme_updated:
        try:
            save_settings()
        except Exception:
            logger.exception("Failed to persist chat theme preference during bot launch")
    if use_langgraph:
        env["CTRLSPK_USE_LANGGRAPH_MEMORY_ORCHESTRATOR"] = "1"

    env["CTRLSPK_METRICS_PATH"] = str(metrics_path)

    memory_dir_arg = str(memory_root)
    env["CTRLSPK_BOT_MEMORY_ROOT"] = memory_dir_arg
    env["BOT_MEMORY_DIR"] = memory_dir_arg
    env["CTRLSPK_CHAT_THEME"] = theme_pref
    if _identity_lock is not None:
        env["CTRLSPK_PARENT_LOCKED"] = "1"
        env["CTRLSPK_IDENTITY_LOCK_PATH"] = str(_identity_lock.lock_path)

    cmd = [sys.executable, str(entry), "--stt", "remote", "--stt-url", stt_url]
    if identity:
        cmd.extend(["--identity", identity])
    if identities_dir:
        cmd.extend(["--identities-dir", identities_dir])
    if prompt_file:
        cmd.extend(["--prompt-file", str(prompt_file)])
    if read_prompt_from_file is True:
        cmd.append("--read-prompt-from-file")
    elif read_prompt_from_file is False:
        cmd.append("--no-read-prompt-from-file")
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    if memory_dir_arg:
        cmd.extend(["--memory-dir", memory_dir_arg])
    if theme_pref:
        cmd.extend(["--theme", theme_pref])

    logger.info("Starting SocialRobot: %s", " ".join(cmd))
    try:
        _bot_proc = subprocess.Popen(
            cmd,
            cwd=str(robot_dir),
            env=env,
            stdin=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        _active_identity = target_identity
        threading.Thread(target=_monitor_bot_exit, args=(_bot_proc,), daemon=True).start()
        time.sleep(0.35)  # give it a moment to open the window
        return True
    except Exception:
        logger.exception("Failed to start SocialRobot")
        _bot_proc = None
        _active_identity = None
        _release_identity_lock()
        return False


def stop_bot() -> None:
    global _bot_proc, _active_identity
    if _bot_proc is None:
        return
    try:
        if _bot_proc.stdin:
            try:
                _bot_proc.stdin.close()
            except Exception:
                logger.warning("Failed to close bot stdin", exc_info=True)
        if _bot_proc.poll() is None:
            _bot_proc.terminate()
            try:
                _bot_proc.wait(timeout=5)
            except Exception:
                logger.exception("Failed to wait for SocialRobot to exit after terminate")
            if _bot_proc.poll() is None:
                _bot_proc.kill()
                try:
                    _bot_proc.wait(timeout=3)
                except Exception:
                    logger.exception("Failed to wait for SocialRobot to exit after kill")
    except Exception:
        logger.exception("Error while stopping SocialRobot")
    finally:
        _bot_proc = None
        _active_identity = None
        _release_identity_lock()


def request_goodbye(identity: Optional[str] = None, timeout: float = 3.0) -> bool:
    """Ask the running bot to shut down via its stdin channel.

    Returns ``True`` when the bot exits within ``timeout`` seconds. ``False``
    indicates that the caller should fall back to :func:`stop_bot`.
    """

    global _bot_proc
    proc = _bot_proc
    identity_name = identity or _active_identity or ""
    if proc is None or proc.poll() is not None or proc.stdin is None:
        logger.debug(
            "request_goodbye skipped because bot process is unavailable (identity=%s)",
            identity_name or "<unknown>",
        )
        return False

    payload = json.dumps(
        {
            "command": "goodbye",
            "identity": (identity or _active_identity or ""),
        }
    )
    try:
        with _bot_stdin_lock:
            proc.stdin.write(payload + "\n")
            proc.stdin.flush()
        logger.debug(
            "Sent goodbye command to SocialRobot (identity=%s)",
            identity_name or "<unknown>",
        )
    except Exception:
        logger.exception("Failed to send 'goodbye' command to bot")
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            logger.info(
                "Bot process exited after goodbye request (identity=%s)",
                identity_name or "<unknown>",
            )
            return True
        time.sleep(0.05)
    logger.info(
        "Bot process still running %.1fs after goodbye request (identity=%s); caller should apply fallback",
        timeout,
        identity_name or "<unknown>",
    )
    return False


def is_bot_running() -> bool:
    return _bot_proc is not None and _bot_proc.poll() is None


def get_active_identity() -> Optional[str]:
    """Return the currently running identity name, if any."""

    if not is_bot_running():
        return None
    return _active_identity


def get_identity_tts_preferences(
    identity: Optional[str],
    *,
    identities_dir: Optional[str] = None,
) -> tuple[Optional[str], dict[str, Any]]:
    """Return the preferred voice and TTS provider settings for an identity.

    Environment variable overrides (``BOT_VOICE`` and the ``BOT_TTS_*`` set)
    take precedence over the values stored in ``identity.json`` so the
    behaviour matches SocialRobot's launch pipeline.
    """

    def _coerce_int(value: Any) -> Optional[int]:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return int(stripped)
            except ValueError:
                logger.warning("Ignoring non-integer device id override: %r", value)
        return None

    def _parse_json_mapping(value: Any, *, context: str) -> Optional[dict]:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            try:
                parsed = json.loads(candidate)
            except Exception:
                logger.warning("Ignoring %s override; JSON decode failed", context, exc_info=True)
                return None
            if isinstance(parsed, dict):
                return parsed
        logger.warning("Ignoring %s override; expected a JSON object", context)
        return None

    def _parse_json_sequence(value: Any, *, context: str) -> Optional[list]:
        if value is None:
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            try:
                parsed = json.loads(candidate)
            except Exception:
                logger.warning("Ignoring %s override; JSON decode failed", context, exc_info=True)
                return None
            if isinstance(parsed, list):
                return parsed
        logger.warning("Ignoring %s override; expected a JSON array", context)
        return None

    normalized_identity = _normalized_identity(identity)
    config, _ = _load_identity_config(normalized_identity, identities_dir)

    voice: Optional[str] = os.getenv("BOT_VOICE")
    if not voice:
        raw_voice = config.get("voice")
        if isinstance(raw_voice, str):
            candidate = raw_voice.strip()
            voice = candidate or None
    else:
        voice = voice.strip() or None

    raw_tts_config = config.get("tts")
    tts_config: dict[str, Any] = raw_tts_config if isinstance(raw_tts_config, dict) else {}

    provider = os.getenv("BOT_TTS_PROVIDER")
    if not provider:
        raw_provider = tts_config.get("onnx_provider") or tts_config.get("provider")
        if isinstance(raw_provider, str):
            provider = raw_provider.strip() or None
        elif raw_provider is not None:
            logger.warning("Ignoring tts.onnx_provider; expected a string value")
    else:
        provider = provider.strip() or None

    device_id = os.getenv("BOT_TTS_DEVICE_ID")
    parsed_device_id = _coerce_int(device_id)
    if parsed_device_id is None:
        parsed_device_id = _coerce_int(tts_config.get("device_id"))

    provider_options = _parse_json_mapping(
        os.getenv("BOT_TTS_PROVIDER_OPTIONS"), context="BOT_TTS_PROVIDER_OPTIONS"
    )
    if provider_options is None:
        provider_options = _parse_json_mapping(
            tts_config.get("provider_options"), context="tts.provider_options"
        )

    providers = _parse_json_sequence(os.getenv("BOT_TTS_PROVIDERS"), context="BOT_TTS_PROVIDERS")
    if providers is None:
        providers = _parse_json_sequence(tts_config.get("providers"), context="tts.providers")
    if providers is None:
        providers = _parse_json_sequence(tts_config.get("onnx_providers"), context="tts.onnx_providers")

    preferences: dict[str, Any] = {}
    if provider:
        preferences["onnx_provider"] = provider
    if parsed_device_id is not None:
        preferences["onnx_device_id"] = parsed_device_id
    if provider_options:
        preferences["onnx_provider_options"] = provider_options
    if providers:
        preferences["onnx_providers"] = providers

    return voice, preferences


def _send_bot_command(command: str, extra: Optional[dict] = None) -> bool:
    """Send a JSON control command to the running SocialRobot process."""

    proc = _bot_proc
    if proc is None or proc.poll() is not None:
        logger.debug(
            "Bot command '%s' skipped because the bot process is unavailable",
            command,
        )
        return False
    if proc.stdin is None:
        logger.debug(
            "Bot command '%s' skipped because stdin is unavailable",
            command,
        )
        return False

    payload: dict[str, object] = {"command": command}
    if extra:
        payload.update(extra)

    message = json.dumps(payload)
    try:
        with _bot_stdin_lock:
            proc.stdin.write(message + "\n")
            proc.stdin.flush()
        return True
    except Exception:
        logger.exception("Failed to send '%s' command to bot", command)
        return False


def request_bot_screenshot() -> bool:
    """Request that the running bot execute the look-at-my-screen workflow."""

    if not _send_bot_command("look_at_my_screen"):
        logger.error("Bot is not running; cannot request screenshot")
        return False
    return True


def pause_bot_vad_listener() -> bool:
    """Ask SocialRobot to pause VAD capture while the push-to-talk hotkey is held."""

    return _send_bot_command("pause_vad")


def resume_bot_vad_listener() -> bool:
    """Ask SocialRobot to resume VAD capture after the push-to-talk hotkey ends."""

    return _send_bot_command("resume_vad")


def update_bot_theme(theme: str) -> bool:
    """Request that SocialRobot switch to the specified chat theme immediately."""

    normalized = str(theme or "").strip().lower()
    if normalized not in {"light", "dark"}:
        normalized = "dark"
    return _send_bot_command("set_theme", {"theme": normalized})


def run_bot_test(
    wav_path: str,
    llm_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    voice: Optional[str] = None,
    stt_url: Optional[str] = None,
    identity: Optional[str] = None,
    identities_dir: Optional[str] = None,
    prompt_file: Optional[str] = None,
    read_prompt_from_file: Optional[bool] = None,
    system_prompt: Optional[str] = None,
    memory_dir: Optional[str] = None,
) -> str:
    """Run SocialRobot in test mode with a WAV file and return its LLM response.

    Args:
        wav_path: Path to the audio sample to feed into the bot.
        llm_url: Optional override for the LLM endpoint.
        llm_model: Optional override for the LLM model name.
        voice: Optional Kokoro voice identifier.
        stt_url: When provided, reuse an existing CtrlSpeak server instead of starting one.
        identity: Optional identity profile name to load.
        identities_dir: Override the root directory that contains identity definitions.
        prompt_file: Explicit prompt file to use for this run.
        read_prompt_from_file: Force reading (or not reading) the prompt file.
        system_prompt: Inline prompt override when not using a file.
        memory_dir: Explicit memory directory for the identity.
    """
    stt_url = stt_url or _resolve_stt_url()
    if not stt_url:
        logger.error("No CtrlSpeak STT server available; cannot run bot test")
        return "Error: No CtrlSpeak STT server available."

    root = Path(__file__).resolve().parents[1]
    wav_file = Path(wav_path).expanduser()
    if not wav_file.is_file():
        logger.error("Test WAV not found: %s", wav_path)
        return f"Error: Test WAV not found: {wav_path}"
    wav_path = str(wav_file.resolve(strict=True))

    robot_dir = root / "third_party" / "social_robot"
    entry = robot_dir / "main.py"
    if not entry.exists():
        logger.error("SocialRobot entrypoint not found at %s", entry)
        return "Error: SocialRobot entrypoint not found."

    if identities_dir:
        identities_root = Path(identities_dir).expanduser()
        if not identities_root.is_absolute():
            identities_root = robot_dir / identities_root
    else:
        identities_root = robot_dir / "personas"
    try:
        identities_root = identities_root.resolve()
    except FileNotFoundError:
        logger.exception("Failed to resolve SocialRobot personas root at %s", identities_root)

    (
        resolved_llm_url,
        resolved_llm_model,
        identity_options,
        identity_hardware,
    ) = _resolve_llm_settings(
        llm_url=llm_url,
        llm_model=llm_model,
        identity=identity,
        identities_root=identities_root,
    )
    normalized_base_url: Optional[str] = None
    resolved_model_name: Optional[str] = None
    if resolved_llm_model:
        resolved_model_name = resolved_llm_model.strip() or None
    if resolved_llm_url:
        normalized_base_url = _normalize_ollama_base_url(resolved_llm_url)

    if normalized_base_url and resolved_model_name:
        state = _ollama_model_state(normalized_base_url, resolved_model_name)
        if state is True:
            _warm_ollama_model(
                resolved_llm_url,
                resolved_model_name,
                identity_options,
                identity_hardware,
            )
        else:
            logger.debug(
                "Skipping Ollama warm-up for model %s in test run (state=%s)",
                resolved_model_name,
                state,
            )

    resolved_identity = (identity or _DEFAULT_IDENTITY_NAME).strip() or _DEFAULT_IDENTITY_NAME
    if memory_dir:
        try:
            candidate = Path(memory_dir).expanduser()
            candidate.mkdir(parents=True, exist_ok=True)
            memory_root = candidate.resolve()
        except Exception:
            logger.exception("Failed to prepare explicit memory directory %s", memory_dir)
            return "Error: Failed to prepare memory directory."
    else:
        memory_root = get_bot_memory_dir(resolved_identity)

    env = os.environ.copy()
    env["CTRLSPEAK_STT_URL"] = stt_url
    env["BOT_STT"] = "remote"
    if llm_url:
        env["BOT_LLM_URL"] = llm_url
    if llm_model:
        env["BOT_LLM_MODEL"] = llm_model
    if voice:
        env["BOT_VOICE"] = voice
    if identity:
        env["BOT_IDENTITY"] = identity
    if identities_dir:
        env["BOT_IDENTITIES_DIR"] = identities_dir
    env["CTRLSPK_BOT_MEMORY_ROOT"] = str(memory_root)
    env["BOT_MEMORY_DIR"] = str(memory_root)

    cmd = [sys.executable, str(entry), "--stt", "remote", "--stt-url", stt_url]
    if identity:
        cmd.extend(["--identity", identity])
    if identities_dir:
        cmd.extend(["--identities-dir", identities_dir])
    if prompt_file:
        cmd.extend(["--prompt-file", str(prompt_file)])
    if read_prompt_from_file is True:
        cmd.append("--read-prompt-from-file")
    elif read_prompt_from_file is False:
        cmd.append("--no-read-prompt-from-file")
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    cmd.extend(["--memory-dir", str(memory_root)])
    cmd.extend(["--test-wav", wav_path])

    logger.info("Running SocialRobot test: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, cwd=str(robot_dir), env=env, capture_output=True, text=True, check=True)
        # Parse the output to find the LLM response
        for line in result.stdout.splitlines():
            if line.startswith("TEST LLM:"):
                return line[len("TEST LLM:"):].strip()
        return f"Error: LLM response not found in output. Full output:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        logger.exception("SocialRobot test failed with error code %d", e.returncode)
        return f"Error: SocialRobot test failed. Stderr:\n{e.stderr}"
    except Exception:
        logger.exception("Failed to run SocialRobot test")
        return "Error: Failed to run SocialRobot test."
