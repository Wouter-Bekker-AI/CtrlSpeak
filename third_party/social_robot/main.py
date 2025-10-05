"""Entrypoint for the robot face and dialogue loop."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import atexit
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Pattern


from audio.stt import FasterWhisperSTT
from audio.remote_stt import RemoteSTT
from audio.tts import KokoroTTS
from audio.vad import VADListener, VADConfig
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
    module="ctranslate2",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

ICON_PATH = _PROJECT_ROOT / "assets" / "icon.ico"

from face_animation.logo import LogoAnimator
from third_party.social_robot.llm.ollama import (
    OllamaClient,
    OllamaUnavailableError,
)
from PySide6.QtWidgets import QApplication

if __package__ in (None, ""):
    from ui.chat_window import ChatWindow
else:
    from .ui.chat_window import ChatWindow

from background_agents.datetime_memory_agent import refresh_datetime_memory
from background_agents.document_memory_agent import refresh_document_memory
from background_agents.manage_think import ManageThinkAgent, load_manage_think_agent
from background_agents.transcript_cleanup_agent import normalize_transcript
from tools import keywords, vision
from tools.message_management import force_plaintext, requires_force_plaintext, strip_emoji, strip_emoji
from utils.config_paths import get_data_dir, get_logger
from utils.image_store import (
    IdentityImageRecord,
    is_image_request,
    load_identity_image,
    write_identity_image_from_base64,
)
from utils.io_atomic import AtomicWriteError, atomic_append_lines, atomic_write_text
from utils.memory_lock import IdentityLock, IdentityLockError, probe_lock_path
from utils.memory_orchestrator import MemoryOrchestrator
from utils.memory_settings import load_identity_settings
from utils.memory_paths import get_bot_memory_dir, get_bot_profile_export_path


logger = get_logger(__name__)

PERSONAS_ROOT = Path(__file__).resolve().parent / "personas"
DEFAULT_IDENTITY_NAME = "reception"
DEFAULT_SYSTEM_PROMPT = "You are a cheerful robotic companion speaking concisely."

CONVERSATION_MAX_BYTES = 10 * 1024 * 1024
CONVERSATION_KEEP = 5

_identity_lock_handle: Optional[IdentityLock] = None
_memory_orchestrator: Optional[MemoryOrchestrator] = None


def _release_identity_lock_handle() -> None:
    global _identity_lock_handle
    if _identity_lock_handle is not None:
        _identity_lock_handle.release()
        _identity_lock_handle = None


def _shutdown_orchestrator() -> None:
    global _memory_orchestrator
    if _memory_orchestrator is None:
        return
    try:
        _memory_orchestrator.close()
    except Exception:
        logger.exception("Failed to close memory orchestrator")
    finally:
        _memory_orchestrator = None


def _ensure_identity_lock(identity_name: str) -> None:
    global _identity_lock_handle
    lock_path_hint = os.getenv("CTRLSPK_IDENTITY_LOCK_PATH")
    parent_locked = os.getenv("CTRLSPK_PARENT_LOCKED") == "1"

    if parent_locked and lock_path_hint:
        try:
            if probe_lock_path(Path(lock_path_hint)):
                return
        except Exception:
            return
    if parent_locked:
        return

    try:
        lock = IdentityLock(identity_name)
        lock.acquire(timeout=0.0)
    except IdentityLockError:
        print("-> Identity in use. Close the running session before starting another.")
        sys.exit(3)

    _identity_lock_handle = lock
    atexit.register(_release_identity_lock_handle)

_VISION_PROMPT_SUFFIX = {
    "screen": "Please describe the attached screenshot and let me know anything important you notice.",
    "clipboard": "Please describe the clipboard image I attached and summarize anything relevant you notice.",
}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    logger.exception("Failed to reconfigure standard streams for UTF-8 output")

def _detect_whisper_device() -> str:
    """Detects the best available device for ctranslate2 (CUDA or CPU)."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API\..*",
                category=UserWarning,
                module="ctranslate2",
            )
            import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        logger.exception("Failed to detect CUDA availability for whisper device selection")
    return "cpu"

class IdentityProfile:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        voice: str,
        llm_model: str,
        llm_url: str,
        memory_path: Optional[Path],
        base_path: Path,
        ollama_options: Optional[Dict[str, object]] = None,
        ollama_hardware: Optional[str] = None,
        vision_enabled: bool = False,
        tool_enabled: bool = False,
        require_text_cleaning: bool = True,
        hide_think: bool = False,
        tts_provider: Optional[str] = None,
        tts_device_id: Optional[int] = None,
        tts_providers: Optional[List[object]] = None,
        tts_provider_options: Optional[Dict[str, object]] = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.voice = voice
        self.llm_model = llm_model
        self.llm_url = llm_url
        self.memory_path = memory_path
        self.base_path = base_path
        self.ollama_options = dict(ollama_options) if ollama_options else {}
        self.ollama_hardware = ollama_hardware
        self.vision_enabled = vision_enabled
        self.tool_enabled = tool_enabled
        self.require_text_cleaning = require_text_cleaning
        self.hide_think = hide_think
        self.tts_provider = tts_provider
        self.tts_device_id = tts_device_id
        if tts_providers:
            normalized: List[object] = []
            for entry in tts_providers:
                if isinstance(entry, dict):
                    item = dict(entry)
                    options = item.get("options")
                    if isinstance(options, dict):
                        item["options"] = dict(options)
                    normalized.append(item)
                else:
                    normalized.append(entry)
            self.tts_providers = normalized
        else:
            self.tts_providers = None
        self.tts_provider_options = dict(tts_provider_options) if tts_provider_options else None

def _resolve_identities_root(arg_value: Optional[str]) -> Path:
    if arg_value:
        return Path(arg_value).expanduser().resolve()
    return PERSONAS_ROOT


def _list_identity_names(root: Path) -> list[str]:
    try:
        return sorted(p.name for p in root.iterdir() if p.is_dir())
    except Exception as exc:
        print(f"-> Failed to enumerate personas under {root}: {exc}")
        return []

def _load_identity_config(root: Path, name: str) -> tuple[dict, Path]:
    identity_path = (root / name).expanduser().resolve()
    config: dict = {}
    if identity_path.is_dir():
        config_path = identity_path / "identity.json"
        if config_path.exists():
            try:
                config.update(json.loads(config_path.read_text(encoding="utf-8")))
            except Exception as exc:
                print(f"-> Failed to parse identity config {config_path}: {exc}")
    else:
        print(f"-> Identity '{name}' not found at {identity_path}; using defaults.")
    return config, identity_path

def _read_prompt(identity_path: Path, prompt_file: str) -> Optional[str]:
    prompt_path = Path(prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = identity_path / prompt_file
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"-> Failed to read system prompt from {prompt_path}: {exc}")
        return None


def _identity_display(name: str) -> str:
    return name.replace("_", " ").strip().title() or name


def _coerce_int(value, *, context: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            print(f"-> Ignoring {context}; expected an integer but received '{value}'.")
            return None
    print(f"-> Ignoring {context}; expected an integer but received {type(value)!r}.")
    return None


def _prepare_memory_dir(identity_name: str, configured: Optional[str]) -> Path:
    default_dir = get_bot_memory_dir(identity_name)
    if not configured:
        return default_dir
    normalized = str(configured).strip()
    if not normalized:
        return default_dir
    replacements = {
        "{appdata}": str(get_data_dir()),
        "{data_root}": str(get_data_dir()),
        "{identity}": identity_name,
    }
    expanded = normalized
    for token, replacement in replacements.items():
        expanded = expanded.replace(token, replacement)
    expanded = os.path.expandvars(expanded)
    candidate = Path(expanded).expanduser()
    try:
        resolved = candidate.resolve()
    except Exception as exc:
        print(f"-> Failed to resolve memory directory {candidate}: {exc}")
        return default_dir
    data_root = get_data_dir().resolve()
    try:
        resolved.relative_to(data_root)
    except ValueError:
        print(f"-> Memory directory {resolved} must reside under {data_root}; using default.")
        return default_dir
    for sub in ("conversation", "screenshots", "chroma", "traces"):
        (resolved / sub).mkdir(parents=True, exist_ok=True)
    return resolved


def _parse_json_list(raw: Optional[str], *, context: str) -> Optional[List[object]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"-> Failed to parse {context}: {exc}")
        return None
    if isinstance(parsed, list):
        return parsed
    print(f"-> Ignoring {context}; expected a JSON list but received {type(parsed)!r}.")
    return None


def _parse_json_dict(raw: Optional[str], *, context: str) -> Optional[Dict[str, object]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"-> Failed to parse {context}: {exc}")
        return None
    if isinstance(parsed, dict):
        return parsed
    print(f"-> Ignoring {context}; expected a JSON object but received {type(parsed)!r}.")
    return None


def resolve_identity(args) -> tuple[IdentityProfile, dict]:
    root = _resolve_identities_root(args.identities_dir)
    identity_name = args.identity or DEFAULT_IDENTITY_NAME
    config, identity_path = _load_identity_config(root, identity_name)

    voice = args.voice or config.get("voice") or os.getenv("BOT_VOICE") or "hm_omega"
    llm_model = args.llm_model or config.get("llm_model") or os.getenv("BOT_LLM_MODEL") or "gemma3:1b"
    llm_url = args.llm_url or config.get("llm_url") or os.getenv("BOT_LLM_URL") or "http://localhost:11434/api/chat"

    prompt_file = args.prompt_file or config.get("prompt_file")
    read_prompt = args.read_prompt_from_file
    if read_prompt is None:
        read_prompt = config.get("read_prompt_from_file")
    if read_prompt is None:
        read_prompt = bool(prompt_file)

    system_prompt = args.system_prompt or config.get("system_prompt")
    if read_prompt:
        if prompt_file is None:
            print("-> read_prompt_from_file requested but no prompt file provided; falling back to defaults.")
        else:
            prompt_text = _read_prompt(identity_path, prompt_file)
            if prompt_text:
                system_prompt = prompt_text

    if not system_prompt:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    memory_setting = (
        args.memory_dir
        or os.getenv("CTRLSPK_BOT_MEMORY_ROOT")
        or config.get("memory_dir")
    )
    memory_path: Optional[Path] = None
    try:
        memory_path = _prepare_memory_dir(identity_name, memory_setting)
    except Exception as exc:
        print(f"-> Failed to prepare memory directory: {exc}")
        memory_path = None

    raw_options = config.get("ollama_options")
    ollama_options: Dict[str, object] = {}
    if isinstance(raw_options, dict):
        ollama_options = dict(raw_options)
    elif raw_options is not None:
        print("-> Ignoring ollama_options because it is not a JSON object.")

    hardware_pref = config.get("ollama_hardware")
    normalized_hardware: Optional[str] = None
    if isinstance(hardware_pref, str):
        candidate = hardware_pref.strip().lower()
        if candidate in {"cpu_only", "cpu_and_gpu", "gpu_only"}:
            normalized_hardware = candidate
        elif candidate:
            print(
                "-> Ignoring ollama_hardware value '%s'; expected cpu_only, cpu_and_gpu, or gpu_only." % hardware_pref
            )
    elif hardware_pref is not None:
        print("-> Ignoring ollama_hardware because it is not a string value.")

    vision_flag = config.get("vision")
    if isinstance(vision_flag, bool):
        vision_enabled = vision_flag
    elif vision_flag is not None:
        print("-> Ignoring vision value; expected a boolean true/false.")
        vision_enabled = False
    else:
        vision_enabled = False

    tool_flag = config.get("tool")
    if isinstance(tool_flag, bool):
        tool_enabled = tool_flag
    elif tool_flag is not None:
        print("-> Ignoring tool value; expected a boolean true/false.")
        tool_enabled = False
    else:
        tool_enabled = False

    cleaning_flag = config.get("require_text_cleaning")
    if isinstance(cleaning_flag, bool):
        require_text_cleaning = cleaning_flag
    elif cleaning_flag is not None:
        print("-> Ignoring require_text_cleaning value; expected a boolean true/false.")
        require_text_cleaning = True
    else:
        require_text_cleaning = True

    hide_think_flag = config.get("hide_think")
    if isinstance(hide_think_flag, bool):
        hide_think = hide_think_flag
    elif hide_think_flag is not None:
        print("-> Ignoring hide_think value; expected a boolean true/false.")
        hide_think = False
    else:
        hide_think = False

    tts_config = config.get("tts", {}) if isinstance(config.get("tts"), dict) else {}
    if config.get("tts") is not None and not isinstance(config.get("tts"), dict):
        print("-> Ignoring tts value; expected an object with provider settings.")

    tts_provider = args.tts_provider
    if not tts_provider:
        tts_provider = os.getenv("BOT_TTS_PROVIDER")
    if not tts_provider:
        raw_provider = tts_config.get("onnx_provider") or tts_config.get("provider")
        if isinstance(raw_provider, str):
            tts_provider = raw_provider
        elif raw_provider is not None:
            print("-> Ignoring tts.onnx_provider; expected a string value.")

    tts_device_id = args.tts_device_id
    if tts_device_id is None:
        env_device = _coerce_int(os.getenv("BOT_TTS_DEVICE_ID"), context="BOT_TTS_DEVICE_ID")
        if env_device is not None:
            tts_device_id = env_device
    if tts_device_id is None:
        tts_device_id = _coerce_int(tts_config.get("device_id"), context="tts.device_id")

    tts_provider_options = _parse_json_dict(
        os.getenv("BOT_TTS_PROVIDER_OPTIONS"), context="BOT_TTS_PROVIDER_OPTIONS"
    )
    if tts_provider_options is None:
        raw_options = tts_config.get("provider_options")
        if isinstance(raw_options, dict):
            tts_provider_options = raw_options
        elif raw_options is not None:
            print("-> Ignoring tts.provider_options; expected an object with option keys.")

    tts_providers = _parse_json_list(args.tts_providers, context="--tts-providers")
    if tts_providers is None:
        tts_providers = _parse_json_list(os.getenv("BOT_TTS_PROVIDERS"), context="BOT_TTS_PROVIDERS")
    if tts_providers is None:
        raw_providers = tts_config.get("providers")
        if isinstance(raw_providers, list):
            tts_providers = raw_providers
        elif raw_providers is not None:
            print("-> Ignoring tts.providers; expected a list of provider definitions.")

    profile = IdentityProfile(
        name=identity_name,
        system_prompt=system_prompt,
        voice=voice,
        llm_model=llm_model,
        llm_url=llm_url,
        memory_path=memory_path,
        base_path=identity_path,
        ollama_options=ollama_options,
        ollama_hardware=normalized_hardware,
        vision_enabled=vision_enabled,
        tool_enabled=tool_enabled,
        require_text_cleaning=require_text_cleaning,
        hide_think=hide_think,
        tts_provider=tts_provider,
        tts_device_id=tts_device_id,
        tts_providers=tts_providers,
        tts_provider_options=tts_provider_options,
    )

    print(f"-> Loaded identity '{profile.name}' (voice={profile.voice}, model={profile.llm_model})")
    if profile.memory_path:
        print(f"-> Identity memory directory: {profile.memory_path}")

    return profile, config

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stt", choices=['local', 'remote'], default=os.getenv("BOT_STT", "local"))
    p.add_argument("--stt-url", default=os.getenv("CTRLSPEAK_STT_URL", "http://127.0.0.1:65432/transcribe"))
    p.add_argument("--llm-url", default=os.getenv("BOT_LLM_URL"))
    p.add_argument("--llm-model", default=os.getenv("BOT_LLM_MODEL"))
    p.add_argument("--voice", default=os.getenv("BOT_VOICE"))
    p.add_argument("--identity", default=os.getenv("BOT_IDENTITY", DEFAULT_IDENTITY_NAME))
    p.add_argument("--identities-dir", default=os.getenv("BOT_IDENTITIES_DIR"))
    p.add_argument("--prompt-file", default=os.getenv("BOT_PROMPT_FILE"))
    prompt_toggle = p.add_mutually_exclusive_group()
    prompt_toggle.add_argument("--read-prompt-from-file", dest="read_prompt_from_file", action="store_true")
    prompt_toggle.add_argument("--no-read-prompt-from-file", dest="read_prompt_from_file", action="store_false")
    p.set_defaults(read_prompt_from_file=None)
    p.add_argument("--system-prompt", default=os.getenv("BOT_SYSTEM_PROMPT"))
    p.add_argument("--memory-dir", default=os.getenv("BOT_MEMORY_DIR"))
    p.add_argument("--tts-provider", help="Override the ONNX Runtime provider for Kokoro TTS.")
    p.add_argument("--tts-device-id", type=int, help="Override the GPU device id for the TTS provider when supported.")
    p.add_argument(
        "--tts-providers",
        help="JSON list of provider entries to pass directly to onnxruntime (advanced).",
    )
    p.add_argument("--test-wav", help="Path to a WAV file to process for testing (bypasses VAD/mic).")
    return p.parse_args()

def load_history(memory_path: Optional[Path]) -> List[dict]:
    if not memory_path:
        return []

    history_file = memory_path / "conversation" / "conversation.jsonl"
    entries: List[dict] = []
    try:
        with history_file.open("r", encoding="utf-8") as src:
            for raw_line in src:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception as exc:
                    print(f"-> Ignoring malformed history line: {exc}")
    except FileNotFoundError:
        return []
    except Exception as exc:
        print(f"-> Failed to load conversation history: {exc}")
    return entries


def save_history(memory_path: Optional[Path], entries: List[dict]) -> None:
    if not memory_path or not entries:
        return

    history_file = memory_path / "conversation" / "conversation.jsonl"
    try:
        atomic_append_lines(
            history_file,
            [json.dumps(entry, ensure_ascii=False) for entry in entries],
            max_bytes=CONVERSATION_MAX_BYTES,
            keep=CONVERSATION_KEEP,
        )
    except AtomicWriteError as exc:
        print(f"-> Failed to save conversation history: {exc}")


def main():
    args = parse_args()
    profile, config = resolve_identity(args)

    _ensure_identity_lock(profile.name)

    identities_root = _resolve_identities_root(args.identities_dir)
    available_identities = _list_identity_names(identities_root)
    if profile.name not in available_identities:
        available_identities.append(profile.name)
    keywords.configure_identity_keywords(available_identities)
    identity_lookup = {name.lower(): name for name in available_identities}

    # STT backend selection
    if args.stt == "remote":
        stt_model = RemoteSTT(args.stt_url)
    else:
        stt_device = _detect_whisper_device()
        stt_model = FasterWhisperSTT(model_size_or_path="tiny.en", device=stt_device, compute_type="int8")

    ollama_client = OllamaClient(
        url=profile.llm_url,
        model=profile.llm_model,
        stream=False,
        system_prompt=profile.system_prompt,
        options=profile.ollama_options,
        hardware_mode=profile.ollama_hardware,
    )

    tts_model = KokoroTTS(
        voice=profile.voice,
        speed=1.0,
        onnx_provider=profile.tts_provider,
        onnx_device_id=profile.tts_device_id,
        onnx_providers=profile.tts_providers,
        onnx_provider_options=profile.tts_provider_options,
    )
    print("-> TTS preprocessing agent is disabled; using deterministic scrub only when needed.")

    identity_settings = load_identity_settings(profile.name)

    use_orchestrator = os.getenv("CTRLSPK_USE_LANGGRAPH_MEMORY_ORCHESTRATOR") == "1"
    forced_tool_orchestrator = False
    if profile.tool_enabled and not use_orchestrator:
        use_orchestrator = True
        forced_tool_orchestrator = True
    metrics_env = os.getenv("CTRLSPK_METRICS_PATH")
    if metrics_env:
        metrics_path = Path(metrics_env).expanduser()
    elif profile.memory_path:
        metrics_path = profile.memory_path / "traces" / "metrics.csv"
    else:
        metrics_path = Path(tempfile.gettempdir()) / "ctrlspeak_metrics.csv"
    think_agent: Optional[ManageThinkAgent] = None
    if profile.hide_think:
        think_agent = load_manage_think_agent()
        print("-> Hide-think mode enabled; suppressing <think> plans from chat history.")

    orchestrator: Optional[MemoryOrchestrator] = None
    global _memory_orchestrator
    if use_orchestrator and profile.memory_path is None:
        print("-> LangGraph orchestrator requires an AppData memory directory; disabling.")
        use_orchestrator = False
    if use_orchestrator and profile.memory_path is not None:
        try:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            orchestrator = MemoryOrchestrator(
                profile.name,
                ollama_client,
                memory_dir=profile.memory_path,
                metrics_path=metrics_path,
                identity_settings=identity_settings,
                think_manager=think_agent if profile.hide_think else None,
                tooling_enabled=profile.tool_enabled,
            )
            _memory_orchestrator = orchestrator
            atexit.register(_shutdown_orchestrator)
            if forced_tool_orchestrator:
                print("-> LangGraph memory orchestrator enabled for tooling support.")
            else:
                print("-> LangGraph memory orchestrator enabled.")
        except Exception as exc:
            print(f"-> Failed to initialize LangGraph orchestrator: {exc}")
            logger.exception("Failed to initialize LangGraph orchestrator")
            orchestrator = None
            use_orchestrator = False

    if args.test_wav:
        import wave
        with open(args.test_wav, "rb") as f:
            wav_file = wave.open(f, "rb")
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2 or wav_file.getframerate() != 16000:
                print("Error: Test WAV must be 16-bit PCM, mono, 16kHz.")
                return
            raw_bytes = wav_file.readframes(wav_file.getnframes())
            wav_file.close()

        try:
            recognized_text = stt_model.run_stt(raw_bytes, sample_rate=16000)
            print(f"TEST STT: {recognized_text}")
            if not recognized_text.strip():
                print("TEST LLM: (no transcription)")
                return

            llm_response = ""
            try:
                for chunk in ollama_client.query(recognized_text, stream=True):
                    llm_response += chunk
                print(f"TEST LLM: {llm_response.strip()}")
            except OllamaUnavailableError as exc:
                print(f"TEST LLM ERROR: {exc}")
        except Exception as e:
            print(f"TEST ERROR: {e}")
        finally:
            ollama_client.unload()
        return

    app = QApplication.instance() or QApplication(sys.argv)
    identity_display = _identity_display(profile.name)

    chat_window = ChatWindow(identity_display, icon_path=ICON_PATH)
    chat_window.set_voice_mode(True)
    chat_window.show()

    animation_style = (config.get("animation_style") or "logo").lower()
    if animation_style not in ("logo", ""):
        raise RuntimeError(
            f"Unsupported animation_style '{animation_style}'. Only 'logo' is supported."
        )

    logo_image = config.get("logo_image")
    logo_path: Optional[Path] = None
    if logo_image:
        candidate = profile.base_path / logo_image
        if candidate.exists():
            logo_path = candidate
        else:
            fallback = _PROJECT_ROOT / "assets" / Path(logo_image).name
            if fallback.exists():
                logo_path = fallback
    if logo_path is None:
        raise RuntimeError(f"Logo image not found: {logo_image}")

    animator = LogoAnimator(logo_path=logo_path)
    animator.setup_widget()
    animator.hide_widget()

    vad_config = VADConfig(sample_rate=16000, frame_duration_ms=30, padding_duration_ms=360, aggressiveness=2, deactivation_ratio=0.9)
    vad_listener: Optional[VADListener] = None
    vad_thread: Optional[threading.Thread] = None
    vad_suppressed = False
    voice_mode_active = threading.Event()
    session_history: List[dict] = []
    last_bot_response: str = ""
    processing_lock = threading.Lock()
    shutdown_requested = threading.Event()
    shutdown_complete = threading.Event()
    failsafe_timer: Optional[threading.Timer] = None

    def _ensure_animator_running() -> None:
        animator.show_widget()

    def _suspend_animator() -> None:
        animator.hide_widget()

    def _export_profile_snapshot() -> None:
        orchestrator = _memory_orchestrator
        if orchestrator is None:
            chat_window.append_status_message("Memory", "Profile export unavailable; memory offline.")
            return
        try:
            slots = orchestrator.vector_store.read_all_profile(orchestrator.profile_user_id)
        except Exception:
            logger.exception("Failed to read profile slots for export")
            chat_window.append_status_message("Memory", "Failed to read profile slots.")
            return
        payload = {
            "identity": profile.name,
            "user_id": orchestrator.profile_user_id,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "slots": [
                {
                    "attribute": (entry.get("metadata") or {}).get("attribute"),
                    "value": (entry.get("metadata") or {}).get("value"),
                    "status": (entry.get("metadata") or {}).get("status", "current"),
                }
                for entry in slots
                if isinstance(entry, dict)
            ],
        }
        export_path = get_bot_profile_export_path(profile.name)
        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(export_path, json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception:
            logger.exception("Failed to write profile snapshot")
            chat_window.append_status_message("Memory", "Failed to export profile snapshot.")
            return
        chat_window.append_status_message("Memory", f"Profile exported to {export_path}")

    chat_window.export_profile_requested.connect(_export_profile_snapshot)

    def on_speech_detected(raw_bytes: bytes) -> None:
        nonlocal vad_listener
        if vad_listener is None:
            return

        try:
            recognized_text = stt_model.run_stt(
                raw_bytes,
                sample_rate=vad_listener.sample_rate,
            )
        except Exception as exc:
            # print("STT error:", exc)
            recognized_text = ""

        with processing_lock:
            _handle_user_request(
                recognized_text,
                source="voice",
                input_medium="voice",
            )

    def _enable_voice_mode() -> None:
        nonlocal vad_listener, vad_thread, vad_suppressed
        if voice_mode_active.is_set():
            return
        voice_mode_active.set()
        chat_window.set_voice_mode(True)
        _ensure_animator_running()
        print("-> Voice mode enabled; starting the VAD listener...")
        vad_listener = VADListener(
            config=vad_config,
            device_index=None,
            on_speech_callback=on_speech_detected,
        )
        vad_thread = threading.Thread(target=vad_listener.start, daemon=True)
        vad_thread.start()
        vad_suppressed = False

    def _disable_voice_mode() -> None:
        nonlocal vad_listener, vad_thread, vad_suppressed
        if not voice_mode_active.is_set():
            return
        voice_mode_active.clear()
        print("-> Voice mode disabled; returning to text chat.")

        def _ui_teardown() -> None:
            chat_window.set_voice_mode(False)
            _suspend_animator()

        chat_window.invoke(_ui_teardown)
        if vad_listener is not None:
            vad_listener.stop()
        if vad_thread is not None:
            vad_thread.join(timeout=2.0)
            vad_thread = None
        vad_listener = None
        vad_suppressed = False
        if tts_model.is_playing:
            tts_model.stop_playback()
        animator.update_amplitude(0.0)

    def _pause_vad_listener() -> None:
        nonlocal vad_listener, vad_suppressed
        if vad_listener is None or not voice_mode_active.is_set():
            return
        try:
            vad_listener.disable_vad()
            vad_suppressed = True
            logger.debug("VAD listener paused via control command")
        except Exception:
            logger.exception("Failed to pause VAD listener on control command")

    def _resume_vad_listener() -> None:
        nonlocal vad_listener, vad_suppressed
        if vad_listener is None:
            vad_suppressed = False
            return
        if not voice_mode_active.is_set():
            vad_suppressed = False
            return
        try:
            vad_listener.enable_vad()
            vad_suppressed = False
            logger.debug("VAD listener resumed after control command")
        except Exception:
            vad_suppressed = False
            logger.exception("Failed to resume VAD listener after control command")

    def _on_text_submitted(message: str) -> None:
        cleaned = message.strip()
        if not cleaned:
            return

        def _worker() -> None:
            with processing_lock:
                _handle_user_request(
                    cleaned,
                    source="voice" if voice_mode_active.is_set() else "text",
                    input_medium="text",
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_voice_mode_requested(enabled: bool) -> None:
        if enabled:
            _enable_voice_mode()
        else:
            _disable_voice_mode()

    chat_window.send_text.connect(_on_text_submitted)
    chat_window.voice_mode_requested.connect(_on_voice_mode_requested)
    chat_window.closed.connect(lambda: shutdown_requested.set())

    _enable_voice_mode()

    def _resolve_identity_name(candidate: str) -> Optional[str]:
        if not candidate:
            return None
        normalized = candidate.strip().lower()
        if not normalized:
            return None
        if normalized == profile.name.lower():
            return profile.name
        return identity_lookup.get(normalized)

    def _restart_with_identity(target_identity: str) -> None:
        print(f"-> Voice command switching to identity '{target_identity}'.")
        shutdown_requested.set()
        try:
            if vad_listener is not None:
                vad_listener.stop()
        except Exception:
            logger.exception("Failed to stop VAD listener during identity restart")
        try:
            if tts_model.is_playing:
                tts_model.stop_playback()
        except Exception:
            logger.exception("Failed to stop TTS playback during identity restart")
        try:
            if animator is not None:
                animator.stop()
        except Exception:
            logger.exception("Failed to stop animator during identity restart")

        argv = sys.argv[1:]
        new_args: list[str] = []
        skip_next = False
        replaced = False
        for index, value in enumerate(argv):
            if skip_next:
                skip_next = False
                continue
            if value == "--identity":
                replaced = True
                new_args.extend(["--identity", target_identity])
                skip_next = True
            else:
                new_args.append(value)
        if not replaced:
            new_args.extend(["--identity", target_identity])

        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())] + new_args)

    def _request_shutdown() -> None:
        nonlocal failsafe_timer

        first_request = not shutdown_requested.is_set()
        shutdown_requested.set()
        if first_request:
            logger.info("Shutdown requested for identity '%s'", profile.name)
        else:
            logger.debug("Shutdown already requested for identity '%s'", profile.name)

        print(f"-> Ending conversation with '{profile.name}'.")

        try:
            _disable_voice_mode()
        except Exception:
            logger.exception("Failed to disable voice mode during shutdown request")

        logger.debug("Shutdown stage: stopping VAD listener")
        try:
            if vad_listener is not None:
                vad_listener.stop()
                logger.debug("Shutdown stage complete: VAD listener stopped")
        except Exception:
            logger.exception("Failed to stop VAD listener during shutdown")

        logger.debug("Shutdown stage: stopping active TTS playback")
        try:
            if tts_model.is_playing:
                tts_model.stop_playback()
                logger.debug("Shutdown stage complete: TTS playback halted")
        except Exception:
            logger.exception("Failed to stop TTS playback during shutdown")

        logger.debug("Shutdown stage: stopping animator")
        try:
            if animator is not None:
                animator.stop()
                logger.debug("Shutdown stage complete: animator stop requested")
        except Exception:
            logger.exception("Failed to stop animator during shutdown")

        logger.debug("Shutdown stage: closing stdin control pipe")
        try:
            if sys.stdin is not None and not sys.stdin.closed:
                sys.stdin.close()
                logger.debug("Shutdown stage complete: stdin closed")
        except Exception:
            logger.exception("Failed to close stdin during shutdown")

        def _force_exit() -> None:
            if not shutdown_requested.is_set() or shutdown_complete.is_set():
                logger.debug("Goodbye failsafe aborted; shutdown already resolved")
                return
            logger.error("Goodbye failsafe triggered; forcing process exit for '%s'", profile.name)
            print("-> Goodbye failsafe: forcing process exit.")
            try:
                sys.stdout.flush()
            except Exception:
                logger.exception("Failed to flush stdout before forced exit")
            os._exit(0)

        if failsafe_timer is not None:
            failsafe_timer.cancel()

        if app is None:
            timer = threading.Timer(3.0, _force_exit)
            timer.daemon = True
            failsafe_timer = timer
            timer.start()
            logger.debug("Goodbye failsafe armed for identity '%s'", profile.name)

    def _handle_conversation_start(identity_name: str) -> None:
        target = _resolve_identity_name(identity_name)
        if not target:
            print(f"-> Ignoring conversation start request for unknown identity '{identity_name}'.")
            return
        if target == profile.name:
            print(f"-> Already chatting with '{target}'.")
            return

        display_target = _identity_display(target)
        display_current = _identity_display(profile.name)
        print(f"-> Switching from {display_current} to {display_target} on user request.")
        _restart_with_identity(target)

    def _handle_conversation_end(identity_name: str) -> None:
        logger.debug(
            "Conversation end requested for '%s' (active '%s')",
            identity_name,
            profile.name,
        )
        normalized = identity_name.strip().lower()
        if normalized != profile.name.lower():
            print(
                f"-> Ignoring goodbye for '{identity_name}' because the active identity is '{profile.name}'."
            )
            logger.debug(
                "Ignored goodbye for '%s' because '%s' remains active",
                identity_name,
                profile.name,
            )
            return

        display_name = _identity_display(profile.name)
        logger.info("Conversation end keyword accepted for '%s'", profile.name)
        print(f"-> Ending conversation with {display_name} on user request.")
        _request_shutdown()

    def _handle_user_request(
        transcript: str,
        *,
        capture_override: Optional[str] = None,
        source: str = "voice",
        input_medium: Optional[str] = None,
    ) -> None:
        nonlocal last_bot_response, vad_listener, session_history

        if shutdown_requested.is_set():
            logger.debug(
                "Discarding user input because shutdown is in progress for '%s'",
                profile.name,
            )
            return

        cleaned = transcript.strip()
        if not cleaned:
            animator.update_amplitude(0.0)
            return

        logger.debug(
            "Processing user input from %s channel: %s",
            source,
            cleaned,
        )
        if source == "command":
            print("-> Simulating user request:", transcript)
        else:
            print("-> User said:", transcript)

        normalized = cleaned
        try:
            cleanup_result = normalize_transcript(cleaned)
            normalized = cleanup_result.text.strip() or cleaned
            if cleanup_result.corrections:
                logger.debug(
                    "Transcript cleanup applied for SocialRobot input: %s",
                    cleanup_result.corrections,
                )
                if source != "command":
                    print("-> Normalized transcript:", normalized)
        except Exception:
            logger.exception("Failed to normalize SocialRobot transcript")
            normalized = cleaned

        cleaned = normalized

        if tts_model.is_playing:
            tts_model.stop_playback()

        display_name = _identity_display(profile.name)
        maintenance_match = keywords.detect_memory_refresh_keyword(cleaned)
        if maintenance_match:
            payload = maintenance_match.keyword.payload.lower()
            if payload == "datetime":
                print(
                    "-> Update date/time keyword detected; refreshing temporal memory "
                    f"for {display_name}..."
                )
                try:
                    success = refresh_datetime_memory(profile.name, force=True, reason="keyword")
                except Exception:
                    logger.exception(
                        "Unexpected error refreshing date/time memory for identity %s", profile.name
                    )
                    success = False
                if success:
                    print("-> Date/time refresh complete.")
                else:
                    print("-> Date/time refresh failed; check logs for details.")
            else:
                print(
                    "-> Update documentation keyword detected; refreshing documentation memory "
                    f"for {display_name}..."
                )
                try:
                    success = refresh_document_memory(profile.name, force=True, reason="keyword")
                except Exception:
                    logger.exception(
                        "Unexpected error refreshing documentation memory for identity %s", profile.name
                    )
                    success = False
                if success:
                    print("-> Documentation refresh complete.")
                else:
                    print("-> Documentation refresh failed; check logs for details.")
            animator.update_amplitude(0.0)
            return

        medium = input_medium or ("voice" if source == "voice" else None)
        chat_window.append_user_message(cleaned, medium=medium)

        normalized_user = cleaned.lower()
        normalized_bot = last_bot_response.strip().lower()

        if (
            normalized_user
            and normalized_bot
            and (
                normalized_user == normalized_bot
                or normalized_user in normalized_bot
                or normalized_bot in normalized_user
            )
        ):
            print("-> Ignoring self-echo from recent response.")
            animator.update_amplitude(0.0)
            return

        if not shutdown_requested.is_set():
            start_match = keywords.detect_conversation_start_keyword(cleaned)
            if start_match:
                logger.debug(
                    "Detected conversation start keyword for '%s' via %s input",
                    start_match.keyword.payload,
                    source,
                )
                _handle_conversation_start(start_match.keyword.payload)
                animator.update_amplitude(0.0)
                return

            if source == "command":
                end_match = keywords.detect_conversation_end_keyword(cleaned)
                if end_match:
                    logger.debug(
                        "Detected conversation end keyword targeting '%s' via %s input",
                        end_match.keyword.payload,
                        source,
                    )
                    _handle_conversation_end(end_match.keyword.payload)
                    animator.update_amplitude(0.0)
                    return
            elif source != "text":
                if not voice_mode_active.is_set():
                    logger.debug(
                        "Ignoring conversation end keyword via %s input because voice mode is inactive",
                        source,
                    )
                else:
                    end_match = keywords.detect_conversation_end_keyword(cleaned)
                    if end_match:
                        logger.debug(
                            "Detected conversation end keyword targeting '%s' via %s input",
                            end_match.keyword.payload,
                            source,
                        )
                        _handle_conversation_end(end_match.keyword.payload)
                        animator.update_amplitude(0.0)
                        return

        nonlocal use_orchestrator

        capture_result: Optional[vision.VisionCapture] = None
        identity_image: Optional[IdentityImageRecord] = None
        augmented_text = cleaned
        identity_name = profile.name.strip().lower()
        enforce_thinking = identity_name == "einstein"
        capture_request = capture_override
        capture_pattern: Optional[Pattern[str]] = None
        placeholder_displayed = False
        placeholder_text: Optional[str] = None
        if profile.hide_think:
            if think_agent is not None:
                candidate_placeholder = think_agent.placeholder_text
                if isinstance(candidate_placeholder, str) and candidate_placeholder.strip():
                    placeholder_text = candidate_placeholder
            else:
                placeholder_text = "Thinking..."

        if placeholder_text and not placeholder_displayed:
            chat_window.append_bot_message(placeholder_text)
            placeholder_displayed = True

        if profile.vision_enabled:
            if capture_request is None:
                keyword_match = keywords.detect_vision_keyword(cleaned)
                if keyword_match:
                    capture_request = keyword_match.keyword.payload
                    capture_pattern = keyword_match.keyword.pattern
            else:
                keyword_config = keywords.get_vision_keyword(capture_request)
                if keyword_config:
                    capture_pattern = keyword_config.pattern
        elif capture_request is not None:
            print("-> Vision capture is disabled for this identity.")
            capture_request = None

        if capture_request:
            if capture_request == "screen":
                capture_result = vision.capture_screenshot()
            elif capture_request == "clipboard":
                capture_result = vision.capture_clipboard_image()
            else:
                print(f"-> Unknown vision capture request: {capture_request}")
                capture_result = None
                capture_request = None

            if capture_result and capture_result.success:
                prompt_suffix = _VISION_PROMPT_SUFFIX.get(capture_result.source, "")
                trimmed_prompt = cleaned
                if capture_pattern:
                    trimmed_prompt = capture_pattern.sub(" ", cleaned)
                has_additional_prompt = bool(trimmed_prompt.strip())
                if not has_additional_prompt and prompt_suffix:
                    augmented_text = prompt_suffix
                else:
                    augmented_text = cleaned
                identity_image = write_identity_image_from_base64(
                    profile.name,
                    capture_result.image_b64,
                    source=capture_result.source,
                )
                if identity_image is None:
                    identity_image = load_identity_image(profile.name)
                print(f"-> Captured {capture_result.source} image for analysis.")
            elif capture_result:
                reason = capture_result.error or "capture failure"
                print(f"-> Proceeding without vision capture due to: {reason}")

        if identity_image is None and is_image_request(augmented_text):
            identity_image = load_identity_image(profile.name)

        vision_metadata: Optional[dict[str, str]] = None
        if identity_image is not None:
            vision_metadata = identity_image.as_metadata(vision_request=capture_request)
        elif capture_request:
            vision_metadata = {"vision_request": capture_request}

        if enforce_thinking:
            lowered_augmented = augmented_text.casefold()
            if "/no_think" not in lowered_augmented:
                if "/think" not in lowered_augmented:
                    stripped = augmented_text.rstrip()
                    augmented_text = (f"{stripped} /think" if stripped else "/think").strip()

        fallback_content: Optional[List[dict]] = None
        if identity_image is not None and (
            capture_request or is_image_request(augmented_text)
        ):
            fallback_content = [
                {"type": "text", "text": augmented_text},
                {"type": "image", "image": identity_image.image_b64},
            ]

        llm_response = ""
        raw_llm_response = ""
        used_orchestrator = False
        think_hidden = False
        think_placeholder: Optional[str] = None
        history = list(session_history)
        history_baseline = len(history)
        turn_result = None
        if use_orchestrator and orchestrator is not None:
            try:
                turn_result = orchestrator.run_turn(
                    transcript,
                    augmented_text=augmented_text,
                    vision_metadata=vision_metadata,
                )
                llm_response = turn_result.response_text
                raw_llm_response = turn_result.raw_response_text
                think_hidden = turn_result.think_hidden
                think_placeholder = turn_result.think_placeholder
                used_orchestrator = True
            except Exception as exc:
                print("-> LangGraph orchestrator failure; falling back to legacy conversation.")
                logger.exception("LangGraph orchestrator failure", exc_info=exc)
                use_orchestrator = False
                _shutdown_orchestrator()

        if used_orchestrator and turn_result is not None:
            session_history.extend(turn_result.history_entries)

        if not used_orchestrator:
            try:
                llm_response = ollama_client.query(
                    augmented_text,
                    history=history,
                    content=fallback_content,
                )
            except OllamaUnavailableError as exc:
                failure_message = str(exc)
                print(f"-> Ollama error: {failure_message}")
                animator.update_amplitude(0.0)
                return
            if not llm_response.strip():
                animator.update_amplitude(0.0)
                return

            history_entry: dict
            if identity_image is not None and (
                capture_request or is_image_request(augmented_text)
            ):
                entry_content = [
                    {"type": "text", "text": transcript},
                    {"type": "image_file", "path": str(identity_image.path)},
                ]
                history_entry = {"role": "user", "content": entry_content}
                if vision_metadata:
                    sanitized_metadata = dict(vision_metadata)
                    sanitized_metadata.pop("vision_request", None)
                    history_entry["metadata"] = sanitized_metadata
            else:
                history_entry = {"role": "user", "content": transcript}

            history.append(history_entry)

        if not raw_llm_response:
            raw_llm_response = llm_response

        manage_result = None
        if think_agent is not None and profile.hide_think and not used_orchestrator:
            try:
                manage_result = think_agent.filter_response(raw_llm_response)
            except Exception:
                logger.exception("Manage-think agent failed to filter response")
                manage_result = None
            else:
                llm_response = manage_result.visible_text
                think_hidden = manage_result.removed
                think_placeholder = manage_result.placeholder_text if manage_result.removed else None

        raw_response = raw_llm_response or llm_response
        print("-> Raw LLM reply:", raw_response)

        if think_hidden and think_placeholder and not placeholder_displayed:
            chat_window.append_status_message("Bot (thinking)", think_placeholder)
            placeholder_displayed = True

        display_response = llm_response

        sanitized_input = strip_emoji(llm_response)



        if requires_force_plaintext(sanitized_input):

            print("-> Handing reply to force_plaintext().")

            sanitized_response = force_plaintext(sanitized_input)

            print("-> Scrubbed reply:", sanitized_response)

            if sanitized_response != sanitized_input:

                print("-> Applied deterministic TTS scrub.")

        else:

            print("-> Reply does not require deterministic scrub.")

            sanitized_response = sanitized_input



        sanitized_response = sanitized_response.strip()

        if not sanitized_response and display_response:

            sanitized_response = "..."

        if sanitized_response != display_response:

            print("-> Reply shown in chat window:", display_response)

        print("-> Final reply for chat history and TTS:", sanitized_response)



        if not used_orchestrator:
            history.append({"role": "assistant", "content": sanitized_response})
            new_entries = history[history_baseline:]
            if new_entries:
                session_history.extend(new_entries)
                save_history(profile.memory_path, new_entries)

        chat_window.append_bot_message(display_response)
        last_bot_response = display_response

        if not voice_mode_active.is_set():
            animator.update_amplitude(0.0)
            return

        try:
            if not sanitized_response:
                animator.update_amplitude(0.0)
                return
            audio_data = tts_model.synthesize(sanitized_response)
        except Exception as exc:
            print("TTS error:", exc)
            animator.update_amplitude(0.0)
            return

        def amplitude_callback(level: float) -> None:
            animator.update_amplitude(level)

        def play_tts_in_thread() -> None:
            tts_model.play_audio_with_amplitude(audio_data, amplitude_callback)
            if vad_listener is not None:
                vad_listener.set_aggressiveness(1)  # Restore VAD aggressiveness after bot playback
            animator.update_amplitude(0.0)

        tts_thread = threading.Thread(target=play_tts_in_thread, daemon=True)
        tts_thread.start()

    def _trigger_look_at_screen() -> None:
        if not profile.vision_enabled:
            print("-> Look at my Screen is disabled for this identity.")
            return
        with processing_lock:
            _handle_user_request(
                "look at my screen",
                capture_override="screen",
                source="command",
            )

    def _trigger_look_at_clipboard() -> None:
        if not profile.vision_enabled:
            print("-> Look at my Clipboard is disabled for this identity.")
            return
        with processing_lock:
            _handle_user_request(
                "look at my clipboard",
                capture_override="clipboard",
                source="command",
            )

    if isinstance(animator, LogoAnimator):
        if profile.vision_enabled:
            animator.set_look_at_screen_callback(_trigger_look_at_screen)
            animator.set_look_at_clipboard_callback(_trigger_look_at_clipboard)
        else:
            animator.set_look_at_screen_callback(None)
            animator.set_look_at_clipboard_callback(None)

    def _stdin_command_listener() -> None:
        if sys.stdin is None or sys.stdin.closed:
            logger.debug("stdin control channel unavailable; skipping listener for '%s'", profile.name)
            return
        logger.debug("Starting stdin control listener for '%s'", profile.name)
        while not shutdown_requested.is_set():
            try:
                line = sys.stdin.readline()
            except Exception as exc:
                print(f"-> Control listener error: {exc}")
                logger.exception("Control listener error while reading stdin")
                break
            if not line:
                logger.debug("stdin control listener received EOF")
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except Exception as exc:
                print(f"-> Ignoring malformed control payload: {exc}")
                logger.debug("Malformed control payload ignored: %s", stripped)
                continue
            command = str(payload.get("command") or "").strip().lower()
            if not command:
                continue
            if command == "look_at_my_screen":
                logger.debug("stdin control command received: look_at_my_screen")
                _trigger_look_at_screen()
            elif command == "look_at_my_clipboard":
                logger.debug("stdin control command received: look_at_my_clipboard")
                _trigger_look_at_clipboard()
            elif command == "pause_vad":
                logger.debug("stdin control command received: pause_vad")
                _pause_vad_listener()
            elif command == "resume_vad":
                logger.debug("stdin control command received: resume_vad")
                _resume_vad_listener()
            elif command == "chat_with":
                target_identity = str(payload.get("identity") or "")
                if target_identity:
                    logger.debug(
                        "stdin control command received: chat_with %s",
                        target_identity,
                    )
                    with processing_lock:
                        _handle_conversation_start(target_identity)
            elif command == "goodbye":
                target_identity = str(payload.get("identity") or "")
                if target_identity:
                    logger.debug(
                        "stdin control command received: goodbye %s",
                        target_identity,
                    )
                    with processing_lock:
                        _handle_conversation_end(target_identity)
            else:
                print(f"-> Unknown control command: {command}")
                logger.debug("Unknown control command received: %s", command)

        logger.debug("stdin control listener exiting for '%s'", profile.name)

    command_thread = threading.Thread(target=_stdin_command_listener, daemon=True)
    command_thread.start()

    try:
        if app:
            app.exec()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        logger.info("Shutdown cleanup starting for '%s'", profile.name)
        shutdown_complete.set()
        try:
            _disable_voice_mode()
        except Exception:
            logger.exception("Failed to disable voice mode during cleanup")
        try:
            if vad_listener is not None:
                logger.debug("Cleanup stage: stopping VAD listener")
                vad_listener.stop()
                logger.debug("Cleanup stage complete: VAD listener stopped")
        except Exception:
            logger.exception("Failed to stop VAD listener during cleanup")
        try:
            if animator is not None:
                logger.debug("Cleanup stage: stopping animator")
                animator.stop()
                logger.debug("Cleanup stage complete: animator stopped")
        except Exception:
            logger.exception("Failed to stop animator during cleanup")
        try:
            logger.debug("Cleanup stage: unloading Ollama client")
            ollama_client.unload()
            logger.debug("Cleanup stage complete: Ollama client unloaded")
        except Exception:
            logger.exception("Failed to unload Ollama client during cleanup")

        if failsafe_timer is not None:
            failsafe_timer.cancel()
            failsafe_timer = None
            logger.debug("Goodbye failsafe timer cancelled after cleanup")
        logger.info("Shutdown cleanup finished for '%s'", profile.name)

if __name__ == "__main__":
    main()


