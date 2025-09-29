"""Entrypoint for the robot face and dialogue loop."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

from audio.stt import FasterWhisperSTT
from audio.remote_stt import RemoteSTT
from audio.tts import KokoroTTS
from audio.vad import VADListener, VADConfig
from face_animation.face import FaceAnimator, FaceSettings
from face_animation.logo import LogoAnimator
from llm.ollama import OllamaClient, OllamaUnavailableError
from PySide6.QtWidgets import QApplication

IDENTITIES_ROOT = Path(__file__).resolve().parent / "identities"
DEFAULT_IDENTITY_NAME = "default"
DEFAULT_SYSTEM_PROMPT = "You are a cheerful robotic companion speaking concisely."

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def _detect_whisper_device() -> str:
    """Detects the best available device for ctranslate2 (CUDA or CPU)."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
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

def _resolve_identities_root(arg_value: Optional[str]) -> Path:
    if arg_value:
        return Path(arg_value).expanduser().resolve()
    return IDENTITIES_ROOT

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

    memory_dir = args.memory_dir or config.get("memory_dir")
    memory_path: Optional[Path] = None
    if memory_dir:
        memory_path = Path(memory_dir)
        if not memory_path.is_absolute():
            memory_path = (identity_path / memory_dir).resolve()
        try:
            memory_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"-> Failed to ensure memory directory {memory_path}: {exc}")
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
    p.add_argument("--test-wav", help="Path to a WAV file to process for testing (bypasses VAD/mic).")
    return p.parse_args()

def load_history(memory_path: Optional[Path]) -> List[dict]:
    if memory_path:
        history_file = memory_path / "conversation.json"
        if history_file.exists():
            try:
                return json.loads(history_file.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"-> Failed to load conversation history: {exc}")
    return []

def save_history(memory_path: Optional[Path], history: List[dict]) -> None:
    if memory_path:
        try:
            history_file = memory_path / "conversation.json"
            history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"-> Failed to save conversation history: {exc}")

def main():
    args = parse_args()
    profile, config = resolve_identity(args)

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

    tts_model = KokoroTTS(voice=profile.voice, speed=1.0)

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

    animator: object = None
    app: Optional[QApplication] = None
    animator_thread: Optional[threading.Thread] = None
    animation_style = config.get("animation_style")

    if animation_style == "logo":
        app = QApplication(sys.argv)
        logo_image = config.get("logo_image")
        logo_path = profile.base_path / logo_image if logo_image else None
        if not logo_path or not logo_path.exists():
            raise RuntimeError(f"Logo image not found: {logo_path}")
        animator = LogoAnimator(logo_path=logo_path)
        animator.setup_widget()
    else:
        face_settings = FaceSettings(window_size=(1920, 1080), rotation_degrees=0)
        face_image_rotation = config.get("face_image_rotation")
        if isinstance(face_image_rotation, (int, float)):
            face_settings.face_image_rotation = float(face_image_rotation)
        mouth_anchor = config.get("mouth_anchor")
        if isinstance(mouth_anchor, list) and len(mouth_anchor) == 2:
            face_settings.mouth_anchor = (float(mouth_anchor[0]), float(mouth_anchor[1]))
        face_image = config.get("face_image")
        if face_image:
            face_image_path = profile.base_path / face_image
            if face_image_path.exists():
                face_settings.face_image_path = str(face_image_path)
        mouth_image = config.get("mouth_image")
        if mouth_image:
            mouth_image_path = profile.base_path / mouth_image
            if mouth_image_path.exists():
                face_settings.mouth_image_path = str(mouth_image_path)
        animator = FaceAnimator(settings=face_settings)
        animator_thread = threading.Thread(target=animator.run, daemon=True)
        animator_thread.start()

    vad_config = VADConfig(sample_rate=16000, frame_duration_ms=30, padding_duration_ms=360, aggressiveness=2, deactivation_ratio=0.9)
    vad_listener: Optional[VADListener] = None
    last_bot_response: str = ""
    processing_lock = threading.Lock()

    def _handle_user_request(
        transcript: str,
        *,
        force_screenshot: bool = False,
        source: str = "voice",
    ) -> None:
        nonlocal last_bot_response, vad_listener

        cleaned = transcript.strip()
        if not cleaned:
            animator.update_amplitude(0.0)
            return

        if source == "command":
            print("-> Simulating user request:", transcript)
        else:
            print("-> User said:", transcript)

        if tts_model.is_playing:
            tts_model.stop_playback()

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

        screenshot_b64: Optional[str] = None
        screenshot_path: Optional[Path] = None
        augmented_text = transcript

        should_capture = force_screenshot or ("look at my screen" in normalized_user)
        if should_capture:
            screenshot_b64, screenshot_path = _capture_screenshot()
            if screenshot_b64:
                prompt_suffix = (
                    "Please describe the attached screenshot and let me know anything important you notice."
                )
                if augmented_text.strip():
                    augmented_text = f"{augmented_text.strip()}\n\n{prompt_suffix}"
                else:
                    augmented_text = prompt_suffix
            else:
                print("-> Proceeding without screenshot due to capture failure.")

        history = load_history(profile.memory_path)
        try:
            user_content: Optional[List[dict]] = None
            if screenshot_b64:
                user_content = [
                    {"type": "text", "text": augmented_text},
                    {"type": "image", "image": screenshot_b64},
                ]
            llm_response = ollama_client.query(
                augmented_text,
                history=history,
                content=user_content,
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
        if user_content is not None:
            history_entry = {"role": "user", "content": user_content}
            if screenshot_path:
                history_entry["metadata"] = {"screenshot_file": str(screenshot_path)}
        else:
            history_entry = {"role": "user", "content": transcript}

        history.append(history_entry)
        history.append({"role": "assistant", "content": llm_response})
        save_history(profile.memory_path, history)

        print("-> Bot replied:", llm_response)

        try:
            audio_data = tts_model.synthesize(llm_response)
        except Exception as exc:
            print("TTS error:", exc)
            animator.update_amplitude(0.0)
            return

        last_bot_response = llm_response

        def amplitude_callback(level: float) -> None:
            animator.update_amplitude(level)

        def play_tts_in_thread() -> None:
            tts_model.play_audio_with_amplitude(audio_data, amplitude_callback)
            if vad_listener is not None:
                vad_listener.set_aggressiveness(1)  # Restore VAD aggressiveness after bot playback
            animator.update_amplitude(0.0)

        tts_thread = threading.Thread(target=play_tts_in_thread, daemon=True)
        tts_thread.start()

    def _capture_screenshot() -> tuple[Optional[str], Optional[Path]]:
        try:
            import pyautogui  # Local import to avoid heavy dependency on startup
        except Exception as exc:
            print(f"-> Screenshot capture unavailable: {exc}")
            return None, None

        try:
            image = pyautogui.screenshot()
        except Exception as exc:
            print(f"-> Failed to capture screenshot: {exc}")
            return None, None

        buffer = io.BytesIO()
        try:
            image.save(buffer, format="PNG")
        except Exception as exc:
            print(f"-> Failed to encode screenshot: {exc}")
            return None, None

        screenshot_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(screenshot_bytes).decode("ascii")

        saved_path: Optional[Path] = None
        if profile.memory_path:
            try:
                screenshot_dir = profile.memory_path / "screenshots"
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                saved_path = screenshot_dir / f"screenshot_{timestamp}.png"
                saved_path.write_bytes(screenshot_bytes)
            except Exception as exc:
                print(f"-> Failed to persist screenshot to disk: {exc}")
                saved_path = None

        print("-> Captured screenshot for analysis.")
        return image_b64, saved_path

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
            print("STT error:", exc)
            recognized_text = ""

        with processing_lock:
            _handle_user_request(recognized_text, force_screenshot=False, source="voice")

    def _trigger_look_at_screen() -> None:
        with processing_lock:
            _handle_user_request(
                "look at my screen",
                force_screenshot=True,
                source="command",
            )

    if isinstance(animator, LogoAnimator):
        animator.set_look_at_screen_callback(_trigger_look_at_screen)

    def _stdin_command_listener() -> None:
        if sys.stdin is None or sys.stdin.closed:
            return
        while True:
            try:
                line = sys.stdin.readline()
            except Exception as exc:
                print(f"-> Control listener error: {exc}")
                break
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except Exception as exc:
                print(f"-> Ignoring malformed control payload: {exc}")
                continue
            command = str(payload.get("command") or "").strip().lower()
            if not command:
                continue
            if command == "look_at_my_screen":
                _trigger_look_at_screen()
            else:
                print(f"-> Unknown control command: {command}")

    command_thread = threading.Thread(target=_stdin_command_listener, daemon=True)
    command_thread.start()

    vad_listener = VADListener(
        config=vad_config,
        device_index=None,
        on_speech_callback=on_speech_detected,
    )
    print("-> Starting the VAD listener...")
    vad_thread = threading.Thread(target=vad_listener.start, daemon=True)
    vad_thread.start()

    try:
        if app:
            app.exec()
        elif animator_thread:
            animator_thread.join()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        vad_listener.stop()
        animator.stop()
        ollama_client.unload()

if __name__ == "__main__":
    main()
