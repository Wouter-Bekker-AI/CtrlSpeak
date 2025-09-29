# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional

import requests

from utils.config_paths import get_logger
from utils.system import (
    CLIENT_ONLY_BUILD, start_server, settings, settings_lock,
    get_best_server, load_settings, save_settings,
)

from utils.models import DEFAULT_MODEL_NAME

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

_bot_proc: Optional[subprocess.Popen] = None


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


def _load_identity_llm_config(identity: str, identities_root: Path) -> tuple[Optional[str], Optional[str]]:
    config_path = identities_root / identity / "identity.json"
    if not config_path.exists():
        return None, None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to parse identity configuration for '%s'", identity)
        return None, None
    return data.get("llm_url"), data.get("llm_model")


def _resolve_llm_settings(
    *,
    llm_url: Optional[str],
    llm_model: Optional[str],
    identity: Optional[str],
    identities_root: Path,
) -> tuple[Optional[str], Optional[str]]:
    resolved_url = llm_url or os.environ.get("BOT_LLM_URL")
    resolved_model = llm_model or os.environ.get("BOT_LLM_MODEL")
    if identity:
        identity_url, identity_model = _load_identity_llm_config(identity, identities_root)
        if not resolved_url:
            resolved_url = identity_url
        if not resolved_model:
            resolved_model = identity_model
    return resolved_url, resolved_model


def _warm_ollama_model(llm_url: Optional[str], llm_model: Optional[str]) -> None:
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

    try:
        response = requests.post(warm_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Preloaded Ollama model %s", llm_model)
    except Exception:
        logger.warning("Failed to preload Ollama model %s", llm_model, exc_info=True)


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
    global _bot_proc
    if _bot_proc and _bot_proc.poll() is None:
        logger.info("Bot already running")
        return True

    stt_url = _resolve_stt_url()
    if not stt_url:
        logger.error("No CtrlSpeak STT server available; cannot start bot")
        return False

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
        identities_root = robot_dir / "identities"
    try:
        identities_root = identities_root.resolve()
    except FileNotFoundError:
        pass

    resolved_llm_url, resolved_llm_model = _resolve_llm_settings(
        llm_url=llm_url,
        llm_model=llm_model,
        identity=identity,
        identities_root=identities_root,
    )
    _warm_ollama_model(resolved_llm_url, resolved_llm_model)

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
    if memory_dir:
        cmd.extend(["--memory-dir", memory_dir])

    logger.info("Starting SocialRobot: %s", " ".join(cmd))
    try:
        _bot_proc = subprocess.Popen(cmd, cwd=str(robot_dir), env=env)
        time.sleep(0.35)  # give it a moment to open the window
        return True
    except Exception:
        logger.exception("Failed to start SocialRobot")
        _bot_proc = None
        return False


def stop_bot() -> None:
    global _bot_proc
    if _bot_proc is None:
        return
    try:
        if _bot_proc.poll() is None:
            _bot_proc.terminate()
            try:
                _bot_proc.wait(timeout=5)
            except Exception:
                pass
            if _bot_proc.poll() is None:
                _bot_proc.kill()
                try:
                    _bot_proc.wait(timeout=3)
                except Exception:
                    pass
    except Exception:
        logger.exception("Error while stopping SocialRobot")
    finally:
        _bot_proc = None


def is_bot_running() -> bool:
    return _bot_proc is not None and _bot_proc.poll() is None


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
        identities_root = robot_dir / "identities"
    try:
        identities_root = identities_root.resolve()
    except FileNotFoundError:
        pass

    resolved_llm_url, resolved_llm_model = _resolve_llm_settings(
        llm_url=llm_url,
        llm_model=llm_model,
        identity=identity,
        identities_root=identities_root,
    )
    _warm_ollama_model(resolved_llm_url, resolved_llm_model)

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
    if memory_dir:
        cmd.extend(["--memory-dir", memory_dir])
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
