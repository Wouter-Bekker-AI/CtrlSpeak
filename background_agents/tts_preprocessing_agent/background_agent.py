"""Helpers for running the TTS preprocessing background agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from third_party.social_robot.llm.ollama import (
    OllamaClient,
    OllamaUnavailableError,
)

from utils.config_paths import get_logger

_DEFAULT_AGENT_DIR = Path(__file__).resolve().parent


logger = get_logger(__name__)

_TRIGGER_CHARACTERS = frozenset({"*", "#"})


@dataclass(frozen=True)
class BackgroundAgentResources:
    """Static resources required by the preprocessing agent."""

    base_path: Path
    identity_config: dict
    header_text: Optional[str]
    system_prompt: Optional[str]
    preamble_mode: str


class TTSPreprocessingAgent:
    """Wraps an Ollama-backed agent that rewrites text prior to TTS playback."""

    def __init__(
        self,
        resources: BackgroundAgentResources,
        *,
        client: Optional[OllamaClient] = None,
    ) -> None:
        self._resources = resources
        config = resources.identity_config
        llm_model = str(config.get("llm_model") or "").strip()
        llm_url = str(config.get("llm_url") or "").strip()
        if not llm_model or not llm_url:
            raise ValueError("Background agent configuration must define llm_model and llm_url")
        options = config.get("ollama_options")
        if not isinstance(options, dict):
            options = {}
        options.setdefault("temperature", 0.0)
        options.setdefault("top_p", 1.0)
        options.setdefault("repeat_penalty", 1.0)
        options.setdefault("mirostat", 0)
        options.setdefault("seed", 0)
        options.setdefault("stop", ["\n\n"])
        preamble_mode = resources.preamble_mode
        self._use_header = preamble_mode in {"header", "both"}
        self._use_prompt = preamble_mode in {"system", "both"}

        header_text = (resources.header_text or "") if self._use_header else ""
        system_prompt = resources.system_prompt if self._use_prompt else None

        self._header_text = header_text.strip()
        self._client = client or OllamaClient(
            url=llm_url,
            model=llm_model,
            stream=False,
            system_prompt=system_prompt,
            options=options,
        )

    @property
    def header_text(self) -> str:
        return self._header_text

    def rewrite(self, text: str) -> str:
        """Send *text* through the preprocessing agent, falling back on errors."""

        cleaned = text.strip()
        if not cleaned:
            return text

        payload = cleaned
        if self._header_text:
            payload = "\n\n".join(part for part in (self._header_text.rstrip(), cleaned) if part)

        try:
            rewritten = self._client.query(payload, stream=False)
        except OllamaUnavailableError as exc:
            print(f"-> TTS preprocessing agent unavailable: {exc}")
            return text
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"-> TTS preprocessing agent failed: {exc}")
            return text

        rewritten = (rewritten or "").strip()
        if not rewritten:
            return text
        return rewritten


def text_requires_cleaning(text: str) -> bool:
    """Return ``True`` only when *text* includes the characters we strip (``*``/``#``)."""

    if not text:
        return False

    return any(char in _TRIGGER_CHARACTERS for char in text)


def _read_optional_text(path: Path) -> Optional[str]:
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"-> Failed to read background agent text file {path}: {exc}")
        return None
    return contents


def load_background_agent_resources(base_path: Optional[Path] = None) -> Optional[BackgroundAgentResources]:
    """Load the preprocessing agent resources if they are available."""

    agent_path = Path(base_path) if base_path else _DEFAULT_AGENT_DIR
    agent_path = agent_path.expanduser()
    try:
        agent_path = agent_path.resolve()
    except FileNotFoundError:
        logger.exception("Background agent directory %s could not be resolved", agent_path)

    identity_path = agent_path / "identity.json"

    if not identity_path.exists():
        print(f"-> Background agent resources missing at {agent_path}; continuing without preprocessing.")
        return None

    try:
        identity_config = json.loads(identity_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"-> Failed to parse background agent identity at {identity_path}: {exc}")
        return None

    header_path = _resolve_agent_path(
        agent_path, identity_config.get("header_text_file") or "header_text.txt"
    )
    system_prompt_path = _resolve_agent_path(
        agent_path, identity_config.get("prompt_file") or "system_prompt.txt"
    )

    preamble_mode = str(identity_config.get("preamble") or "both").strip().lower()
    valid_modes = {"header", "system", "both"}
    if preamble_mode not in valid_modes:
        print(
            "-> Background agent identity must set 'preamble' to 'header', 'system', or 'both'."
        )
        return None

    header_text: Optional[str] = None
    if preamble_mode in {"header", "both"}:
        header_text = _read_optional_text(header_path)
        if not header_text:
            print(
                f"-> Background agent header text missing at {header_path}; continuing without preprocessing."
            )
            return None

    system_prompt: Optional[str] = None
    if preamble_mode in {"system", "both"}:
        system_prompt = _read_optional_text(system_prompt_path)
        if not system_prompt:
            print(
                f"-> Background agent system prompt missing at {system_prompt_path}; continuing without preprocessing."
            )
            return None

    return BackgroundAgentResources(
        base_path=agent_path,
        identity_config=identity_config,
        header_text=header_text.strip() if header_text else None,
        system_prompt=system_prompt.strip() if system_prompt else None,
        preamble_mode=preamble_mode,
    )


def _resolve_agent_path(base: Path, path_value: Optional[str]) -> Path:
    if path_value:
        path = Path(path_value).expanduser()
    else:
        path = Path()
    if not path.is_absolute():
        path = (base / path).expanduser()
    return path


def load_tts_preprocessing_agent(base_path: Optional[Path] = None) -> Optional[TTSPreprocessingAgent]:
    """Create the preprocessing agent when all required resources are present."""

    resources = load_background_agent_resources(base_path)
    if not resources:
        return None
    try:
        return TTSPreprocessingAgent(resources)
    except Exception as exc:
        print(f"-> Failed to initialize TTS preprocessing agent: {exc}")
        return None


def main() -> None:  # pragma: no cover - manual utility
    import argparse

    parser = argparse.ArgumentParser(description="Run the TTS preprocessing agent on sample text")
    parser.add_argument("text", help="Assistant reply to rewrite for TTS", nargs="+")
    parser.add_argument(
        "--agent-dir",
        help="Override the background agent directory",
        default=None,
    )
    args = parser.parse_args()
    agent = load_tts_preprocessing_agent(args.agent_dir)
    if not agent:
        raise SystemExit("TTS preprocessing agent resources are unavailable")
    payload = " ".join(args.text)
    rewritten = agent.rewrite(payload)
    print(rewritten)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    main()


__all__ = [
    "BackgroundAgentResources",
    "TTSPreprocessingAgent",
    "load_background_agent_resources",
    "load_tts_preprocessing_agent",
]
