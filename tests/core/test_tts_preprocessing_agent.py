from __future__ import annotations

import json
from pathlib import Path
import sys
import types

try:
    import requests  # type: ignore  # Ensure the real package is loaded before optional stubbing
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    requests = None  # type: ignore[assignment]

import pytest

SOCIAL_ROBOT_DIR = Path(__file__).resolve().parents[2] / "third_party" / "social_robot"
if str(SOCIAL_ROBOT_DIR) not in sys.path:
    sys.path.insert(0, str(SOCIAL_ROBOT_DIR))

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.__path__ = []  # type: ignore[attr-defined]

    class HTTPError(Exception):
        def __init__(self, *args, response=None, request=None):  # pragma: no cover - simple stub
            super().__init__(*args)
            self.response = response
            self.request = request

    class Response:  # pragma: no cover - simple stub
        def __init__(self, status_code: int = 200):
            self.status_code = status_code

    requests_stub.HTTPError = HTTPError
    requests_stub.Response = Response
    exceptions_module = types.ModuleType("requests.exceptions")
    exceptions_module.HTTPError = HTTPError
    exceptions_module.RequestException = HTTPError
    requests_stub.exceptions = exceptions_module  # type: ignore[attr-defined]

    def _unused_post(*args, **kwargs):  # pragma: no cover - simple stub
        raise RuntimeError("requests stub invoked")

    requests_stub.post = _unused_post
    sys.modules["requests"] = requests_stub
    sys.modules["requests.exceptions"] = exceptions_module

from background_agents.tts_preprocessing_agent.background_agent import (
    TTSPreprocessingAgent,
    load_background_agent_resources,
    load_tts_preprocessing_agent,
    text_requires_cleaning,
)
from third_party.social_robot.llm.ollama import OllamaUnavailableError


class DummyClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict] = []

    def query(self, payload: str, stream: bool = False):  # pragma: no cover - simple stub
        self.calls.append({"payload": payload, "stream": stream})
        return self.response


class FailingClient:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def query(self, payload: str, stream: bool = False):  # pragma: no cover - simple stub
        raise self._exc


@pytest.fixture()
def agent_directory(tmp_path: Path) -> Path:
    agent_dir = tmp_path / "tts_preprocessing_agent"
    agent_dir.mkdir()
    (agent_dir / "identity.json").write_text(
        json.dumps(
            {
                "llm_model": "gemma3:1b",
                "llm_url": "http://localhost:11434/api/chat",
                "ollama_options": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "repeat_penalty": 1.0,
                    "mirostat": 0,
                    "seed": 0,
                    "stop": ["\\n\\n"],
                },
                "prompt_file": "system_prompt.txt",
                "preamble": "system",
            }
        ),
        encoding="utf-8",
    )
    (agent_dir / "system_prompt.txt").write_text("Cleanup prompt", encoding="utf-8")
    return agent_dir


def test_load_resources(agent_directory: Path) -> None:
    resources = load_background_agent_resources(agent_directory)
    assert resources is not None
    assert resources.system_prompt == "Cleanup prompt"
    assert resources.header_text is None


def test_load_resources_with_header(agent_directory: Path) -> None:
    identity_path = agent_directory / "identity.json"
    config = json.loads(identity_path.read_text(encoding="utf-8"))
    config["preamble"] = "both"
    config["header_text_file"] = "header_text.txt"
    identity_path.write_text(json.dumps(config), encoding="utf-8")
    (agent_directory / "header_text.txt").write_text("Cleanup header", encoding="utf-8")

    resources = load_background_agent_resources(agent_directory)
    assert resources is not None
    assert resources.header_text == "Cleanup header"
    assert resources.system_prompt == "Cleanup prompt"


def test_load_resources_invalid_preamble(agent_directory: Path) -> None:
    identity_path = agent_directory / "identity.json"
    config = json.loads(identity_path.read_text(encoding="utf-8"))
    config["preamble"] = "invalid"
    identity_path.write_text(json.dumps(config), encoding="utf-8")

    assert load_background_agent_resources(agent_directory) is None


def test_rewrite_appends_header(agent_directory: Path) -> None:
    identity_path = agent_directory / "identity.json"
    config = json.loads(identity_path.read_text(encoding="utf-8"))
    config["preamble"] = "both"
    config["header_text_file"] = "header_text.txt"
    identity_path.write_text(json.dumps(config), encoding="utf-8")
    (agent_directory / "header_text.txt").write_text("Cleanup header", encoding="utf-8")

    resources = load_background_agent_resources(agent_directory)
    assert resources is not None
    dummy = DummyClient("rewritten")
    agent = TTSPreprocessingAgent(resources, client=dummy)
    output = agent.rewrite("Hello world!")
    assert output == "rewritten"
    assert dummy.calls
    assert dummy.calls[0]["payload"] == "Cleanup header\n\nHello world!"


def test_rewrite_without_header(agent_directory: Path) -> None:
    resources = load_background_agent_resources(agent_directory)
    assert resources is not None
    dummy = DummyClient("rewritten")
    agent = TTSPreprocessingAgent(resources, client=dummy)
    agent.rewrite("Hello world!")
    assert dummy.calls
    assert dummy.calls[0]["payload"] == "Hello world!"


@pytest.mark.parametrize(
    "exception",
    [OllamaUnavailableError("offline"), RuntimeError("boom")],
)
def test_rewrite_falls_back_to_original(agent_directory: Path, exception: Exception) -> None:
    resources = load_background_agent_resources(agent_directory)
    assert resources is not None
    agent = TTSPreprocessingAgent(resources, client=FailingClient(exception))
    assert agent.rewrite("Keep this") == "Keep this"


def test_load_agent_handles_missing(agent_directory: Path) -> None:
    # Remove the system prompt to force a graceful failure
    (agent_directory / "system_prompt.txt").unlink()
    assert load_tts_preprocessing_agent(agent_directory) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Simple sentence for playback.", False),
        ("Hello  world!", False),
        ("This *should* be cleaned", True),
        ("Heading #1", True),
        ("Normal punctuation, nothing fancy.", False),
    ],
)
def test_text_requires_cleaning(text: str, expected: bool) -> None:
    assert text_requires_cleaning(text) is expected
