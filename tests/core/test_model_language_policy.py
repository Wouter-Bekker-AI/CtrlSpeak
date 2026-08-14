from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from utils import models, system, transcription_backend
from utils.transcription_backend import BackendConfig, DEFAULT_API_URL


pytestmark = pytest.mark.core_headless


class FakeWhisperModel:
    def __init__(self, *, reported_language: str | None = None) -> None:
        self.reported_language = reported_language
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, path: str, **kwargs: Any):
        self.calls.append({"path": path, **kwargs})
        language = self.reported_language or kwargs.get("language") or "en"
        segment = SimpleNamespace(text="hello", start=0.0, end=1.0)
        return [segment], SimpleNamespace(language=language)


def configure_test_transcription(
    monkeypatch,
    model: FakeWhisperModel,
    allowed: tuple[str, ...],
) -> None:
    monkeypatch.setattr(models, "initialize_transcriber", lambda **_kwargs: model)
    monkeypatch.setattr(models, "_split_wav_into_chunks", lambda path: [path])
    monkeypatch.setattr(system, "_apply_last_connected", lambda server: server)
    config = BackendConfig("bundled", DEFAULT_API_URL, None, "disabled", allowed)
    monkeypatch.setattr(transcription_backend, "get_runtime_backend_config", lambda: config)


def test_local_single_language_is_forced_with_transcribe_task(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    model = FakeWhisperModel()
    configure_test_transcription(monkeypatch, model, ("en",))

    result = models.transcribe_local(str(audio), play_feedback=False)

    assert result == "hello"
    assert model.calls[0]["language"] == "en"
    assert model.calls[0]["task"] == "transcribe"


def test_local_multiple_languages_use_allowed_detection_or_ordered_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    model = FakeWhisperModel()
    configure_test_transcription(monkeypatch, model, ("en", "af"))
    monkeypatch.setattr(models, "_detect_spoken_language", lambda *_args: "af")

    assert models.transcribe_local(str(audio), play_feedback=False) == "hello"
    assert model.calls[0]["language"] == "af"

    fallback_model = FakeWhisperModel()
    configure_test_transcription(monkeypatch, fallback_model, ("en", "af"))
    monkeypatch.setattr(models, "_detect_spoken_language", lambda *_args: "zh")

    assert models.transcribe_local(str(audio), play_feedback=False) == "hello"
    assert fallback_model.calls[0]["language"] == "en"


def test_local_transcription_refuses_reported_language_outside_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    model = FakeWhisperModel(reported_language="zh")
    configure_test_transcription(monkeypatch, model, ("en",))
    errors: list[str] = []
    monkeypatch.setattr(models, "notify_error", lambda _title, detail: errors.append(detail))

    assert models.transcribe_local(str(audio), play_feedback=False) is None
    assert errors and "outside the configured output-language allowlist" in errors[0]
