from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.corrections import CorrectionStore
from app.main import FasterWhisperBackend, SERVICE_VERSION, create_app


class FakeBackend:
    name = "fake-whisper"
    device = "fake-cuda"
    compute_type = "fake-float16"

    def __init__(self, *, response_language: str | None = None) -> None:
        self.response_language = response_language
        self.calls: list[dict[str, Any]] = []
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def transcribe(
        self,
        audio_path: Path,
        allowed_languages: tuple[str, ...],
        initial_prompt: str | None,
        word_timestamps: bool,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "audio": audio_path.read_bytes(),
                "allowed_languages": allowed_languages,
                "initial_prompt": initial_prompt,
                "word_timestamps": word_timestamps,
            }
        )
        language = self.response_language or (allowed_languages[0] if allowed_languages else "en")
        return {
            "raw_text": "hello Acme corp",
            "language": language,
            "detected_language": language,
            "allowed_languages": list(allowed_languages),
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello Acme corp", "words": []}],
        }


def make_client(tmp_path: Path, backend: FakeBackend, *, token: str | None = None) -> TestClient:
    app = create_app(
        store=CorrectionStore(tmp_path / "corrections.sqlite3"),
        backend=backend,
        temp_dir=tmp_path / "uploads",
        bearer_token=token,
    )
    return TestClient(app, client=("127.0.0.1", 50000))


def audio_form(**data: str) -> dict[str, Any]:
    return {
        "files": {"audio": ("sample.wav", b"RIFF-fake", "audio/wav")},
        "data": data,
    }


def test_health_and_openapi_report_current_language_contract(tmp_path: Path) -> None:
    backend = FakeBackend()
    with make_client(tmp_path, backend) as client:
        response = client.get("/health")
        document = client.get("/openapi.json").json()

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "version": SERVICE_VERSION,
        "role": "standalone",
        "model": "fake-whisper",
        "device": "fake-cuda",
        "compute_type": "fake-float16",
    }
    assert SERVICE_VERSION == "0.7.0"
    rendered = str(document)
    assert "allowed_languages" in rendered
    assert "server-enforced" in rendered


def test_non_loopback_access_requires_service_configuration_and_valid_bearer(
    tmp_path: Path,
) -> None:
    backend = FakeBackend()
    unconfigured_app = create_app(
        store=CorrectionStore(tmp_path / "unconfigured.sqlite3"),
        backend=backend,
        temp_dir=tmp_path / "uploads-unconfigured",
        bearer_token=None,
    )
    with TestClient(unconfigured_app, client=("192.168.1.50", 50000)) as client:
        assert client.get("/health").status_code == 403

    configured_app = create_app(
        store=CorrectionStore(tmp_path / "configured.sqlite3"),
        backend=FakeBackend(),
        temp_dir=tmp_path / "uploads-configured",
        bearer_token="test-secret",
    )
    with TestClient(configured_app, client=("192.168.1.50", 50000)) as client:
        assert client.get("/health").status_code == 401
        assert client.get(
            "/health", headers={"Authorization": "Bearer wrong"}
        ).status_code == 401
        assert client.get(
            "/health", headers={"Authorization": "Bearer test-secret"}
        ).status_code == 200


def test_transcribe_propagates_and_returns_ordered_language_policy(tmp_path: Path) -> None:
    backend = FakeBackend(response_language="af")
    with make_client(tmp_path, backend) as client:
        response = client.post(
            "/v1/transcribe",
            **audio_form(
                allowed_languages="en,af",
                initial_prompt="CtrlSpeak vocabulary",
                word_timestamps="true",
            ),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "af"
    assert body["allowed_languages"] == ["en", "af"]
    assert body["language_policy"] == "restricted"
    assert backend.calls == [
        {
            "audio": b"RIFF-fake",
            "allowed_languages": ("en", "af"),
            "initial_prompt": "CtrlSpeak vocabulary",
            "word_timestamps": True,
        }
    ]


@pytest.mark.parametrize(
    "data",
    [
        {"allowed_languages": "en,xx"},
        {"allowed_languages": "en,af,de,fr,es,it"},
        {"allowed_languages": "en,af", "language": "de"},
    ],
)
def test_invalid_language_policies_are_rejected_before_transcription(
    tmp_path: Path,
    data: dict[str, str],
) -> None:
    backend = FakeBackend()
    with make_client(tmp_path, backend) as client:
        response = client.post("/v1/transcribe", **audio_form(**data))

    assert response.status_code == 422
    assert backend.calls == []


def test_server_blocks_backend_output_outside_requested_allowlist(tmp_path: Path) -> None:
    backend = FakeBackend(response_language="zh")
    with make_client(tmp_path, backend) as client:
        response = client.post(
            "/v1/transcribe",
            **audio_form(allowed_languages="en,af"),
        )

    assert response.status_code == 502
    assert "violated the requested language policy" in response.json()["detail"]


def test_legacy_language_and_automatic_modes_remain_compatible(tmp_path: Path) -> None:
    backend = FakeBackend()
    with make_client(tmp_path, backend) as client:
        legacy = client.post("/v1/transcribe", **audio_form(language="af"))
        automatic = client.post("/v1/transcribe", **audio_form())

    assert legacy.status_code == 200
    assert legacy.json()["allowed_languages"] == ["af"]
    assert automatic.status_code == 200
    assert automatic.json()["allowed_languages"] == []
    assert automatic.json()["language_policy"] == "automatic"


def test_correction_and_confirmed_text_routes_remain_operational(tmp_path: Path) -> None:
    backend = FakeBackend()
    with make_client(tmp_path, backend) as client:
        created = client.post(
            "/v1/corrections",
            json={
                "source_phrase": "Acme corp",
                "replacement_phrase": "ACME Corp",
                "tags": ["company"],
            },
        )
        assert created.status_code == 201
        rule_id = created.json()["id"]

        transcription = client.post(
            "/v1/transcribe",
            **audio_form(allowed_languages="en"),
        )
        assert transcription.status_code == 200
        assert transcription.json()["text"] == "hello ACME Corp"
        transcription_id = transcription.json()["id"]

        feedback = client.post(
            f"/v1/transcriptions/{transcription_id}/feedback",
            json={
                "rule_ids": [rule_id],
                "confirmed_text": "hello ACME Corporation",
                "capture_method": "active_field_on_enter",
                "client_metadata": {"client": "CtrlSpeak", "version": SERVICE_VERSION},
            },
        )

    assert feedback.status_code == 200
    assert feedback.json()["transcription_id"] == transcription_id
    assert feedback.json()["rule_ids"] == [rule_id]
    assert feedback.json()["override_id"]


def test_faster_whisper_backend_forces_selected_language_and_transcribe_task(tmp_path: Path) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def transcribe(self, path: str, **kwargs: Any):
            self.calls.append({"path": path, **kwargs})
            segment = SimpleNamespace(start=0.0, end=1.0, text="hello", words=[])
            return [segment], SimpleNamespace(language=kwargs["language"])

    backend = FasterWhisperBackend(tmp_path / "models")
    backend.model = FakeModel()
    backend._detect_language = lambda _path: "af"  # type: ignore[method-assign]
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")

    result = backend.transcribe(audio, ("en", "af"), None, False)

    assert result["language"] == "af"
    assert backend.model.calls[0]["language"] == "af"
    assert backend.model.calls[0]["task"] == "transcribe"


def test_faster_whisper_backend_loads_explicit_cpu_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeModel:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            calls["model_name"] = model_name
            calls.update(kwargs)

    def get_supported_compute_types(device: str) -> set[str]:
        calls["device_probe"] = device
        return {"int8", "float32"}

    fake_ctranslate2 = SimpleNamespace(get_supported_compute_types=get_supported_compute_types)
    fake_faster_whisper = SimpleNamespace(WhisperModel=FakeModel)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ctranslate2)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_faster_whisper)

    backend = FasterWhisperBackend(
        tmp_path / "models",
        model_name="large-v3-turbo",
        device="cpu",
        compute_type="int8",
        cpu_threads=3,
        num_workers=2,
    )
    backend.load()

    assert calls == {
        "device_probe": "cpu",
        "model_name": "large-v3-turbo",
        "device": "cpu",
        "compute_type": "int8",
        "download_root": str(tmp_path / "models"),
        "cpu_threads": 3,
        "num_workers": 2,
    }
    assert backend.model is not None
