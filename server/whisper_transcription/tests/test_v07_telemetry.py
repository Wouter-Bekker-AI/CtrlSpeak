from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from app.corrections import CorrectionStore
from app.main import create_app
from app.providers import (
    ProviderContext,
    ProviderFailure,
    ProviderRouter,
    RemoteWorkerProvider,
)


class TimedProvider:
    def __init__(
        self,
        provider_id: str,
        clock: list[float],
        duration_seconds: float,
        result: dict[str, Any] | ProviderFailure,
    ) -> None:
        self.id = provider_id
        self.clock = clock
        self.duration_seconds = duration_seconds
        self.result = result

    def describe(self) -> dict[str, Any]:
        return {"id": self.id, "status": "ready"}

    def transcribe(self, _audio_path: Path, _context: ProviderContext) -> dict[str, Any]:
        self.clock[0] += self.duration_seconds
        if isinstance(self.result, ProviderFailure):
            raise self.result
        return dict(self.result)


class WorkerBackend:
    name = "large-v3-turbo"
    device = "cuda"
    compute_type = "float16"

    def load(self) -> None:
        return None

    def transcribe(self, *_args: Any) -> dict[str, Any]:
        return {
            "raw_text": "safe result",
            "language": "en",
            "detected_language": "en",
            "detected_languages": ["en"],
            "segments": [],
        }


def test_router_records_truthful_per_attempt_and_total_durations(tmp_path: Path) -> None:
    clock = [100.0]
    unavailable = TimedProvider(
        "worker",
        clock,
        0.125,
        ProviderFailure(
            "offline",
            category="worker_unavailable",
            retryable=True,
        ),
    )
    cloud = TimedProvider(
        "cloud",
        clock,
        0.875,
        {
            "raw_text": "safe result",
            "language": "en",
            "segments": [],
        },
    )
    router = ProviderRouter(
        [unavailable, cloud],
        {"preferred": (unavailable.id, cloud.id)},
        default_strategy="preferred",
        monotonic=lambda: clock[0],
    )

    result = router.transcribe(
        tmp_path / "audio.wav",
        ProviderContext((), None, (), False, None),
        strategy=None,
        provider_id=None,
        allowed_provider_ids={unavailable.id, cloud.id},
    )

    assert result.routing_duration_ms == 1000.0
    assert result.attempts == (
        {
            "provider": "worker",
            "status": "failed",
            "category": "worker_unavailable",
            "retryable": True,
            "duration_ms": 125.0,
        },
        {"provider": "cloud", "status": "succeeded", "duration_ms": 875.0},
    )


def test_error_attempts_include_duration_without_sensitive_values(tmp_path: Path) -> None:
    clock = [50.0]
    failed = TimedProvider(
        "cloud",
        clock,
        0.25,
        ProviderFailure(
            "rejected",
            category="openai_invalid_key",
            retryable=False,
            status_code=401,
        ),
    )
    router = ProviderRouter(
        [failed],
        {"only": (failed.id,)},
        default_strategy="only",
        monotonic=lambda: clock[0],
    )

    try:
        router.transcribe(
            tmp_path / "audio.wav",
            ProviderContext((), None, (), False, "never-expose-this-key"),
            strategy=None,
            provider_id=None,
            allowed_provider_ids={failed.id},
        )
    except ProviderFailure as exc:
        assert exc.routing_duration_ms == 250.0  # type: ignore[attr-defined]
        assert exc.attempts[0]["duration_ms"] == 250.0  # type: ignore[attr-defined]
        assert "never-expose-this-key" not in str(exc.attempts)  # type: ignore[attr-defined]
    else:
        raise AssertionError("expected provider failure")


def test_worker_capabilities_report_bounded_probe_health() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(
            200,
            json={"model": "large-v3-turbo", "device": "cuda", "compute_type": "float16"},
        )

    provider = RemoteWorkerProvider(
        "ubuntu-gpu-large-v3-turbo",
        "http://10.83.233.2:8765",
        "worker-secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    description = provider.describe()
    health = description["health"]

    assert calls == ["/health"]
    assert description["status"] == "ready"
    assert health["status"] == "ready"
    assert health["probe_status"] == "ready"
    assert isinstance(health["probe_duration_ms"], float)
    assert health["probe_duration_ms"] >= 0
    assert health["probe_timeout_ms"] == 500.0
    assert health["connect_timeout_ms"] == 350.0
    assert health["cache_remaining_ms"] > 0
    assert health["circuit_retry_after_ms"] == 0.0


def test_gateway_response_and_audit_store_routing_telemetry(tmp_path: Path) -> None:
    clock = [10.0]
    provider = TimedProvider(
        "local",
        clock,
        0.4,
        {
            "raw_text": "safe result",
            "language": "en",
            "detected_language": "en",
            "detected_languages": ["en"],
            "segments": [],
        },
    )
    router = ProviderRouter(
        [provider],
        {"local": (provider.id,)},
        default_strategy="local",
        monotonic=lambda: clock[0],
    )
    store = CorrectionStore(tmp_path / "gateway.sqlite3")
    app = create_app(
        store=store,
        backend=WorkerBackend(),
        router=router,
        temp_dir=tmp_path / "uploads",
        service_role="gateway",
        clients_json='{"alice":"alice-token"}',
        environ={},
    )

    with TestClient(app, client=("10.83.233.3", 50000)) as client:
        response = client.post(
            "/v1/transcribe",
            headers={"Authorization": "Bearer alice-token"},
            files={"audio": ("sample.wav", b"RIFF-fake", "audio/wav")},
        )
        capabilities = client.get(
            "/v1/capabilities",
            headers={"Authorization": "Bearer alice-token"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["routing_duration_ms"] == 400.0
    assert payload["attempts"][0]["duration_ms"] == 400.0
    audit = store.get_transcription(payload["id"])
    assert audit is not None
    assert audit["metadata"]["routing_duration_ms"] == 400.0
    assert audit["metadata"]["attempts"][0]["duration_ms"] == 400.0
    assert capabilities.json()["telemetry"] == {
        "attempt_duration_ms": True,
        "routing_duration_ms": True,
        "worker_inference_duration_ms": True,
        "worker_health": True,
    }


def test_worker_response_includes_measured_inference_duration(tmp_path: Path) -> None:
    app = create_app(
        backend=WorkerBackend(),
        temp_dir=tmp_path / "uploads",
        service_role="worker",
        worker_token="worker-secret",
        environ={},
    )

    with TestClient(app, client=("10.83.233.1", 50000)) as client:
        response = client.post(
            "/v1/worker/transcribe",
            headers={"Authorization": "Bearer worker-secret"},
            files={"audio": ("sample.wav", b"RIFF-fake", "audio/wav")},
        )

    assert response.status_code == 200
    assert isinstance(response.json()["inference_duration_ms"], float)
    assert response.json()["inference_duration_ms"] >= 0
