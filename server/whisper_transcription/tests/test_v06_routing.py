from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any

import httpx
from fastapi.testclient import TestClient

from app.corrections import CorrectionStore
from app.main import create_app
from app.providers import (
    OpenAITranscriptionProvider,
    ProviderContext,
    ProviderFailure,
    ProviderRouter,
)


class FakeBackend:
    name = "large-v3-turbo"
    device = "cuda"
    compute_type = "float16"

    def __init__(self) -> None:
        self.loaded = False
        self.calls: list[dict[str, Any]] = []

    def load(self) -> None:
        self.loaded = True

    def transcribe(self, audio_path: Path, allowed_languages, initial_prompt, word_timestamps):
        self.calls.append(
            {
                "audio": audio_path.read_bytes(),
                "allowed_languages": allowed_languages,
                "initial_prompt": initial_prompt,
                "word_timestamps": word_timestamps,
            }
        )
        language = allowed_languages[0] if allowed_languages else "en"
        return {
            "raw_text": "raw Acme corp",
            "language": language,
            "detected_language": language,
            "detected_languages": [language],
            "segments": [],
        }


class FakeProvider:
    def __init__(self, provider_id: str, result: dict[str, Any] | ProviderFailure) -> None:
        self.id = provider_id
        self.result = result
        self.calls: list[ProviderContext] = []

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": "fake",
            "status": "ready",
            "requires_credential": False,
            "paid": False,
        }

    def transcribe(self, _audio_path: Path, context: ProviderContext) -> dict[str, Any]:
        self.calls.append(context)
        if isinstance(self.result, ProviderFailure):
            raise self.result
        return dict(self.result)


def audio_form(**data: str) -> dict[str, Any]:
    return {
        "files": {"audio": ("sample.wav", b"RIFF-fake", "audio/wav")},
        "data": data,
    }


def test_worker_role_exposes_raw_worker_contract_only(tmp_path: Path) -> None:
    backend = FakeBackend()
    app = create_app(
        store=CorrectionStore(tmp_path / "worker.sqlite3"),
        backend=backend,
        temp_dir=tmp_path / "uploads",
        service_role="worker",
        worker_token="worker-secret",
        environ={},
    )
    with TestClient(app, client=("10.83.233.1", 50000)) as client:
        headers = {"Authorization": "Bearer worker-secret"}
        capabilities = client.get("/v1/capabilities", headers=headers)
        transcription = client.post(
            "/v1/worker/transcribe",
            headers=headers,
            **audio_form(
                allowed_languages="en",
                keywords='["CtrlSpeak", "ACME Corp"]',
            ),
        )
        client_route = client.post("/v1/transcribe", headers=headers, **audio_form())
        corrections = client.get("/v1/corrections", headers=headers)

    assert capabilities.status_code == 200
    assert capabilities.json()["role"] == "worker"
    assert capabilities.json()["accepts_client_transcriptions"] is False
    assert transcription.status_code == 200
    assert transcription.json()["corrected"] is False
    assert transcription.json()["raw_text"] == "raw Acme corp"
    assert "Known spellings: CtrlSpeak, ACME Corp" in backend.calls[0]["initial_prompt"]
    assert client_route.status_code == 404
    assert corrections.status_code == 404


def test_gateway_cascades_and_applies_identity_scoped_correction_once(tmp_path: Path) -> None:
    unavailable = FakeProvider(
        "ubuntu-gpu-large-v3-turbo",
        ProviderFailure(
            "offline",
            category="worker_unavailable",
            retryable=True,
        ),
    )
    cloud = FakeProvider(
        "openai-gpt-transcribe",
        {
            "raw_text": "hello Acme corp",
            "language": "en",
            "detected_language": "en",
            "detected_languages": ["en"],
            "segments": [],
            "usage": {"type": "tokens", "total_tokens": 12},
        },
    )
    tiny = FakeProvider("nova-tiny-whisper", {})
    router = ProviderRouter(
        [unavailable, cloud, tiny],
        {
            "resilient-quality": (
                unavailable.id,
                cloud.id,
                tiny.id,
            )
        },
        default_strategy="resilient-quality",
    )
    store = CorrectionStore(tmp_path / "gateway.sqlite3")
    app = create_app(
        store=store,
        backend=FakeBackend(),
        router=router,
        temp_dir=tmp_path / "uploads",
        service_role="gateway",
        clients_json='{"alice":{"token":"alice-token"},"bob":{"token":"bob-token"}}',
        environ={},
    )
    with TestClient(app, client=("10.83.233.3", 50000)) as client:
        alice_headers = {"Authorization": "Bearer alice-token"}
        bob_headers = {"Authorization": "Bearer bob-token"}
        correction = client.post(
            "/v1/corrections",
            headers=alice_headers,
            json={
                "source_phrase": "Acme corp",
                "replacement_phrase": "ACME Corp",
                "send_as_keyword": True,
                "language_codes": ["en"],
            },
        )
        alice = client.post(
            "/v1/transcribe",
            headers={**alice_headers, "X-CtrlSpeak-OpenAI-Key": "alice-openai-key"},
            **audio_form(allowed_languages="en"),
        )
        bob = client.post(
            "/v1/transcribe",
            headers=bob_headers,
            **audio_form(allowed_languages="en"),
        )

    assert correction.status_code == 201
    assert alice.status_code == 200
    assert alice.json()["text"] == "hello ACME Corp"
    assert alice.json()["provider_used"] == "openai-gpt-transcribe"
    assert alice.json()["degraded"] is True
    assert [item["status"] for item in alice.json()["attempts"]] == ["failed", "succeeded"]
    assert cloud.calls[0].openai_api_key == "alice-openai-key"
    assert "Acme corp" in cloud.calls[0].keywords
    assert bob.status_code == 200
    assert bob.json()["text"] == "hello Acme corp"
    assert cloud.calls[1].openai_api_key is None


def test_terminal_openai_quota_error_does_not_fall_through_to_tiny(tmp_path: Path) -> None:
    quota = FakeProvider(
        "openai-gpt-transcribe",
        ProviderFailure(
            "quota",
            category="openai_quota_exhausted",
            retryable=False,
            status_code=402,
            action="Top up the supplied account.",
        ),
    )
    tiny = FakeProvider(
        "nova-tiny-whisper",
        {"raw_text": "fallback", "language": "en", "segments": []},
    )
    router = ProviderRouter(
        [quota, tiny],
        {"resilient-quality": (quota.id, tiny.id)},
        default_strategy="resilient-quality",
    )
    app = create_app(
        store=CorrectionStore(tmp_path / "gateway.sqlite3"),
        backend=FakeBackend(),
        router=router,
        temp_dir=tmp_path / "uploads",
        service_role="gateway",
        clients_json='{"alice":"alice-token"}',
        environ={},
    )
    with TestClient(app, client=("10.83.233.3", 50000)) as client:
        response = client.post(
            "/v1/transcribe",
            headers={
                "Authorization": "Bearer alice-token",
                "X-CtrlSpeak-OpenAI-Key": "alice-openai-key",
            },
            **audio_form(allowed_languages="en"),
        )

    assert response.status_code == 402
    assert response.json()["detail"]["category"] == "openai_quota_exhausted"
    assert not tiny.calls


def test_openai_adapter_uses_request_key_languages_and_keywords_without_logging_them(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = request.content
        return httpx.Response(
            200,
            json={
                "text": "hello CtrlSpeak",
                "languages": [{"code": "en"}],
                "usage": {"type": "tokens", "total_tokens": 5},
            },
        )

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF-fake")
    provider = OpenAITranscriptionProvider(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    result = provider.transcribe(
        audio,
        ProviderContext(
            allowed_languages=("en", "af"),
            initial_prompt=None,
            keywords=("CtrlSpeak",),
            word_timestamps=False,
            openai_api_key="request-scoped-secret",
        ),
    )

    assert result["raw_text"] == "hello CtrlSpeak"
    assert captured["authorization"] == "Bearer request-scoped-secret"
    assert b'name="languages[]"' in captured["body"]
    assert b'name="keywords[]"' in captured["body"]


def test_v05_database_migrates_rules_globally_and_exact_overrides_to_legacy(
    tmp_path: Path,
) -> None:
    database = tmp_path / "v05.sqlite3"
    with sqlite3.connect(database) as conn:
        conn.executescript(
            """
            CREATE TABLE correction_rules (
              id TEXT PRIMARY KEY, source_phrase TEXT NOT NULL,
              replacement_phrase TEXT NOT NULL, context_terms TEXT NOT NULL DEFAULT '[]',
              tags TEXT NOT NULL DEFAULT '[]', enabled INTEGER NOT NULL DEFAULT 1,
              priority INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL, use_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE transcriptions (
              id TEXT PRIMARY KEY, raw_text TEXT NOT NULL, corrected_text TEXT NOT NULL,
              language TEXT, segments TEXT NOT NULL, applied_rule_ids TEXT NOT NULL,
              created_at TEXT NOT NULL, context TEXT, metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE transcription_feedback (
              transcription_id TEXT NOT NULL, rule_id TEXT NOT NULL, note TEXT,
              created_at TEXT NOT NULL, PRIMARY KEY (transcription_id, rule_id),
              FOREIGN KEY(transcription_id) REFERENCES transcriptions(id) ON DELETE CASCADE,
              FOREIGN KEY(rule_id) REFERENCES correction_rules(id) ON DELETE CASCADE
            );
            CREATE TABLE transcript_edit_feedback (
              id TEXT PRIMARY KEY, transcription_id TEXT NOT NULL,
              confirmed_text TEXT NOT NULL, capture_method TEXT NOT NULL,
              client_metadata TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
              FOREIGN KEY(transcription_id) REFERENCES transcriptions(id) ON DELETE CASCADE
            );
            CREATE TABLE exact_transcript_overrides (
              id TEXT PRIMARY KEY, raw_text TEXT NOT NULL UNIQUE,
              corrected_text TEXT NOT NULL, source_feedback_id TEXT NOT NULL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              use_count INTEGER NOT NULL DEFAULT 0,
              FOREIGN KEY(source_feedback_id) REFERENCES transcript_edit_feedback(id) ON DELETE RESTRICT
            );
            INSERT INTO correction_rules VALUES
              ('rule-1','Acme corp','ACME Corp','[]','[]',1,0,'now','now',0);
            INSERT INTO transcriptions VALUES
              ('tx-1','raw old','returned old','en','[]','[]','now',NULL,'{}');
            INSERT INTO transcript_edit_feedback VALUES
              ('feedback-1','tx-1','confirmed old','enter','{}','now');
            INSERT INTO exact_transcript_overrides VALUES
              ('override-1','raw old','confirmed old','feedback-1','now','now',0);
            """
        )

    store = CorrectionStore(database)

    migrated_rule = store.get_rule("rule-1")
    assert migrated_rule is not None
    assert migrated_rule["scope"] == "global"
    assert migrated_rule["owner_id"] is None
    assert migrated_rule["language_codes"] == []
    assert migrated_rule["send_as_keyword"] is False
    assert store.get_transcription("tx-1")["principal_id"] == "legacy"  # type: ignore[index]
    assert store.apply_with_metadata("raw old", principal_id="legacy")[0] == "confirmed old"
    assert store.apply_with_metadata("raw old", principal_id="alice")[0] == "raw old"
