from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils import config_paths
from utils.transcription_backend import (
    ApiBackendError,
    ApiTranscriptionClient,
    BackendConfig,
    BackendPersistenceError,
    DEFAULT_API_URL,
    activate_runtime_backend_config,
    backend_from_display_name,
    backend_display_name,
    get_backend_config,
    get_backend_status,
    get_runtime_backend_config,
    set_session_openai_api_key,
    uses_bundled_runtime,
    save_backend_config,
    transcribe_selected,
)
from utils.local_corrections import LocalCorrectionLibrary


pytestmark = pytest.mark.core_headless


class FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict[str, object]:
        return self._payload


class RecordingSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs):
        files = kwargs.get("files")
        audio_bytes = files["audio"][1].read() if files else None
        self.calls.append({"url": url, "audio_bytes": audio_bytes, **kwargs})
        return self.response

    def get(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response

    def patch(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response

    def delete(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response


def test_backend_display_names_are_explicit_and_round_trip() -> None:
    assert backend_display_name("bundled") == "Embedded / local"
    assert backend_display_name("api") == "Remote API"
    assert backend_from_display_name("Embedded / local") == "bundled"
    assert backend_from_display_name("Remote API") == "api"
    assert get_backend_status(
        BackendConfig("bundled", DEFAULT_API_URL, None, "disabled")
    ) == (
        "Embedded / local · bundled CtrlSpeak model · feedback: disabled · "
        "output languages: Automatic (no restriction)"
    )


def test_backend_config_defaults_to_bundled_and_loopback_and_env_takes_precedence(monkeypatch) -> None:
    assert get_backend_config() == BackendConfig(
        backend="bundled",
        api_url="http://127.0.0.1:8765",
        api_token=None,
        feedback_capture_method="active_field_on_enter",
    )

    with config_paths.settings_lock:
        config_paths.settings.update(
            transcription_backend="api",
            api_url="http://settings.invalid:9000/",
            api_token="settings-secret",
            feedback_capture_method="clipboard_on_enter",
        )
    monkeypatch.setenv("CTRLSPEAK_BACKEND", "bundled")
    monkeypatch.setenv("CTRLSPEAK_API_URL", "https://whisper.example.test/base/")
    monkeypatch.setenv("CTRLSPEAK_API_TOKEN", "environment-secret")

    assert get_backend_config() == BackendConfig(
        backend="bundled",
        api_url="https://whisper.example.test/base",
        api_token="environment-secret",
        feedback_capture_method="active_field_on_enter",
    )
    assert uses_bundled_runtime(BackendConfig("api", DEFAULT_API_URL, None, "disabled")) is False
    assert uses_bundled_runtime(BackendConfig("bundled", DEFAULT_API_URL, None, "disabled")) is True


def test_ordered_output_languages_are_validated_persisted_and_overridden_by_environment(
    monkeypatch,
) -> None:
    saved = save_backend_config(
        backend="api",
        api_url=DEFAULT_API_URL,
        api_token=None,
        feedback_capture_method="disabled",
        allowed_output_languages=["English", "af", "en"],
    )

    assert saved.allowed_output_languages == ("en", "af")
    persisted = json.loads(config_paths.get_config_file_path().read_text("utf-8"))
    assert persisted["allowed_output_languages"] == ["en", "af"]
    assert "English (en), Afrikaans (af)" in get_backend_status(saved)

    monkeypatch.setenv("CTRLSPEAK_OUTPUT_LANGUAGES", "de, en")
    assert get_backend_config().allowed_output_languages == ("de", "en")

    with pytest.raises(ValueError, match="Unsupported Whisper language"):
        save_backend_config(
            backend="api",
            api_url=DEFAULT_API_URL,
            api_token=None,
            feedback_capture_method="disabled",
            allowed_output_languages=["not-a-language"],
        )


def test_invalid_saved_api_url_is_rejected_even_for_bundled_backend(monkeypatch) -> None:
    monkeypatch.setenv("CTRLSPEAK_BACKEND", "bundled")
    monkeypatch.setenv("CTRLSPEAK_API_URL", "not-a-url")

    with pytest.raises(ValueError, match=r"API URL.*https?://"):
        get_backend_config()

    with pytest.raises(ValueError, match=r"API URL.*https?://"):
        save_backend_config(
            backend="bundled",
            api_url="still-not-a-url",
            api_token=None,
            feedback_capture_method="disabled",
        )


@pytest.mark.parametrize(
    ("api_url", "api_token"),
    [
        ("http://127.0.0.1:8765", None),
        ("http://localhost:8765", None),
        ("http://10.1.2.3:8765", None),
        ("http://172.16.0.10:8765/base", None),
        ("http://192.168.50.4:8765", "optional-lan-token"),
        ("http://example.com", None),
        ("http://8.8.8.8:8765", "optional-public-token"),
        ("https://whisper.example.com", None),
        ("https://192.0.2.20:8765", None),
    ],
)
def test_any_valid_http_or_https_endpoint_is_supported_with_an_optional_token(
    api_url: str,
    api_token: str | None,
) -> None:
    saved = save_backend_config(
        backend="api",
        api_url=api_url,
        api_token=api_token,
        feedback_capture_method="disabled",
    )
    assert saved.api_url == api_url


def test_runtime_backend_remains_pinned_until_restart(monkeypatch) -> None:
    monkeypatch.setattr("utils.transcription_backend._runtime_backend_config", None)
    active = BackendConfig(
        "api",
        "http://192.168.1.50:8765",
        None,
        "active_field_on_enter",
    )
    activate_runtime_backend_config(active)

    saved = save_backend_config(
        backend="bundled",
        api_url="https://cloud.example.test/whisper",
        api_token="later-token",
        feedback_capture_method="disabled",
    )

    assert saved.backend == "bundled"
    assert get_backend_config().backend == "bundled"
    assert get_runtime_backend_config() == active


def test_backend_persistence_failure_is_actionable_and_not_claimed_successful(
    monkeypatch,
) -> None:
    with config_paths.settings_lock:
        before = dict(config_paths.settings)
    monkeypatch.setattr(config_paths, "save_settings", lambda: False)

    with pytest.raises(BackendPersistenceError, match=r"settings.*could not be saved"):
        save_backend_config(
            backend="api",
            api_url="http://192.168.1.50:8765",
            api_token=None,
            feedback_capture_method="active_field_on_enter",
        )

    with config_paths.settings_lock:
        assert config_paths.settings == before


def test_invalid_backend_from_settings_or_environment_is_not_silently_defaulted(
    monkeypatch,
) -> None:
    with config_paths.settings_lock:
        config_paths.settings["transcription_backend"] = "typo"
    with pytest.raises(ValueError, match=r"transcription_backend.*bundled.*api"):
        get_backend_config(environ={})

    monkeypatch.setenv("CTRLSPEAK_BACKEND", "also-wrong")
    with pytest.raises(ValueError, match=r"CTRLSPEAK_BACKEND.*bundled.*api"):
        get_backend_config()


def test_backend_status_and_persistence_never_disclose_token() -> None:
    saved = save_backend_config(
        backend="api",
        api_url="http://127.0.0.1:8765/",
        api_token="do-not-print-me",
        feedback_capture_method="active_field_on_enter",
    )

    status = get_backend_status(saved)
    assert status == (
        "API · http://127.0.0.1:8765 · bearer token configured · "
        "feedback: automatic active-field capture on Enter · "
        "output languages: Automatic (no restriction)"
    )
    assert "do-not-print-me" not in status
    assert "do-not-print-me" not in repr(saved)
    persisted = json.loads(config_paths.get_config_file_path().read_text("utf-8"))
    assert persisted["api_token"] == "do-not-print-me"


def test_api_transcription_uploads_audio_and_retains_audit_metadata(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wave-data")
    response_payload = {
        "id": "tx-123",
        "raw_text": "raw words",
        "text": "corrected words",
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": "raw words"}],
        "applied_correction_rule_ids": ["rule-1"],
        "exact_override_id": None,
    }
    session = RecordingSession(FakeResponse(200, response_payload))
    client = ApiTranscriptionClient(
        BackendConfig("api", "http://127.0.0.1:8765", "secret", "disabled"),
        session=session,
    )

    result = client.transcribe(audio)

    assert result.text == "corrected words"
    assert result.transcription_id == "tx-123"
    assert result.raw_text == "raw words"
    assert result.corrected_text == "corrected words"
    assert result.metadata == response_payload
    assert result.feedback_target is not None
    assert result.feedback_target.backend == "api"
    assert result.feedback_target.api_url == "http://127.0.0.1:8765"
    assert result.feedback_target.api_token == "secret"
    call = session.calls[0]
    assert call["url"] == "http://127.0.0.1:8765/v1/transcribe"
    assert call["audio_bytes"] == b"wave-data"
    assert call["headers"] == {"Authorization": "Bearer secret"}


def test_api_transcription_sends_and_enforces_the_ordered_language_allowlist(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wave-data")
    payload = {
        "id": "tx-allowed",
        "raw_text": "hello",
        "text": "hello",
        "language": "en",
    }
    session = RecordingSession(FakeResponse(200, payload))
    config = BackendConfig("api", DEFAULT_API_URL, None, "disabled", ("en", "af"))

    result = ApiTranscriptionClient(config, session=session).transcribe(audio)

    assert result.text == "hello"
    assert session.calls[0]["data"] == {"allowed_languages": "en,af"}

    rejected = RecordingSession(FakeResponse(200, {**payload, "language": "zh"}))
    with pytest.raises(ApiBackendError, match="outside.*allowlist"):
        ApiTranscriptionClient(config, session=rejected).transcribe(audio)


def test_api_failure_is_actionable_and_does_not_fall_back_to_bundled(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wave-data")
    session = RecordingSession(FakeResponse(503, {"detail": "model is not ready"}))
    client = ApiTranscriptionClient(
        BackendConfig("api", "http://127.0.0.1:8765", None, "disabled"),
        session=session,
    )
    bundled_calls: list[Path] = []

    with pytest.raises(ApiBackendError, match=r"HTTP 503.*model is not ready"):
        transcribe_selected(
            audio,
            config=client.config,
            bundled_transcriber=lambda path: bundled_calls.append(path) or "fallback",
            api_client=client,
        )

    assert bundled_calls == []


def test_gateway_strategy_is_persisted_and_sent_with_request_scoped_openai_key(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"wave-data")
    payload = {
        "id": "tx-routed",
        "raw_text": "hello",
        "text": "hello",
        "language": "en",
        "provider_used": "openai-gpt-transcribe",
    }
    session = RecordingSession(FakeResponse(200, payload))
    saved = save_backend_config(
        backend="api",
        api_url=DEFAULT_API_URL,
        api_token="client-token",
        feedback_capture_method="disabled",
        provider_strategy="openai-then-local",
    )
    set_session_openai_api_key("request-only-key")
    try:
        ApiTranscriptionClient(saved, session=session).transcribe(audio)
    finally:
        set_session_openai_api_key(None)

    call = session.calls[0]
    assert call["data"] == {"strategy": "openai-then-local"}
    assert call["headers"] == {
        "Authorization": "Bearer client-token",
        "X-CtrlSpeak-OpenAI-Key": "request-only-key",
    }
    persisted = json.loads(config_paths.get_config_file_path().read_text("utf-8"))
    assert persisted["provider_strategy"] == "openai-then-local"
    assert "openai_api_key" not in persisted
    assert "request-only-key" not in json.dumps(persisted)
    assert "request-only-key" not in repr(saved)


def test_capability_probe_rejects_worker_endpoint() -> None:
    session = RecordingSession(
        FakeResponse(
            200,
            {
                "version": "0.6.0",
                "role": "worker",
                "accepts_client_transcriptions": False,
            },
        )
    )
    client = ApiTranscriptionClient(
        BackendConfig("api", DEFAULT_API_URL, "client-token", "disabled"),
        session=session,
    )

    with pytest.raises(ApiBackendError, match="worker, not.*gateway"):
        client.get_capabilities()


def test_api_client_creates_identity_scoped_correction_rule() -> None:
    response_payload = {
        "id": "rule-123",
        "source_phrase": "control speak",
        "replacement_phrase": "CtrlSpeak",
        "enabled": True,
        "scope": "user",
        "owner_id": "desktop-user",
    }
    session = RecordingSession(FakeResponse(201, response_payload))
    client = ApiTranscriptionClient(
        BackendConfig("api", "https://gateway.example.test", "client-token", "disabled"),
        session=session,
    )

    created = client.create_correction("  control speak  ", " CtrlSpeak ")

    assert created == response_payload
    assert session.calls == [
        {
            "url": "https://gateway.example.test/v1/corrections",
            "audio_bytes": None,
            "headers": {"Authorization": "Bearer client-token"},
            "timeout": 300.0,
            "json": {
                "source_phrase": "control speak",
                "replacement_phrase": "CtrlSpeak",
                "enabled": True,
                "scope": "user",
            },
        }
    ]


@pytest.mark.parametrize(
    ("source", "replacement", "scope", "message"),
    [
        ("", "CtrlSpeak", "user", "phrase to replace"),
        ("control speak", "", "user", "replacement phrase"),
        ("CtrlSpeak", "CtrlSpeak", "user", "must differ"),
        ("control speak", "CtrlSpeak", "everyone", "scope"),
    ],
)
def test_api_client_rejects_invalid_correction_submission(
    source: str,
    replacement: str,
    scope: str,
    message: str,
) -> None:
    session = RecordingSession(FakeResponse(500, {"detail": "must not be called"}))
    client = ApiTranscriptionClient(
        BackendConfig("api", DEFAULT_API_URL, None, "disabled"),
        session=session,
    )

    with pytest.raises(ValueError, match=message):
        client.create_correction(source, replacement, scope=scope)

    assert session.calls == []


def test_api_client_lists_identity_visible_corrections_with_filters() -> None:
    items = [
        {
            "id": "rule-user",
            "source_phrase": "control speak",
            "replacement_phrase": "CtrlSpeak",
            "enabled": True,
            "scope": "user",
            "owner_id": "desktop-user",
        },
        {
            "id": "rule-global",
            "source_phrase": "open ai",
            "replacement_phrase": "OpenAI",
            "enabled": True,
            "scope": "global",
            "owner_id": None,
        },
    ]
    session = RecordingSession(FakeResponse(200, {"items": items}))
    client = ApiTranscriptionClient(
        BackendConfig("api", "https://gateway.example.test", "client-token", "disabled"),
        session=session,
    )

    assert client.list_corrections(enabled=True, include_all=True) == items
    assert session.calls == [
        {
            "url": "https://gateway.example.test/v1/corrections",
            "headers": {"Authorization": "Bearer client-token"},
            "timeout": 300.0,
            "params": {"enabled": "true", "include_all": "true"},
        }
    ]


def test_api_client_reads_and_url_quotes_one_correction_id() -> None:
    item = {
        "id": "team/rule?one",
        "source_phrase": "control speak",
        "replacement_phrase": "CtrlSpeak",
    }
    session = RecordingSession(FakeResponse(200, item))
    client = ApiTranscriptionClient(
        BackendConfig("api", DEFAULT_API_URL, "client-token", "disabled"),
        session=session,
    )

    assert client.get_correction("team/rule?one") == item
    assert session.calls[0]["url"] == (
        f"{DEFAULT_API_URL}/v1/corrections/team%2Frule%3Fone"
    )
    assert session.calls[0]["headers"] == {"Authorization": "Bearer client-token"}


def test_api_client_updates_all_mutable_correction_fields() -> None:
    updated = {
        "id": "rule-123",
        "source_phrase": "control suite",
        "replacement_phrase": "CtrlSpeak",
        "context_terms": ["speech"],
        "tags": ["product"],
        "enabled": False,
        "priority": 20,
        "language_codes": ["en", "af"],
        "send_as_keyword": True,
        "scope": "user",
        "owner_id": "desktop-user",
    }
    session = RecordingSession(FakeResponse(200, updated))
    client = ApiTranscriptionClient(
        BackendConfig("api", "https://gateway.example.test", "client-token", "disabled"),
        session=session,
    )

    result = client.update_correction(
        "rule-123",
        source_phrase="  control suite ",
        replacement_phrase=" CtrlSpeak ",
        context_terms=("speech",),
        tags=["product"],
        enabled=False,
        priority=20,
        language_codes=["English", "af", "en"],
        send_as_keyword=True,
    )

    assert result == updated
    assert session.calls == [
        {
            "url": "https://gateway.example.test/v1/corrections/rule-123",
            "headers": {"Authorization": "Bearer client-token"},
            "timeout": 300.0,
            "json": {
                "source_phrase": "control suite",
                "replacement_phrase": "CtrlSpeak",
                "context_terms": ["speech"],
                "tags": ["product"],
                "enabled": False,
                "send_as_keyword": True,
                "priority": 20,
                "language_codes": ["en", "af"],
            },
        }
    ]


def test_api_client_deletes_correction_without_parsing_204_body() -> None:
    session = RecordingSession(FakeResponse(204, {}))
    client = ApiTranscriptionClient(
        BackendConfig("api", DEFAULT_API_URL, "client-token", "disabled"),
        session=session,
    )

    assert client.delete_correction("rule-123") is None
    assert session.calls == [
        {
            "url": f"{DEFAULT_API_URL}/v1/corrections/rule-123",
            "headers": {"Authorization": "Bearer client-token"},
            "timeout": 300.0,
        }
    ]


@pytest.mark.parametrize(
    "operation",
    [
        lambda client: client.list_corrections(enabled="yes"),
        lambda client: client.get_correction("  "),
        lambda client: client.update_correction("rule-1"),
        lambda client: client.update_correction("rule-1", enabled="yes"),
        lambda client: client.update_correction("rule-1", priority=True),
        lambda client: client.update_correction("rule-1", language_codes=["not-a-language"]),
        lambda client: client.delete_correction(""),
    ],
)
def test_api_client_rejects_invalid_correction_management_before_network(operation) -> None:
    session = RecordingSession(FakeResponse(500, {"detail": "must not be called"}))
    client = ApiTranscriptionClient(
        BackendConfig("api", DEFAULT_API_URL, None, "disabled"),
        session=session,
    )

    with pytest.raises(ValueError):
        operation(client)
    assert session.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"items": "not-a-list"},
        {"items": [{"id": "rule-1", "source_phrase": "missing replacement"}]},
    ],
)
def test_api_client_rejects_malformed_correction_lists(payload: dict[str, object]) -> None:
    session = RecordingSession(FakeResponse(200, payload))
    client = ApiTranscriptionClient(
        BackendConfig("api", DEFAULT_API_URL, None, "disabled"),
        session=session,
    )

    with pytest.raises(ApiBackendError, match="correction-rule list|correction rule"):
        client.list_corrections()


@pytest.mark.parametrize(
    "operation",
    [
        lambda client: client.list_corrections(),
        lambda client: client.get_correction("rule-1"),
        lambda client: client.update_correction("rule-1", enabled=False),
        lambda client: client.delete_correction("rule-1"),
    ],
)
def test_correction_management_never_calls_gateway_for_bundled_backend(operation) -> None:
    session = RecordingSession(FakeResponse(500, {"detail": "must not be called"}))
    client = ApiTranscriptionClient(
        BackendConfig("bundled", DEFAULT_API_URL, None, "disabled"),
        session=session,
    )

    with pytest.raises(ValueError, match="Remote API backend"):
        operation(client)
    assert session.calls == []


def test_api_client_rejects_mismatched_updated_correction() -> None:
    session = RecordingSession(
        FakeResponse(
            200,
            {
                "id": "different-rule",
                "source_phrase": "control speak",
                "replacement_phrase": "CtrlSpeak",
                "enabled": False,
            },
        )
    )
    client = ApiTranscriptionClient(
        BackendConfig("api", DEFAULT_API_URL, None, "disabled"),
        session=session,
    )

    with pytest.raises(ApiBackendError, match="mismatched correction rule"):
        client.update_correction("rule-1", enabled=False)


def test_bundled_transcription_records_and_applies_only_local_exact_overrides(
    tmp_path: Path,
) -> None:
    library = LocalCorrectionLibrary(tmp_path / "local-corrections.sqlite3")
    config = BackendConfig("bundled", DEFAULT_API_URL, None, "active_field_on_enter")

    first = transcribe_selected(
        tmp_path / "unused.wav",
        config=config,
        bundled_transcriber=lambda _path: "write Acme tomorrow",
        local_correction_library=library,
    )
    assert first is not None
    assert first.raw_text == "write Acme tomorrow"
    assert first.corrected_text == "write Acme tomorrow"
    assert first.transcription_id
    assert first.feedback_target is not None
    assert first.feedback_target.backend == "bundled"

    library.approve_exact_override(
        first.transcription_id,
        confirmed_text="write ACME tomorrow",
        capture_method="active_field_on_enter",
    )
    repeated = transcribe_selected(
        tmp_path / "unused.wav",
        config=config,
        bundled_transcriber=lambda _path: "write Acme tomorrow",
        local_correction_library=library,
    )
    similar = transcribe_selected(
        tmp_path / "unused.wav",
        config=config,
        bundled_transcriber=lambda _path: "write Acme today",
        local_correction_library=library,
    )

    assert repeated is not None and repeated.text == "write ACME tomorrow"
    assert repeated.metadata and repeated.metadata["exact_override_id"]
    assert similar is not None and similar.text == "write Acme today"
    assert similar.metadata and similar.metadata["exact_override_id"] is None


def test_feedback_posts_only_confirmed_text_to_matching_transcription() -> None:
    session = RecordingSession(FakeResponse(200, {"transcription_id": "tx-123", "override_id": "override-1"}))
    client = ApiTranscriptionClient(
        BackendConfig("api", "http://127.0.0.1:8765", None, "active_field_on_enter"),
        session=session,
    )

    response = client.submit_feedback(
        "tx-123",
        final_text="confirmed final words",
        capture_method="active_field_on_enter",
        client_metadata={"client": "CtrlSpeak", "version": "0.3.0"},
    )

    assert response["override_id"] == "override-1"
    call = session.calls[0]
    assert call["url"] == "http://127.0.0.1:8765/v1/transcriptions/tx-123/feedback"
    assert call["json"] == {
        "confirmed_text": "confirmed final words",
        "capture_method": "active_field_on_enter",
        "client_metadata": {"client": "CtrlSpeak", "version": "0.3.0"},
    }
