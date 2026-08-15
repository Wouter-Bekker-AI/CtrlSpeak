from __future__ import annotations

import json

from utils import config_paths, credential_store
from utils.transcription_backend import (
    forget_openai_api_key,
    get_session_openai_api_key,
    initialize_openai_api_key_from_secure_storage,
    persist_openai_api_key,
)


class FakeCredentialBackend:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def read(self, target: str) -> str | None:
        return self.values.get(target)

    def write(self, target: str, value: str) -> None:
        self.values[target] = value

    def delete(self, target: str) -> None:
        self.values.pop(target, None)


def _install_fake_backend(monkeypatch) -> FakeCredentialBackend:
    backend = FakeCredentialBackend()
    monkeypatch.setattr(credential_store, "_backend", backend)
    monkeypatch.setattr(credential_store, "_backend_initialized", True)
    return backend


def test_openai_key_round_trips_through_native_store_without_settings(monkeypatch) -> None:
    backend = _install_fake_backend(monkeypatch)

    persist_openai_api_key("  project-secret  ", remember=True)
    assert get_session_openai_api_key() == "project-secret"
    assert backend.values[credential_store.OPENAI_CREDENTIAL_TARGET] == "project-secret"

    config_path = config_paths.get_config_file_path()
    if config_path.exists():
        persisted = config_path.read_text("utf-8")
        assert "project-secret" not in persisted
        assert "openai_api_key" not in json.loads(persisted)

    persist_openai_api_key(None, remember=False)
    assert get_session_openai_api_key() is None
    assert backend.values == {}


def test_saved_openai_key_loads_on_startup_and_can_be_forgotten(monkeypatch) -> None:
    backend = _install_fake_backend(monkeypatch)
    backend.values[credential_store.OPENAI_CREDENTIAL_TARGET] = "saved-secret"

    assert initialize_openai_api_key_from_secure_storage() is True
    assert get_session_openai_api_key() == "saved-secret"

    forget_openai_api_key()
    assert get_session_openai_api_key() is None
    assert backend.values == {}
