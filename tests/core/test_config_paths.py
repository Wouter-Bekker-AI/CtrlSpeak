from __future__ import annotations

import json
import stat

import pytest

from utils import config_paths

pytestmark = pytest.mark.core_headless


def test_get_config_dir_creates_expected_structure(tmp_path, monkeypatch):
    config_home = tmp_path / "cfg"
    config_home.mkdir()
    if config_paths.sys.platform.startswith("win"):
        monkeypatch.setenv("APPDATA", str(config_home))
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    # Reload so the helper reads the new environment variable without leaking
    # the fixture-wide isolation performed in conftest.
    import importlib

    importlib.reload(config_paths)

    config_dir = config_paths.get_config_dir()
    expected = config_home / "CtrlSpeak"
    assert config_dir == expected
    for subdir in ("models", "cuda", "temp", config_paths.LOG_DIR_NAME):
        assert (config_dir / subdir).exists()


def test_linux_config_dir_falls_back_to_home_dot_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config_paths.sys, "platform", "linux")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(config_paths.Path, "home", lambda: tmp_path)

    assert config_paths.get_config_dir() == tmp_path / ".config" / "CtrlSpeak"


def test_settings_round_trip(tmp_path, monkeypatch):
    config_home = tmp_path / "cfg"
    config_home.mkdir()
    if config_paths.sys.platform.startswith("win"):
        monkeypatch.setenv("APPDATA", str(config_home))
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    import importlib

    importlib.reload(config_paths)

    loaded = config_paths.load_settings()
    assert loaded["model_name"] == config_paths.DEFAULT_SETTINGS["model_name"]
    assert loaded["allowed_output_languages"] == []

    with config_paths.settings_lock:
        config_paths.settings["unit_test_marker"] = "ok"
    assert config_paths.save_settings() is True

    reloaded = config_paths.load_settings()
    assert reloaded["unit_test_marker"] == "ok"

    settings_file = config_paths.get_config_file_path()
    assert settings_file.exists()
    assert "unit_test_marker" in settings_file.read_text(encoding="utf-8")
    if not config_paths.sys.platform.startswith("win"):
        assert stat.S_IMODE(settings_file.stat().st_mode) == 0o600


def test_v04_settings_migration_salvages_valid_fields(tmp_path, monkeypatch):
    config_home = tmp_path / "cfg"
    config_home.mkdir()
    if config_paths.sys.platform.startswith("win"):
        monkeypatch.setenv("APPDATA", str(config_home))
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    import importlib

    importlib.reload(config_paths)
    settings_file = config_paths.get_config_file_path()
    settings_file.write_text(
        json.dumps(
            {
                "mode": "client_server",
                "device_preference": "cuda",
                "model_name": "large-v3",
                "transcription_backend": "api",
                "api_url": "http://192.168.1.22:8765",
                "api_token": "keep-this-secret",
                "show_whats_new_on_update": "not-a-boolean",
                "future_field": {"preserve": True},
            }
        ),
        encoding="utf-8",
    )

    loaded = config_paths.load_settings()

    assert loaded["settings_schema_version"] == config_paths.SETTINGS_SCHEMA_VERSION
    assert loaded["mode"] == "client_server"
    assert loaded["device_preference"] == "cuda"
    assert loaded["model_name"] == "large-v3"
    assert loaded["transcription_backend"] == "api"
    assert loaded["api_url"] == "http://192.168.1.22:8765"
    assert loaded["api_token"] == "keep-this-secret"
    assert loaded["show_whats_new_on_update"] is True
    assert loaded["allowed_output_languages"] == []
    assert loaded["future_field"] == {"preserve": True}

    backups = list(settings_file.parent.glob("settings.pre-migration-v2.*.json"))
    assert len(backups) == 1
    original = json.loads(backups[0].read_text(encoding="utf-8"))
    assert original["api_token"] == "keep-this-secret"
    migrated = json.loads(settings_file.read_text(encoding="utf-8"))
    assert migrated["api_token"] == "keep-this-secret"
    assert migrated["show_whats_new_on_update"] is True


def test_corrupt_settings_are_backed_up_before_defaults_are_written(tmp_path, monkeypatch):
    config_home = tmp_path / "cfg"
    config_home.mkdir()
    if config_paths.sys.platform.startswith("win"):
        monkeypatch.setenv("APPDATA", str(config_home))
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    import importlib

    importlib.reload(config_paths)
    settings_file = config_paths.get_config_file_path()
    settings_file.write_text("{definitely not json", encoding="utf-8")

    loaded = config_paths.load_settings()

    assert loaded["settings_schema_version"] == config_paths.SETTINGS_SCHEMA_VERSION
    assert loaded["transcription_backend"] == "bundled"
    backups = list(settings_file.parent.glob("settings.pre-migration-v2.*.json"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{definitely not json"
    assert json.loads(settings_file.read_text(encoding="utf-8"))["model_name"] == "small"


def test_settings_persistence_failure_is_returned_to_the_caller(monkeypatch):
    monkeypatch.setattr(config_paths, "get_config_file_path", lambda: object())

    assert config_paths.save_settings() is False


def test_create_and_cleanup_recording_file(tmp_path, monkeypatch):
    config_home = tmp_path / "cfg"
    config_home.mkdir()
    if config_paths.sys.platform.startswith("win"):
        monkeypatch.setenv("APPDATA", str(config_home))
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    import importlib

    importlib.reload(config_paths)

    file_path = config_paths.create_recording_file_path()
    assert file_path.parent == config_paths.get_temp_dir()
    file_path.write_bytes(b"WAV")
    assert file_path.exists()
    config_paths.cleanup_recording_file(file_path)
    assert not file_path.exists()
