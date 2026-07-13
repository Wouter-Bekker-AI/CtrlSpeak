from __future__ import annotations

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
    monkeypatch.setenv("HOME", str(tmp_path))

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
