from __future__ import annotations

import pytest

from utils import config_paths

pytestmark = pytest.mark.core_headless


def _reload_with_appdata(tmp_path, monkeypatch):
    data_home = tmp_path / "data"
    config_home = tmp_path / "cfg"
    data_home.mkdir()
    config_home.mkdir()

    if config_paths.sys.platform.startswith("win"):
        monkeypatch.setenv("APPDATA", str(data_home))
    else:
        monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    import importlib

    importlib.reload(config_paths)
    return data_home, (data_home if config_paths.sys.platform.startswith("win") else config_home)


def test_get_data_and_config_dirs(tmp_path, monkeypatch):
    data_home, config_home = _reload_with_appdata(tmp_path, monkeypatch)

    data_dir = config_paths.get_data_dir()
    expected_data = data_home / "CtrlSpeak"
    assert data_dir == expected_data
    for subdir in ("models", "cuda", "bot_memory", config_paths.LOG_DIR_NAME, "temp"):
        assert (data_dir / subdir).exists()

    config_dir = config_paths.get_config_dir()
    expected_config = config_home / "CtrlSpeak"
    assert config_dir == expected_config


def test_settings_round_trip(tmp_path, monkeypatch):
    _, _ = _reload_with_appdata(tmp_path, monkeypatch)

    loaded = config_paths.load_settings()
    assert loaded["model_name"] == config_paths.DEFAULT_SETTINGS["model_name"]

    with config_paths.settings_lock:
        config_paths.settings["unit_test_marker"] = "ok"
    config_paths.save_settings()

    reloaded = config_paths.load_settings()
    assert reloaded["unit_test_marker"] == "ok"

    settings_file = config_paths.get_config_file_path()
    assert settings_file.exists()
    assert "unit_test_marker" in settings_file.read_text(encoding="utf-8")


def test_create_and_cleanup_recording_file(tmp_path, monkeypatch):
    _, _ = _reload_with_appdata(tmp_path, monkeypatch)

    file_path = config_paths.create_recording_file_path()
    assert file_path.parent == config_paths.get_temp_dir()
    file_path.write_bytes(b"WAV")
    assert file_path.exists()
    config_paths.cleanup_recording_file(file_path)
    assert not file_path.exists()
