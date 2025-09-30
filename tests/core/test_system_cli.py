from __future__ import annotations

import pytest

from utils import system

pytestmark = pytest.mark.core_headless


def test_parse_cli_args_transcribe():
    args = system.parse_cli_args(["ctrlspeak", "--transcribe", "sample.wav", "--force-sendinput"])
    assert args.transcribe == "sample.wav"
    assert args.force_sendinput is True
    assert args.auto_setup is None
    assert args.uninstall is False


def test_parse_cli_args_auto_setup():
    args = system.parse_cli_args(["ctrlspeak", "--auto-setup", "client_server"])
    assert args.auto_setup == "client_server"
    assert args.transcribe is None


def test_acquire_single_instance_lock(tmp_path, monkeypatch):
    data_home = tmp_path / "data"
    config_home = tmp_path / "cfg"
    data_home.mkdir()
    config_home.mkdir()
    if system.sys.platform.startswith("win"):
        monkeypatch.setenv("APPDATA", str(data_home))
    else:
        monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    import importlib
    from utils import config_paths

    importlib.reload(config_paths)
    importlib.reload(system)

    assert system.acquire_single_instance_lock() is True
    first_handle = system.instance_lock_handle
    assert system.acquire_single_instance_lock() is False
    assert system.instance_lock_handle is not None

    lock_path = system.get_data_dir() / system.LOCK_FILENAME
    assert lock_path.exists()

    system.release_single_instance_lock()
    assert system.instance_lock_handle is None
    assert not lock_path.exists()
