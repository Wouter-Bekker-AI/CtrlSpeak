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
    lock_dir = tmp_path / "cfg"
    lock_dir.mkdir()
    if system.sys.platform.startswith("win"):
        monkeypatch.setenv("APPDATA", str(lock_dir))
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(lock_dir))

    import importlib

    importlib.reload(system)

    assert system.acquire_single_instance_lock() is True
    first_handle = system.instance_lock_handle
    assert first_handle is not None

    assert system.acquire_single_instance_lock() is True
    assert system.instance_lock_handle is not None

    lock_path = system.get_config_dir() / system.LOCK_FILENAME
    assert lock_path.exists()

    system.release_single_instance_lock()
    assert system.instance_lock_handle is None
    assert not lock_path.exists()
