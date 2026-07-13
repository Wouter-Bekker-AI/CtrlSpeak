from __future__ import annotations

import sys
import types

import pytest

from utils import system

pytestmark = pytest.mark.core_headless


def test_parse_cli_args_transcribe():
    args = system.parse_cli_args([
        "ctrlspeak", "--transcribe", "sample.wav", "--force-sendinput",
        "--backend", "api", "--api-url", "http://127.0.0.1:8765/",
    ])
    assert args.transcribe == "sample.wav"
    assert args.force_sendinput is True
    assert args.auto_setup is None
    assert args.uninstall is False
    assert args.backend == "api"
    assert args.api_url == "http://127.0.0.1:8765/"


def test_parse_cli_args_auto_setup():
    args = system.parse_cli_args(["ctrlspeak", "--auto-setup", "client_server"])
    assert args.auto_setup == "client_server"
    assert args.transcribe is None


def test_cli_backend_selection_persists_without_accepting_a_token_argument(capsys):
    args = system.parse_cli_args([
        "ctrlspeak", "--backend", "api", "--api-url", "http://127.0.0.1:9000",
        "--backend-status",
    ])

    assert system.apply_backend_cli_config(args) is True

    output = capsys.readouterr().out
    assert output == (
        "API · http://127.0.0.1:9000 · no bearer token · "
        "feedback: automatic active-field capture on Enter\n"
    )
    assert not hasattr(args, "api_token")
    with system.settings_lock:
        assert system.settings["transcription_backend"] == "api"
        assert system.settings["api_url"] == "http://127.0.0.1:9000"


def test_api_backend_hotkey_does_not_require_legacy_server():
    with system.settings_lock:
        system.settings["mode"] = "client"
        system.settings["transcription_backend"] = "api"
    system.last_connected_server = None

    assert system._client_hotkey_available() is True


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


def test_windows_reacquire_single_instance_lock_is_idempotent(tmp_path, monkeypatch):
    lock_attempts = 0

    def locking(_fileno, mode, _length):
        nonlocal lock_attempts
        if mode == fake_msvcrt.LK_NBLCK:
            lock_attempts += 1
            if lock_attempts > 1:
                raise OSError("lock already held")

    fake_msvcrt = types.SimpleNamespace(LK_NBLCK=1, LK_UNLCK=2, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setattr(system.sys, "platform", "win32")
    monkeypatch.setattr(system, "get_config_dir", lambda: tmp_path)

    try:
        assert system.acquire_single_instance_lock() is True
        first_handle = system.instance_lock_handle

        assert system.acquire_single_instance_lock() is True
        assert system.instance_lock_handle is first_handle
        assert lock_attempts == 1
    finally:
        system.release_single_instance_lock()
