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


def test_parse_cli_args_supports_packaging_and_internal_update_modes():
    args = system.parse_cli_args(
        [
            "ctrlspeak",
            "--version",
            "--health-check-file",
            "health.json",
            "--apply-update",
            "transaction.json",
            "--post-update",
            "abc",
            "--rollback-notice",
            "def",
        ]
    )

    assert args.show_version is True
    assert args.health_check_file == "health.json"
    assert args.apply_update == "transaction.json"
    assert args.post_update == "abc"
    assert args.rollback_notice == "def"


def test_cli_backend_selection_persists_without_accepting_a_token_argument(capsys):
    args = system.parse_cli_args([
        "ctrlspeak", "--backend", "api", "--api-url", "http://127.0.0.1:9000",
        "--backend-status",
    ])

    assert system.apply_backend_cli_config(args) is True

    output = capsys.readouterr().out
    assert output == (
        "API · http://127.0.0.1:9000 · no bearer token · "
        "feedback: automatic active-field capture on Enter · "
        "output languages: Automatic (no restriction)\n"
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


def test_tray_menu_exposes_correction_submission(monkeypatch):
    labels: list[object] = []
    gui_calls: list[str] = []

    fake_gui = types.ModuleType("utils.gui")
    fake_gui.ensure_management_ui_thread = lambda: gui_calls.append("ensure")
    fake_gui.run_management_ui_loop = lambda: gui_calls.append("run")
    fake_gui.request_management_ui_shutdown = lambda: gui_calls.append("shutdown")
    monkeypatch.setitem(sys.modules, "utils.gui", fake_gui)

    fake_pystray = types.ModuleType("pystray")

    def menu_item(label, *_args, **_kwargs):
        labels.append(label)
        return object()

    class FakeIcon:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self):
            return None

        def stop(self):
            return None

    fake_pystray.MenuItem = menu_item
    fake_pystray.Menu = lambda *_items: object()
    fake_pystray.Icon = FakeIcon
    monkeypatch.setitem(sys.modules, "pystray", fake_pystray)
    monkeypatch.setattr(system, "start_client_listener", lambda: None)
    monkeypatch.setattr(system, "create_icon_image", lambda: None)
    monkeypatch.setattr(system, "_tray_icon", None)

    system.run_tray()

    resolved = [label(None) if callable(label) else str(label) for label in labels]
    assert "Show / hide quick panel" in resolved
    assert "Submit correction…" in resolved
    assert "run" in gui_calls


def test_tray_correction_action_queues_the_dialog(monkeypatch):
    calls: list[tuple[object, tuple[object, ...]]] = []
    icon = object()
    dialog = lambda _icon: None
    fake_gui = types.ModuleType("utils.gui")
    fake_gui._show_correction_submission_dialog = dialog
    monkeypatch.setitem(sys.modules, "utils.gui", fake_gui)
    monkeypatch.setattr(
        system,
        "enqueue_management_task",
        lambda callback, *args: calls.append((callback, args)),
    )

    system.submit_correction_from_tray(icon, None)

    assert calls == [(dialog, (icon,))]


def test_native_tray_actions_only_enqueue_tk_work(monkeypatch):
    calls: list[tuple[object, tuple[object, ...]]] = []
    icon = object()
    fake_gui = types.ModuleType("utils.gui")
    flyout = lambda _icon: None
    management = lambda _icon: None
    fake_gui._show_tray_flyout = flyout
    fake_gui._show_management_window = management
    monkeypatch.setitem(sys.modules, "utils.gui", fake_gui)
    monkeypatch.setattr(
        system,
        "enqueue_management_task",
        lambda callback, *args: calls.append((callback, args)),
    )

    system.open_tray_flyout(icon, None)
    system.open_management_dialog(icon, None)

    assert calls == [(flyout, (icon,)), (management, (icon,))]


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
