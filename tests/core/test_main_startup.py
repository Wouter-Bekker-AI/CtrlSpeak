from __future__ import annotations

import importlib
import json
import sys
import types

import pytest

from utils import config_paths


pytestmark = pytest.mark.core_headless


def _import_main_without_runtime_gui(monkeypatch):
    gui = types.ModuleType("utils.gui")
    startup_errors: list[tuple[str, str]] = []
    gui.show_splash_screen = lambda *_args: None
    gui.ensure_mode_selected = lambda: None
    gui.ensure_management_ui_thread = lambda: None
    gui.show_startup_error = lambda title, message: startup_errors.append((title, message))
    models = types.ModuleType("utils.models")
    models.initialize_transcriber = lambda **_kwargs: None
    models.ensure_model_ready_for_local_server = lambda: True
    models.ensure_initial_model_installation = lambda: True
    monkeypatch.setitem(sys.modules, "utils.gui", gui)
    monkeypatch.setitem(sys.modules, "utils.models", models)
    sys.modules.pop("main", None)
    app = importlib.import_module("main")
    app._test_startup_errors = startup_errors
    return app


@pytest.mark.parametrize(
    ("environment", "expected_detail"),
    [
        ({"CTRLSPEAK_BACKEND": "wrong"}, "CTRLSPEAK_BACKEND"),
        (
            {
                "CTRLSPEAK_BACKEND": "api",
                "CTRLSPEAK_API_URL": "ftp://not-an-http-api.example.com",
            },
            "http:// or https://",
        ),
    ],
)
def test_invalid_environment_backend_configuration_exits_cleanly(
    monkeypatch,
    capsys,
    environment: dict[str, str],
    expected_detail: str,
) -> None:
    app = _import_main_without_runtime_gui(monkeypatch)
    for name in ("CTRLSPEAK_BACKEND", "CTRLSPEAK_API_URL", "CTRLSPEAK_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(app, "acquire_single_instance_lock", lambda: True)
    monkeypatch.setattr(app, "load_settings", config_paths.load_settings)

    exit_code = app.main(["ctrlspeak"])

    assert exit_code == 2
    error = capsys.readouterr().err
    assert "Invalid backend configuration" in error
    assert expected_detail in error
    assert len(app._test_startup_errors) == 1
    assert app._test_startup_errors[0][0] == "Invalid backend configuration"
    assert expected_detail in app._test_startup_errors[0][1]


def test_invalid_saved_backend_configuration_exits_cleanly(monkeypatch, capsys) -> None:
    app = _import_main_without_runtime_gui(monkeypatch)
    for name in ("CTRLSPEAK_BACKEND", "CTRLSPEAK_API_URL", "CTRLSPEAK_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(app, "acquire_single_instance_lock", lambda: True)

    def load_invalid_settings() -> None:
        config_paths.load_settings()
        with config_paths.settings_lock:
            config_paths.settings["api_url"] = "missing-a-scheme"

    monkeypatch.setattr(app, "load_settings", load_invalid_settings)

    exit_code = app.main(["ctrlspeak"])

    assert exit_code == 2
    error = capsys.readouterr().err
    assert "Invalid backend configuration" in error
    assert "API URL" in error
    assert len(app._test_startup_errors) == 1
    assert "API URL" in app._test_startup_errors[0][1]


def test_invalid_configuration_is_shown_when_windowed_build_has_no_stderr(
    monkeypatch,
) -> None:
    app = _import_main_without_runtime_gui(monkeypatch)
    monkeypatch.setenv("CTRLSPEAK_BACKEND", "api")
    monkeypatch.setenv("CTRLSPEAK_API_URL", "missing-a-scheme")
    monkeypatch.setattr(app, "acquire_single_instance_lock", lambda: True)
    monkeypatch.setattr(app, "load_settings", config_paths.load_settings)
    monkeypatch.setattr(app.sys, "stderr", None)

    assert app.main(["ctrlspeak"]) == 2
    assert len(app._test_startup_errors) == 1
    assert "http:// or https://" in app._test_startup_errors[0][1]


def test_packaged_health_check_file_exits_before_normal_startup(tmp_path, monkeypatch) -> None:
    app = _import_main_without_runtime_gui(monkeypatch)
    target = tmp_path / "health.json"
    monkeypatch.setattr(
        app,
        "acquire_single_instance_lock",
        lambda: (_ for _ in ()).throw(AssertionError("normal startup must not run")),
    )

    assert app.main(["ctrlspeak", "--health-check-file", str(target)]) == 0

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["product"] == "ctrlspeak"
    assert payload["version"] == app.APP_VERSION
    assert isinstance(payload["frozen"], bool)


def test_version_mode_exits_before_normal_startup(monkeypatch, capsys) -> None:
    app = _import_main_without_runtime_gui(monkeypatch)
    monkeypatch.setattr(
        app,
        "acquire_single_instance_lock",
        lambda: (_ for _ in ()).throw(AssertionError("normal startup must not run")),
    )

    assert app.main(["ctrlspeak", "--version"]) == 0
    assert capsys.readouterr().out.strip() == app.APP_VERSION
