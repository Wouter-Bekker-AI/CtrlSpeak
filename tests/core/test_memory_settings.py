import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.core_headless


def _prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import utils.config_paths as cfg

    data_home = tmp_path / "data"
    config_home = tmp_path / "cfg"
    data_home.mkdir()
    config_home.mkdir()

    if cfg.sys.platform.startswith("win"):
        monkeypatch.setenv("APPDATA", str(data_home))
    else:
        monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    modules = {}
    for name in [
        "utils.config_paths",
        "utils.memory_settings",
    ]:
        modules[name] = importlib.reload(importlib.import_module(name))
    return modules


def test_identity_settings_roundtrip(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    memory_settings = modules["utils.memory_settings"]

    identity = "Test Persona"
    defaults = memory_settings.load_identity_settings(identity)
    assert defaults["retrieval_top_k"] == 5
    assert defaults["pii_redaction"] is False

    overrides = {"retrieval_top_k": 7, "pii_redaction": True}
    memory_settings.save_identity_settings(identity, overrides)

    loaded = memory_settings.load_identity_settings(identity)
    assert loaded["retrieval_top_k"] == 7
    assert loaded["pii_redaction"] is True
    assert loaded["store_vector_memory"] is True

    settings_path = memory_settings.get_identity_settings_path(identity)
    assert settings_path.exists()
