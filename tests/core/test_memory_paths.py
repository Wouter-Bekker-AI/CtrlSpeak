import importlib
from pathlib import Path

import pytest

from utils import config_paths, memory_paths

pytestmark = pytest.mark.core_headless


def _reload_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_home = tmp_path / "data"
    config_home = tmp_path / "cfg"
    data_home.mkdir()
    config_home.mkdir()

    if config_paths.sys.platform.startswith("win"):
        monkeypatch.setenv("APPDATA", str(data_home))
    else:
        monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    import utils.config_paths as cfg
    import utils.memory_paths as mem

    importlib.reload(cfg)
    importlib.reload(mem)
    globals()["config_paths"] = cfg
    globals()["memory_paths"] = mem


def test_bot_memory_dirs_created(tmp_path, monkeypatch):
    _reload_paths(tmp_path, monkeypatch)

    identity = "Assistant Persona"
    root = memory_paths.get_bot_memory_dir(identity)

    expected_parent = config_paths.get_data_dir() / "bot_memory"
    assert root.parent == expected_parent
    assert root.name == "Assistant_Persona"

    assert (root / "conversation").is_dir()
    assert (root / "screenshots").is_dir()
    assert (root / "chroma").is_dir()
    assert (root / "traces").is_dir()

    log_path = memory_paths.get_bot_conversation_log(identity)
    assert log_path.parent == root / "conversation"
    assert log_path.name == "conversation.jsonl"


def test_identity_lock_path(tmp_path, monkeypatch):
    _reload_paths(tmp_path, monkeypatch)

    lock_path = memory_paths.get_identity_lock_path("test")
    assert lock_path.parent.name == ".locks"
    assert lock_path.name.endswith(".lock")
    assert lock_path.parent.parent == config_paths.get_data_dir()
