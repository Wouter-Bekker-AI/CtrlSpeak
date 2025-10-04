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
        "utils.memory_paths",
        "utils.vector_memory",
    ]:
        modules[name] = importlib.reload(importlib.import_module(name))
    return modules


def test_profile_slot_roundtrip(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    vector_memory = modules["utils.vector_memory"]

    store = vector_memory.VectorMemoryStore("Persona")
    metadata = store.upsert_profile_slot("user123", "name", "Alice", "test")
    assert metadata["status"] == "current"
    current = store.read_profile_slot("user123", "name")
    assert current is not None
    assert current["metadata"]["value"] == "Alice"

    store.upsert_profile_slot("user123", "name", "Alicia", "test")
    updated = store.read_profile_slot("user123", "name")
    assert updated is not None
    assert updated["metadata"]["value"] == "Alicia"

    payload = store.profile_collection.get(include=["metadatas"])
    statuses = []
    for meta in payload.get("metadatas", []) or []:
        if not isinstance(meta, dict):
            continue
        if meta.get("user_id") != "user123" or meta.get("attribute") != "name":
            continue
        statuses.append(meta.get("status"))
    assert "superseded" in statuses


def test_profile_query(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    vector_memory = modules["utils.vector_memory"]

    store = vector_memory.VectorMemoryStore("Persona")
    store.upsert_profile_slot("default_user", "name", "Casey", "test")
    store.upsert_profile_slot("default_user", "age", "29", "test")

    specific = store.query_profile("user.name?", "default_user", attribute="name")
    assert specific
    assert specific[0].metadata.get("value") == "Casey"

    general = store.query_profile("user.profile?", "default_user")
    attributes = {item.metadata.get("attribute") for item in general}
    assert {"name", "age"}.issubset(attributes)
