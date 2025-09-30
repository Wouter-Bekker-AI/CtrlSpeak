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


def test_vector_memory_retention_and_ttl(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    vector_memory = modules["utils.vector_memory"]

    store = vector_memory.VectorMemoryStore("Persona")
    store.add_memories(["first"], metadata=[{"role": "user"}], max_items=2)
    store.add_memories(["second"], metadata=[{"role": "assistant"}], max_items=2)
    assert store.count() == 2

    result = store.add_memories(["third"], metadata=[{"role": "assistant"}], max_items=2)
    assert result["evicted"] >= 1

    store.add_memories(["ephemeral"], ttl_days=0)
    purged = store.purge_expired()
    assert purged >= 1


def test_vector_memory_embedder_upgrade(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    vector_memory = modules["utils.vector_memory"]

    initial = vector_memory.VectorMemoryStore("Tester", embedder_version="1")
    first_collection = initial.collection.name

    upgraded = vector_memory.VectorMemoryStore("Tester", embedder_version="2")
    assert upgraded.collection.name != first_collection


def test_vector_memory_retrieval_threshold(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    vector_memory = modules["utils.vector_memory"]

    store = vector_memory.VectorMemoryStore("Assistant")
    store.add_memories(["hello world"], metadata=[{"role": "user"}])

    no_hits = store.retrieve("something unrelated", top_k=5, threshold=0.95)
    assert no_hits == []

    hits = store.retrieve("hello world", top_k=5, threshold=0.1)
    assert hits
