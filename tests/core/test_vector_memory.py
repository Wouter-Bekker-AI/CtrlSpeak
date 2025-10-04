import importlib
import types
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

    store = vector_memory.VectorMemoryStore("Vision")
    store.add_memories(["hello world"], metadata=[{"role": "user"}])

    no_hits = store.retrieve("something unrelated", top_k=5, threshold=0.95)
    assert no_hits == []

    hits = store.retrieve("hello world", top_k=5, threshold=0.1)
    assert hits


def test_vector_memory_documentation_fallback(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    vector_memory = modules["utils.vector_memory"]

    store = vector_memory.VectorMemoryStore("Vision")
    store.add_memories(
        ["Press Chat with Bot in the management window to launch Vision."],
        metadata=[
            {
                "category": "documentation",
                "source": "docs/usage.md",
                "chunk": 1,
                "chunks": 1,
                "doc_hash": "abc123",
            }
        ],
    )

    results = store.retrieve(
        "how do i use this program?",
        top_k=3,
        threshold=0.95,
        category_thresholds={"documentation": 0.2},
        fallback_categories={"documentation": 1},
    )

    assert results, "expected documentation fallback to produce a match"
    assert results[0].metadata.get("category") == "documentation"


def test_vector_memory_retrieve_scoped_queries(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    vector_memory = modules["utils.vector_memory"]

    store = vector_memory.VectorMemoryStore("Vision")

    monkeypatch.setattr(store, "count", lambda: 5)

    calls = []

    def fake_query(self, *, query_embeddings, n_results, include, where=None):
        calls.append({"n_results": n_results, "include": list(include), "where": where})
        if where is None:
            return {
                "documents": [[
                    "Launch Vision from the management window.",
                    "Documentation filler entry.",
                ]],
                "metadatas": [[
                    {
                        "category": "documentation",
                        "sequence": 1,
                        "created_at": "2024-01-01T00:00:00Z",
                    },
                    {
                        "category": "documentation",
                        "sequence": 2,
                        "created_at": "2024-01-01T00:00:01Z",
                    },
                ]],
                "distances": [[0.1, 0.95]],
            }
        if where == {"category": "chat_history"}:
            return {
                "documents": [["Remember to check yesterday's notes."]],
                "metadatas": [[
                    {
                        "category": "chat_history",
                        "sequence": 3,
                        "created_at": "2024-01-01T00:00:02Z",
                    }
                ]],
                "distances": [[0.8]],
            }
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    monkeypatch.setattr(store.collection, "query", types.MethodType(fake_query, store.collection))

    results = store.retrieve(
        "assistant launch help",
        top_k=2,
        threshold=0.9,
        category_thresholds={"documentation": 0.2, "chat_history": 0.7},
        fallback_categories={"documentation": 1, "chat_history": 1},
    )

    assert results
    assert [item.metadata.get("category") for item in results] == ["documentation", "chat_history"]

    assert calls[0]["n_results"] == 4
    assert "embeddings" not in calls[0]["include"]
    assert any(call["where"] == {"category": "chat_history"} for call in calls)
