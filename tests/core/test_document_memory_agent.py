import importlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.core_headless


def _prepare_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    app_base = tmp_path / "app"
    docs_dir = app_base / "docs"
    data_home.mkdir(exist_ok=True)
    config_home.mkdir(exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    (app_base / "README.md").write_text("Readme overview for CtrlSpeak", encoding="utf-8")
    (docs_dir / "bot_integration.md").write_text(
        "Detailed usage instructions live here.",
        encoding="utf-8",
    )
    (docs_dir / "tooling.md").write_text("Tooling reference for CtrlSpeak.", encoding="utf-8")
    (docs_dir / "user_flow.md").write_text("User journey walkthrough.", encoding="utf-8")

    monkeypatch.setenv("APPDATA", str(data_home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    modules = {}
    for name in [
        "utils.config_paths",
        "utils.memory_paths",
        "utils.memory_settings",
        "utils.vector_memory",
    ]:
        modules[name] = importlib.reload(importlib.import_module(name))

    doc_agent = importlib.reload(importlib.import_module("background_agents.document_memory_agent"))
    monkeypatch.setattr(doc_agent, "get_app_base_dir", lambda: app_base)

    return modules, doc_agent, app_base


def test_refresh_document_memory_populates_and_skips_within_cooldown(tmp_path, monkeypatch, capsys):
    modules, doc_agent, _ = _prepare_environment(tmp_path, monkeypatch)
    vector_memory = modules["utils.vector_memory"]

    assert doc_agent.refresh_document_memory("assistant") is True

    store = vector_memory.VectorMemoryStore("assistant")
    payload = store.collection.get(include=["metadatas", "documents"])
    raw_documents = payload.get("documents") or []
    raw_metadata = payload.get("metadatas") or []
    documents = []
    metadata = []
    for doc, meta in zip(raw_documents, raw_metadata):
        if isinstance(meta, dict) and meta.get("category") == "documentation":
            documents.append(doc)
            metadata.append(meta)
    assert documents, "expected documentation chunks to be stored"
    assert len(documents) == len(metadata)
    assert {entry.get("category") for entry in metadata} == {"documentation"}

    tracker_path = doc_agent._tracker_path("assistant")  # noqa: SLF001 - internal helper for tests
    tracker_data = json.loads(tracker_path.read_text(encoding="utf-8"))
    assert tracker_data["doc_hash"] == metadata[0]["doc_hash"]

    doc_count = store.collection.count()
    assert doc_agent.refresh_document_memory("assistant") is True
    captured = capsys.readouterr()
    assert "already current" in captured.out
    assert store.collection.count() == doc_count


def test_refresh_document_memory_detects_changes_and_force(tmp_path, monkeypatch, capsys):
    modules, doc_agent, app_base = _prepare_environment(tmp_path, monkeypatch)
    vector_memory = modules["utils.vector_memory"]

    assert doc_agent.refresh_document_memory("assistant") is True
    tracker_path = doc_agent._tracker_path("assistant")
    baseline = json.loads(tracker_path.read_text(encoding="utf-8"))
    initial_hash = baseline["doc_hash"]

    guide_path = app_base / "docs" / "bot_integration.md"
    guide_path.write_text(
        "Updated usage guide with new sections and elaboration.",
        encoding="utf-8",
    )

    assert doc_agent.refresh_document_memory("assistant") is True
    captured = capsys.readouterr()
    assert "injecting documentation" in captured.out.lower()

    updated = json.loads(tracker_path.read_text(encoding="utf-8"))
    assert updated["doc_hash"] != initial_hash

    store = vector_memory.VectorMemoryStore("assistant")
    payload = store.collection.get(include=["metadatas"])
    metadata = [
        meta
        for meta in (payload.get("metadatas") or [])
        if isinstance(meta, dict) and meta.get("category") == "documentation"
    ]
    assert metadata, "expected documentation metadata after update"
    assert {entry.get("doc_hash") for entry in metadata} == {updated["doc_hash"]}

    assert doc_agent.refresh_document_memory("assistant", force=True, reason="test") is True
    forced = capsys.readouterr()
    assert "forced refresh" in forced.out.lower()
