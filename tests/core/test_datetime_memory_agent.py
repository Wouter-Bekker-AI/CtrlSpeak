from __future__ import annotations

import importlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.core_headless


def _prepare_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    utc_now: datetime,
    local_now: datetime,
):
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    data_home.mkdir(exist_ok=True)
    config_home.mkdir(exist_ok=True)

    monkeypatch.setenv("APPDATA", str(data_home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")

    modules = {}
    for name in [
        "utils.config_paths",
        "utils.memory_paths",
        "utils.memory_settings",
        "utils.vector_memory",
    ]:
        modules[name] = importlib.reload(importlib.import_module(name))

    dt_agent = importlib.reload(importlib.import_module("background_agents.datetime_memory_agent"))
    monkeypatch.setattr(dt_agent, "_utc_now", lambda: utc_now)
    monkeypatch.setattr(dt_agent, "_local_now", lambda: local_now)

    return modules, dt_agent


def test_refresh_datetime_memory_populates_and_skips_within_cooldown(tmp_path, monkeypatch, capsys):
    utc_now = datetime(2025, 1, 15, 6, tzinfo=timezone.utc)
    local_now = datetime(2025, 1, 15, 8, tzinfo=timezone(timedelta(hours=2)))
    modules, dt_agent = _prepare_environment(tmp_path, monkeypatch, utc_now, local_now)

    vector_memory = modules["utils.vector_memory"]

    assert dt_agent.refresh_datetime_memory("vision") is True

    store = vector_memory.VectorMemoryStore("vision")
    payload = store.collection.get(include=["metadatas", "documents"])
    documents = payload.get("documents") or []
    metadata = payload.get("metadatas") or []

    assert len(documents) == 1
    assert len(metadata) == 1
    meta = metadata[0]
    assert meta["category"] == "temporal_context"
    assert meta["utc_offset"].startswith("UTC+")

    tracker_path = dt_agent._tracker_path("vision")  # noqa: SLF001 - internal helper for tests
    tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
    assert tracker["snapshot_hash"] == meta["snapshot_hash"]

    assert dt_agent.refresh_datetime_memory("vision") is True
    captured = capsys.readouterr()
    assert "already current" in captured.out
    assert store.collection.count() == 1


def test_refresh_datetime_memory_detects_changes_and_force(tmp_path, monkeypatch, capsys):
    utc_now = datetime(2025, 1, 15, 6, tzinfo=timezone.utc)
    local_now = datetime(2025, 1, 15, 8, tzinfo=timezone(timedelta(hours=2)))
    modules, dt_agent = _prepare_environment(tmp_path, monkeypatch, utc_now, local_now)

    vector_memory = modules["utils.vector_memory"]

    assert dt_agent.refresh_datetime_memory("vision") is True

    tracker_path = dt_agent._tracker_path("vision")
    baseline = json.loads(tracker_path.read_text(encoding="utf-8"))

    new_utc = utc_now + timedelta(hours=25)
    new_local = local_now + timedelta(hours=25)
    monkeypatch.setattr(dt_agent, "_utc_now", lambda: new_utc)
    monkeypatch.setattr(dt_agent, "_local_now", lambda: new_local)

    assert dt_agent.refresh_datetime_memory("vision") is True
    captured = capsys.readouterr()
    assert "injecting temporal context" in captured.out.lower()

    updated = json.loads(tracker_path.read_text(encoding="utf-8"))
    assert updated["snapshot_hash"] != baseline["snapshot_hash"]

    store = vector_memory.VectorMemoryStore("vision")
    payload = store.collection.get(include=["metadatas"])
    metadata = payload.get("metadatas") or []
    assert metadata
    hashes = {meta.get("snapshot_hash") for meta in metadata if isinstance(meta, dict)}
    assert updated["snapshot_hash"] in hashes

    assert dt_agent.refresh_datetime_memory("vision", force=True, reason="test") is True
    forced = capsys.readouterr()
    assert "forced refresh" in forced.out.lower()
