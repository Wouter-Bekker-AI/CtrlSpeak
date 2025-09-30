import importlib
import threading
import importlib
import threading
import time
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
        "utils.memory_settings",
        "utils.metrics",
        "utils.image_store",
        "utils.vector_memory",
        "utils.memory_orchestrator",
    ]:
        modules[name] = importlib.reload(importlib.import_module(name))
    return modules


class DummyLLM:
    def __init__(self) -> None:
        self.calls = []

    def query(self, user_text: str, history=None, content=None):
        history_copy = list(history or [])
        content_copy = list(content or [])
        self.calls.append({"text": user_text, "history": history_copy, "content": content_copy})
        return f"echo:{user_text}"


def test_orchestrator_persists_and_retrieves(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]
    memory_paths = modules["utils.memory_paths"]

    flush_event = threading.Event()
    original_flush = orchestrator_module.MemoryPersistenceWorker._flush_history

    def tracking_flush(self, entries):  # type: ignore[override]
        flush_event.set()
        time.sleep(0.05)
        return original_flush(self, entries)

    monkeypatch.setattr(orchestrator_module.MemoryPersistenceWorker, "_flush_history", tracking_flush)

    identity_settings = {
        "store_vector_memory": True,
        "store_screenshots": True,
        "retrieval_top_k": 5,
        "retrieval_threshold": 0.1,
        "max_vector_items": 10,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    llm = DummyLLM()
    metrics_path = tmp_path / "metrics.csv"
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "Tester",
        llm,
        memory_dir=tmp_path,
        metrics_path=metrics_path,
        identity_settings=identity_settings,
    )

    start = time.perf_counter()
    result1 = orchestrator.run_turn("hello world")
    duration = time.perf_counter() - start
    assert duration < 0.05
    assert not flush_event.is_set()
    flush_event.wait(timeout=2.0)
    assert flush_event.is_set()

    time.sleep(0.1)
    result2 = orchestrator.run_turn("hello again")
    time.sleep(0.1)

    orchestrator.close()

    assert result1.response_text == "echo:hello world"
    assert result2.response_text == "echo:hello again"
    assert len(llm.calls) >= 2
    assert any(
        entry.get("role") == "system" and "Relevant memory" in entry.get("content", "")
        for entry in llm.calls[1]["history"]
    )

    conversation_log = memory_paths.get_bot_conversation_log("Tester")
    assert conversation_log.exists()
    metrics_path = Path(metrics_path)
    assert metrics_path.exists()


_SAMPLE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wwAAoMBgK8mBp0AAAAASUVORK5CYII="
)


def test_orchestrator_attaches_identity_image(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]
    memory_paths = modules["utils.memory_paths"]
    image_store = modules["utils.image_store"]

    identity_settings = {
        "store_vector_memory": False,
        "retrieval_top_k": 2,
        "retrieval_threshold": 0.5,
        "max_vector_items": 5,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    assert image_store.write_identity_image_from_base64("VisionTester", _SAMPLE_PNG_B64, source="screen") is not None

    llm = DummyLLM()
    metrics_path = tmp_path / "metrics_vision.csv"
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "VisionTester",
        llm,
        memory_dir=tmp_path,
        metrics_path=metrics_path,
        identity_settings=identity_settings,
    )

    result = orchestrator.run_turn("please describe the image on my screen")
    time.sleep(0.1)
    orchestrator.close()

    assert llm.calls, "expected the LLM to be invoked"
    payload = llm.calls[0]["content"]
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[1]["type"] == "image"
    assert isinstance(payload[1]["image"], str) and payload[1]["image"]

    history_entries = result.history_entries
    assert history_entries[0]["content"][1]["type"] == "image_file"

    conversation_log = memory_paths.get_bot_conversation_log("VisionTester")
    assert conversation_log.exists()
    assert _SAMPLE_PNG_B64 not in conversation_log.read_text(encoding="utf-8")
