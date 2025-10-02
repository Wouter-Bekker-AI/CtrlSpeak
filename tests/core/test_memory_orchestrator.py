import importlib
import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

import pytest

_MANAGE_THINK_PATH = Path(__file__).resolve().parents[2] / "background_agents" / "manage_think.py"
_MANAGE_THINK_SPEC = importlib.util.spec_from_file_location(
    "_manage_think_for_memory_tests", _MANAGE_THINK_PATH
)
assert _MANAGE_THINK_SPEC and _MANAGE_THINK_SPEC.loader, "Failed to load manage_think helper"
_MANAGE_THINK_MODULE = importlib.util.module_from_spec(_MANAGE_THINK_SPEC)
sys.modules[_MANAGE_THINK_SPEC.name] = _MANAGE_THINK_MODULE
_MANAGE_THINK_SPEC.loader.exec_module(_MANAGE_THINK_MODULE)
ManageThinkAgent = _MANAGE_THINK_MODULE.ManageThinkAgent

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
    def __init__(self, planner_script: Optional[Iterable[str]] = None) -> None:
        self.calls = []
        self._planner_script = list(planner_script or [])

    def query(self, user_text: str, history=None, content=None):
        history_copy = list(history or [])
        content_copy = list(content or [])
        self.calls.append({"text": user_text, "history": history_copy, "content": content_copy})
        planner_requested = any(
            isinstance(entry, dict)
            and entry.get("role") == "system"
            and isinstance(entry.get("content"), str)
            and "retrieval planner" in entry.get("content", "").lower()
            for entry in history_copy
        )
        if planner_requested and self._planner_script:
            return self._planner_script.pop(0)
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

    llm = DummyLLM(["none", "chat_history"])
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
    assert result1.vector_query_attempted is False
    assert result1.vector_query_result_count == 0
    assert result1.retrieval_plan == {
        "documentation": False,
        "chat_history": False,
        "date": False,
    }
    assert len(llm.calls) >= 3
    assert result2.vector_query_attempted is True
    assert result2.retrieval_plan.get("chat_history") is True
    response_calls = [
        call
        for call in llm.calls
        if not any(
            isinstance(entry, dict)
            and entry.get("role") == "system"
            and isinstance(entry.get("content"), str)
            and "retrieval planner" in entry.get("content", "").lower()
            for entry in call["history"]
        )
    ]
    assert response_calls, "expected the LLM to be invoked for a user-facing response"
    latest_response = response_calls[-1]
    assert any(
        entry.get("role") == "system" and "Relevant memory" in entry.get("content", "")
        for entry in latest_response["history"]
    )
    assert result1.retrieval_plan_used_llm is True

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

    llm = DummyLLM(["none"])
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
    response_calls = [
        call
        for call in llm.calls
        if not any(
            isinstance(entry, dict)
            and entry.get("role") == "system"
            and isinstance(entry.get("content"), str)
            and "retrieval planner" in entry.get("content", "").lower()
            for entry in call["history"]
        )
    ]
    assert response_calls, "expected at least one user-facing LLM invocation"
    payload = response_calls[-1]["content"]
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[1]["type"] == "image"
    assert isinstance(payload[1]["image"], str) and payload[1]["image"]

    history_entries = result.history_entries
    assert history_entries[0]["content"][1]["type"] == "image_file"

    conversation_log = memory_paths.get_bot_conversation_log("VisionTester")
    assert conversation_log.exists()
    assert _SAMPLE_PNG_B64 not in conversation_log.read_text(encoding="utf-8")


def test_orchestrator_scrubs_responses_before_persistence(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]
    memory_paths = modules["utils.memory_paths"]

    flush_event = threading.Event()
    original_history_flush = orchestrator_module.MemoryPersistenceWorker._flush_history

    def capture_history(self, entries):  # type: ignore[override]
        flush_event.set()
        return original_history_flush(self, entries)

    monkeypatch.setattr(
        orchestrator_module.MemoryPersistenceWorker,
        "_flush_history",
        capture_history,
    )

    captured_vector_docs: list[list[str]] = []

    def capture_vector(self, task):  # type: ignore[override]
        captured_vector_docs.append(list(task.vector_documents))
        return 0

    monkeypatch.setattr(
        orchestrator_module.MemoryPersistenceWorker,
        "_flush_vector_store",
        capture_vector,
    )

    class MarkdownLLM(DummyLLM):
        def __init__(self) -> None:
            super().__init__(planner_script=["chat_history"])

        def query(self, user_text: str, history=None, content=None):
            result = super().query(user_text, history=history, content=content)
            if result == "chat_history":
                return result
            return "*   **Name:** Wouter\n# Header"

    identity_settings = {
        "store_vector_memory": True,
        "store_screenshots": False,
        "retrieval_top_k": 1,
        "retrieval_threshold": 0.0,
        "max_vector_items": 5,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    llm = MarkdownLLM()
    metrics_path = tmp_path / "metrics_scrub.csv"
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "MarkdownTester",
        llm,
        memory_dir=tmp_path,
        metrics_path=metrics_path,
        identity_settings=identity_settings,
    )

    result = orchestrator.run_turn("tell me about myself")
    flush_event.wait(timeout=2.0)
    time.sleep(0.1)
    orchestrator.close()

    assistant_entries = [entry for entry in result.history_entries if entry.get("role") == "assistant"]
    assert assistant_entries, "expected an assistant entry in the turn history"
    assistant_text = assistant_entries[-1]["content"]
    assert "*" not in assistant_text
    assert "#" not in assistant_text

    conversation_log = memory_paths.get_bot_conversation_log("MarkdownTester")
    log_text = conversation_log.read_text(encoding="utf-8")
    assert "*" not in log_text
    assert "#" not in log_text

    assert captured_vector_docs, "expected vector persistence to run"
    scrubbed_vector = captured_vector_docs[-1][-1]
    assert "*" not in scrubbed_vector
    assert "#" not in scrubbed_vector


def test_orchestrator_prioritizes_documentation(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]

    identity_settings = {
        "store_vector_memory": True,
        "retrieval_top_k": 5,
        "retrieval_threshold": 0.95,
        "max_vector_items": 10,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    llm = DummyLLM(["none"])
    metrics_path = tmp_path / "metrics_docs.csv"
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "DocTester",
        llm,
        memory_dir=tmp_path,
        metrics_path=metrics_path,
        identity_settings=identity_settings,
    )

    orchestrator.vector_store.add_memories(
        [
            "Open Manage CtrlSpeak and press Chat with Bot to start the assistant session."
        ],
        metadata=[
            {
                "category": orchestrator_module.DOCUMENTATION_CATEGORY,
                "source": "docs/user_flow.md",
                "chunk": 1,
                "chunks": 1,
                "doc_hash": "deadbeef",
            }
        ],
        max_items=10,
    )

    result = orchestrator.run_turn("how do i use this program?")
    orchestrator.close()

    assert any(
        item.metadata.get("category") == orchestrator_module.DOCUMENTATION_CATEGORY
        for item in result.retrieved
    ), "expected documentation memory to be returned"
    assert result.vector_query_attempted is True
    assert result.vector_query_documentation_count >= 1

    assert llm.calls, "expected the LLM to be invoked"
    response_calls = [
        call
        for call in llm.calls
        if not any(
            isinstance(entry, dict)
            and entry.get("role") == "system"
            and isinstance(entry.get("content"), str)
            and "retrieval planner" in entry.get("content", "").lower()
            for entry in call["history"]
        )
    ]
    assert response_calls, "expected a user-facing LLM response"
    history_entry = next(
        (
            entry
            for entry in response_calls[-1]["history"]
            if entry.get("role") == "system"
        ),
        {},
    )
    assert "Documentation excerpts" in history_entry.get("content", "")
    assert result.retrieval_plan.get("documentation") is True
    assert result.retrieval_plan_used_llm is False


def test_orchestrator_returns_temporal_context(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]

    identity_settings = {
        "store_vector_memory": True,
        "retrieval_top_k": 5,
        "retrieval_threshold": 0.95,
        "max_vector_items": 10,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    llm = DummyLLM(["date"])
    metrics_path = tmp_path / "metrics_time.csv"
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "TimeTester",
        llm,
        memory_dir=tmp_path,
        metrics_path=metrics_path,
        identity_settings=identity_settings,
    )

    orchestrator.vector_store.add_memories(
        ["Today is Thursday, 2 October 2025. Local calendar date: 2025-10-02."],
        metadata=[
            {
                "category": orchestrator_module.TEMPORAL_CATEGORY,
                "kind": "current_date",
                "snapshot_hash": "hash123",
            }
        ],
        max_items=10,
    )

    result = orchestrator.run_turn("what is the current date?")
    orchestrator.close()

    assert any(
        item.metadata.get("category") == orchestrator_module.TEMPORAL_CATEGORY
        for item in result.retrieved
    ), "expected temporal context to be returned"
    assert result.vector_query_attempted is True
    assert result.vector_query_temporal_count >= 1

    assert llm.calls, "expected the LLM to be invoked"
    response_calls = [
        call
        for call in llm.calls
        if not any(
            isinstance(entry, dict)
            and entry.get("role") == "system"
            and isinstance(entry.get("content"), str)
            and "retrieval planner" in entry.get("content", "").lower()
            for entry in call["history"]
        )
    ]
    assert response_calls, "expected a user-facing LLM response"
    history_entry = next(
        (
            entry
            for entry in response_calls[-1]["history"]
            if entry.get("role") == "system"
        ),
        {},
    )
    assert "Temporal context" in history_entry.get("content", "")
    assert "Today is" in history_entry.get("content", "")
    assert result.retrieval_plan.get("date") is True
    assert result.retrieval_plan_used_llm is False


def test_orchestrator_skips_planner_when_heuristics_trigger(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]

    identity_settings = {
        "store_vector_memory": True,
        "retrieval_top_k": 5,
        "retrieval_threshold": 0.1,
        "max_vector_items": 10,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    llm = DummyLLM(["none"])
    metrics_path = tmp_path / "metrics_chat.csv"
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "HistoryTester",
        llm,
        memory_dir=tmp_path,
        metrics_path=metrics_path,
        identity_settings=identity_settings,
    )

    orchestrator.vector_store.add_memories(
        ["You told me your favourite title is Captain."],
        metadata=[{"category": "chat_history"}],
        max_items=10,
    )

    result = orchestrator.run_turn("what do you know about me?")
    orchestrator.close()

    assert result.retrieval_plan.get("chat_history") is True
    assert result.retrieval_plan_used_llm is False


def test_orchestrator_appends_no_think_for_einstein_planner(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]

    identity_settings = {
        "store_vector_memory": False,
        "retrieval_top_k": 5,
        "retrieval_threshold": 0.75,
        "max_vector_items": 10,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    llm = DummyLLM(["none"])
    metrics_path = tmp_path / "metrics_einstein.csv"
    think_agent = ManageThinkAgent()
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "einstein",
        llm,
        memory_dir=tmp_path,
        metrics_path=metrics_path,
        identity_settings=identity_settings,
        think_manager=think_agent,
    )

    result = orchestrator.run_turn("hi there", augmented_text="hi there /think")
    orchestrator.close()

    planner_calls = [
        call
        for call in llm.calls
        if any(
            isinstance(entry, dict)
            and entry.get("role") == "system"
            and isinstance(entry.get("content"), str)
            and "retrieval planner" in entry["content"].lower()
            for entry in call["history"]
        )
    ]
    assert planner_calls, "expected the planner prompt to invoke the LLM"
    planner_text = planner_calls[0]["text"]
    assert "hi there" in planner_text
    assert "/no_think" in planner_text
    assert "/think" not in planner_text.lower().replace("/no_think", "")
    assert result.retrieval_plan_used_llm is True


def test_orchestrator_hides_think_blocks_with_manager(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]
    memory_paths = modules["utils.memory_paths"]

    class ThinkingLLM(DummyLLM):
        def __init__(self):
            super().__init__(planner_script=["none"])

        def query(self, user_text: str, history=None, content=None):
            result = super().query(user_text, history=history, content=content)
            if result.startswith("echo:"):
                return "<think>Plan the steps carefully.</think>\nAnswer: Provide guidance."
            return result

    identity_settings = {
        "store_vector_memory": True,
        "retrieval_top_k": 1,
        "retrieval_threshold": 0.0,
        "max_vector_items": 5,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    think_manager = ManageThinkAgent()
    metrics_path = tmp_path / "metrics_think.csv"
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "ThinkTester",
        ThinkingLLM(),
        memory_dir=tmp_path,
        metrics_path=metrics_path,
        identity_settings=identity_settings,
        think_manager=think_manager,
    )

    result = orchestrator.run_turn("please think aloud about this request")
    time.sleep(0.1)
    orchestrator.close()

    assert result.think_hidden is True
    assert result.think_placeholder == "Thinking..."
    assert result.response_text == "Provide guidance."
    assert "<think>" not in result.response_text
    assert "<think>" in result.raw_response_text

    conversation_log = memory_paths.get_bot_conversation_log("ThinkTester")
    assert conversation_log.exists()
    log_text = conversation_log.read_text(encoding="utf-8")
    assert "<think>" not in log_text
