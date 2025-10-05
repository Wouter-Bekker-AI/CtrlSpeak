import difflib
import importlib
import importlib.util
import importlib
import sys
import json
import types
import copy
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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

    if "requests" not in sys.modules:
        requests_stub = types.SimpleNamespace()
        requests_stub.exceptions = types.SimpleNamespace(RequestException=RuntimeError)

        def _stubbed_request(*_args, **_kwargs):
            raise RuntimeError("requests stub invoked")

        requests_stub.get = _stubbed_request
        requests_stub.post = _stubbed_request
        sys.modules["requests"] = requests_stub
        sys.modules["requests.exceptions"] = requests_stub.exceptions

    modules = {}
    for name in [
        "utils.config_paths",
        "utils.memory_paths",
        "utils.memory_settings",
        "utils.metrics",
        "utils.image_store",
        "utils.vector_memory",
        "utils.memory_orchestrator",
        "tools.goose_tool",
    ]:
        modules[name] = importlib.reload(importlib.import_module(name))
    return modules


class DummyLLM:
    def __init__(
        self,
        planner_script: Optional[Iterable[str]] = None,
        tool_script: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        self.calls = []
        self.tool_calls = []
        self._planner_script = list(planner_script or [])
        self._tool_script = list(tool_script or [])
        self.system_prompt = ""

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

    def chat(self, messages, *, tools=None, tool_choice=None, stream=None):
        record = {
            "messages": copy.deepcopy(list(messages)),
            "tools": copy.deepcopy(list(tools or [])),
            "tool_choice": tool_choice,
        }
        self.tool_calls.append(record)
        if self._tool_script:
            return self._tool_script.pop(0)
        return {"message": {"role": "assistant", "content": "none", "tool_calls": []}}


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
    if orchestrator.tooling_enabled:
        assert llm.tool_calls, "expected the LLM chat endpoint to be invoked with tools"
        chat_payload = llm.tool_calls[0]
        assert chat_payload.get("tool_choice") == "auto"
        user_payloads = [
            str(msg.get("content", ""))
            for msg in chat_payload.get("messages", [])
            if isinstance(msg, dict) and msg.get("role") == "user"
        ]
        assert user_payloads, "expected the tool-enabled call to include the user prompt"
        assert any("hello world" in payload.lower() for payload in user_payloads)
        assert all("/no_think" not in payload.lower() for payload in user_payloads)
        guard_messages = [
            msg
            for msg in chat_payload.get("messages", [])
            if isinstance(msg, dict)
            and msg.get("role") == "system"
            and "goose_tool_query" in str(msg.get("content", ""))
        ]
        assert guard_messages, "expected Goose tool instructions to accompany the user prompt"
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


def test_orchestrator_profile_write_and_read(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]
    memory_paths = modules["utils.memory_paths"]

    identity_settings = {
        "store_vector_memory": True,
        "store_screenshots": False,
        "retrieval_top_k": 3,
        "retrieval_threshold": 0.2,
        "max_vector_items": 50,
        "vector_ttl_days": None,
        "pii_redaction": False,
        "profile_rerank": False,
    }

    llm = DummyLLM()
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "Tester",
        llm,
        memory_dir=tmp_path,
        metrics_path=tmp_path / "metrics.csv",
        identity_settings=identity_settings,
    )

    result = orchestrator.run_turn("My name is Alice.")
    orchestrator._persistence._queue.join()  # type: ignore[attr-defined]
    assert "I'll remember that your name is Alice" in result.response_text
    slot = orchestrator.vector_store.read_profile_slot(orchestrator.profile_user_id, "name")
    assert slot is not None
    assert slot["metadata"]["value"] == "Alice"

    export_path = memory_paths.get_bot_profile_export_path("Tester")
    assert export_path.exists()
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["identity"] == "Tester"
    assert payload["user_id"] == orchestrator.profile_user_id
    assert any(slot_entry.get("value") == "Alice" for slot_entry in payload.get("slots", []))

    read_result = orchestrator.run_turn("what's my name?")
    assert read_result.response_text.lower().startswith("your name is alice")

    orchestrator.close()


def test_orchestrator_profile_write_requires_confirmation(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]

    identity_settings = {
        "store_vector_memory": True,
        "store_screenshots": False,
        "retrieval_top_k": 3,
        "retrieval_threshold": 0.2,
        "max_vector_items": 50,
        "vector_ttl_days": None,
        "pii_redaction": False,
        "profile_rerank": False,
    }

    llm = DummyLLM()

    def fake_extract(self, text: str):
        return orchestrator_module.ProfileSlotExtraction("name", "Charlie", 0.5, "test")

    monkeypatch.setattr(
        orchestrator_module.MemoryOrchestrator,
        "_extract_profile_slot",
        fake_extract,
    )

    orchestrator = orchestrator_module.MemoryOrchestrator(
        "Tester",
        llm,
        memory_dir=tmp_path,
        metrics_path=tmp_path / "metrics.csv",
        identity_settings=identity_settings,
    )

    result = orchestrator.run_turn("call me Charlie maybe")
    orchestrator._persistence._queue.join()  # type: ignore[attr-defined]
    assert "could you confirm" in result.response_text.lower()
    slot = orchestrator.vector_store.read_profile_slot(orchestrator.profile_user_id, "name")
    assert slot is None

    orchestrator.close()
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
    assert not response_calls, "low-confidence extraction should not trigger a user-facing LLM call"


_SAMPLE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wwAAoMBgK8mBp0AAAAASUVORK5CYII="
)


def test_orchestrator_starts_with_fresh_session_history(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]
    memory_paths = modules["utils.memory_paths"]

    conversation_log = memory_paths.get_bot_conversation_log("HistoryTester")
    conversation_log.parent.mkdir(parents=True, exist_ok=True)
    with conversation_log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"role": "user", "content": "old session question"}) + "\n")
        handle.write(json.dumps({"role": "assistant", "content": "old session answer"}) + "\n")

    identity_settings = {
        "store_vector_memory": False,
        "retrieval_top_k": 2,
        "retrieval_threshold": 0.5,
        "max_vector_items": 5,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    llm = DummyLLM(["none"])
    metrics_path = tmp_path / "metrics_history.csv"
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "HistoryTester",
        llm,
        memory_dir=tmp_path,
        metrics_path=metrics_path,
        identity_settings=identity_settings,
    )

    result = orchestrator.run_turn("hello from a new session")
    time.sleep(0.1)
    orchestrator.close()

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
    final_call = response_calls[-1]
    assert "hello from a new session" in final_call["text"]
    history_entries = final_call["history"]
    assert all("old session question" not in str(entry.get("content", "")) for entry in history_entries)
    assert all("old session answer" not in str(entry.get("content", "")) for entry in history_entries)

    assert len(orchestrator.history) == 2
    assert orchestrator.history[0]["content"] == "hello from a new session"
    assert orchestrator.history[1]["content"].startswith("echo:hello from a new session")

    log_text = conversation_log.read_text(encoding="utf-8")
    assert "old session question" in log_text
    assert "hello from a new session" in log_text

    assert result.history_entries[0]["content"] == "hello from a new session"

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


def test_orchestrator_recovers_when_llm_only_thinks(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]

    class PlanOnlyLLM(DummyLLM):
        def __init__(self):
            tool_script = [
                {"message": {"role": "assistant", "content": "<think>Plan</think>", "tool_calls": []}},
                {
                    "message": {
                        "role": "assistant",
                        "content": "<think>Plan</think>\nAnswer: Completed.",
                        "tool_calls": [],
                    }
                },
            ]
            super().__init__(planner_script=["none"], tool_script=tool_script)

    identity_settings = {
        "store_vector_memory": False,
        "retrieval_top_k": 1,
        "retrieval_threshold": 0.0,
        "max_vector_items": 1,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    think_manager = ManageThinkAgent()
    metrics_path = tmp_path / "metrics_plan_only.csv"
    llm = PlanOnlyLLM()
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "PlanOnlyTester",
        llm,
        memory_dir=tmp_path,
        metrics_path=metrics_path,
        identity_settings=identity_settings,
        tooling_enabled=True,
        think_manager=think_manager,
    )

    result = orchestrator.run_turn("plan something", augmented_text="plan something /think")
    orchestrator.close()

    assert result.response_text == "Completed."
    assert result.think_hidden is True
    assert len(llm.tool_calls) >= 2
    reminder_payloads = []
    for call in llm.tool_calls[1:]:
        for message in call.get("messages", []):
            if isinstance(message, dict) and message.get("role") == "user":
                reminder_payloads.append(str(message.get("content", "")))
    assert any("Provide the final Answer" in payload for payload in reminder_payloads)


def test_orchestrator_leaves_goose_planning_to_llm(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]

    identity_settings = {
        "store_vector_memory": False,
        "retrieval_top_k": 2,
        "retrieval_threshold": 0.1,
        "max_vector_items": 5,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    llm = DummyLLM(["none"])
    metrics_path = tmp_path / "metrics_goose_hint.csv"
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "GooseHintTester",
        llm,
        memory_dir=tmp_path,
        metrics_path=metrics_path,
        identity_settings=identity_settings,
        tooling_enabled=True,
    )

    result = orchestrator.run_turn("please read the README.md file")
    time.sleep(0.1)
    orchestrator.close()

    trace_payload = json.loads(result.trace_path.read_text(encoding="utf-8"))
    assert trace_payload.get("tool_plan_summary") == "none"
    assert not trace_payload.get("tool_plan"), "tool planning should be left to the LLM"
    assert llm.tool_calls, "expected the LLM to receive the tool-enabled prompt"


def _make_goose_tool_call(prompt: str, **kwargs) -> Dict[str, Any]:
    payload = {"prompt": prompt}
    payload.update(kwargs)
    return {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "goose_tool_query",
                        "arguments": json.dumps(payload),
                    },
                }
            ],
        }
    }


def test_orchestrator_executes_goose_tool(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]

    captured_calls: list[Dict[str, Any]] = []

    def fake_goose_query(prompt: str, **kwargs):
        captured_calls.append({"prompt": prompt, "kwargs": kwargs})
        return '{"final": "done"}'

    monkeypatch.setattr(orchestrator_module, "goose_query", fake_goose_query)

    identity_settings = {
        "store_vector_memory": False,
        "retrieval_top_k": 1,
        "retrieval_threshold": 0.0,
        "max_vector_items": 5,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    llm = DummyLLM(tool_script=[
        _make_goose_tool_call("list all python files"),
        {"message": {"role": "assistant", "content": "Task complete."}},
    ])

    metrics_path = tmp_path / "metrics_goose_success.csv"
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "GooseExecTester",
        llm,
        memory_dir=tmp_path,
        metrics_path=metrics_path,
        identity_settings=identity_settings,
        tooling_enabled=True,
    )

    result = orchestrator.run_turn("list all python files")
    time.sleep(0.1)
    orchestrator.close()

    assert captured_calls, "expected goose_query to be invoked"
    assert captured_calls[0]["prompt"] == "list all python files"
    assert captured_calls[0]["kwargs"].get("stream") is True

    trace_payload = json.loads(result.trace_path.read_text(encoding="utf-8"))
    tool_results = trace_payload.get("tool_results", [])
    assert tool_results and tool_results[0]["success"] is True
    assert tool_results[0]["output"] == '{"final": "done"}'

    history_entries = result.history_entries
    assert [entry.get("role") for entry in history_entries] == [
        "user",
        "tool",
        "assistant",
    ]
    tool_entry = history_entries[1]
    assert tool_entry.get("name") == "goose_tool_query"
    assert tool_entry.get("content") == '{"final": "done"}'
    assert tool_entry.get("metadata", {}).get("success") is True


def test_orchestrator_reports_goose_failure(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]

    def failing_goose_query(prompt: str, **kwargs):
        raise RuntimeError("simulated goose failure")

    monkeypatch.setattr(orchestrator_module, "goose_query", failing_goose_query)

    identity_settings = {
        "store_vector_memory": False,
        "retrieval_top_k": 1,
        "retrieval_threshold": 0.0,
        "max_vector_items": 5,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    llm = DummyLLM(tool_script=[
        _make_goose_tool_call("read docs/tooling.md"),
        {"message": {"role": "assistant", "content": "Goose error handled."}},
    ])

    metrics_path = tmp_path / "metrics_goose_failure.csv"
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "GooseFailureTester",
        llm,
        memory_dir=tmp_path,
        metrics_path=metrics_path,
        identity_settings=identity_settings,
        tooling_enabled=True,
    )

    result = orchestrator.run_turn("read docs/tooling.md")
    time.sleep(0.1)
    orchestrator.close()

    trace_payload = json.loads(result.trace_path.read_text(encoding="utf-8"))
    tool_results = trace_payload.get("tool_results", [])
    assert tool_results and tool_results[0]["success"] is False
    assert "simulated goose failure" in tool_results[0]["message"]

    history_entries = result.history_entries
    assert any(entry.get("role") == "tool" for entry in history_entries)
    failure_entry = next(entry for entry in history_entries if entry.get("role") == "tool")
    assert "simulated goose failure" in failure_entry.get("content", "")
    assert failure_entry.get("metadata", {}).get("success") is False


def test_orchestrator_passes_stream_flag(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]

    captured_calls: list[Dict[str, Any]] = []

    def fake_goose_query(prompt: str, **kwargs):
        captured_calls.append({"prompt": prompt, "kwargs": kwargs})
        return '{"final": "done"}'

    monkeypatch.setattr(orchestrator_module, "goose_query", fake_goose_query)

    identity_settings = {
        "store_vector_memory": False,
        "retrieval_top_k": 1,
        "retrieval_threshold": 0.0,
        "max_vector_items": 5,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    llm = DummyLLM(tool_script=[
        _make_goose_tool_call("tail logs", stream=False),
        {"message": {"role": "assistant", "content": "Task complete."}},
    ])

    orchestrator = orchestrator_module.MemoryOrchestrator(
        "GooseStreamTester",
        llm,
        memory_dir=tmp_path,
        metrics_path=tmp_path / "metrics_goose_stream.csv",
        identity_settings=identity_settings,
        tooling_enabled=True,
    )

    orchestrator.run_turn("tail logs")
    time.sleep(0.1)
    orchestrator.close()

    assert captured_calls and captured_calls[0]["kwargs"].get("stream") is True
