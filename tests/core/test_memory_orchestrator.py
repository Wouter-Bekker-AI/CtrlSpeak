import difflib
import importlib
import importlib.util
import sys
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
        assert llm.tool_calls, "expected the tool router to be consulted"
        probe_payload = llm.tool_calls[0]
        assert probe_payload.get("tool_choice") == "auto"
        assert any(
            isinstance(msg, dict)
            and msg.get("role") == "user"
            and "/no_think" in str(msg.get("content", ""))
            for msg in probe_payload.get("messages", [])
        )
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


def test_orchestrator_reads_file_via_workspace_tools(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]

    workspace_module = importlib.reload(importlib.import_module("tools.workspace"))
    original_root = workspace_module.WORKSPACE_ROOT
    workspace_module.set_workspace_root(tmp_path)

    desktop = tmp_path / "Desktop"
    desktop.mkdir()
    target = desktop / "linux commands.txt"
    target.write_text("ls -la\npwd\n", encoding="utf-8")

    preflight = workspace_module.search_files("linux commands.txt")
    assert preflight.get("paths"), "workspace search should locate the staged file"
    orchestrator_view = orchestrator_module.workspace.search_files("linux commands.txt")
    assert orchestrator_view.get("paths"), "orchestrator workspace view should locate the staged file"

    identity_settings = {
        "store_vector_memory": False,
        "retrieval_top_k": 1,
        "retrieval_threshold": 0.0,
        "max_vector_items": 5,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    tool_events: list[str] = []

    try:
        tool_script = [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "workspace_read_file",
                                "arguments": {
                                    "path": "linux commands.txt",
                                    "location_hint": "Desktop",
                                },
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "Answer: ls -la\npwd"}},
        ]
        llm = DummyLLM(tool_script=tool_script)
        metrics_path = tmp_path / "metrics_tools.csv"
        orchestrator = orchestrator_module.MemoryOrchestrator(
            "Einstein",
            llm,
            memory_dir=tmp_path,
            metrics_path=metrics_path,
            identity_settings=identity_settings,
            tooling_enabled=True,
            tool_logger=tool_events.append,
        )

        user_prompt = "Hi please tell me what is in the linux commands.txt file on my Desktop"
        result = orchestrator.run_turn(user_prompt)
        time.sleep(0.1)
        orchestrator.close()
    finally:
        workspace_module.set_workspace_root(original_root)

    user_calls = [call for call in llm.calls if call.get("text") == user_prompt]
    assert not user_calls, "LLM should not receive the user prompt when tooling satisfies the request"
    assert llm.tool_calls and llm.tool_calls[0].get("tool_choice") == "required"
    assert "ls -la" in result.response_text, f"tool events: {tool_events}"
    assert any("search_files" in entry for entry in tool_events)
    assert any("read_file" in entry for entry in tool_events)


def test_orchestrator_reads_external_desktop_file(tmp_path, tmp_path_factory, monkeypatch):
    fake_home = tmp_path_factory.mktemp("fake_home")
    desktop = fake_home / "Desktop"
    desktop.mkdir()
    target = desktop / "linux commands.txt"
    target.write_text("dir\n", encoding="utf-8")

    if "requests" not in sys.modules:
        requests_stub = types.SimpleNamespace()
        requests_stub.exceptions = types.SimpleNamespace(RequestException=RuntimeError)
        requests_stub.get = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("requests stub invoked"))
        requests_stub.post = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("requests stub invoked"))
        sys.modules["requests"] = requests_stub
        sys.modules["requests.exceptions"] = requests_stub.exceptions

    monkeypatch.setattr(
        Path,
        "home",
        classmethod(lambda cls: fake_home),
    )

    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]

    workspace_module = importlib.reload(importlib.import_module("tools.workspace"))
    original_root = workspace_module.WORKSPACE_ROOT
    original_extras = workspace_module.list_additional_roots()
    workspace_module.set_workspace_root(tmp_path)
    workspace_module.set_additional_allowed_roots(original_extras)

    identity_settings = {
        "store_vector_memory": False,
        "retrieval_top_k": 1,
        "retrieval_threshold": 0.0,
        "max_vector_items": 5,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    tool_events: list[str] = []

    try:
        workspace_module.register_allowed_root(fake_home)
        workspace_module.register_allowed_root(desktop)

        tool_script = [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "workspace_read_file",
                                "arguments": {
                                    "path": "linux commands.txt",
                                    "location_hint": "Desktop",
                                },
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "Answer: dir"}},
        ]
        llm = DummyLLM(tool_script=tool_script)
        metrics_path = tmp_path / "metrics_external.csv"
        orchestrator = orchestrator_module.MemoryOrchestrator(
            "Einstein",
            llm,
            memory_dir=tmp_path,
            metrics_path=metrics_path,
            identity_settings=identity_settings,
            tooling_enabled=True,
            tool_logger=tool_events.append,
        )

        user_prompt = "Please tell me the content of linux commands.txt on my Desktop."
        result = orchestrator.run_turn(user_prompt)
        time.sleep(0.1)
        orchestrator.close()
    finally:
        workspace_module.set_workspace_root(original_root)
        workspace_module.set_additional_allowed_roots(original_extras)

    assert "dir" in result.response_text
    assert not any(call.get("text") == user_prompt for call in llm.calls)
    assert llm.tool_calls and llm.tool_calls[0].get("tool_choice") == "required"
    assert any("read_file" in entry for entry in tool_events)
    assert any("Expanded candidate path" in entry for entry in tool_events)


def test_tool_plan_apply_text_patch(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]
    workspace_module = importlib.reload(importlib.import_module("tools.workspace"))

    workspace_module.set_workspace_root(tmp_path)
    target = tmp_path / "notes.txt"
    target.write_text("alpha\n", encoding="utf-8")
    original_sha = workspace_module.read_file("notes.txt")["sha256"]

    diff = "\n".join(
        difflib.unified_diff(
            ["alpha\n"],
            ["alpha\n", "beta\n"],
            fromfile="a/notes.txt",
            tofile="b/notes.txt",
            lineterm="",
        )
    ) + "\n"

    llm = DummyLLM(["none"])
    orchestrator = orchestrator_module.MemoryOrchestrator(
        "PatchTester",
        llm,
        memory_dir=tmp_path,
        metrics_path=tmp_path / "metrics_patch.csv",
        identity_settings={
            "store_vector_memory": False,
            "store_screenshots": False,
            "retrieval_top_k": 2,
            "retrieval_threshold": 0.0,
            "max_vector_items": 5,
            "vector_ttl_days": None,
            "pii_redaction": False,
        },
        tooling_enabled=True,
    )

    action = orchestrator_module.ToolAction(
        kind="apply_text_patch",
        description="Append beta line",
        candidate_path="notes.txt",
        source="llm",
        parameters={"diff": diff, "expect_sha256": original_sha},
    )

    results = orchestrator._execute_tool_plan([action])
    orchestrator.close()
    workspace_module.set_workspace_root(tmp_path)

    assert results and results[0]["success"]
    updated = target.read_text(encoding="utf-8")
    assert updated == "alpha\nbeta\n"


def test_llm_tool_flow_applies_patch(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]
    workspace_module = importlib.reload(importlib.import_module("tools.workspace"))

    original_root = workspace_module.WORKSPACE_ROOT
    original_extras = workspace_module.list_additional_roots()
    workspace_module.set_workspace_root(tmp_path)
    workspace_module.set_additional_allowed_roots([])

    try:
        target = tmp_path / "notes.txt"
        target.write_text("alpha\n", encoding="utf-8")
        current_sha = workspace_module.read_file("notes.txt")["sha256"]

        diff = "\n".join(
            difflib.unified_diff(
                ["alpha\n"],
                ["alpha\n", "beta\n"],
                fromfile="a/notes.txt",
                tofile="b/notes.txt",
                lineterm="",
            )
        ) + "\n"

        tool_script = [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "workspace_apply_text_patch",
                                "arguments": {
                                    "path": "notes.txt",
                                    "diff": diff,
                                    "expect_sha256": current_sha,
                                },
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "Answer: Done."}},
        ]

        llm = DummyLLM(tool_script=tool_script)
        orchestrator = orchestrator_module.MemoryOrchestrator(
            "Einstein",
            llm,
            memory_dir=tmp_path,
            metrics_path=tmp_path / "metrics_tool_flow.csv",
            identity_settings={
                "store_vector_memory": False,
                "store_screenshots": False,
                "retrieval_top_k": 2,
                "retrieval_threshold": 0.0,
                "max_vector_items": 5,
                "vector_ttl_days": None,
                "pii_redaction": False,
            },
            tooling_enabled=True,
        )

        result = orchestrator.run_turn("please append beta to notes.txt")
        orchestrator.close()

        assert "beta" in target.read_text(encoding="utf-8")
        assert result.response_text == "Answer: Done."
        assert len(llm.tool_calls) >= 2, "expected LLM to be invoked again after tool execution"
        first_payload = llm.tool_calls[0]
        assert first_payload.get("tool_choice") == "auto"
        first_messages = first_payload.get("messages", [])
        assert any(
            isinstance(msg, dict)
            and msg.get("role") == "user"
            and "/no_think" in str(msg.get("content", ""))
            for msg in first_messages
        )
    finally:
        workspace_module.set_workspace_root(original_root)
        workspace_module.set_additional_allowed_roots(original_extras)


def test_tool_probe_includes_recent_snapshot(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]
    workspace_module = importlib.reload(importlib.import_module("tools.workspace"))

    original_root = workspace_module.WORKSPACE_ROOT
    original_extras = workspace_module.list_additional_roots()
    workspace_module.set_workspace_root(tmp_path)
    workspace_module.set_additional_allowed_roots(original_extras)

    try:
        file_path = tmp_path / "notes.txt"
        file_path.write_text("alpha\n", encoding="utf-8")

        llm = DummyLLM()
        orchestrator = orchestrator_module.MemoryOrchestrator(
            "SnapshotTester",
            llm,
            memory_dir=tmp_path,
            metrics_path=tmp_path / "metrics_snapshot.csv",
            identity_settings={
                "store_vector_memory": False,
                "store_screenshots": False,
                "retrieval_top_k": 2,
                "retrieval_threshold": 0.0,
                "max_vector_items": 5,
                "vector_ttl_days": None,
                "pii_redaction": False,
            },
            tooling_enabled=True,
        )

        action = orchestrator_module.ToolAction(
            kind="read_file",
            description="Inspect notes",
            candidate_path="notes.txt",
            search_term="notes.txt",
            location_hint=None,
            source="langgraph",
            parameters={},
        )

        orchestrator._execute_tool_plan([action])

        state = {"user_text": "please append beta to that file"}
        orchestrator._probe_llm_for_tool_actions(state)

        orchestrator.close()

        assert llm.tool_calls, "expected tool probe to contact the LLM"
        payload = llm.tool_calls[-1]
        messages = payload.get("messages", [])
        assert payload.get("tool_choice") == "required"
        assert any(
            isinstance(msg, dict)
            and "Most recent workspace file snapshots" in str(msg.get("content", ""))
            for msg in messages
        )
        assert any(
            isinstance(msg, dict)
            and msg.get("role") == "user"
            and "/no_think" in str(msg.get("content", ""))
            for msg in messages
        )
    finally:
        workspace_module.set_workspace_root(original_root)
        workspace_module.set_additional_allowed_roots(original_extras)


def test_llm_tool_call_reuses_heuristic_location_hint(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]
    workspace_module = importlib.reload(importlib.import_module("tools.workspace"))

    original_root = workspace_module.WORKSPACE_ROOT
    original_extras = workspace_module.list_additional_roots()
    workspace_module.set_workspace_root(tmp_path)
    workspace_module.set_additional_allowed_roots(original_extras)

    fake_home = tmp_path / "home"
    desktop = fake_home / "Desktop"
    desktop.mkdir(parents=True)
    target = desktop / "casings.txt"
    target.write_text("hello\n", encoding="utf-8")

    tool_events: list[str] = []

    monkeypatch.setattr(orchestrator_module.Path, "home", staticmethod(lambda: fake_home))
    workspace_module.register_allowed_root(desktop)

    orchestrator = None

    try:
        llm = DummyLLM()
        orchestrator = orchestrator_module.MemoryOrchestrator(
            "Einstein",
            llm,
            memory_dir=tmp_path,
            metrics_path=tmp_path / "metrics_llm_location.csv",
            identity_settings={
                "store_vector_memory": False,
                "store_screenshots": False,
                "retrieval_top_k": 2,
                "retrieval_threshold": 0.0,
                "max_vector_items": 5,
                "vector_ttl_days": None,
                "pii_redaction": False,
            },
            tooling_enabled=True,
            tool_logger=tool_events.append,
        )

        tool_calls = [
            {
                "function": {
                    "name": "workspace_read_file",
                    "arguments": {"path": "casings.txt"},
                }
            }
        ]
        actions = orchestrator._actions_from_tool_calls(
            tool_calls,
            user_text="please show me casings.txt on my Desktop",
        )

        assert actions, "expected tool actions to be generated"
        action = actions[0]
        assert action.location_hint, "location hint should be inherited from heuristics"

        results = orchestrator._execute_tool_plan(actions)
        assert results and results[0]["success"], "read should succeed once path is expanded"
        assert any("Expanded candidate path" in event for event in tool_events)
    finally:
        if orchestrator is not None:
            orchestrator.close()
        workspace_module.set_workspace_root(original_root)
        workspace_module.set_additional_allowed_roots(original_extras)


def test_tool_router_context_prefers_most_recent_file_for_pronouns(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]
    workspace_module = importlib.reload(importlib.import_module("tools.workspace"))

    original_root = workspace_module.WORKSPACE_ROOT
    original_extras = workspace_module.list_additional_roots()
    workspace_module.set_workspace_root(tmp_path)
    workspace_module.set_additional_allowed_roots([])

    (tmp_path / "first.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "second.txt").write_text("bravo\n", encoding="utf-8")

    identity_settings = {
        "store_vector_memory": False,
        "store_screenshots": False,
        "retrieval_top_k": 2,
        "retrieval_threshold": 0.0,
        "max_vector_items": 5,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    orchestrator = None
    try:
        orchestrator = orchestrator_module.MemoryOrchestrator(
            "Einstein",
            DummyLLM(),
            memory_dir=tmp_path,
            metrics_path=tmp_path / "metrics_recent.csv",
            identity_settings=identity_settings,
            tooling_enabled=True,
        )

        read_first = orchestrator_module.ToolAction(
            kind="read_file",
            description="Inspect first",
            candidate_path="first.txt",
            search_term="first.txt",
        )
        read_second = orchestrator_module.ToolAction(
            kind="read_file",
            description="Inspect second",
            candidate_path="second.txt",
            search_term="second.txt",
        )

        orchestrator._execute_tool_plan([read_first])
        orchestrator._execute_tool_plan([read_second])

        context = orchestrator._build_tool_router_context("please append beta to that file")
        assert context is not None
        assert "second.txt" in context
        assert "first.txt" not in context
    finally:
        if orchestrator is not None:
            orchestrator.close()
        workspace_module.set_workspace_root(original_root)
        workspace_module.set_additional_allowed_roots(original_extras)


def test_directory_listing_heuristic_detects_desktop_request(tmp_path, monkeypatch):
    modules = _prepare(tmp_path, monkeypatch)
    orchestrator_module = modules["utils.memory_orchestrator"]
    workspace_module = importlib.reload(importlib.import_module("tools.workspace"))

    original_root = workspace_module.WORKSPACE_ROOT
    original_extras = workspace_module.list_additional_roots()
    workspace_module.set_workspace_root(tmp_path)
    workspace_module.set_additional_allowed_roots([])

    identity_settings = {
        "store_vector_memory": False,
        "store_screenshots": False,
        "retrieval_top_k": 2,
        "retrieval_threshold": 0.0,
        "max_vector_items": 5,
        "vector_ttl_days": None,
        "pii_redaction": False,
    }

    orchestrator = None
    try:
        orchestrator = orchestrator_module.MemoryOrchestrator(
            "Einstein",
            DummyLLM(),
            memory_dir=tmp_path,
            metrics_path=tmp_path / "metrics_listdir.csv",
            identity_settings=identity_settings,
            tooling_enabled=True,
        )

        actions = orchestrator._plan_tool_actions(
            {"user_text": "list all the text files on my Desktop"}
        )
        assert actions, "expected heuristics to return a list_directory action"
        action = actions[0]
        assert action.kind == "list_directory"
        assert action.source == "langgraph"
        assert action.parameters.get("extensions") == ["txt"]
        assert action.candidate_path
    finally:
        if orchestrator is not None:
            orchestrator.close()
        workspace_module.set_workspace_root(original_root)
        workspace_module.set_additional_allowed_roots(original_extras)
