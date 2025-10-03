# -*- coding: utf-8 -*-
"""LangGraph-based memory orchestrator for SocialRobot."""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from background_agents.manage_think import ManageThinkAgent

from utils.config_paths import get_logger
from utils.image_store import is_image_request, load_identity_image
from utils.io_atomic import atomic_append_lines, atomic_write_text
from utils.metrics import MetricsRecorder
from utils.memory_paths import get_bot_conversation_log, get_bot_traces_dir
from utils.memory_settings import load_identity_settings
from utils.vector_memory import RetrievedMemory, VectorMemoryStore
from tools.message_management import force_plaintext, requires_force_plaintext
from tools import workspace


CONVERSATION_MAX_BYTES = 10 * 1024 * 1024
CONVERSATION_KEEP = 5
PERSIST_START_DELAY_SECONDS = 0.05

DOCUMENTATION_CATEGORY = "documentation"
TEMPORAL_CATEGORY = "temporal_context"
_REASONING_TAG_PATTERN = re.compile(r"/(?:no_)?think\b", re.IGNORECASE)
_DOCUMENTATION_INTENT_PHRASES = (
    "how do i",
    "how can i",
    "how to",
    "help me",
    "help with",
    "help on",
    "guide",
    "manual",
    "documentation",
    "docs",
    "instructions",
    "use this program",
    "use this app",
    "use ctrlspeak",
    "what can i do",
    "what should i do",
    "what does this program",
    "what is ctrlspeak",
)

_TEMPORAL_INTENT_PHRASES = (
    "what is the date",
    "what's the date",
    "current date",
    "today",
    "what day is it",
    "which day is it",
    "what is today",
    "tell me the date",
    "what is the time",
    "current time",
    "time right now",
    "timezone",
    "time zone",
    "utc offset",
)

_CHAT_HISTORY_INTENT_PHRASES = (
    "what do you know about me",
    "what's my name",
    "what is my name",
    "do you remember",
    "remember what i",
    "what did i say",
    "what was i talking",
    "who am i",
    "about our conversation",
    "previous conversation",
    "earlier we",
    "chat history",
    "conversation history",
    "our history",
    "history with me",
)

TOOL_RESPONSE_CHAR_LIMIT = 4000

_TOOL_ROUTER_PROMPT = (
    "You are the CtrlSpeak tool router. Decide whether the request requires calling a workspace tool. "
    "Only call a tool when you must read or modify files. If no tool is needed, respond with the single word 'none'."
)

_WORKSPACE_TOOL_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "workspace_read_file",
            "description": (
                "Read a file from the CtrlSpeak workspace or allowed host directories. "
                "Use this when you must inspect the exact file contents before answering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the target file.",
                    },
                    "location_hint": {
                        "type": "string",
                        "description": (
                            "Optional location hint (for example 'Desktop' or 'AppData/Logs') to guide path expansion."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "Short summary for logging the read operation.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workspace_apply_text_patch",
            "description": (
                "Apply a unified diff to a UTF-8 text file. Always provide the diff you want to apply and, when available, "
                "the SHA-256 of the file before your edit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file that should be patched.",
                    },
                    "diff": {
                        "type": "string",
                        "description": "Unified diff covering only the intended change.",
                    },
                    "expect_sha256": {
                        "type": ["string", "null"],
                        "description": "SHA-256 digest of the file before applying the diff, when known.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "One-line summary of the modification for status messages.",
                    },
                },
                "required": ["path", "diff"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workspace_list_directory",
            "description": (
                "List files within a directory under the CtrlSpeak workspace or allowed host directories, "
                "optionally filtering by pattern or extension."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the directory to inspect.",
                    },
                    "pattern": {
                        "type": ["string", "null"],
                        "description": "Optional glob-style pattern (e.g., '*.txt') to filter entries.",
                    },
                    "extensions": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Optional list of file extensions to include (without leading dots).",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to include files from subdirectories recursively.",
                        "default": False,
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": "Maximum number of entries to return (defaults to 200).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short summary for logging the listing operation.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
]

_ASSESSMENT_INSTRUCTIONS = (
    "You are the CtrlSpeak retrieval planner."
    " Decide whether the assistant needs extra context before answering a user."
    " You can request any of the following resources:"
    " documentation (instruction manuals and software guidance),"
    " chat_history (conversation memories about the current user),"
    " and date (current date, time, and timezone snapshot)."
    " Only request documentation for programming, software, or CtrlSpeak usage questions."
    " Only request chat_history when the query references previous conversation or personal details."
    " Only request date when the user asks about the current date, time, or timezone."
    " Respond ONLY with one of these options in lowercase:"
    " 'none', 'documentation', 'chat_history', 'date', or a comma-separated combination such as"
    " 'documentation,chat_history'."
    " Do not add explanations or punctuation beyond commas."
)


logger = get_logger(__name__)


class _TurnState(TypedDict, total=False):
    correlation_id: str
    user_text: str
    augmented_text: str
    content_blocks: Optional[List[dict]]
    history: List[dict]
    retrieved: List[Dict[str, Any]]
    response_text: str
    raw_response_text: str
    metrics: Dict[str, float]
    errors: List[Dict[str, Any]]
    vision_metadata: Optional[Dict[str, Any]]
    vision_attached: bool
    retrieval_plan: Dict[str, bool]
    retrieval_plan_summary: str
    vector_query_planned: bool
    think_hidden: bool
    think_placeholder: str
    hidden_think: str


@dataclass
class TurnResult:
    correlation_id: str
    response_text: str
    raw_response_text: str
    history_entries: List[dict]
    retrieved: List[RetrievedMemory]
    trace_path: Path
    metrics: Dict[str, float]
    vector_query_attempted: bool
    vector_query_result_count: int
    vector_query_documentation_count: int
    vector_query_temporal_count: int
    retrieval_plan: Dict[str, bool]
    retrieval_plan_used_llm: bool = False
    think_hidden: bool = False
    think_placeholder: Optional[str] = None
    hidden_think: Optional[str] = None


class PersistenceTask:
    def __init__(
        self,
        *,
        correlation_id: str,
        entries: List[dict],
        vector_documents: List[str],
        vector_metadata: List[Dict[str, Any]],
        settings: Dict[str, Any],
    ) -> None:
        self.correlation_id = correlation_id
        self.entries = entries
        self.vector_documents = vector_documents
        self.vector_metadata = vector_metadata
        self.settings = settings
        self.retries = 0


@dataclass
@dataclass
class ToolAction:
    kind: str
    description: str
    candidate_path: Optional[str] = None
    search_term: Optional[str] = None
    location_hint: Optional[str] = None
    source: str = "langgraph"
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.parameters is None:
            self.parameters = {}


class MemoryPersistenceWorker:
    """Background worker that flushes conversation logs and vector store."""

    def __init__(
        self,
        identity: str,
        vector_store: VectorMemoryStore,
        metrics: MetricsRecorder,
        *,
        conversation_log: Path,
    ) -> None:
        self.identity = identity
        self.vector_store = vector_store
        self.metrics = metrics
        self.conversation_log = conversation_log
        self._queue: "queue.Queue[Optional[PersistenceTask]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop = threading.Event()
        self._thread.start()

    def enqueue(self, task: PersistenceTask) -> None:
        self._queue.put(task)

    def close(self, timeout: float = 5.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                break
            if PERSIST_START_DELAY_SECONDS > 0:
                time.sleep(PERSIST_START_DELAY_SECONDS)
            start = time.perf_counter()
            metrics: Dict[str, float] = {}
            try:
                self._flush_history(task.entries)
                evictions = self._flush_vector_store(task)
                metrics["evictions"] = float(evictions)
                latency_ms = (time.perf_counter() - start) * 1000.0
                metrics["persist_latency_ms"] = round(latency_ms, 2)
                self.metrics.record(task.correlation_id, metrics)
            except Exception as exc:
                if task.retries < 1:
                    task.retries += 1
                    self._queue.put(task)
                else:
                    metrics["persist_error"] = 1.0
                    self.metrics.record(task.correlation_id, metrics)
            finally:
                self._queue.task_done()

    def _flush_history(self, entries: List[dict]) -> None:
        if not entries:
            return
        lines = [json.dumps(entry, ensure_ascii=False) for entry in entries]
        atomic_append_lines(
            self.conversation_log,
            lines,
            max_bytes=CONVERSATION_MAX_BYTES,
            keep=CONVERSATION_KEEP,
        )

    def _flush_vector_store(self, task: PersistenceTask) -> int:
        settings = task.settings
        if not settings.get("store_vector_memory", True):
            return 0
        documents = task.vector_documents
        if not documents:
            return 0
        ttl_days = settings.get("vector_ttl_days")
        try:
            ttl_value = None if ttl_days is None else float(ttl_days)
        except Exception:
            ttl_value = None
        evicted = 0
        try:
            evicted += self.vector_store.purge_expired()
        except Exception:
            pass
        result = self.vector_store.add_memories(
            documents,
            metadata=task.vector_metadata,
            max_items=int(settings.get("max_vector_items", 5000)),
            ttl_days=ttl_value,
            pii_redaction=bool(settings.get("pii_redaction", False)),
        )
        evicted += int(result.get("evicted", 0))
        return evicted


class MemoryOrchestrator:
    """Orchestrate conversation turns via LangGraph."""

    def __init__(
        self,
        identity: str,
        llm_client,
        *,
        memory_dir: Path,
        metrics_path: Path,
        identity_settings: Optional[Dict[str, Any]] = None,
        think_manager: Optional[ManageThinkAgent] = None,
        tooling_enabled: bool = False,
        tool_logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.identity = identity
        self.llm_client = llm_client
        self.memory_dir = memory_dir
        self.identity_settings = (
            dict(identity_settings)
            if identity_settings is not None
            else load_identity_settings(identity)
        )
        self.think_manager = think_manager
        self.conversation_log = get_bot_conversation_log(identity)
        self.traces_dir = get_bot_traces_dir(identity)
        self.metrics = MetricsRecorder(metrics_path)
        self.vector_store = VectorMemoryStore(identity)
        self.history: List[dict] = self._load_history()
        self.tooling_enabled = bool(tooling_enabled)
        self._custom_tool_logger: Optional[Callable[[str], None]] = tool_logger
        self._system_info: Optional[Dict[str, Any]] = None
        self._recent_file_reads: List[Dict[str, Any]] = []
        if self.tooling_enabled:
            try:
                self._system_info = workspace.get_system_info()
                self._emit_tool_event(
                    "Workspace system info resolved: "
                    f"platform={self._system_info.get('platform')} "
                    f"is_windows={self._system_info.get('is_windows')}"
                )
            except Exception as exc:
                self._emit_tool_event(f"Failed to resolve workspace system info: {exc}")
        self._graph = self._build_graph()
        self._persistence = MemoryPersistenceWorker(
            identity,
            self.vector_store,
            self.metrics,
            conversation_log=self.conversation_log,
        )

    # ------------------------------------------------------------------
    # History utilities
    # ------------------------------------------------------------------
    def _load_history(self) -> List[dict]:
        entries: List[dict] = []
        try:
            with self.conversation_log.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        continue
        except FileNotFoundError:
            return []
        except Exception:
            return []
        return entries

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(_TurnState)
        graph.add_node("assess_context", self._node_assess_context)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("plan_tools", self._node_plan_tools)
        graph.add_node("call_tools", self._node_call_tools)
        graph.add_node("llm", self._node_llm)
        graph.add_node("persist", self._node_persist)
        graph.set_entry_point("assess_context")
        graph.add_edge("assess_context", "retrieve")
        graph.add_edge("retrieve", "plan_tools")
        graph.add_edge("plan_tools", "call_tools")
        graph.add_edge("call_tools", "llm")
        graph.add_edge("llm", "persist")
        graph.add_edge("persist", END)
        return graph.compile()

    def _emit_tool_event(self, message: str) -> None:
        if not self.tooling_enabled:
            return
        prefixed = f"[Tools] {message}"
        if self._custom_tool_logger is not None:
            try:
                self._custom_tool_logger(prefixed)
                return
            except Exception:
                pass
        print(prefixed)

    def _plan_tool_actions(self, state: _TurnState) -> List[ToolAction]:
        if not self.tooling_enabled:
            return []
        user_text = state.get("user_text") or ""
        action = self._detect_file_read_request(user_text)
        if action is not None:
            action.source = "langgraph"
            return [action]
        directory_action = self._detect_directory_listing_request(user_text)
        if directory_action is not None:
            directory_action.source = "langgraph"
            return [directory_action]
        return []

    @staticmethod
    def _serialize_tool_action(action: ToolAction) -> Dict[str, Any]:
        return {
            "kind": action.kind,
            "description": action.description,
            "candidate_path": action.candidate_path,
            "search_term": action.search_term,
            "location_hint": action.location_hint,
            "source": action.source,
            "parameters": action.parameters,
        }

    @staticmethod
    def _deserialize_tool_action(payload: Dict[str, Any]) -> ToolAction:
        return ToolAction(
            kind=str(payload.get("kind", "")),
            description=str(payload.get("description", "")),
            candidate_path=payload.get("candidate_path"),
            search_term=payload.get("search_term"),
            location_hint=payload.get("location_hint"),
            source=str(payload.get("source", "langgraph")),
            parameters=dict(payload.get("parameters") or {}),
        )

    def _record_file_read(self, path: Optional[str], content: str, sha256: Optional[str]) -> None:
        if not path:
            return
        normalized_path = str(path)
        trimmed_content = content
        max_snapshot_length = 4000
        if len(trimmed_content) > max_snapshot_length:
            trimmed_content = trimmed_content[:max_snapshot_length] + "\n…"
        entry = {
            "path": normalized_path,
            "base_name": Path(normalized_path).name.casefold(),
            "sha256": sha256 or "",
            "content": trimmed_content,
            "timestamp": time.time(),
        }
        existing = [item for item in self._recent_file_reads if item.get("path") != normalized_path]
        existing.append(entry)
        self._recent_file_reads = existing[-5:]

    def _record_file_snapshot_from_disk(self, path: str) -> None:
        try:
            read_back = workspace.read_file(path)
        except Exception:
            return
        if not isinstance(read_back, dict) or not read_back.get("ok"):
            return
        content = read_back.get("content")
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8", errors="replace")
            except Exception:
                content = ""
        if content is None:
            content = ""
        stored_path = read_back.get("path")
        display_path = path or stored_path
        self._record_file_read(display_path or stored_path, str(content), read_back.get("sha256"))

    def _build_tool_router_context(self, user_text: str) -> Optional[str]:
        if not self._recent_file_reads:
            return None

        lowered = (user_text or "").casefold()
        pronoun_keywords = {"that file", "this file", "the file", "same file"}

        explicit_targets: List[Dict[str, Any]] = []
        for entry in reversed(self._recent_file_reads):
            base = entry.get("base_name")
            path_value = str(entry.get("path") or "").casefold()
            if (base and base in lowered) or (path_value and path_value in lowered):
                explicit_targets.append(entry)

        if explicit_targets:
            targets = explicit_targets[:3]
        elif any(token in lowered for token in pronoun_keywords):
            targets = [self._recent_file_reads[-1]]
        elif "file" in lowered:
            targets = [self._recent_file_reads[-1]]
        else:
            return None

        snippets: List[str] = []
        for entry in targets:
            content = str(entry.get("content") or "").rstrip()
            sha = entry.get("sha256") or "unknown"
            path_display = entry.get("path") or "unknown"
            snippets.append(
                f"File path: {path_display}\nSHA-256: {sha}\nCurrent content:\n{content}"
            )

        return (
            "Most recent workspace file snapshots (for tool planning):\n" + "\n\n".join(snippets)
        )

    def _normalize_location_hint(self, location: str) -> str:
        cleaned = re.sub(r"\b(folder|directory)\b", "", location or "", flags=re.IGNORECASE)
        cleaned = cleaned.replace("\\", "/")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return ""
        parts: List[str] = []
        for chunk in re.split(r"[\\/]", cleaned):
            for token in re.split(r"\s+", chunk):
                candidate = token.strip()
                if not candidate:
                    continue
                lowered = candidate.lower()
                if lowered in {"my", "the", "this", "that", "these", "those", "your", "our", "a", "an"}:
                    continue
                candidate = candidate.strip(".,;:'\"")
                if not candidate:
                    continue
                parts.append(candidate)
        return "/".join(parts)

    def _infer_location_hint_from_text(self, file_name: str, user_text: str) -> Optional[str]:
        if not file_name or not user_text:
            return None
        pattern = re.compile(re.escape(file_name), re.IGNORECASE)
        locations_to_check = []
        match = pattern.search(user_text)
        if match:
            locations_to_check.append(user_text[match.end() :])
        locations_to_check.append(user_text)
        for segment in locations_to_check:
            location_match = re.search(
                r"(?i)(?:on|in)\s+(?:my\s+)?([\w\s\./\\-]+)",
                segment,
            )
            if location_match:
                candidate = location_match.group(1).strip()
                if candidate:
                    return candidate
        return None

    @staticmethod
    def _clean_candidate_filename(candidate: str) -> str:
        """Reduce a natural-language file description to a probable filename."""

        if not candidate:
            return ""

        cleaned = candidate.strip().strip("'\"")
        cleaned = re.sub(r"\s+", " ", cleaned)
        lowered = cleaned.lower()

        # Drop the final article or pronoun that precedes the actual name.
        markers = [
            " the ",
            " this ",
            " that ",
            " these ",
            " those ",
            " my ",
            " our ",
            " your ",
            " a ",
            " an ",
        ]
        for marker in markers:
            idx = lowered.rfind(marker)
            if idx != -1:
                trimmed = cleaned[idx + len(marker) :].strip()
                if trimmed:
                    cleaned = trimmed
                    lowered = cleaned.lower()
                break

        # Remove leading determiners at the start of the candidate.
        prefix_markers = [
            "the ",
            "this ",
            "that ",
            "these ",
            "those ",
            "my ",
            "our ",
            "your ",
            "a ",
            "an ",
        ]
        for marker in prefix_markers:
            if lowered.startswith(marker):
                cleaned = cleaned[len(marker) :].strip()
                lowered = cleaned.lower()
                break

        return cleaned

    def _detect_file_read_request(self, user_text: str) -> Optional[ToolAction]:
        if not user_text:
            return None
        lowered = user_text.lower()
        trigger_phrases = (
            "what is in",
            "what's in",
            "what is inside",
            "show me",
            "display",
            "open",
            "read",
            "list the contents",
            "content of",
            "contents of",
        )
        if not any(phrase in lowered for phrase in trigger_phrases):
            return None

        candidate: Optional[str] = None
        quoted_match = re.search(r'"([^"\n]+)"', user_text)
        if quoted_match:
            candidate = quoted_match.group(1).strip()

        file_match = None
        content_match = None
        if not candidate:
            file_match = re.search(r"(?i)(?:the|this)?\s*([\w ._\\/\-]+?)\s+file", user_text)
            if file_match:
                candidate = file_match.group(1).strip()
        if not candidate:
            content_match = re.search(
                r"(?i)content(?:s)?\s+of\s+([\w ._\\/\-]+)", user_text
            )
            if content_match:
                candidate = content_match.group(1).strip()
        if not candidate:
            return None

        location_hint: Optional[str] = None
        trailing_source = ""
        if file_match:
            trailing_source = user_text[file_match.end() :]
        elif quoted_match:
            trailing_source = user_text[quoted_match.end() :]
        elif content_match:
            trailing_source = user_text[content_match.end() :]
        if trailing_source:
            location_match = re.search(r"(?i)(?:on|in)\s+(?:my\s+)?([\w\s\./\\-]+)", trailing_source)
            if location_match:
                location_hint = location_match.group(1).strip()

        if not location_hint:
            candidate_lower = candidate.lower()
            for marker in (" on ", " in "):
                idx = candidate_lower.rfind(marker)
                if idx != -1:
                    possible_location = candidate[idx + len(marker) :].strip()
                    if possible_location and not location_hint:
                        location_hint = possible_location
                    candidate = candidate[:idx].strip()
                    break
                candidate_lower = candidate.lower()

        candidate = self._clean_candidate_filename(candidate or "")
        if not candidate:
            return None

        candidate = candidate.rstrip(".?!,;:")

        path_parts: List[str] = []
        if location_hint:
            normalized_location = self._normalize_location_hint(location_hint)
            if normalized_location:
                path_parts.append(normalized_location)
        path_parts.append(candidate)
        candidate_path = "/".join(part for part in path_parts if part)
        candidate_path = candidate_path.replace("\\", "/")
        search_term = Path(candidate).name
        description = f"Inspect '{candidate}'"
        return ToolAction(
            kind="read_file",
            description=description,
            candidate_path=candidate_path or None,
            search_term=search_term,
            location_hint=location_hint,
        )

    def _detect_directory_listing_request(self, user_text: str) -> Optional[ToolAction]:
        if not user_text:
            return None

        lowered = user_text.casefold()
        if not any(token in lowered for token in ("list", "show", "display", "what", "which")):
            return None
        if not any(keyword in lowered for keyword in ("file", "files", "folder", "directory", "contents")):
            return None

        extensions: Optional[List[str]] = None
        if any(phrase in lowered for phrase in ("text file", "text files", "txt file", "txt files", "*.txt")):
            extensions = ["txt"]

        candidate: Optional[str] = None
        quoted_match = re.search(r'"([^"\n]+)"', user_text)
        if quoted_match:
            candidate = quoted_match.group(1).strip()

        if not candidate:
            path_match = re.search(r"([A-Za-z]:\\\\[^\s\"']+|/[^\s\"']+)", user_text)
            if path_match:
                candidate = path_match.group(1).strip()

        location_hint: Optional[str] = None
        if not candidate:
            location_match = re.search(
                r"(?i)(?:on|in|inside|within|under)\s+(?:my\s+|the\s+)?([\w\s\./\\-]+?)(?:\s+(?:folder|directory))?(?:\b|$)",
                user_text,
            )
            if location_match:
                candidate = location_match.group(1).strip()

        if not candidate:
            return None

        cleaned_candidate = re.sub(r"\b(?:files?|documents?|items)\b", "", candidate, flags=re.IGNORECASE)
        cleaned_candidate = cleaned_candidate.strip().rstrip(".?!,;:")
        normalized_hint = self._normalize_location_hint(cleaned_candidate)
        location_hint = normalized_hint or cleaned_candidate or None

        candidate_path = (normalized_hint or cleaned_candidate).replace("\\", "/") if (normalized_hint or cleaned_candidate) else ""
        if not candidate_path:
            return None

        parameters: Dict[str, Any] = {}
        if extensions:
            parameters["extensions"] = extensions

        description = f"List directory '{candidate_path}'"
        search_term = Path(candidate_path).name or candidate_path
        return ToolAction(
            kind="list_directory",
            description=description,
            candidate_path=candidate_path,
            search_term=search_term,
            location_hint=location_hint,
            parameters=parameters,
        )

    def _actions_from_tool_calls(
        self,
        tool_calls: Iterable[Dict[str, Any]],
        *,
        user_text: Optional[str] = None,
    ) -> List[ToolAction]:
        actions: List[ToolAction] = []
        heuristic_action: Optional[ToolAction] = None
        if user_text:
            try:
                heuristic_action = self._detect_file_read_request(user_text)
            except Exception:
                heuristic_action = None
        for index, call in enumerate(tool_calls):
            function_payload = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function_payload, dict):
                continue
            name = str(function_payload.get("name") or "").strip()
            raw_arguments = function_payload.get("arguments", {})
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(raw_arguments, dict):
                arguments = dict(raw_arguments)
            else:
                arguments = {}

            if name == "workspace_read_file":
                path_value = str(arguments.get("path") or "").strip()
                if not path_value:
                    continue
                hint = arguments.get("location_hint")
                description = str(arguments.get("description") or f"Inspect '{path_value}'")
                action = ToolAction(
                    kind="read_file",
                    description=description,
                    candidate_path=path_value,
                    search_term=Path(path_value).name,
                    location_hint=str(hint) if hint else None,
                    source="llm",
                    parameters={"raw_arguments": arguments, "tool_call_index": index},
                )
                if heuristic_action:
                    if not action.location_hint and heuristic_action.location_hint:
                        action.location_hint = heuristic_action.location_hint
                    if not action.search_term and heuristic_action.search_term:
                        action.search_term = heuristic_action.search_term
                    # When the LLM only supplied a bare filename, reuse the
                    # heuristic's candidate path so location hints such as
                    # "Desktop" carry through to path expansion.
                    bare_candidate = action.candidate_path.replace("\\", "/").strip()
                    if bare_candidate and "/" not in bare_candidate and not Path(bare_candidate).is_absolute():
                        heuristic_candidate = (heuristic_action.candidate_path or "").strip()
                        if heuristic_candidate:
                            action.candidate_path = heuristic_candidate
                if not action.location_hint and user_text:
                    inferred_hint = self._infer_location_hint_from_text(
                        Path(path_value).name,
                        user_text,
                    )
                    if inferred_hint:
                        action.location_hint = inferred_hint
                if action.location_hint:
                    bare_candidate = action.candidate_path.replace("\\", "/").strip()
                    if (
                        bare_candidate
                        and "/" not in bare_candidate
                        and not Path(bare_candidate).is_absolute()
                    ):
                        normalized_hint = self._normalize_location_hint(action.location_hint)
                        if normalized_hint:
                            action.candidate_path = f"{normalized_hint}/{Path(path_value).name}"
                actions.append(action)
            elif name == "workspace_apply_text_patch":
                path_value = str(arguments.get("path") or "").strip()
                diff_value = arguments.get("diff")
                if not path_value or not isinstance(diff_value, str) or not diff_value.strip():
                    continue
                expect_sha = arguments.get("expect_sha256")
                summary = str(arguments.get("summary") or f"Apply text patch to '{path_value}'")
                parameters: Dict[str, Any] = {
                    "diff": diff_value,
                    "expect_sha256": str(expect_sha).strip() if expect_sha else None,
                    "raw_arguments": arguments,
                    "tool_call_index": index,
                }
                actions.append(
                    ToolAction(
                        kind="apply_text_patch",
                        description=summary,
                        candidate_path=path_value,
                        search_term=Path(path_value).name,
                        source="llm",
                        parameters=parameters,
                    )
                )
            elif name == "workspace_list_directory":
                path_value = str(arguments.get("path") or "").strip()
                if not path_value:
                    continue
                recursive = bool(arguments.get("recursive"))
                pattern = arguments.get("pattern")
                extensions = arguments.get("extensions")
                limit = arguments.get("limit")
                summary = str(arguments.get("description") or f"List directory '{path_value}'")
                parameters = {
                    "recursive": recursive,
                    "pattern": pattern,
                    "extensions": extensions,
                    "limit": limit,
                }
                actions.append(
                    ToolAction(
                        kind="list_directory",
                        description=summary,
                        candidate_path=path_value,
                        search_term=Path(path_value).name,
                        source="llm",
                        parameters=parameters,
                    )
                )
        return actions

    def _prioritize_matches(self, matches: List[str], action: ToolAction) -> List[str]:
        if not matches:
            return []
        candidate = (action.candidate_path or "").replace("\\", "/").strip("/")
        target_name = Path((action.search_term or "")).name.lower()

        def sort_key(path: str) -> tuple[int, str]:
            normalized = path.replace("\\", "/").strip("/")
            lowered = normalized.lower()
            if candidate and lowered == candidate.lower():
                return (0, lowered)
            if target_name and Path(path).name.lower() == target_name:
                return (1, lowered)
            return (2, lowered)

        unique_matches = list(dict.fromkeys(matches))
        unique_matches.sort(key=sort_key)
        return unique_matches

    def _resolve_candidate_path(self, action: ToolAction) -> Optional[str]:
        candidate = action.candidate_path
        if not candidate:
            return None
        normalized = candidate.replace("\\", "/").strip()
        if not normalized:
            return None
        path_obj = Path(normalized)
        if path_obj.is_absolute():
            return str(path_obj)
        expanded = self._expand_special_path(normalized, action)
        if expanded is not None:
            return str(expanded)
        return normalized

    def _probe_llm_for_tool_actions(self, state: _TurnState) -> List[ToolAction]:
        user_text = state.get("user_text") or ""
        if not user_text.strip():
            return []
        messages: List[Dict[str, Any]] = []
        if getattr(self.llm_client, "system_prompt", None):
            messages.append({"role": "system", "content": str(self.llm_client.system_prompt)})
        messages.append({"role": "system", "content": _TOOL_ROUTER_PROMPT})

        context_message = self._build_tool_router_context(user_text)
        if context_message:
            messages.append({"role": "assistant", "content": context_message})
            self._emit_tool_event("Tool probe enriched with recent file context.")

        cleaned_user_text = user_text.rstrip()
        if "/no_think" not in cleaned_user_text.lower():
            suffix = "\n/no_think" if cleaned_user_text else "/no_think"
            cleaned_user_text = cleaned_user_text + suffix
        messages.append({"role": "user", "content": cleaned_user_text})

        try:
            response = self.llm_client.chat(
                messages,
                tools=_WORKSPACE_TOOL_SCHEMA,
                tool_choice="required",
                stream=False,
            )
        except Exception as exc:
            self._emit_tool_event(f"Tool probe failed: {exc}")
            state.setdefault("errors", []).append({"node": "plan_tools", "error": str(exc)})
            return []

        message = response.get("message") if isinstance(response, dict) else None
        if not isinstance(message, dict):
            return []
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            content = message.get("content")
            if isinstance(content, str) and content.strip() and content.strip().lower() != "none":
                self._emit_tool_event("Tool probe returned no actionable tool calls.")
            return []

        actions = self._actions_from_tool_calls(tool_calls, user_text=state.get("user_text"))
        if actions:
            summary = ", ".join(action.description for action in actions)
            self._emit_tool_event(
                f"Planner requested {len(actions)} tool action(s) via LLM: {summary}."
            )
        return actions

    def _expand_special_path(self, candidate: str, action: ToolAction) -> Optional[Path]:
        segments = [segment for segment in candidate.split("/") if segment]
        if not segments:
            return None
        remainder = segments[1:]
        hint_tokens = {
            piece.lower()
            for piece in re.split(r"[\\s/\\\\]+", action.location_hint or "")
            if piece
        }
        hint_tokens.add(segments[0].lower())
        root = self._lookup_special_root(hint_tokens)
        if root is None:
            return None
        try:
            resolved_root = root.expanduser().resolve()
        except Exception:
            resolved_root = root
        workspace.register_allowed_root(resolved_root)
        if remainder:
            return resolved_root.joinpath(*remainder)
        candidate_name = action.search_term or Path(candidate).name
        if candidate_name:
            return resolved_root / candidate_name
        return resolved_root

    def _lookup_special_root(self, tokens: Iterable[str]) -> Optional[Path]:
        normalized_tokens = {token.strip().lower() for token in tokens if token}
        if not normalized_tokens:
            return None

        def _candidate(path_like: Optional[Path | str]) -> Optional[Path]:
            if not path_like:
                return None
            try:
                return Path(path_like).expanduser().resolve()
            except Exception:
                try:
                    return Path(path_like).expanduser()
                except Exception:
                    return None

        home = _candidate(Path.home())
        is_windows = bool((self._system_info or {}).get("is_windows"))

        mapping: List[tuple[set[str], Optional[Path]]] = [
            ({"home", "homedir", "homefolder", "home directory"}, home),
            ({"desktop"}, _candidate(Path.home() / "Desktop")),
            (
                {"documents", "document", "docs", "mydocuments"},
                _candidate(Path.home() / "Documents"),
            ),
            ({"downloads", "download"}, _candidate(Path.home() / "Downloads")),
            ({"pictures", "photos", "images"}, _candidate(Path.home() / "Pictures")),
            ({"music", "songs"}, _candidate(Path.home() / "Music")),
            ({"videos", "video", "movies"}, _candidate(Path.home() / "Videos")),
        ]

        if is_windows:
            mapping.extend(
                [
                    ({"appdata", "roaming"}, _candidate(os.environ.get("APPDATA"))),
                    ({"localappdata", "local"}, _candidate(os.environ.get("LOCALAPPDATA"))),
                    ({"userprofile"}, _candidate(os.environ.get("USERPROFILE"))),
                ]
            )

        for keys, candidate_root in mapping:
            if not candidate_root:
                continue
            if normalized_tokens & keys:
                return candidate_root
        return None

    def _execute_tool_plan(self, actions: List[ToolAction]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for action in actions:
            if action.kind == "read_file":
                results.append(self._execute_read_file_action(action))
            elif action.kind == "apply_text_patch":
                results.append(self._execute_apply_text_patch_action(action))
            elif action.kind == "list_directory":
                results.append(self._execute_list_directory_action(action))
            else:
                self._emit_tool_event(f"Unsupported tool action '{action.kind}' ignored.")
                results.append({"kind": action.kind, "success": False, "message": "unsupported"})
        return results

    def _execute_read_file_action(self, action: ToolAction) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "kind": action.kind,
            "description": action.description,
            "success": False,
        }
        candidate_path = action.candidate_path
        stat_info: Optional[Dict[str, Any]] = None
        selected_path: Optional[str] = None

        if candidate_path:
            resolved_candidate = self._resolve_candidate_path(action)
            if resolved_candidate and resolved_candidate != candidate_path:
                self._emit_tool_event(
                    f"Expanded candidate path to '{resolved_candidate}'."
                )
                candidate_path = resolved_candidate
            self._emit_tool_event(f"stat_file('{candidate_path}')")
            stat_result = workspace.stat_file(candidate_path)
            result["stat"] = stat_result
            if stat_result.get("ok") and stat_result.get("exists") and stat_result.get("is_file"):
                selected_path = candidate_path
                stat_info = stat_result
                self._emit_tool_event(
                    f"Candidate path found (size={stat_result.get('size', 'unknown')} bytes)."
                )
                self._emit_tool_event("search_files skipped; direct path resolved.")
            else:
                error_msg = "not found"
                if not stat_result.get("ok", True):
                    errors = stat_result.get("errors") or []
                    if errors and isinstance(errors, list):
                        error_msg = str(errors[0].get("message", error_msg))
                self._emit_tool_event(
                    f"Candidate path missing; proceeding to search (reason: {error_msg})."
                )

        search_term = action.search_term or ""
        glob_pattern: Optional[str] = None
        if action.location_hint:
            normalized_hint = self._normalize_location_hint(action.location_hint)
            if normalized_hint:
                hint_tokens = {
                    piece.strip().lower()
                    for chunk in re.split(r"[\\/]+", normalized_hint)
                    for piece in re.split(r"\s+", chunk)
                    if piece
                }
                root_hint = self._lookup_special_root(hint_tokens)
                if root_hint is not None:
                    try:
                        resolved_root = root_hint.expanduser().resolve()
                    except Exception:
                        resolved_root = root_hint
                    workspace.register_allowed_root(resolved_root)
                    glob_pattern = "**/*"
                else:
                    glob_pattern = f"{normalized_hint}/**/*"
        if selected_path is None and search_term:
            self._emit_tool_event(
                f"search_files(query='{search_term}', glob={glob_pattern or 'default'})"
            )
            search_result = workspace.search_files(search_term, glob=glob_pattern)
            result["search"] = search_result
            matches = list(search_result.get("paths", [])) if isinstance(search_result, dict) else []
            prioritized = self._prioritize_matches(matches, action)

            if not prioritized and glob_pattern:
                self._emit_tool_event(
                    "Scoped search returned no matches; retrying without glob.")
                fallback_result = workspace.search_files(search_term)
                result["search_fallback"] = fallback_result
                prioritized = self._prioritize_matches(
                    list(fallback_result.get("paths", []))
                    if isinstance(fallback_result, dict)
                    else [],
                    action,
                )

            if prioritized:
                for candidate_match in prioritized:
                    self._emit_tool_event(f"Evaluating search match '{candidate_match}'.")
                    stat_info = workspace.stat_file(candidate_match)
                    result["stat"] = stat_info
                    if stat_info.get("ok") and stat_info.get("exists") and stat_info.get("is_file"):
                        selected_path = candidate_match
                        break
                    message = "unavailable"
                    if not stat_info.get("ok", True):
                        errors = stat_info.get("errors") or []
                        if errors and isinstance(errors, list):
                            message = str(errors[0].get("message", message))
                    self._emit_tool_event(
                        f"Search candidate '{candidate_match}' unavailable ({message})."
                    )
            else:
                self._emit_tool_event("search_files returned no matches.")

        if selected_path is None:
            result["message"] = "No matching file found in the workspace."
            return result

        self._emit_tool_event(f"read_file('{selected_path}')")
        read_result = workspace.read_file(selected_path)
        result["read"] = read_result
        if not read_result.get("ok"):
            errors = read_result.get("errors") or []
            message = "read failure"
            if errors and isinstance(errors, list):
                message = str(errors[0].get("message", message))
            self._emit_tool_event(f"Failed to read '{selected_path}': {message}.")
            result["message"] = message
            return result

        content = str(read_result.get("content", ""))
        truncated = False
        if len(content) > TOOL_RESPONSE_CHAR_LIMIT:
            truncated = True
            content = content[:TOOL_RESPONSE_CHAR_LIMIT].rstrip() + "\n...[truncated]"
            self._emit_tool_event(
                f"Response truncated to {TOOL_RESPONSE_CHAR_LIMIT} characters for safety."
            )

        self._emit_tool_event(
            f"Read {len(content)} characters from '{selected_path}'."
        )
        result.update(
            {
                "success": True,
                "path": selected_path,
                "content": content,
                "sha256": read_result.get("sha256"),
                "size": (stat_info or {}).get("size"),
                "truncated": truncated,
            }
        )
        self._record_file_read(selected_path, content, read_result.get("sha256"))
        return result

    def _execute_apply_text_patch_action(self, action: ToolAction) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "kind": action.kind,
            "description": action.description,
            "success": False,
        }
        candidate_path = action.candidate_path or ""
        diff_payload = action.parameters.get("diff") if isinstance(action.parameters, dict) else None
        expect_sha = action.parameters.get("expect_sha256") if isinstance(action.parameters, dict) else None
        if not candidate_path:
            result["message"] = "Missing target path for text patch."
            self._emit_tool_event("Text patch request missing target path; aborting.")
            return result
        if not isinstance(diff_payload, str) or not diff_payload.strip():
            result["message"] = "Missing unified diff for text patch."
            self._emit_tool_event("Text patch request missing diff payload; aborting.")
            return result

        resolved_candidate = self._resolve_candidate_path(action)
        if resolved_candidate and resolved_candidate != candidate_path:
            self._emit_tool_event(f"Expanded candidate path to '{resolved_candidate}'.")
            candidate_path = resolved_candidate

        expect_sha_str: Optional[str]
        if isinstance(expect_sha, str) and expect_sha.strip():
            expect_sha_str = expect_sha.strip()
        else:
            expect_sha_str = None

        self._emit_tool_event(f"dry_run_text_patch('{candidate_path}')")
        dry_run = workspace.dry_run_text_patch(
            candidate_path,
            diff_payload,
            expect_sha256=expect_sha_str,
        )
        if not dry_run.get("ok"):
            errors = dry_run.get("errors") or []
            message = "dry run failed"
            if errors and isinstance(errors, list):
                message = str(errors[0].get("message", message))
            self._emit_tool_event(f"Dry run failed for '{candidate_path}': {message}.")
            result["message"] = message
            result["dry_run"] = dry_run
            return result

        self._emit_tool_event(f"apply_text_patch('{candidate_path}')")
        apply_result = workspace.apply_text_patch(
            candidate_path,
            diff_payload,
            expect_sha256=expect_sha_str,
        )
        if not apply_result.get("ok"):
            errors = apply_result.get("errors") or []
            message = "apply failed"
            if errors and isinstance(errors, list):
                message = str(errors[0].get("message", message))
            self._emit_tool_event(f"Failed to apply text patch to '{candidate_path}': {message}.")
            result["message"] = message
            result["apply"] = apply_result
            return result

        summary = action.description or "Text patch applied."
        relative_path = apply_result.get("path") or candidate_path
        new_sha = apply_result.get("new_sha256")
        previous_sha = apply_result.get("previous_sha256")
        self._emit_tool_event(
            f"Applied text patch to '{relative_path}' (new_sha={new_sha or 'unknown'})."
        )

        result.update(
            {
                "success": True,
                "path": relative_path,
                "new_sha256": new_sha,
                "previous_sha256": previous_sha,
                "message": summary,
            }
        )
        if relative_path:
            self._record_file_snapshot_from_disk(relative_path)
        return result

    def _execute_list_directory_action(self, action: ToolAction) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "kind": action.kind,
            "description": action.description,
            "success": False,
        }
        target_path = action.candidate_path or ""
        if not target_path:
            result["message"] = "Missing directory path."
            self._emit_tool_event("Directory listing request missing target path; aborting.")
            return result

        resolved_candidate = self._resolve_candidate_path(action)
        if resolved_candidate and resolved_candidate != target_path:
            self._emit_tool_event(f"Expanded candidate path to '{resolved_candidate}'.")
            target_path = resolved_candidate

        recursive = bool(action.parameters.get("recursive")) if isinstance(action.parameters, dict) else False
        pattern = None
        extensions = None
        limit = None
        if isinstance(action.parameters, dict):
            pattern = action.parameters.get("pattern")
            extensions = action.parameters.get("extensions")
            limit_value = action.parameters.get("limit")
            try:
                limit = int(limit_value) if limit_value is not None else None
            except (TypeError, ValueError):
                limit = None

        self._emit_tool_event(
            f"list_directory('{target_path}', pattern={pattern or '*'}, recursive={recursive}, limit={limit or 'default'})"
        )
        list_result = workspace.list_directory(
            target_path,
            pattern=pattern,
            extensions=extensions,
            recursive=recursive,
            limit=limit,
        )
        result.update(list_result)
        if list_result.get("ok"):
            entries = list_result.get("entries") or list_result.get("files") or []
            result.update(
                {
                    "success": True,
                    "path": list_result.get("path", target_path),
                    "entries": entries,
                    "message": list_result.get("message")
                    or f"Found {len(entries)} item(s) in {list_result.get('path', target_path)}.",
                }
            )
        else:
            errors = list_result.get("errors") or []
            message = "listing failed"
            if errors and isinstance(errors, list):
                message = str(errors[0].get("message", message))
            self._emit_tool_event(f"Directory listing failed for '{target_path}': {message}.")
            result["message"] = message
        return result

    def _documentation_threshold(self) -> float:
        value = self.identity_settings.get("documentation_retrieval_threshold")
        try:
            return float(value)
        except Exception:
            return 0.35

    def _build_planner_prompt(self, query_text: str) -> str:
        cleaned = _REASONING_TAG_PATTERN.sub("", query_text or "")
        cleaned = cleaned.strip()
        if not cleaned:
            cleaned = (query_text or "").strip()
        prompt = (
            "Decide which additional context is required for this user question:\n"
            f"{cleaned}"
        )
        if self.identity.casefold() == "einstein":
            prompt = f"{prompt}\n/no_think"
        return prompt

    @staticmethod
    def _should_prioritize_documentation(text: str) -> bool:
        if not text:
            return False
        lowered = text.casefold()
        stripped = lowered.strip()
        if stripped in {"help", "docs", "documentation", "manual", "guide"}:
            return True
        if "explain" in lowered and any(
            phrase in lowered for phrase in {"program", "software", "ctrlspeak", "app", "bot"}
        ):
            return True
        return any(phrase in lowered for phrase in _DOCUMENTATION_INTENT_PHRASES)

    def _temporal_threshold(self) -> float:
        value = self.identity_settings.get("temporal_retrieval_threshold")
        try:
            return float(value)
        except Exception:
            return 0.15

    @staticmethod
    def _should_prioritize_temporal(text: str) -> bool:
        if not text:
            return False
        lowered = text.casefold()
        stripped = lowered.strip()
        if stripped in {"date", "time", "timezone", "time zone"}:
            return True
        return any(phrase in lowered for phrase in _TEMPORAL_INTENT_PHRASES)

    @staticmethod
    def _should_prioritize_chat_history(text: str) -> bool:
        if not text:
            return False
        lowered = text.casefold()
        if "history" in lowered and any(
            token in lowered for token in {"chat", "conversation", "with me", "about us"}
        ):
            return True
        return any(phrase in lowered for phrase in _CHAT_HISTORY_INTENT_PHRASES)

    @staticmethod
    def _parse_assessment_response(response: str) -> Optional[Dict[str, bool]]:
        if not response:
            return None
        lowered = response.strip().casefold()
        if not lowered:
            return None
        if lowered in {"no", "nope", "nah"}:
            lowered = "none"
        tokens = [token for token in re.split(r"[^a-z]+", lowered) if token]
        if not tokens:
            return None
        if any(token not in {"documentation", "chat", "chat_history", "history", "date", "time", "timezone", "none"} for token in tokens):
            return None
        if "none" in tokens:
            return {"documentation": False, "chat_history": False, "date": False}
        normalized: Dict[str, bool] = {"documentation": False, "chat_history": False, "date": False}
        for token in tokens:
            if token == "documentation":
                normalized["documentation"] = True
            elif token in {"chat_history", "history", "chat"}:
                normalized["chat_history"] = True
            elif token in {"date", "time", "timezone"}:
                normalized["date"] = True
        return normalized

    @staticmethod
    def _summarize_plan(plan: Dict[str, bool]) -> str:
        if not plan:
            return "none"
        requested = [key for key, value in plan.items() if value]
        return "+".join(requested) if requested else "none"

    def _default_retrieval_plan(self, text: str) -> Dict[str, bool]:
        return {
            "documentation": self._should_prioritize_documentation(text),
            "chat_history": self._should_prioritize_chat_history(text),
            "date": self._should_prioritize_temporal(text),
        }

    # ------------------------------------------------------------------
    def _resolve_vision_payload(
        self,
        augmented_text: str,
        vision_metadata: Optional[Dict[str, Any]],
    ) -> tuple[Optional[List[dict]], Optional[Dict[str, Any]], bool]:
        metadata: Dict[str, Any] = {}
        if isinstance(vision_metadata, dict):
            metadata = dict(vision_metadata)

        request_hint = metadata.get("vision_request")
        include_image = bool(request_hint)
        if not include_image and is_image_request(augmented_text):
            include_image = True

        if not include_image:
            metadata.pop("vision_attached", None)
            return None, (metadata or None), False

        record = load_identity_image(self.identity)
        if record is None:
            logger.debug(
                "Requested image attachment for identity '%s' but no stored PNG was found.",
                self.identity,
            )
            metadata.pop("vision_attached", None)
            return None, (metadata or None), False

        metadata.update(
            {
                "vision_file": str(record.path),
                "vision_updated": record.updated_at_iso,
            }
        )
        if record.source and not metadata.get("vision_source"):
            metadata["vision_source"] = record.source
        metadata["vision_attached"] = True
        metadata.pop("vision_request", None)

        content = [
            {"type": "text", "text": augmented_text},
            {"type": "image", "image": record.image_b64},
        ]
        return content, metadata, True

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------
    def _node_assess_context(self, state: _TurnState) -> _TurnState:
        query_text = state.get("augmented_text") or state.get("user_text") or ""
        plan: Dict[str, bool] = {"documentation": False, "chat_history": False, "date": False}
        summary = "none"
        planned = False
        planner_invoked = False

        if query_text.strip():
            heuristic_plan = self._default_retrieval_plan(query_text)
            for key, value in heuristic_plan.items():
                if value:
                    plan[key] = True
            planned = any(plan.values())
            parsed: Optional[Dict[str, bool]] = None
            if not planned:
                try:
                    planner_prompt = self._build_planner_prompt(query_text)
                    response = self.llm_client.query(
                        planner_prompt,
                        history=[{"role": "system", "content": _ASSESSMENT_INSTRUCTIONS}],
                    )
                    planner_invoked = True
                except Exception as exc:
                    logger.debug("Context assessment failed: %s", exc)
                    response = ""
                parsed = self._parse_assessment_response(str(response))
                if parsed is not None:
                    plan.update(parsed)
                    planned = any(plan.values())
            if not planned:
                plan.update(heuristic_plan)
                planned = any(plan.values())
            summary = self._summarize_plan(plan)
        state["retrieval_plan"] = plan
        state["retrieval_plan_summary"] = summary
        state["vector_query_planned"] = planned
        state["retrieval_plan_used_llm"] = planner_invoked
        return state

    def _node_retrieve(self, state: _TurnState) -> _TurnState:
        threshold = float(self.identity_settings.get("retrieval_threshold", 0.75))
        top_k = int(self.identity_settings.get("retrieval_top_k", 5))
        query_text = state.get("augmented_text") or state.get("user_text") or ""

        plan = state.get("retrieval_plan") or {}
        doc_intent = bool(plan.get("documentation"))
        temporal_intent = bool(plan.get("date"))
        chat_intent = bool(plan.get("chat_history"))
        doc_threshold = self._documentation_threshold()
        doc_limit = min(top_k, 3 if doc_intent else 1)
        category_thresholds: dict[str, float] = {}
        fallback_categories: dict[str, int] = {}
        if doc_intent:
            category_thresholds[DOCUMENTATION_CATEGORY] = doc_threshold
            fallback_categories[DOCUMENTATION_CATEGORY] = doc_limit
        if temporal_intent:
            category_thresholds[TEMPORAL_CATEGORY] = self._temporal_threshold()
            fallback_categories[TEMPORAL_CATEGORY] = 1
        if not category_thresholds:
            category_thresholds = None  # type: ignore[assignment]
        if not fallback_categories:
            fallback_categories = None  # type: ignore[assignment]

        results = []
        metrics: Dict[str, float] = {}
        attempted = False

        if query_text and (doc_intent or temporal_intent or chat_intent):
            attempted = True
            try:
                results = self.vector_store.retrieve(
                    query_text,
                    top_k=top_k,
                    threshold=threshold,
                    category_thresholds=category_thresholds,
                    fallback_categories=fallback_categories,
                )
            except Exception as exc:
                state.setdefault("errors", []).append({"node": "retrieve", "error": str(exc)})

        if (
            attempted
            and not results
            and not doc_intent
            and not temporal_intent
            and chat_intent
            and query_text
        ):
            try:
                results = self.vector_store.retrieve(
                    query_text,
                    top_k=top_k,
                    threshold=threshold,
                    category_thresholds={DOCUMENTATION_CATEGORY: doc_threshold},
                    fallback_categories={DOCUMENTATION_CATEGORY: doc_limit},
                )
            except Exception as exc:
                state.setdefault("errors", []).append({"node": "retrieve", "error": str(exc)})

        if attempted:
            filtered = []
            for item in results:
                metadata = item.metadata or {}
                category = str(metadata.get("category", ""))
                if category == DOCUMENTATION_CATEGORY and not doc_intent:
                    continue
                if category == TEMPORAL_CATEGORY and not temporal_intent:
                    continue
                if category not in {DOCUMENTATION_CATEGORY, TEMPORAL_CATEGORY} and not chat_intent:
                    continue
                filtered.append(item)
            results = filtered

        if results:
            avg_similarity = sum(item.similarity for item in results) / len(results)
            metrics["retrieval_hits"] = float(len(results))
            metrics["avg_similarity"] = round(avg_similarity, 4)
        doc_matches = 0
        if results:
            doc_matches = sum(
                1
                for item in results
                if str(item.metadata.get("category", "")) == DOCUMENTATION_CATEGORY
            )
        temporal_matches = 0
        if results:
            temporal_matches = sum(
                1
                for item in results
                if str(item.metadata.get("category", "")) == TEMPORAL_CATEGORY
            )
        state["retrieved"] = [
            {"content": item.content, "metadata": item.metadata, "similarity": item.similarity}
            for item in results
        ]
        state["vector_query_attempted"] = attempted
        state["vector_query_result_count"] = len(results)
        state["vector_query_documentation_count"] = doc_matches
        state["vector_query_temporal_count"] = temporal_matches
        if not attempted:
            state["retrieved"] = []
        if metrics:
            container = state.get("metrics")
            if not isinstance(container, dict):
                container = {}
                state["metrics"] = container
            container.update(metrics)
        return state

    def _node_plan_tools(self, state: _TurnState) -> _TurnState:
        if not isinstance(state.get("metrics"), dict):
            state["metrics"] = {}
        if not isinstance(state.get("errors"), list):
            state["errors"] = []
        actions = self._plan_tool_actions(state)
        state["tool_plan"] = [self._serialize_tool_action(action) for action in actions]

        probe_enabled = bool(self.tooling_enabled)
        state["tool_probe_pending"] = probe_enabled
        force_required = probe_enabled and bool(actions)
        state["tool_probe_force_required"] = force_required

        if actions:
            summary = ", ".join(
                f"{action.description} (source={action.source})" for action in actions
            )
            state["tool_plan_summary"] = summary
            hint_lines: List[str] = []
            for action in actions:
                tool_name = self._tool_name_for_action(action)
                target = action.candidate_path or action.search_term or action.description
                hint_lines.append(
                    f"LangGraph suggests calling {tool_name} for {target}."
                )
            hint_text = "LangGraph identified required workspace tool usage:\n" + "\n".join(
                hint_lines
            )
            state["tool_probe_hint"] = hint_text
            self._emit_tool_event(
                f"Planned {len(actions)} action(s): {summary}. Deferring to LLM."
            )
        else:
            state["tool_plan_summary"] = "none"
            state.pop("tool_probe_hint", None)
        return state

    def _node_call_tools(self, state: _TurnState) -> _TurnState:
        plan_entries = state.get("tool_plan")
        if not self.tooling_enabled or not isinstance(plan_entries, list) or not plan_entries:
            if not self.tooling_enabled:
                state["tool_probe_pending"] = False
                state["tool_probe_force_required"] = False
            return state

        actions: List[ToolAction] = []
        for entry in plan_entries:
            if isinstance(entry, dict):
                actions.append(self._deserialize_tool_action(entry))
        if not actions:
            return state

        if state.get("tool_probe_hint"):
            self._emit_tool_event("Tool hint prepared for LLM execution.")
        else:
            self._emit_tool_event("Tool plan ready; awaiting LLM decision.")
        return state

    def _prepare_history_with_context(self, state: _TurnState) -> List[dict]:
        history = list(self.history)
        retrieved = state.get("retrieved") or []
        if retrieved:
            memory_lines: List[str] = []
            documentation_lines: List[str] = []
            temporal_lines: List[str] = []
            for item in retrieved:
                metadata = item.get("metadata") or {}
                category = str(metadata.get("category", ""))
                if category == DOCUMENTATION_CATEGORY:
                    source = metadata.get("source")
                    prefix = f"[{source}] " if source else ""
                    documentation_lines.append(f"- {prefix}{item['content']}")
                elif category == TEMPORAL_CATEGORY:
                    temporal_lines.append(f"- {item['content']}")
                else:
                    memory_lines.append(f"- {item['content']}")
            sections: List[str] = []
            if memory_lines:
                sections.append("Relevant memory:\n" + "\n".join(memory_lines))
            if documentation_lines:
                sections.append("Documentation excerpts:\n" + "\n".join(documentation_lines))
            if temporal_lines:
                sections.append("Temporal context:\n" + "\n".join(temporal_lines))
            if sections:
                history.append({"role": "system", "content": "\n\n".join(sections)})
        return history

    def _store_llm_response(self, state: _TurnState, raw_response: str) -> _TurnState:
        state["raw_response_text"] = raw_response
        filtered_response = raw_response
        if self.think_manager is not None:
            try:
                think_result = self.think_manager.filter_response(raw_response)
            except Exception as exc:
                logger.debug("Think manager failed: %s", exc)
            else:
                filtered_response = think_result.visible_text
                if think_result.removed:
                    state["think_hidden"] = True
                    if think_result.placeholder_text:
                        state["think_placeholder"] = think_result.placeholder_text
                    if think_result.hidden_think:
                        state["hidden_think"] = think_result.hidden_think
        state["response_text"] = filtered_response or ""
        return state

    def _normalize_history_for_chat(self, history: List[dict]) -> List[dict]:
        normalizer = getattr(self.llm_client, "_normalize_history_entry", None)
        normalized: List[dict] = []
        for entry in history:
            if not isinstance(entry, dict):
                continue
            if callable(normalizer):
                normalized.append(normalizer(entry))
            else:
                role = entry.get("role", "user")
                content = entry.get("content", "")
                normalized.append({"role": role, "content": content})
        return normalized

    def _build_user_message(self, state: _TurnState, text: str) -> dict:
        content_blocks = state.get("content_blocks")
        builder = getattr(self.llm_client, "_build_user_message", None)
        if content_blocks is not None and callable(builder):
            try:
                return builder(content_blocks)
            except Exception:
                pass
        return {"role": "user", "content": text}

    def _node_llm_plain(self, state: _TurnState) -> _TurnState:
        augmented = state.get("augmented_text") or state.get("user_text") or ""
        history = self._prepare_history_with_context(state)
        content_blocks = state.get("content_blocks")
        try:
            response = self.llm_client.query(
                augmented,
                history=history,
                content=content_blocks,
            )
        except Exception as exc:
            state.setdefault("errors", []).append({"node": "llm", "error": str(exc)})
            raw_response = "I'm having trouble responding right now."
        else:
            raw_response = response or ""
        return self._store_llm_response(state, raw_response)

    def _tool_name_for_action(self, action: ToolAction) -> str:
        if action.kind == "apply_text_patch":
            return "workspace_apply_text_patch"
        if action.kind == "list_directory":
            return "workspace_list_directory"
        return "workspace_read_file"

    def _node_llm_with_tools(self, state: _TurnState) -> Optional[_TurnState]:
        augmented = state.get("augmented_text") or state.get("user_text") or ""
        history = self._prepare_history_with_context(state)
        messages: List[dict] = []
        system_prompt = getattr(self.llm_client, "system_prompt", None)
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.extend(self._normalize_history_for_chat(history))

        context_message = self._build_tool_router_context(state.get("user_text", ""))
        if context_message:
            messages.append({"role": "assistant", "content": context_message})

        probe_hint = str(state.get("tool_probe_hint") or "").strip()
        if probe_hint:
            messages.append({"role": "assistant", "content": probe_hint})

        user_message = self._build_user_message(state, augmented)
        probe_pending = bool(state.get("tool_probe_pending"))
        if probe_pending and isinstance(user_message, dict):
            content_value = user_message.get("content")
            if isinstance(content_value, str):
                existing = content_value.rstrip()
                if "/no_think" not in existing.lower():
                    suffix = "\n/no_think" if existing else "/no_think"
                    user_message["content"] = existing + suffix
        messages.append(user_message)

        loop_results: List[Dict[str, Any]] = []
        max_iterations = 6
        force_tool_required = bool(state.get("tool_probe_force_required"))
        for iteration in range(max_iterations):
            try:
                tool_choice_value = (
                    "required"
                    if iteration == 0 and probe_pending and force_tool_required
                    else "auto"
                )
                response = self.llm_client.chat(
                    messages,
                    tools=_WORKSPACE_TOOL_SCHEMA,
                    tool_choice=tool_choice_value,
                    stream=False,
                )
            except Exception as exc:
                self._emit_tool_event(f"Tool-enabled LLM call failed: {exc}")
                state.setdefault("errors", []).append({"node": "llm", "error": str(exc)})
                return None

            if iteration == 0 and probe_pending:
                probe_pending = False
                state["tool_probe_pending"] = False
                state["tool_probe_force_required"] = False

            message = response.get("message") if isinstance(response, dict) else None
            if not isinstance(message, dict):
                break

            tool_calls = message.get("tool_calls")
            content = str(message.get("content") or "")

            if tool_calls:
                actions = self._actions_from_tool_calls(tool_calls, user_text=state.get("user_text"))
                if not actions:
                    if content:
                        messages.append({"role": "assistant", "content": content})
                        return self._store_llm_response(state, content)
                    break

                assistant_entry = {"role": "assistant", "content": content or "", "tool_calls": tool_calls}
                messages.append(assistant_entry)

                results = self._execute_tool_plan(actions)
                loop_results.extend(results)
                state.setdefault("tool_results", []).extend(results)

                for tool_call, action, result in zip(tool_calls, actions, results):
                    tool_name = str(tool_call.get("function", {}).get("name") or self._tool_name_for_action(action))
                    try:
                        serialized = json.dumps(result, ensure_ascii=False)
                    except (TypeError, ValueError):
                        serialized = str(result)
                    messages.append({"role": "tool", "name": tool_name, "content": serialized})
                continue

            messages.append({"role": "assistant", "content": content})
            return self._store_llm_response(state, content)

        if loop_results:
            last_success = next((item for item in reversed(loop_results) if item.get("success")), None)
            if last_success and last_success.get("message"):
                return self._store_llm_response(state, str(last_success.get("message")))
        return None

    def _node_llm(self, state: _TurnState) -> _TurnState:
        if state.get("skip_llm"):
            if "raw_response_text" not in state:
                state["raw_response_text"] = state.get("response_text", "")
            return state
        if self.tooling_enabled:
            updated_state = self._node_llm_with_tools(state)
            if updated_state is not None:
                return updated_state
        return self._node_llm_plain(state)

    def _node_persist(self, state: _TurnState) -> _TurnState:
        correlation_id = state["correlation_id"]
        response = state.get("response_text", "")
        if requires_force_plaintext(response):
            scrubbed_response = force_plaintext(response)
        else:
            scrubbed_response = response
        state["scrubbed_response"] = scrubbed_response
        user_text = state.get("user_text", "")
        vision_metadata = state.get("vision_metadata")
        attached_image = bool(state.get("vision_attached"))
        entries = self._build_history_entries(
            user_text,
            scrubbed_response,
            attached_image,
            vision_metadata,
        )
        self.history.extend(entries)
        vector_documents = [user_text, scrubbed_response]
        vector_metadata = [
            {"role": "user", "correlation_id": correlation_id},
            {"role": "assistant", "correlation_id": correlation_id},
        ]
        task = PersistenceTask(
            correlation_id=correlation_id,
            entries=entries,
            vector_documents=vector_documents,
            vector_metadata=vector_metadata,
            settings=self.identity_settings,
        )
        self._persistence.enqueue(task)
        return state

    # ------------------------------------------------------------------
    def _build_history_entries(
        self,
        user_text: str,
        response_text: str,
        attached_image: bool,
        vision_metadata: Optional[Dict[str, Any]],
    ) -> List[dict]:
        entries: List[dict] = []
        if attached_image:
            entry_content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
            metadata: Dict[str, Any] = dict(vision_metadata or {})
            file_path = metadata.get("vision_file")
            if file_path:
                entry_content.append({"type": "image_file", "path": file_path})
            else:
                entry_content.append({"type": "image_reference", "description": "latest identity image"})
            metadata.pop("vision_request", None)
            metadata["vision_attached"] = True
            entries.append({"role": "user", "content": entry_content, "metadata": metadata})
        else:
            entries.append({"role": "user", "content": user_text})
        entries.append({"role": "assistant", "content": response_text})
        return entries

    # ------------------------------------------------------------------
    def run_turn(
        self,
        user_text: str,
        *,
        augmented_text: Optional[str] = None,
        content_blocks: Optional[List[dict]] = None,
        vision_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnResult:
        correlation_id = uuid.uuid4().hex
        augmented = augmented_text or user_text
        resolved_blocks = content_blocks
        resolved_metadata: Optional[Dict[str, Any]] = vision_metadata
        attached_image = False
        if resolved_blocks is None:
            resolved_blocks, resolved_metadata, attached_image = self._resolve_vision_payload(
                augmented,
                vision_metadata,
            )
        else:
            attached_image = any(
                isinstance(block, dict) and block.get("type") == "image"
                for block in resolved_blocks
            )
            if attached_image:
                meta_container: Dict[str, Any] = {}
                if isinstance(vision_metadata, dict):
                    meta_container.update(vision_metadata)
                meta_container["vision_attached"] = True
                resolved_metadata = meta_container
        initial_state: _TurnState = {
            "correlation_id": correlation_id,
            "user_text": user_text,
            "augmented_text": augmented,
            "content_blocks": resolved_blocks,
            "history": list(self.history),
            "vision_metadata": resolved_metadata,
            "vision_attached": attached_image,
        }
        result_state = self._graph.invoke(initial_state)
        metrics = result_state.get("metrics", {})
        if metrics:
            self.metrics.record(correlation_id, metrics)
        trace_path = self._write_trace(result_state)
        retrieved = [
            RetrievedMemory(item["content"], item.get("metadata", {}), float(item.get("similarity", 0)))
            for item in result_state.get("retrieved", [])
        ]
        raw_response_text = result_state.get("raw_response_text")
        if raw_response_text is None:
            raw_response_text = result_state.get("response_text", "")
        attempted_flag = bool(result_state.get("vector_query_attempted"))
        result_count = int(result_state.get("vector_query_result_count", len(retrieved)))
        doc_count = int(
            result_state.get(
                "vector_query_documentation_count",
                sum(
                    1
                    for item in retrieved
                    if str(item.metadata.get("category", "")) == DOCUMENTATION_CATEGORY
                ),
            )
        )
        temporal_count = int(
            result_state.get(
                "vector_query_temporal_count",
                sum(
                    1
                    for item in retrieved
                    if str(item.metadata.get("category", "")) == TEMPORAL_CATEGORY
                ),
            )
        )
        plan = result_state.get("retrieval_plan")
        plan_summary = result_state.get("retrieval_plan_summary") or "none"
        if attempted_flag:
            print(
                "[Memory] Vector store queried "
                f"(plan={plan_summary}, method={'llm' if result_state.get('retrieval_plan_used_llm') else 'heuristic'}, "
                f"results={result_count}, documentation={doc_count}, temporal={temporal_count})."
            )
        else:
            print(
                "[Memory] Vector store not queried "
                f"(plan={plan_summary}, method={'llm' if result_state.get('retrieval_plan_used_llm') else 'heuristic'})."
            )
        result_metadata = result_state.get("vision_metadata")
        result_attached = bool(result_state.get("vision_attached"))
        scrubbed_response = result_state.get("scrubbed_response")
        if scrubbed_response is None:
            scrubbed_response = force_plaintext(result_state.get("response_text", ""))
        think_hidden = bool(result_state.get("think_hidden"))
        think_placeholder = result_state.get("think_placeholder")
        hidden_think = result_state.get("hidden_think")
        entries = self._build_history_entries(
            user_text,
            scrubbed_response,
            result_attached,
            result_metadata,
        )
        return TurnResult(
            correlation_id=correlation_id,
            response_text=result_state.get("response_text", ""),
            raw_response_text=str(raw_response_text or ""),
            history_entries=entries,
            retrieved=retrieved,
            trace_path=trace_path,
            metrics=metrics,
            vector_query_attempted=attempted_flag,
            vector_query_result_count=result_count,
            vector_query_documentation_count=doc_count,
            vector_query_temporal_count=temporal_count,
            retrieval_plan=dict(plan) if isinstance(plan, dict) else {},
            retrieval_plan_used_llm=bool(result_state.get("retrieval_plan_used_llm")),
            think_hidden=think_hidden,
            think_placeholder=str(think_placeholder)
            if isinstance(think_placeholder, str) and think_placeholder
            else None,
            hidden_think=str(hidden_think)
            if isinstance(hidden_think, str) and hidden_think
            else None,
        )

    def _write_trace(self, state: _TurnState) -> Path:
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        path = self.traces_dir / f"run_{timestamp}_{state['correlation_id']}.json"
        trace = {
            "correlation_id": state["correlation_id"],
            "user_text": state.get("user_text"),
            "augmented_text": state.get("augmented_text"),
            "retrieved": state.get("retrieved", []),
            "response_text": state.get("response_text"),
            "errors": state.get("errors", []),
        }
        if state.get("tool_plan"):
            trace["tool_plan"] = state.get("tool_plan")
        if state.get("tool_results"):
            trace["tool_results"] = state.get("tool_results")
        if state.get("tool_plan_summary"):
            trace["tool_plan_summary"] = state.get("tool_plan_summary")
        trace["vision_attached"] = bool(state.get("vision_attached"))
        if state.get("vision_metadata"):
            trace["vision_metadata"] = state.get("vision_metadata")
        if state.get("retrieval_plan"):
            trace["retrieval_plan"] = state.get("retrieval_plan")
        if state.get("retrieval_plan_summary"):
            trace["retrieval_plan_summary"] = state.get("retrieval_plan_summary")
        atomic_write_text(path, json.dumps(trace, indent=2, ensure_ascii=False))
        return path

    def close(self) -> None:
        self._persistence.close()


__all__ = ["MemoryOrchestrator", "TurnResult", "CONVERSATION_MAX_BYTES", "CONVERSATION_KEEP"]
