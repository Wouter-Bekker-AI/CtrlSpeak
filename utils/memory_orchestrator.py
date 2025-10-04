# -*- coding: utf-8 -*-
"""LangGraph-based memory orchestrator for SocialRobot."""
from __future__ import annotations

import json
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
from tools.message_management import force_plaintext, requires_force_plaintext, strip_emoji
from tools.goose_tool import goose_query, ALLOWED_MODES as GOOSE_ALLOWED_MODES


CONVERSATION_MAX_BYTES = 10 * 1024 * 1024
CONVERSATION_KEEP = 5
PERSIST_START_DELAY_SECONDS = 0.05

DOCUMENTATION_CATEGORY = "documentation"
TEMPORAL_CATEGORY = "temporal_context"
CHAT_HISTORY_CATEGORY = "chat_history"
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
    "tell me everything you know about me",
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



def _chat_query_variants(query_text: str) -> list[str]:
    lowered = query_text.lower()
    variants: list[str] = []
    seen = set()

    def _add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            variants.append(candidate)
            seen.add(candidate)

    _add(query_text)

    if any(phrase in lowered for phrase in ("what's my name", "what is my name", "tell me my name", "everything you know about me")):
        _add("my name is")

    if any(term in lowered for term in ("how old am i", "what's my age", "what is my age", "age")):
        _add("my age is")
        _add("i am")

    if 'favorite' in lowered or 'favourite' in lowered:
        after = lowered.split('favorite', 1)[1] if 'favorite' in lowered else lowered.split('favourite', 1)[1]
        after = after.strip(' ?!.,')
        if after:
            first_words = after.split()[:2]
            for length in range(1, len(first_words) + 1):
                phrase = ' '.join(first_words[:length])
                _add(f"my favorite {phrase} is")
        _add("my favorite")

    if 'sport' in lowered and 'favorite' not in lowered:
        _add("my favorite sport is")

    if 'everything you know about me' in lowered or 'tell me everything you know about me' in lowered:
        _add("my name is")
        _add("i am")
        _add("i like")

    if lowered.startswith('look at our chat history') or 'chat history' in lowered:
        _add("my name is")
        _add("my age is")

    if not variants:
        variants.append(query_text)
    return variants


def _format_chat_history_response(retrieved_items: List[dict]) -> str:
    facts: List[str] = []
    seen: set[str] = set()
    for item in retrieved_items:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        if str(metadata.get("category", "")) != CHAT_HISTORY_CATEGORY:
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        role = str(metadata.get("role") or "")
        if role == "turn":
            for line in content.splitlines():
                trimmed = line.strip()
                if trimmed and trimmed not in seen:
                    facts.append(trimmed)
                    seen.add(trimmed)
        else:
            if content not in seen:
                facts.append(content)
                seen.add(content)
    if not facts:
        return ""
    summary_lines = ["Here is what our previous conversations mention:"]
    summary_lines.extend(f"- {fact}" for fact in facts[:8])
    return "\\n".join(summary_lines)


TOOL_RESPONSE_CHAR_LIMIT = 4000

_TOOL_EXECUTION_GUARD = (
    "You have access to the goose_tool_query function. Use it for any filesystem or execution task by describing the request in natural language. "
    "Return a tool call whenever the user asks you to inspect, modify, search, list, or run files. Do not fabricate resultsÃ¢â‚¬â€let the tool perform the work."
)

_GOOSE_TOOL_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "goose_tool_query",
            "description": (
                "Delegate complex workspace operations to the Goose CLI. Provide a detailed natural-language prompt that explains"
                " the desired file action (read, edit, create, list, search, execute, etc.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed instruction for Goose describing the desired workspace task.",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Optional Goose mode ('auto', 'smart_approve', 'approve', or 'chat').",
                        "enum": sorted(GOOSE_ALLOWED_MODES),
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional Goose model identifier to override the default.",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Optional Goose provider name (defaults to 'ollama').",
                    },
                    "goose_exe": {
                        "type": "string",
                        "description": "Optional path to the Goose executable (defaults to 'goose').",
                    },
                    "stream": {
                        "type": "boolean",
                        "description": "Stream Goose output live (defaults to false).",
                    },
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        },
    }
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
        prior_history = self._load_history()
        self._prior_history_count = len(prior_history)
        if self._prior_history_count:
            logger.debug(
                "Loaded %d persisted history entries for identity '%s'; starting new session with a fresh runtime history.",
                self._prior_history_count,
                self.identity,
            )
        self.history: List[dict] = []
        self.tooling_enabled = bool(tooling_enabled)
        self._custom_tool_logger: Optional[Callable[[str], None]] = tool_logger
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

    @staticmethod
    def _log_turn_event(message: str) -> None:
        print(f"[LangGraph] {message}")

    @staticmethod
    def _preview_text(value: str, *, limit: int = 200) -> str:
        cleaned = value.replace("\n", " ").strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 1].rstrip() + "â€¦"

    def _plan_tool_actions(self, state: _TurnState) -> List[ToolAction]:
        """Tool selection is delegated entirely to the LLM."""
        return []

    def _default_retrieval_plan(self, text: str) -> Dict[str, bool]:
        lowered = text.lower()
        plan = {"documentation": False, "chat_history": False, "date": False}
        if any(phrase in lowered for phrase in _DOCUMENTATION_INTENT_PHRASES):
            plan["documentation"] = True
        if any(phrase in lowered for phrase in _CHAT_HISTORY_INTENT_PHRASES):
            plan["chat_history"] = True
        if any(phrase in lowered for phrase in _TEMPORAL_INTENT_PHRASES):
            plan["date"] = True
        return plan

    def _build_planner_prompt(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = _REASONING_TAG_PATTERN.sub("", cleaned).strip()
        if "/no_think" not in cleaned.lower():
            cleaned = f"{cleaned} /no_think".strip()
        return cleaned

    @staticmethod
    def _parse_assessment_response(payload: str) -> Optional[Dict[str, bool]]:
        if not payload:
            return None
        normalized = payload.strip().lower()
        if not normalized:
            return None
        tokens = [token.strip() for token in normalized.split(",") if token.strip()]
        if not tokens:
            if normalized == "none":
                return {"documentation": False, "chat_history": False, "date": False}
            return None
        valid = {"documentation", "chat_history", "date", "none"}
        if any(token not in valid for token in tokens):
            return None
        if tokens == ["none"]:
            return {"documentation": False, "chat_history": False, "date": False}
        plan = {"documentation": False, "chat_history": False, "date": False}
        for token in tokens:
            if token == "none":
                continue
            plan[token] = True
        return plan

    @staticmethod
    def _summarize_plan(plan: Dict[str, bool]) -> str:
        if not plan:
            return "none"
        enabled = [name for name, enabled in plan.items() if enabled]
        return ",".join(enabled) if enabled else "none"

    def _documentation_threshold(self) -> float:
        raw = self.identity_settings.get("documentation_threshold")
        try:
            value = float(raw)
        except Exception:
            base = float(self.identity_settings.get("retrieval_threshold", 0.75))
            value = min(base, 0.6)
        return max(0.0, value)

    def _temporal_threshold(self) -> float:
        raw = self.identity_settings.get("temporal_threshold")
        try:
            value = float(raw)
        except Exception:
            base = float(self.identity_settings.get("retrieval_threshold", 0.75))
            value = min(base, 0.5)
        return max(0.0, value)

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
        if chat_intent:
            category_thresholds[CHAT_HISTORY_CATEGORY] = threshold
            fallback_categories[CHAT_HISTORY_CATEGORY] = max(1, min(top_k, 4))
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
        used_query = query_text

        if chat_intent:
            query_candidates = _chat_query_variants(query_text)
        else:
            query_candidates = [query_text]

        if doc_intent or temporal_intent or chat_intent:
            for candidate in query_candidates:
                if not candidate.strip():
                    continue
                attempted = True
                try:
                    candidate_results = self.vector_store.retrieve(
                        candidate,
                        top_k=top_k,
                        threshold=threshold,
                        category_thresholds=category_thresholds,
                        fallback_categories=fallback_categories,
                    )
                except Exception as exc:
                    state.setdefault("errors", []).append({"node": "retrieve", "error": str(exc)})
                    continue
                if candidate_results:
                    results = candidate_results
                    used_query = candidate
                    if candidate != query_text:
                        print(f"[Memory] Chat query adjusted to: {candidate}")
                    break

        if (
            attempted
            and not results
            and chat_intent
            and not doc_intent
            and not temporal_intent
        ):
            try:
                relaxed_threshold = min(threshold, 0.6)
                results = self.vector_store.retrieve(
                    used_query,
                    top_k=top_k,
                    threshold=relaxed_threshold,
                    category_thresholds={CHAT_HISTORY_CATEGORY: relaxed_threshold},
                    fallback_categories={CHAT_HISTORY_CATEGORY: max(1, min(top_k, 4))},
                )
            except Exception as exc:
                state.setdefault("errors", []).append({"node": "retrieve", "error": str(exc)})

        if attempted:
            filtered = []
            for item in results:
                metadata = item.metadata or {}
                category = str(metadata.get("category", "") or "")
                if not category:
                    category = CHAT_HISTORY_CATEGORY
                if category == DOCUMENTATION_CATEGORY:
                    if not doc_intent:
                        continue
                elif category == TEMPORAL_CATEGORY:
                    if not temporal_intent:
                        continue
                elif category == CHAT_HISTORY_CATEGORY:
                    if not chat_intent:
                        continue
                else:
                    if not chat_intent:
                        continue
                filtered.append(item)
            results = filtered

        if attempted and chat_intent:
            chat_results = [
                item
                for item in results
                if str(item.metadata.get("category", "")) == CHAT_HISTORY_CATEGORY
            ]
            if chat_results:
                print("[Memory] Chat history results:")
                for index, item in enumerate(chat_results[:top_k], start=1):
                    role = str(item.metadata.get("role") or "unknown")
                    snippet = item.content.replace("\n", " ").strip()
                    if len(snippet) > 200:
                        snippet = snippet[:197] + "..."
                    print(f"  {index}. ({role}) {snippet}")

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


    def _actions_from_tool_calls(
        self,
        tool_calls: Iterable[Dict[str, Any]],
        *,
        user_text: Optional[str] = None,
    ) -> List[ToolAction]:
        actions: List[ToolAction] = []
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

            if name != "goose_tool_query":
                continue

            prompt_value = str(arguments.get("prompt") or "").strip()
            if not prompt_value:
                continue

            parameters: Dict[str, Any] = {"prompt": prompt_value, "tool_call_index": index}
            for optional_key in ("mode", "model", "provider", "goose_exe"):
                value = arguments.get(optional_key)
                if isinstance(value, str) and value.strip():
                    parameters[optional_key] = value.strip()
            stream_value = arguments.get("stream")
            if isinstance(stream_value, bool):
                parameters["stream"] = stream_value

            preview = prompt_value.replace("\n", " ")
            if len(preview) > 60:
                preview = preview[:57].rstrip() + "â€¦"
            description = f"Goose query: {preview}"
            actions.append(
                ToolAction(
                    kind="goose_query",
                    description=description,
                    source="llm",
                    parameters=parameters,
                )
            )
        return actions

    def _execute_tool_plan(self, actions: List[ToolAction]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for action in actions:
            if action.kind == "goose_query":
                results.append(self._execute_goose_tool_query_action(action))
            else:
                self._emit_tool_event(f"Unsupported tool action '{action.kind}' ignored.")
                results.append({"kind": action.kind, "success": False, "message": "unsupported"})
        return results

    def _execute_goose_tool_query_action(self, action: ToolAction) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "kind": action.kind,
            "description": action.description,
            "success": False,
        }
        parameters = action.parameters or {}
        prompt = str(parameters.get("prompt") or "").strip()
        if not prompt:
            result["message"] = "Goose query missing prompt."
            self._emit_tool_event("Goose tool call rejected: missing prompt.")
            return result

        goose_kwargs: Dict[str, Any] = {}
        for optional_key in ("mode", "model", "provider", "goose_exe"):
            value = parameters.get(optional_key)
            if isinstance(value, str) and value.strip():
                goose_kwargs[optional_key] = value.strip()

        stream_value = parameters.get("stream")
        if isinstance(stream_value, bool) and not stream_value:
            self._emit_tool_event(
                "Goose tool requested non-streaming output; forcing stream=True for terminal visibility."
            )

        goose_kwargs["stream"] = True

        try:
            self._log_turn_event(
                f"Calling Goose tool with prompt: {self._preview_text(prompt)}"
            )
            self._emit_tool_event("Invoking goose_tool_query.")
            output = goose_query(prompt, **goose_kwargs)
        except Exception as exc:
            message = str(exc)
            result["message"] = message
            self._emit_tool_event(f"Goose tool failed: {message}")
            return result

        result.update({
            "success": True,
            "output": output,
            "prompt": prompt,
        })
        self._log_turn_event(
            f"Goose tool completed successfully: {self._preview_text(output)}"
        )
        return result
    def _node_plan_tools(self, state: _TurnState) -> _TurnState:
        if not isinstance(state.get("metrics"), dict):
            state["metrics"] = {}
        if not isinstance(state.get("errors"), list):
            state["errors"] = []
        actions = self._plan_tool_actions(state)
        state["tool_plan"] = [self._serialize_tool_action(action) for action in actions]
        state["tool_plan_summary"] = "none"
        state.pop("tool_probe_hint", None)
        state.pop("tool_probe_pending", None)
        state.pop("tool_probe_force_required", None)
        return state

    def _node_call_tools(self, state: _TurnState) -> _TurnState:
        plan_entries = state.get("tool_plan")
        if not self.tooling_enabled or not isinstance(plan_entries, list) or not plan_entries:
            state.pop("tool_probe_pending", None)
            state.pop("tool_probe_force_required", None)
            return state

        actions: List[ToolAction] = []
        for entry in plan_entries:
            if isinstance(entry, dict):
                actions.append(self._deserialize_tool_action(entry))
        if not actions:
            return state

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
        state.pop("think_missing_answer", None)
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
                    if not filtered_response.strip():
                        state["think_missing_answer"] = True
        state["response_text"] = filtered_response or ""
        if filtered_response:
            self._log_turn_event(
                f"LLM returned response: {self._preview_text(filtered_response)}"
            )
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
        self._log_turn_event(
            f"Sending to LLM without tools: {self._preview_text(str(augmented))}"
        )
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
        stored_state = self._store_llm_response(state, raw_response)
        if stored_state.pop("think_missing_answer", False):
            if stored_state.get("_think_retry_attempted"):
                apology = "I'm sorry, I wasn't able to produce a final answer. Please try again."
                stored_state["response_text"] = apology
                stored_state["raw_response_text"] = apology
                stored_state.pop("_think_retry_attempted", None)
                return stored_state

            stored_state["_think_retry_attempted"] = True
            reminder = (
                "Your previous response only contained a <think> plan. Provide the final Answer now "
                "without using <think>."
            )
            self._log_turn_event(
                "LLM response contained only a hidden plan; requesting the visible answer."
            )
            history.append({"role": "assistant", "content": raw_response})
            try:
                retry_raw = self.llm_client.query(
                    reminder,
                    history=history,
                    content=content_blocks,
                )
            except Exception as exc:
                state.setdefault("errors", []).append({"node": "llm", "error": str(exc)})
                apology = "I'm sorry, I wasn't able to produce a final answer. Please try again."
                stored_state["response_text"] = apology
                stored_state["raw_response_text"] = apology
                stored_state.pop("_think_retry_attempted", None)
                return stored_state

            for key in (
                "response_text",
                "raw_response_text",
                "think_hidden",
                "think_placeholder",
                "hidden_think",
            ):
                stored_state.pop(key, None)

            return self._store_llm_response(state, retry_raw or "")

        stored_state.pop("_think_retry_attempted", None)
        return stored_state

    def _tool_name_for_action(self, action: ToolAction) -> str:
        if action.kind == "goose_query":
            return "goose_tool_query"
        return action.kind

    def _node_llm_with_tools(self, state: _TurnState) -> Optional[_TurnState]:
        augmented = state.get("augmented_text") or state.get("user_text") or ""
        history = self._prepare_history_with_context(state)
        messages: List[dict] = []
        system_prompt = getattr(self.llm_client, "system_prompt", None)
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "system", "content": _TOOL_EXECUTION_GUARD})
        messages.extend(self._normalize_history_for_chat(history))

        user_message = self._build_user_message(state, augmented)
        messages.append(user_message)

        preview_source = augmented
        if isinstance(user_message, dict):
            content = user_message.get("content")
            if isinstance(content, str) and content.strip():
                preview_source = content
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(str(block.get("text") or ""))
                if text_parts:
                    preview_source = " ".join(text_parts)
        self._log_turn_event(
            "Sending to LLM with tools (goose_tool_query available): "
            f"{self._preview_text(str(preview_source))}"
        )

        loop_results: List[Dict[str, Any]] = []
        max_iterations = 6
        for iteration in range(max_iterations):
            try:
                response = self.llm_client.chat(
                    messages,
                    tools=_GOOSE_TOOL_SCHEMA,
                    tool_choice="auto",
                    stream=False,
                )
            except Exception as exc:
                self._emit_tool_event(f"Tool-enabled LLM call failed: {exc}")
                state.setdefault("errors", []).append({"node": "llm", "error": str(exc)})
                return None

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

                for action in actions:
                    tool_name = self._tool_name_for_action(action)
                    parameters = action.parameters or {}
                    prompt_text = str(parameters.get("prompt") or "")
                    if prompt_text:
                        self._log_turn_event(
                            f"LLM requested tool '{tool_name}' with prompt: {self._preview_text(prompt_text)}"
                        )

                results = self._execute_tool_plan(actions)
                loop_results.extend(results)
                state.setdefault("tool_results", []).extend(results)

                for tool_call, action, result in zip(tool_calls, actions, results):
                    tool_name = str(tool_call.get("function", {}).get("name") or self._tool_name_for_action(action))
                    try:
                        serialized = json.dumps(result, ensure_ascii=False)
                    except (TypeError, ValueError):
                        serialized = str(result)
                    self._log_turn_event(
                        f"Handing tool output back to LLM for '{tool_name}': {self._preview_text(serialized)}"
                    )
                    messages.append({"role": "tool", "name": tool_name, "content": serialized})
                continue

            messages.append({"role": "assistant", "content": content})
            stored_state = self._store_llm_response(state, content)
            if stored_state.pop("think_missing_answer", False):
                if stored_state.get("_think_retry_attempted"):
                    apology = "I'm sorry, I wasn't able to produce a final answer. Please try again."
                    stored_state["response_text"] = apology
                    stored_state["raw_response_text"] = apology
                    stored_state.pop("_think_retry_attempted", None)
                    return stored_state

                stored_state["_think_retry_attempted"] = True
                reminder = (
                    "Your previous response only contained a <think> plan. Provide the final Answer now "
                    "without using <think>."
                )
                self._log_turn_event(
                    "LLM response contained only a hidden plan; requesting the visible answer."
                )
                messages.append({"role": "user", "content": reminder})
                for key in (
                    "response_text",
                    "raw_response_text",
                    "think_hidden",
                    "think_placeholder",
                    "hidden_think",
                ):
                    stored_state.pop(key, None)
                continue

            stored_state.pop("_think_retry_attempted", None)
            return stored_state

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
        sanitized_response = strip_emoji(response)
        if requires_force_plaintext(sanitized_response):
            scrubbed_response = force_plaintext(sanitized_response)
        else:
            scrubbed_response = sanitized_response
        scrubbed_response = scrubbed_response.strip()
        if response and not scrubbed_response:
            scrubbed_response = "..."
        state["scrubbed_response"] = scrubbed_response
        user_text = state.get("user_text", "")
        vision_metadata = state.get("vision_metadata")
        attached_image = bool(state.get("vision_attached"))
        tool_results = state.get("tool_results") if isinstance(state.get("tool_results"), list) else None
        entries = self._build_history_entries(
            user_text,
            scrubbed_response,
            attached_image,
            vision_metadata,
            tool_results=tool_results,
        )
        self.history.extend(entries)
        vector_documents = []
        vector_metadata = []

        user_entry = strip_emoji(user_text) if user_text else ""
        if user_entry:
            vector_documents.append(user_entry)
            vector_metadata.append({"role": "user", "correlation_id": correlation_id, "category": CHAT_HISTORY_CATEGORY})

        if scrubbed_response:
            vector_documents.append(scrubbed_response)
            vector_metadata.append({"role": "assistant", "correlation_id": correlation_id, "category": CHAT_HISTORY_CATEGORY})

        if user_entry and scrubbed_response:
            combined_turn = f"User: {user_entry}\nAssistant: {scrubbed_response}"
            vector_documents.append(combined_turn)
            vector_metadata.append({"role": "turn", "correlation_id": correlation_id, "category": CHAT_HISTORY_CATEGORY, "kind": "user_assistant_pair"})
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
        *,
        tool_results: Optional[List[Dict[str, Any]]] = None,
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

        if tool_results:
            entries.extend(self._build_tool_history_entries(tool_results))

        entries.append({"role": "assistant", "content": response_text})
        return entries

    def _build_tool_history_entries(self, tool_results: List[Dict[str, Any]]) -> List[dict]:
        entries: List[dict] = []
        for result in tool_results:
            if not isinstance(result, dict):
                continue

            kind = str(result.get("kind") or "").strip()
            tool_name = "goose_tool_query" if kind == "goose_query" else (kind or "tool")
            output_text = ""
            if isinstance(result.get("output"), str) and result["output"].strip():
                output_text = result["output"].strip()
            elif isinstance(result.get("message"), str) and result["message"].strip():
                output_text = result["message"].strip()
            if not output_text:
                continue

            truncated = self._truncate_tool_output(output_text)
            metadata: Dict[str, Any] = {"tool_name": tool_name, "success": bool(result.get("success"))}
            prompt = result.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                metadata["prompt"] = prompt.strip()
            description = result.get("description")
            if isinstance(description, str) and description.strip():
                metadata["description"] = description.strip()
            if isinstance(result.get("message"), str) and result["message"].strip() and not result.get("success"):
                metadata.setdefault("error", result["message"].strip())
            if len(output_text) > TOOL_RESPONSE_CHAR_LIMIT:
                metadata["truncated"] = True

            entry: Dict[str, Any] = {"role": "tool", "name": tool_name, "content": truncated}
            if metadata:
                entry["metadata"] = metadata
            entries.append(entry)
        return entries

    @staticmethod
    def _truncate_tool_output(text: str) -> str:
        normalized = text.strip()
        if len(normalized) <= TOOL_RESPONSE_CHAR_LIMIT:
            return normalized
        truncated = normalized[:TOOL_RESPONSE_CHAR_LIMIT].rstrip()
        return f"{truncated}\nâ€¦[truncated]"

    def _resolve_vision_payload(
        self,
        user_text: str,
        vision_metadata: Optional[Dict[str, Any]],
    ) -> tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]], bool]:
        if not user_text or not is_image_request(user_text):
            return None, vision_metadata, False

        record = load_identity_image(self.identity)
        if record is None:
            return None, vision_metadata, False

        content_blocks: List[Dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {"type": "image", "image": record.image_b64},
        ]
        metadata: Dict[str, Any] = dict(vision_metadata or {})
        metadata["vision_file"] = str(record.path)
        metadata["vision_updated"] = record.updated_at_iso
        if record.source:
            metadata["vision_source"] = record.source
        metadata["vision_request"] = user_text
        metadata["vision_attached"] = True
        return content_blocks, metadata, True

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
        self._log_turn_event(f"User request received: {self._preview_text(user_text)}")
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
        tool_results_state = result_state.get("tool_results")
        tool_results: Optional[List[Dict[str, Any]]]
        if isinstance(tool_results_state, list):
            tool_results = list(tool_results_state)
        else:
            tool_results = None
        entries = self._build_history_entries(
            user_text,
            scrubbed_response,
            result_attached,
            result_metadata,
            tool_results=tool_results,
        )
        self._log_turn_event(
            f"Final answer prepared for chat: {self._preview_text(result_state.get('response_text', ''))}"
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






