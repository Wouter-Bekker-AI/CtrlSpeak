# -*- coding: utf-8 -*-
"""LangGraph-based memory orchestrator for SocialRobot."""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from utils.config_paths import get_logger
from utils.image_store import is_image_request, load_identity_image
from utils.io_atomic import atomic_append_lines, atomic_write_text
from utils.metrics import MetricsRecorder
from utils.memory_paths import get_bot_conversation_log, get_bot_traces_dir
from utils.memory_settings import load_identity_settings
from utils.vector_memory import RetrievedMemory, VectorMemoryStore
from tools.message_management import force_plaintext, requires_force_plaintext


CONVERSATION_MAX_BYTES = 10 * 1024 * 1024
CONVERSATION_KEEP = 5
PERSIST_START_DELAY_SECONDS = 0.05

DOCUMENTATION_CATEGORY = "documentation"
TEMPORAL_CATEGORY = "temporal_context"
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


logger = get_logger(__name__)


class _TurnState(TypedDict, total=False):
    correlation_id: str
    user_text: str
    augmented_text: str
    content_blocks: Optional[List[dict]]
    history: List[dict]
    retrieved: List[Dict[str, Any]]
    response_text: str
    metrics: Dict[str, float]
    errors: List[Dict[str, Any]]
    vision_metadata: Optional[Dict[str, Any]]
    vision_attached: bool


@dataclass
class TurnResult:
    correlation_id: str
    response_text: str
    history_entries: List[dict]
    retrieved: List[RetrievedMemory]
    trace_path: Path
    metrics: Dict[str, float]
    vector_query_attempted: bool
    vector_query_result_count: int
    vector_query_documentation_count: int
    vector_query_temporal_count: int


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
    ) -> None:
        self.identity = identity
        self.llm_client = llm_client
        self.memory_dir = memory_dir
        self.identity_settings = (
            dict(identity_settings)
            if identity_settings is not None
            else load_identity_settings(identity)
        )
        self.conversation_log = get_bot_conversation_log(identity)
        self.traces_dir = get_bot_traces_dir(identity)
        self.metrics = MetricsRecorder(metrics_path)
        self.vector_store = VectorMemoryStore(identity)
        self.history: List[dict] = self._load_history()
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
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("plan_tools", self._node_plan_tools)
        graph.add_node("call_tools", self._node_call_tools)
        graph.add_node("llm", self._node_llm)
        graph.add_node("persist", self._node_persist)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "plan_tools")
        graph.add_edge("plan_tools", "call_tools")
        graph.add_edge("call_tools", "llm")
        graph.add_edge("llm", "persist")
        graph.add_edge("persist", END)
        return graph.compile()

    def _documentation_threshold(self) -> float:
        value = self.identity_settings.get("documentation_retrieval_threshold")
        try:
            return float(value)
        except Exception:
            return 0.35

    @staticmethod
    def _should_prioritize_documentation(text: str) -> bool:
        if not text:
            return False
        lowered = text.casefold()
        stripped = lowered.strip()
        if stripped in {"help", "docs", "documentation", "manual", "guide"}:
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
    def _node_retrieve(self, state: _TurnState) -> _TurnState:
        threshold = float(self.identity_settings.get("retrieval_threshold", 0.75))
        top_k = int(self.identity_settings.get("retrieval_top_k", 5))
        query_text = state.get("augmented_text") or state.get("user_text") or ""

        doc_intent = self._should_prioritize_documentation(query_text)
        temporal_intent = self._should_prioritize_temporal(query_text)
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

        if query_text:
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

        if attempted and not results and not doc_intent and not temporal_intent and query_text:
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
        if metrics:
            container = state.get("metrics")
            if not isinstance(container, dict):
                container = {}
                state["metrics"] = container
            container.update(metrics)
        return state

    def _node_plan_tools(self, state: _TurnState) -> _TurnState:
        # Placeholder for future tool planning.
        if not isinstance(state.get("metrics"), dict):
            state["metrics"] = {}
        if not isinstance(state.get("errors"), list):
            state["errors"] = []
        return state

    def _node_call_tools(self, state: _TurnState) -> _TurnState:
        # Currently there are no blocking tool calls.
        return state

    def _node_llm(self, state: _TurnState) -> _TurnState:
        augmented = state.get("augmented_text") or state.get("user_text") or ""
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
        content_blocks = state.get("content_blocks")
        try:
            response = self.llm_client.query(
                augmented,
                history=history,
                content=content_blocks,
            )
        except Exception as exc:
            state.setdefault("errors", []).append({"node": "llm", "error": str(exc)})
            response = "I'm having trouble responding right now."
        state["response_text"] = response or ""
        return state

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
        if attempted_flag:
            print(
                "[Memory] Vector store queried "
                f"(results={result_count}, documentation={doc_count}, temporal={temporal_count})."
            )
        else:
            print("[Memory] Vector store not queried.")
        result_metadata = result_state.get("vision_metadata")
        result_attached = bool(result_state.get("vision_attached"))
        scrubbed_response = result_state.get("scrubbed_response")
        if scrubbed_response is None:
            scrubbed_response = force_plaintext(result_state.get("response_text", ""))
        entries = self._build_history_entries(
            user_text,
            scrubbed_response,
            result_attached,
            result_metadata,
        )
        return TurnResult(
            correlation_id=correlation_id,
            response_text=result_state.get("response_text", ""),
            history_entries=entries,
            retrieved=retrieved,
            trace_path=trace_path,
            metrics=metrics,
            vector_query_attempted=attempted_flag,
            vector_query_result_count=result_count,
            vector_query_documentation_count=doc_count,
            vector_query_temporal_count=temporal_count,
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
        trace["vision_attached"] = bool(state.get("vision_attached"))
        if state.get("vision_metadata"):
            trace["vision_metadata"] = state.get("vision_metadata")
        atomic_write_text(path, json.dumps(trace, indent=2, ensure_ascii=False))
        return path

    def close(self) -> None:
        self._persistence.close()


__all__ = ["MemoryOrchestrator", "TurnResult", "CONVERSATION_MAX_BYTES", "CONVERSATION_KEEP"]
