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

from utils.io_atomic import atomic_append_lines, atomic_write_text
from utils.metrics import MetricsRecorder
from utils.memory_paths import get_bot_conversation_log, get_bot_traces_dir
from utils.memory_settings import load_identity_settings
from utils.vector_memory import RetrievedMemory, VectorMemoryStore


CONVERSATION_MAX_BYTES = 10 * 1024 * 1024
CONVERSATION_KEEP = 5
PERSIST_START_DELAY_SECONDS = 0.01


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


@dataclass
class TurnResult:
    correlation_id: str
    response_text: str
    history_entries: List[dict]
    retrieved: List[RetrievedMemory]
    trace_path: Path
    metrics: Dict[str, float]


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

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------
    def _node_retrieve(self, state: _TurnState) -> _TurnState:
        threshold = float(self.identity_settings.get("retrieval_threshold", 0.75))
        top_k = int(self.identity_settings.get("retrieval_top_k", 5))
        results = []
        metrics: Dict[str, float] = {}
        try:
            results = self.vector_store.retrieve(
                state.get("augmented_text") or state.get("user_text") or "",
                top_k=top_k,
                threshold=threshold,
            )
        except Exception as exc:
            state.setdefault("errors", []).append({"node": "retrieve", "error": str(exc)})
        if results:
            avg_similarity = sum(item.similarity for item in results) / len(results)
            metrics["retrieval_hits"] = float(len(results))
            metrics["avg_similarity"] = round(avg_similarity, 4)
        state["retrieved"] = [
            {"content": item.content, "metadata": item.metadata, "similarity": item.similarity}
            for item in results
        ]
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
            context_lines = [f"- {item['content']}" for item in retrieved]
            history.append({"role": "system", "content": "Relevant memory:\n" + "\n".join(context_lines)})
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
        user_text = state.get("user_text", "")
        content_blocks = state.get("content_blocks")
        vision_metadata = state.get("vision_metadata")
        entries = self._build_history_entries(user_text, response, content_blocks, vision_metadata)
        self.history.extend(entries)
        vector_documents = [user_text, response]
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
        content_blocks: Optional[List[dict]],
        vision_metadata: Optional[Dict[str, Any]],
    ) -> List[dict]:
        entries: List[dict] = []
        if content_blocks:
            record: Dict[str, Any] = {"role": "user", "content": content_blocks}
            if vision_metadata:
                record["metadata"] = vision_metadata
            entries.append(record)
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
        initial_state: _TurnState = {
            "correlation_id": correlation_id,
            "user_text": user_text,
            "augmented_text": augmented_text or user_text,
            "content_blocks": content_blocks,
            "history": list(self.history),
            "vision_metadata": vision_metadata,
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
        entries = self._build_history_entries(
            user_text,
            result_state.get("response_text", ""),
            content_blocks,
            vision_metadata,
        )
        return TurnResult(
            correlation_id=correlation_id,
            response_text=result_state.get("response_text", ""),
            history_entries=entries,
            retrieved=retrieved,
            trace_path=trace_path,
            metrics=metrics,
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
        atomic_write_text(path, json.dumps(trace, indent=2, ensure_ascii=False))
        return path

    def close(self) -> None:
        self._persistence.close()


__all__ = ["MemoryOrchestrator", "TurnResult", "CONVERSATION_MAX_BYTES", "CONVERSATION_KEEP"]
