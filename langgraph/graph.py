"""Minimal graph executor compatible with utils.memory_orchestrator tests."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

END = "__langgraph_end__"


class _CompiledGraph:
    def __init__(
        self,
        nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
        edges: Dict[str, List[str]],
        entry: str,
    ) -> None:
        self._nodes = nodes
        self._edges = edges
        self._entry = entry

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current = self._entry
        current_state = dict(state)
        while current and current != END:
            node = self._nodes[current]
            result = node(current_state)
            if result is not None:
                current_state = result
            next_nodes = self._edges.get(current) or []
            if not next_nodes:
                break
            next_node = next_nodes[0]
            if next_node == END:
                break
            current = next_node
        return current_state


class StateGraph:
    def __init__(self, _state_type: Any) -> None:
        self._nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self._edges: Dict[str, List[str]] = {}
        self._entry: Optional[str] = None

    def add_node(self, name: str, func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self._nodes[name] = func

    def set_entry_point(self, name: str) -> None:
        self._entry = name

    def add_edge(self, src: str, dest: str) -> None:
        self._edges.setdefault(src, []).append(dest)

    def compile(self) -> _CompiledGraph:
        if self._entry is None:
            raise ValueError("Entry point not set for StateGraph")
        return _CompiledGraph(dict(self._nodes), dict(self._edges), self._entry)


__all__ = ["END", "StateGraph"]
