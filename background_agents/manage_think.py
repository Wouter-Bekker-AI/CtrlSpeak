"""Helpers for hiding <think> plans from chat history and TTS."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_PREFIX_PATTERN = re.compile(r"^answer:\s*", re.IGNORECASE)


@dataclass
class ManageThinkResult:
    """Result of filtering an LLM response for <think> blocks."""

    visible_text: str
    hidden_think: Optional[str]
    removed: bool
    placeholder_text: str


class ManageThinkAgent:
    """Utility that strips <think> plans when hide_think is enabled."""

    def __init__(self, placeholder_text: str = "Thinking...") -> None:
        self.placeholder_text = placeholder_text

    def filter_response(self, text: str) -> ManageThinkResult:
        if not text:
            return ManageThinkResult("", None, False, self.placeholder_text)

        hidden_segments: List[str] = []

        def _replacer(match: re.Match[str]) -> str:
            content = match.group(1).strip()
            if content:
                hidden_segments.append(content)
            return ""

        visible = _THINK_PATTERN.sub(_replacer, text)
        visible = self._normalize_visible_text(visible)
        hidden = "\n\n".join(hidden_segments) if hidden_segments else None
        return ManageThinkResult(
            visible_text=visible,
            hidden_think=hidden,
            removed=bool(hidden_segments),
            placeholder_text=self.placeholder_text,
        )

    @staticmethod
    def _normalize_visible_text(text: str) -> str:
        collapsed = re.sub(r"\n{3,}", "\n\n", text)
        stripped = collapsed.strip()
        return _ANSWER_PREFIX_PATTERN.sub("", stripped, count=1)


def load_manage_think_agent(placeholder_text: str = "Thinking...") -> ManageThinkAgent:
    """Factory that mirrors the background agent loader pattern."""

    return ManageThinkAgent(placeholder_text=placeholder_text)
