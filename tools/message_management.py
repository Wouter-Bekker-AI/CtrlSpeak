"""Utilities for deterministic assistant message cleanup."""

from __future__ import annotations

from typing import Iterable

import re

DEFAULT_DROP_CHARS: Iterable[str] = ["*", "#", "_", "`", ">", "|"]
DEFAULT_BULLET_PREFIXES: Iterable[str] = ["-", "+", "*"]

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]",
    flags=re.UNICODE,
)
_VARIATION_SELECTORS = ("\uFE0E", "\uFE0F")
_ZWJ = "\u200D"


def requires_force_plaintext(
    text: str,
    drop_chars: Iterable[str] = DEFAULT_DROP_CHARS,
    bullet_prefixes: Iterable[str] = DEFAULT_BULLET_PREFIXES,
) -> bool:
    """Return ``True`` when ``force_plaintext`` would alter ``text``."""

    if not text:
        return False

    if any(char in text for char in drop_chars):
        return True

    for line in text.splitlines():
        trimmed = line.lstrip()
        for prefix in bullet_prefixes:
            if trimmed.startswith(prefix + " ") or trimmed == prefix:
                return True

    return False


def force_plaintext(
    text: str,
    drop_chars: Iterable[str] = DEFAULT_DROP_CHARS,
    bullet_prefixes: Iterable[str] = DEFAULT_BULLET_PREFIXES,
) -> str:
    """Deterministically scrub assistant text for text-to-speech systems.

    The cleanup performs three passes:
    1. Remove all characters listed in ``drop_chars``.
    2. Strip common bullet prefixes/indentation from each line.
    3. Collapse runs of internal whitespace while preserving line breaks.
    """
    if not text:
        return text

    translation_table = {ord(char): None for char in drop_chars}
    scrubbed = text.translate(translation_table)

    cleaned_lines: list[str] = []
    for line in scrubbed.splitlines():
        trimmed = line.lstrip()
        for prefix in bullet_prefixes:
            if trimmed.startswith(prefix + " "):
                trimmed = trimmed[len(prefix) + 1 :]
                break
            if trimmed == prefix:
                trimmed = ""
                break
        cleaned_lines.append(trimmed)

    normalized = "\n".join(" ".join(segment.split()) for segment in cleaned_lines)
    return normalized.strip()


def strip_emoji(text: str) -> str:
    """Remove emoji characters and variation selectors from text."""
    if not text:
        return text
    stripped = _EMOJI_PATTERN.sub('', text)
    for selector in _VARIATION_SELECTORS:
        stripped = stripped.replace(selector, '')
    stripped = stripped.replace(_ZWJ, '')
    return stripped

