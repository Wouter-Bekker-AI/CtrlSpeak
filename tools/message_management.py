"""Utilities for deterministic assistant message cleanup."""

from __future__ import annotations

from typing import Iterable

DEFAULT_DROP_CHARS: Iterable[str] = ["*", "#", "_", "`", ">", "|"]
DEFAULT_BULLET_PREFIXES: Iterable[str] = ["-", "+", "•", "*"]


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
