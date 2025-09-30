# -*- coding: utf-8 -*-
"""Lightweight PII redaction helpers for embeddings."""
from __future__ import annotations

import re
from typing import Iterable

_EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\d)"
)
_ID_RE = re.compile(r"\b(?:[A-Z]{2,5}\d{3,8}|\d{3}-\d{2}-\d{4})\b", re.IGNORECASE)

_PLACEHOLDER = "[REDACTED]"


def _apply(pattern: re.Pattern[str], text: str) -> str:
    return pattern.sub(_PLACEHOLDER, text)


def redact_text(text: str, enabled: bool = True) -> str:
    """Return ``text`` with simple email/phone/ID tokens removed when enabled."""

    if not enabled or not text:
        return text
    redacted = text
    for pattern in (_EMAIL_RE, _PHONE_RE, _ID_RE):
        redacted = _apply(pattern, redacted)
    return redacted


def redact_iterable(values: Iterable[str], enabled: bool = True) -> list[str]:
    """Apply :func:`redact_text` to ``values`` preserving order."""

    return [redact_text(value, enabled=enabled) for value in values]


__all__ = ["redact_text", "redact_iterable"]
