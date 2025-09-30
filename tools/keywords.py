"""Keyword detection helpers for speech-driven tooling triggers."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence


@dataclass(frozen=True, slots=True)
class Keyword:
    """Represents a phrase that should trigger a specific automation."""

    name: str
    pattern: re.Pattern[str]
    category: str
    payload: str

    def search(self, text: str) -> Optional[re.Match[str]]:
        """Search ``text`` for the keyword pattern."""

        return self.pattern.search(text)


@dataclass(frozen=True, slots=True)
class KeywordMatch:
    """A matched keyword along with the regex match details."""

    keyword: Keyword
    match: re.Match[str]


_LOOK_AT_SCREEN = Keyword(
    name="look_at_screen",
    pattern=re.compile(r"\blook at my screen\b", re.IGNORECASE),
    category="vision",
    payload="screen",
)

_LOOK_AT_CLIPBOARD = Keyword(
    name="look_at_clipboard",
    pattern=re.compile(r"\blook at my clipboard\b", re.IGNORECASE),
    category="vision",
    payload="clipboard",
)

_FUZZY_CLIPBOARD_PATTERN = re.compile(
    r"\blook at my\s+(?P<target>[A-Za-z]+(?:[\s_-]+[A-Za-z]+)?)\b",
    re.IGNORECASE,
)

_CLIPBOARD_CANONICAL = "clipboard"
_CLIPBOARD_FUZZY_THRESHOLD = 0.88

VISION_KEYWORDS: tuple[Keyword, ...] = (
    _LOOK_AT_SCREEN,
    _LOOK_AT_CLIPBOARD,
)

_CONVERSATION_START_KEYWORDS: tuple[Keyword, ...] = ()
_CONVERSATION_END_KEYWORDS: tuple[Keyword, ...] = ()

ALL_KEYWORDS: tuple[Keyword, ...] = VISION_KEYWORDS


def _identity_pattern_tokens(name: str) -> str:
    tokens = [token for token in re.split(r"[_\s]+", name.strip()) if token]
    if not tokens:
        return ""
    return r"\s+".join(re.escape(token) for token in tokens)


def _build_identity_keywords(identities: Sequence[str]) -> tuple[tuple[Keyword, ...], tuple[Keyword, ...]]:
    start_keywords: list[Keyword] = []
    end_keywords: list[Keyword] = []
    seen_payloads: set[str] = set()

    for identity in identities:
        normalized_payload = identity.strip()
        if not normalized_payload:
            continue
        if normalized_payload in seen_payloads:
            continue
        seen_payloads.add(normalized_payload)

        pattern_tokens = _identity_pattern_tokens(normalized_payload)
        if not pattern_tokens:
            continue

        start_keywords.append(
            Keyword(
                name=f"chat_with_{normalized_payload.lower()}",
                pattern=re.compile(
                    rf"\bchat with\s+(?P<identity>{pattern_tokens})\b",
                    re.IGNORECASE,
                ),
                category="conversation_start",
                payload=normalized_payload,
            )
        )
        end_keywords.append(
            Keyword(
                name=f"goodbye_{normalized_payload.lower()}",
                pattern=re.compile(
                    rf"\bgoodbye\s+(?P<identity>{pattern_tokens})\b",
                    re.IGNORECASE,
                ),
                category="conversation_end",
                payload=normalized_payload,
            )
        )

    return tuple(start_keywords), tuple(end_keywords)


def configure_identity_keywords(identities: Iterable[str]) -> None:
    """Register conversation keywords for the supplied identity names."""

    normalized_identities = []
    for name in identities:
        if not isinstance(name, str):
            continue
        cleaned = name.strip()
        if not cleaned:
            continue
        normalized_identities.append(cleaned)

    start_keywords, end_keywords = _build_identity_keywords(normalized_identities)

    global _CONVERSATION_START_KEYWORDS, _CONVERSATION_END_KEYWORDS, ALL_KEYWORDS
    _CONVERSATION_START_KEYWORDS = start_keywords
    _CONVERSATION_END_KEYWORDS = end_keywords

    ALL_KEYWORDS = VISION_KEYWORDS + _CONVERSATION_START_KEYWORDS + _CONVERSATION_END_KEYWORDS


def iter_keyword_matches(text: str, keywords: Iterable[Keyword] = ALL_KEYWORDS) -> Iterator[KeywordMatch]:
    """Yield :class:`KeywordMatch` objects for each keyword in ``text``.

    The search is case-insensitive and resilient to surrounding whitespace.
    """

    cleaned = text.strip()
    if not cleaned:
        return

    for keyword in keywords:
        match = keyword.search(cleaned)
        if not match and keyword is _LOOK_AT_CLIPBOARD:
            match = _match_fuzzy_clipboard(cleaned)
        if match:
            yield KeywordMatch(keyword, match)


def find_first_keyword(text: str, keywords: Iterable[Keyword] = ALL_KEYWORDS) -> Optional[KeywordMatch]:
    """Return the first keyword found in ``text`` (if any)."""

    return next(iter_keyword_matches(text, keywords), None)


def detect_vision_keyword(text: str) -> Optional[KeywordMatch]:
    """Detect whether ``text`` contains a vision-related keyword."""

    return find_first_keyword(text, VISION_KEYWORDS)


def get_vision_keyword(payload: str) -> Optional[Keyword]:
    """Retrieve a vision keyword by its payload identifier."""

    normalized = payload.strip().lower()
    for keyword in VISION_KEYWORDS:
        if keyword.payload == normalized:
            return keyword
    return None


def detect_conversation_start_keyword(text: str) -> Optional[KeywordMatch]:
    """Detect whether ``text`` requests switching to another identity."""

    return find_first_keyword(text, _CONVERSATION_START_KEYWORDS)


def detect_conversation_end_keyword(text: str) -> Optional[KeywordMatch]:
    """Detect whether ``text`` requests ending the current conversation."""

    return find_first_keyword(text, _CONVERSATION_END_KEYWORDS)


def get_conversation_start_keyword(payload: str) -> Optional[Keyword]:
    """Retrieve a conversation-start keyword by its payload."""

    normalized = payload.strip()
    for keyword in _CONVERSATION_START_KEYWORDS:
        if keyword.payload.lower() == normalized.lower():
            return keyword
    return None


def get_conversation_end_keyword(payload: str) -> Optional[Keyword]:
    """Retrieve a conversation-end keyword by its payload."""

    normalized = payload.strip()
    for keyword in _CONVERSATION_END_KEYWORDS:
        if keyword.payload.lower() == normalized.lower():
            return keyword
    return None


def _match_fuzzy_clipboard(text: str) -> Optional[re.Match[str]]:
    """Return a regex match when ``text`` approximates the clipboard keyword."""

    for match in _FUZZY_CLIPBOARD_PATTERN.finditer(text):
        target = match.group("target")
        normalized = re.sub(r"[\s_-]+", "", target.lower())
        if not normalized:
            continue
        if normalized == _CLIPBOARD_CANONICAL:
            return match
        similarity = difflib.SequenceMatcher(
            None, normalized, _CLIPBOARD_CANONICAL
        ).ratio()
        if similarity >= _CLIPBOARD_FUZZY_THRESHOLD:
            return match
    return None


__all__ = [
    "Keyword",
    "KeywordMatch",
    "VISION_KEYWORDS",
    "ALL_KEYWORDS",
    "detect_vision_keyword",
    "find_first_keyword",
    "get_vision_keyword",
    "iter_keyword_matches",
    "configure_identity_keywords",
    "detect_conversation_start_keyword",
    "detect_conversation_end_keyword",
    "get_conversation_start_keyword",
    "get_conversation_end_keyword",
]
