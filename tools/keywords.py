"""Keyword detection helpers for speech-driven tooling triggers."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional, Sequence


DEFAULT_FUZZY_THRESHOLD = 0.85


@dataclass(frozen=True, slots=True)
class Keyword:
    """Represents a phrase that should trigger a specific automation."""

    name: str
    pattern: re.Pattern[str]
    category: str
    payload: str
    fuzzy_targets: tuple[str, ...] = ()
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD

    def search(self, text: str) -> Optional[re.Match[str]]:
        """Search ``text`` for the keyword pattern."""

        return self.pattern.search(text)


@dataclass(frozen=True, slots=True)
class KeywordMatch:
    """A matched keyword along with the regex match details."""

    keyword: Keyword
    match: Optional[re.Match[str]] = None


_LOOK_AT_SCREEN = Keyword(
    name="look_at_screen",
    pattern=re.compile(r"\blook at my screen\b", re.IGNORECASE),
    category="vision",
    payload="screen",
    fuzzy_targets=("look at my screen",),
)

_LOOK_AT_CLIPBOARD = Keyword(
    name="look_at_clipboard",
    pattern=re.compile(r"\blook at my clipboard\b", re.IGNORECASE),
    category="vision",
    payload="clipboard",
    fuzzy_targets=("look at my clipboard",),
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

_QUIT_CTRL_SPEAK = Keyword(
    name="quit_ctrlspeak",
    pattern=re.compile(r"\bquit\s+control\s+speak\b", re.IGNORECASE),
    category="system",
    payload="quit_ctrlspeak",
    fuzzy_targets=(
        "quit control speak",
        "quit ctrl speak",
        "quit controlspeak",
    ),
    fuzzy_threshold=0.83,
)

SYSTEM_KEYWORDS: tuple[Keyword, ...] = (
    _QUIT_CTRL_SPEAK,
)

_UPDATE_DOCUMENTATION = Keyword(
    name="update_documentation",
    pattern=re.compile(r"\bupdate documentation\b", re.IGNORECASE),
    category="memory_maintenance",
    payload="documentation",
    fuzzy_targets=("update documentation", "refresh documentation"),
    fuzzy_threshold=0.88,
)

_UPDATE_DATETIME = Keyword(
    name="update_datetime",
    pattern=re.compile(r"\bupdate\s+datetime\b", re.IGNORECASE),
    category="memory_maintenance",
    payload="datetime",
    fuzzy_targets=(
        "update datetime",
        "update date time",
        "refresh datetime",
        "refresh date time",
    ),
    fuzzy_threshold=0.84,
)

MEMORY_KEYWORDS: tuple[Keyword, ...] = (
    _UPDATE_DOCUMENTATION,
    _UPDATE_DATETIME,
)

_CONVERSATION_START_KEYWORDS: tuple[Keyword, ...] = ()
_CONVERSATION_END_KEYWORDS: tuple[Keyword, ...] = ()

ALL_KEYWORDS: tuple[Keyword, ...] = VISION_KEYWORDS + MEMORY_KEYWORDS + SYSTEM_KEYWORDS


_IDENTITY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "reception": ("receptionist",),
}


def _identity_tokens(name: str) -> list[str]:
    return [token for token in re.split(r"[_\s]+", name.strip()) if token]


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

        identity_tokens = _identity_tokens(normalized_payload)
        if not identity_tokens:
            continue

        identity_lower = normalized_payload.lower()
        synonym_tokens: list[list[str]] = []
        for synonym in _IDENTITY_SYNONYMS.get(identity_lower, ()):  # type: ignore[arg-type]
            tokens = _identity_tokens(synonym)
            if tokens:
                synonym_tokens.append(tokens)

        pattern_variants = [r"\s+".join(re.escape(token) for token in identity_tokens)]
        pattern_variants.extend(
            r"\s+".join(re.escape(token) for token in tokens) for tokens in synonym_tokens
        )

        fuzzy_variants = [" ".join(token.lower() for token in identity_tokens)]
        fuzzy_variants.extend(" ".join(token.lower() for token in tokens) for tokens in synonym_tokens)

        combined_pattern = "|".join(pattern_variants)
        start_fuzzy_targets = tuple(f"chat with {variant}" for variant in fuzzy_variants)

        start_keywords.append(
            Keyword(
                name=f"chat_with_{identity_lower}",
                pattern=re.compile(
                    rf"\bchat with\s+(?P<identity>(?:{combined_pattern}))\b",
                    re.IGNORECASE,
                ),
                category="conversation_start",
                payload=normalized_payload,
                fuzzy_targets=start_fuzzy_targets,
            )
        )

        end_fuzzy_targets = tuple(f"goodbye {variant}" for variant in fuzzy_variants)

        end_keywords.append(
            Keyword(
                name=f"goodbye_{identity_lower}",
                pattern=re.compile(
                    rf"\bgoodbye\s+(?P<identity>(?:{combined_pattern}))\b",
                    re.IGNORECASE,
                ),
                category="conversation_end",
                payload=normalized_payload,
                fuzzy_targets=end_fuzzy_targets,
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

    ALL_KEYWORDS = (
        VISION_KEYWORDS + MEMORY_KEYWORDS + _CONVERSATION_START_KEYWORDS + _CONVERSATION_END_KEYWORDS
    )


def iter_keyword_matches(text: str, keywords: Iterable[Keyword] = ALL_KEYWORDS) -> Iterator[KeywordMatch]:
    """Yield :class:`KeywordMatch` objects for each keyword in ``text``.

    The search is case-insensitive and resilient to surrounding whitespace.
    """

    cleaned = text.strip()
    if not cleaned:
        return

    fuzzy_tokens = _tokenize_for_fuzzy(cleaned)

    for keyword in keywords:
        match = keyword.search(cleaned)
        if not match and keyword is _LOOK_AT_CLIPBOARD:
            match = _match_fuzzy_clipboard(cleaned)
        if match:
            yield KeywordMatch(keyword, match)
            continue

        if _fuzzy_keyword_match(keyword, fuzzy_tokens):
            yield KeywordMatch(keyword, None)


def find_first_keyword(text: str, keywords: Iterable[Keyword] = ALL_KEYWORDS) -> Optional[KeywordMatch]:
    """Return the first keyword found in ``text`` (if any)."""

    return next(iter_keyword_matches(text, keywords), None)


def detect_vision_keyword(text: str) -> Optional[KeywordMatch]:
    """Detect whether ``text`` contains a vision-related keyword."""

    return find_first_keyword(text, VISION_KEYWORDS)


def detect_memory_refresh_keyword(text: str) -> Optional[KeywordMatch]:
    """Detect whether ``text`` requests refreshing documentation memory."""

    return find_first_keyword(text, MEMORY_KEYWORDS)


def detect_system_keyword(text: str) -> Optional[KeywordMatch]:
    """Detect whether ``text`` contains a system-level keyword."""

    return find_first_keyword(text, SYSTEM_KEYWORDS)


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


def _tokenize_for_fuzzy(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]


def _fuzzy_keyword_match(keyword: Keyword, tokens: Sequence[str]) -> bool:
    if not keyword.fuzzy_targets or not tokens:
        return False

    for target in keyword.fuzzy_targets:
        target_tokens = [token for token in target.split() if token]
        window_size = len(target_tokens)
        if not window_size or len(tokens) < window_size:
            continue

        for start in range(len(tokens) - window_size + 1):
            candidate_tokens = tokens[start : start + window_size]
            candidate = " ".join(candidate_tokens)
            similarity = difflib.SequenceMatcher(None, candidate, target).ratio()
            if similarity >= keyword.fuzzy_threshold:
                return True

    return False


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
    "MEMORY_KEYWORDS",
    "SYSTEM_KEYWORDS",
    "ALL_KEYWORDS",
    "detect_vision_keyword",
    "detect_memory_refresh_keyword",
    "detect_system_keyword",
    "find_first_keyword",
    "get_vision_keyword",
    "iter_keyword_matches",
    "configure_identity_keywords",
    "detect_conversation_start_keyword",
    "detect_conversation_end_keyword",
]
