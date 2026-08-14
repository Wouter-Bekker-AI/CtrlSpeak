"""Server-side validation and selection for CtrlSpeak language allowlists."""
from __future__ import annotations


MAX_ALLOWED_LANGUAGES = 5
SUPPORTED_LANGUAGE_CODES = frozenset(
    "af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr "
    "gl gu ha haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb ln lo lt lv "
    "mg mi mk ml mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si sk sl sn "
    "so sq sr su sv sw ta te tg th tk tl tr tt uk ur uz vi yi yo yue zh".split()
)


def normalize_language_policy(
    allowed_languages: str | None,
    legacy_language: str | None,
) -> tuple[str, ...]:
    """Return a validated ordered policy while retaining legacy `language` support."""
    candidates = []
    if allowed_languages:
        candidates.extend(allowed_languages.split(","))
    if legacy_language:
        legacy = legacy_language.strip().casefold()
        if candidates and legacy not in {item.strip().casefold() for item in candidates}:
            raise ValueError("language must be included in allowed_languages when both are supplied")
        if not candidates:
            candidates.append(legacy)

    result: list[str] = []
    for candidate in candidates:
        code = candidate.strip().casefold()
        if not code:
            continue
        if code not in SUPPORTED_LANGUAGE_CODES:
            raise ValueError(f"unsupported Whisper language code: {candidate!r}")
        if code not in result:
            result.append(code)
    if len(result) > MAX_ALLOWED_LANGUAGES:
        raise ValueError(
            f"allowed_languages accepts at most {MAX_ALLOWED_LANGUAGES} unique language codes"
        )
    return tuple(result)


def choose_allowed_language(allowed: tuple[str, ...], detected: str | None) -> str | None:
    if not allowed:
        return None
    candidate = str(detected or "").strip().casefold()
    return candidate if candidate in allowed else allowed[0]
