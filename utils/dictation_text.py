"""Desktop-only text policy. API/Telegram consumers retain S1's formatting."""
from __future__ import annotations

import re
import unicodedata


def prepare_dictation_text(text: str, *, preserve_formatting: bool = False) -> str:
    """Never allow model text to supply terminal controls or keyboard shortcuts.

    Only explicit formatting opt-in retains internal paragraph breaks. Tabs are
    always spaces (a typed Tab can move focus). Strip trailing breaks so no mode
    appends a submit action. Preserve ordinary Unicode, emoji and joiners; discard
    lone surrogates, C0/C1 controls and bidi overrides that conceal visible text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chars = []
    for char in text:
        if char in "\n\v\f\x85\u2028\u2029":
            chars.append("\n" if preserve_formatting else " ")
        elif char == "\t":
            chars.append(" ")
        elif unicodedata.category(char) in {"Cc", "Cs"}:
            continue
        elif char in "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069":
            continue
        else:
            chars.append(char)
    result = "".join(chars)
    if not preserve_formatting:
        result = re.sub(r" +", " ", result)
    return result.strip()


def control_character_counts(text: str) -> dict[str, int]:
    """Use codepoint names, never literal terminal controls, in diagnostics."""
    counts: dict[str, int] = {}
    for char in text:
        if unicodedata.category(char) in {"Cc", "Cs", "Zl", "Zp"}:
            key = f"U+{ord(char):04X}"
            counts[key] = counts.get(key, 0) + 1
    return counts
