"""Canonical Whisper language names and ordered output-language policies."""
from __future__ import annotations

from collections.abc import Iterable


MAX_ALLOWED_OUTPUT_LANGUAGES = 5

# Whisper/faster-whisper language codes. Keep this mapping dependency-free so
# settings can be validated before model libraries are imported.
WHISPER_LANGUAGES: dict[str, str] = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "as": "Assamese",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fo": "Faroese",
    "fr": "French",
    "gl": "Galician",
    "gu": "Gujarati",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lb": "Luxembourgish",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mi": "Maori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Myanmar",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Nynorsk",
    "no": "Norwegian",
    "oc": "Occitan",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tr": "Turkish",
    "tt": "Tatar",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "yue": "Cantonese",
    "zh": "Chinese",
}

_NAME_TO_CODE = {name.casefold(): code for code, name in WHISPER_LANGUAGES.items()}


def normalize_allowed_output_languages(
    value: object,
    *,
    source: str = "allowed output languages",
) -> tuple[str, ...]:
    """Validate, canonicalize, deduplicate, and preserve the requested order."""
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        candidates: Iterable[object] = value.split(",")
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        raise ValueError(f"{source} must be a comma-separated string or a list of language codes")

    result: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            raise ValueError(f"{source} must contain only language names or codes")
        normalized = candidate.strip().casefold()
        if not normalized:
            continue
        code = normalized if normalized in WHISPER_LANGUAGES else _NAME_TO_CODE.get(normalized)
        if code is None:
            raise ValueError(f"Unsupported Whisper language in {source}: {candidate!r}")
        if code not in result:
            result.append(code)
    if len(result) > MAX_ALLOWED_OUTPUT_LANGUAGES:
        raise ValueError(
            f"{source} supports at most {MAX_ALLOWED_OUTPUT_LANGUAGES} languages; "
            "put the preferred fallback first"
        )
    return tuple(result)


def is_valid_allowed_output_languages_setting(value: object) -> bool:
    try:
        normalized = normalize_allowed_output_languages(value)
    except ValueError:
        return False
    return isinstance(value, list) and list(normalized) == value


def language_choices() -> tuple[tuple[str, str], ...]:
    """Put the most likely local choices first, then list the rest by name."""
    preferred = ("en", "af")
    remaining = sorted(
        ((code, name) for code, name in WHISPER_LANGUAGES.items() if code not in preferred),
        key=lambda item: item[1].casefold(),
    )
    return tuple((code, WHISPER_LANGUAGES[code]) for code in preferred) + tuple(remaining)


def language_policy_display(codes: Iterable[str]) -> str:
    values = tuple(codes)
    if not values:
        return "Automatic (no restriction)"
    return ", ".join(f"{WHISPER_LANGUAGES[code]} ({code})" for code in values)


def choose_allowed_language(allowed: Iterable[str], detected: str | None) -> str | None:
    """Return an allowed detected language, otherwise the ordered safe fallback."""
    values = tuple(allowed)
    if not values:
        return None
    normalized_detected = str(detected or "").strip().casefold()
    return normalized_detected if normalized_detected in values else values[0]
