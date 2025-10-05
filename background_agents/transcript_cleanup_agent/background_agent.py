from __future__ import annotations

import json
import threading

from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Iterable, Optional, Sequence

from utils.config_paths import get_data_dir, get_logs_dir, get_logger
from utils.io_atomic import atomic_append_lines

_DEFAULT_AGENT_DIR = Path(__file__).resolve().parent

logger = get_logger(__name__)

_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_BOUNDARY_TEMPLATE = r"(?<![A-Za-z0-9]){token}(?![A-Za-z0-9])"


@dataclass(frozen=True)
class BackgroundAgentResources:
    """Static resources and configuration for the transcript cleanup agent."""

    base_path: Path
    config: dict
    base_variant_map: dict[str, tuple[str, ...]]
    supplemental_phrases: tuple[str, ...]
    phrase_thresholds: dict[str, float]
    default_threshold: float
    log_path: Path
    max_log_bytes: int
    log_keep: int
    custom_variant_map_path: Path


@dataclass(frozen=True)
class TranscriptCorrection:
    """A single correction applied to the transcript."""

    original: str
    replacement: str
    reason: str
    score: Optional[float] = None


@dataclass(frozen=True)
class TranscriptCleanupResult:
    """Result payload returned after normalization."""

    text: str
    corrections: tuple[TranscriptCorrection, ...]


class TranscriptCleanupAgent:
    """Normalizes Whisper transcripts prior to keyword routing."""

    def __init__(self, resources: BackgroundAgentResources) -> None:
        self._resources = resources
        self._lock = threading.RLock()
        self._base_variant_map = dict(resources.base_variant_map)
        self._custom_variant_map: dict[str, tuple[str, ...]] = {}
        self._custom_variant_map_path = resources.custom_variant_map_path
        self._custom_variant_mtime: Optional[int] = None
        self._custom_variant_size: Optional[int] = None

        self._variant_patterns: list[tuple[re.Pattern[str], str, str]] = []
        self._supplemental_phrases = tuple(resources.supplemental_phrases)
        self._static_tracked_phrases: set[str] = {
            phrase.strip() for phrase in self._supplemental_phrases if phrase.strip()
        }
        self._static_tracked_phrases.update(_collect_keyword_phrases())
        self._tracked_phrases: set[str] = set(self._static_tracked_phrases)

        self._phrase_thresholds = dict(resources.phrase_thresholds)
        self._default_threshold = max(0.0, min(1.0, resources.default_threshold))

        self._rebuild_variant_state()
        self._refresh_custom_variants_if_needed(force=True)

    @property
    def tracked_phrases(self) -> tuple[str, ...]:
        with self._lock:
            self._refresh_custom_variants_if_needed()
            return tuple(sorted({phrase for phrase in self._tracked_phrases if phrase}))

    def normalize(self, text: str) -> TranscriptCleanupResult:
        with self._lock:
            return self._normalize_locked(text)

    def _normalize_locked(self, text: str) -> TranscriptCleanupResult:
        self._refresh_custom_variants_if_needed()

        if not text:
            return TranscriptCleanupResult(text=text, corrections=())

        working = text
        corrections: list[TranscriptCorrection] = []

        for pattern, canonical, reason in self._variant_patterns:
            def _replacer(match: re.Match[str]) -> str:
                original = match.group(0)
                if original.lower() == canonical.lower():
                    return original
                replacement = _preserve_case(canonical, original)
                corrections.append(
                    TranscriptCorrection(
                        original=original,
                        replacement=replacement,
                        reason=reason,
                    )
                )
                return replacement

            working = pattern.sub(_replacer, working)

        fuzzy_replacements = self._detect_fuzzy_replacements(working)
        if fuzzy_replacements:
            working, fuzzy_corrections = _apply_replacements(working, fuzzy_replacements)
            corrections.extend(fuzzy_corrections)

        result = TranscriptCleanupResult(text=working, corrections=tuple(corrections))

        if corrections:
            self._log_corrections(text, result)

        return result

    def _detect_fuzzy_replacements(self, text: str) -> list[tuple[int, int, str, TranscriptCorrection]]:
        tokens = list(_WORD_PATTERN.finditer(text))
        if not tokens:
            return []

        replacements: list[tuple[int, int, str, TranscriptCorrection]] = []
        for phrase in self._tracked_phrases:
            canonical = phrase.strip()
            if not canonical:
                continue
            canonical_lower = canonical.lower()
            word_count = len(canonical_lower.split())
            if word_count == 0 or word_count > len(tokens):
                continue
            threshold = self._phrase_thresholds.get(canonical_lower, self._default_threshold)
            for index in range(len(tokens) - word_count + 1):
                start = tokens[index].start()
                end = tokens[index + word_count - 1].end()
                candidate = text[start:end]
                candidate_lower = candidate.lower()
                if candidate_lower == canonical_lower:
                    continue
                score = SequenceMatcher(None, candidate_lower, canonical_lower).ratio()
                if score >= threshold:
                    replacement = _preserve_case(canonical, candidate)
                    correction = TranscriptCorrection(
                        original=candidate,
                        replacement=replacement,
                        reason=f"fuzzy:{canonical_lower}",
                        score=round(score, 4),
                    )
                    replacements.append((start, end, replacement, correction))
        return replacements

    def _refresh_custom_variants_if_needed(self, *, force: bool = False) -> None:
        path = self._custom_variant_map_path

        try:
            stat_result = path.stat()
        except FileNotFoundError:
            if force or self._custom_variant_map:
                self._custom_variant_map = {}
                self._custom_variant_mtime = None
                self._custom_variant_size = None
                self._rebuild_variant_state()
            return
        except Exception:
            logger.exception("Failed to stat custom transcript variant map at %s", path)
            return

        mtime = stat_result.st_mtime_ns
        size = stat_result.st_size
        if not force and self._custom_variant_mtime == mtime and self._custom_variant_size == size:
            return

        updated_map = _load_variant_map(path)
        if updated_map != self._custom_variant_map:
            self._custom_variant_map = updated_map
            self._rebuild_variant_state()

        self._custom_variant_mtime = mtime
        self._custom_variant_size = size

    def _rebuild_variant_state(self) -> None:
        combined = _merge_variant_maps(self._base_variant_map, self._custom_variant_map)

        patterns: list[tuple[re.Pattern[str], str, str]] = []
        for canonical, variants in combined.items():
            normalized_canonical = canonical.strip()
            if not normalized_canonical:
                continue
            for variant in variants:
                normalized_variant = variant.strip()
                if not normalized_variant:
                    continue
                pattern = re.compile(
                    _BOUNDARY_TEMPLATE.format(token=re.escape(normalized_variant)),
                    re.IGNORECASE,
                )
                reason = f"variant:{normalized_canonical.lower()}"
                patterns.append((pattern, normalized_canonical, reason))

        self._variant_patterns = patterns

        tracked = set(self._static_tracked_phrases)
        tracked.update(canonical for canonical in combined.keys() if canonical)
        self._tracked_phrases = tracked

    def _log_corrections(self, original: str, result: TranscriptCleanupResult) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        lines = [
            f"{timestamp} original={original!r}",
        ]
        for correction in result.corrections:
            if correction.score is not None:
                lines.append(
                    f"    {correction.reason} score={correction.score}"
                    f" {correction.original!r} -> {correction.replacement!r}"
                )
            else:
                lines.append(
                    f"    {correction.reason} {correction.original!r}"
                    f" -> {correction.replacement!r}"
                )
        lines.append(f"    normalized={result.text!r}")
        try:
            atomic_append_lines(
                self._resources.log_path,
                lines,
                max_bytes=self._resources.max_log_bytes,
                keep=self._resources.log_keep,
            )
        except Exception:
            logger.exception("Failed to append transcript cleanup audit log")


def _apply_replacements(
    text: str,
    replacements: Sequence[tuple[int, int, str, TranscriptCorrection]],
) -> tuple[str, list[TranscriptCorrection]]:
    if not replacements:
        return text, []

    sorted_replacements = sorted(replacements, key=lambda item: (item[0], -(item[1] - item[0])))
    filtered: list[tuple[int, int, str, TranscriptCorrection]] = []
    last_end = -1
    for start, end, replacement, correction in sorted_replacements:
        if start < last_end:
            continue
        filtered.append((start, end, replacement, correction))
        last_end = end

    if not filtered:
        return text, []

    segments: list[str] = []
    cursor = 0
    applied: list[TranscriptCorrection] = []
    for start, end, replacement, correction in filtered:
        segments.append(text[cursor:start])
        segments.append(replacement)
        cursor = end
        applied.append(correction)
    segments.append(text[cursor:])

    return "".join(segments), applied


def _merge_variant_maps(
    base: dict[str, tuple[str, ...]],
    custom: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    combined: dict[str, list[str]] = {}

    for source in (base, custom):
        for canonical, variants in source.items():
            canonical_clean = canonical.strip()
            if not canonical_clean:
                continue
            bucket = combined.setdefault(canonical_clean, [])
            for variant in variants:
                normalized_variant = variant.strip()
                if normalized_variant and normalized_variant not in bucket:
                    bucket.append(normalized_variant)

    return {canonical: tuple(variants) for canonical, variants in combined.items() if variants}


def _preserve_case(canonical: str, sample: str) -> str:
    if sample.isupper():
        return canonical.upper()
    if sample.islower():
        return canonical.lower()
    if sample[:1].isupper() and sample[1:].islower():
        if not canonical:
            return canonical
        return canonical[:1].upper() + canonical[1:]
    tokens = sample.split()
    if tokens and all(token[:1].isupper() and token[1:].islower() for token in tokens):
        return " ".join(part.capitalize() for part in canonical.split())
    return canonical


def _collect_keyword_phrases() -> set[str]:
    phrases: set[str] = set()
    try:
        from tools import keywords

        for keyword in keywords.ALL_KEYWORDS:
            category = (keyword.category or "").strip().lower()

            if category == "conversation_start":
                payload = keyword.payload.strip()
                if payload:
                    phrases.add(f"chat with {payload.lower()}")
                continue

            if category == "conversation_end":
                payload = keyword.payload.strip()
                if payload:
                    phrases.add(f"goodbye {payload.lower()}")
                continue

            for target in keyword.fuzzy_targets:
                cleaned = target.strip()
                if cleaned:
                    phrases.add(cleaned)
    except Exception:
        logger.exception("Failed to collect keyword phrases for transcript cleanup")
    return phrases


def load_background_agent_resources(
    base_path: Optional[Path] = None,
) -> Optional[BackgroundAgentResources]:
    agent_path = Path(base_path) if base_path else _DEFAULT_AGENT_DIR
    agent_path = agent_path.expanduser()
    try:
        agent_path = agent_path.resolve()
    except FileNotFoundError:
        logger.exception("Transcript cleanup agent directory %s not found", agent_path)
        return None

    identity_path = agent_path / "identity.json"
    if not identity_path.exists():
        logger.warning("Transcript cleanup agent identity missing at %s", identity_path)
        return None

    try:
        config = json.loads(identity_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to parse transcript cleanup identity from %s", identity_path)
        return None

    variant_path = _resolve_agent_path(agent_path, config.get("variant_map"))
    variant_map = _load_variant_map(variant_path)

    supplemental_path = _resolve_agent_path(agent_path, config.get("supplemental_phrases"))
    supplemental_phrases = _load_supplemental_phrases(supplemental_path)

    raw_thresholds = config.get("phrase_thresholds", {})
    phrase_thresholds: dict[str, float] = {}
    if isinstance(raw_thresholds, dict):
        for key, value in raw_thresholds.items():
            if not isinstance(key, str):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            numeric = max(0.0, min(1.0, numeric))
            phrase_thresholds[key.strip().lower()] = numeric

    default_threshold = config.get("default_threshold", 0.84)
    try:
        default_threshold = float(default_threshold)
    except (TypeError, ValueError):
        default_threshold = 0.84

    log_file = str(config.get("log_file") or "transcript_cleanup.log").strip() or "transcript_cleanup.log"
    log_path = get_logs_dir() / log_file

    max_log_bytes = config.get("max_log_bytes", 262144)
    try:
        max_log_bytes = int(max_log_bytes)
    except (TypeError, ValueError):
        max_log_bytes = 262144

    log_keep = config.get("log_keep", 5)
    try:
        log_keep = int(log_keep)
    except (TypeError, ValueError):
        log_keep = 5

    return BackgroundAgentResources(
        base_path=agent_path,
        config=config,
        base_variant_map=variant_map,
        supplemental_phrases=supplemental_phrases,
        phrase_thresholds=phrase_thresholds,
        default_threshold=default_threshold,
        log_path=log_path,
        max_log_bytes=max_log_bytes,
        log_keep=log_keep,
        custom_variant_map_path=_ensure_custom_variant_map_path(),
    )


def _resolve_agent_path(base: Path, value: Optional[str]) -> Path:
    if value:
        candidate = Path(value)
    else:
        candidate = Path()
    if not candidate.is_absolute():
        candidate = (base / candidate).expanduser()
    return candidate


def _ensure_custom_variant_map_path() -> Path:
    base_dir = get_data_dir() / "langgraph_agents" / "transcript_cleanup_agent"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "custom_variant_map.json"
    if not path.exists():
        try:
            path.write_text("{}\n", encoding="utf-8")
        except Exception:
            logger.exception("Failed to initialize custom transcript variant map at %s", path)
    return path


def _load_variant_map(path: Path) -> dict[str, tuple[str, ...]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Transcript cleanup variant map missing at %s", path)
        return {}
    except Exception:
        logger.exception("Failed to parse transcript cleanup variant map at %s", path)
        return {}

    variants: dict[str, tuple[str, ...]] = {}
    if isinstance(payload, dict):
        for canonical, values in payload.items():
            if not isinstance(canonical, str):
                continue
            items: list[str] = []
            if isinstance(values, str):
                items.append(values)
            elif isinstance(values, Iterable):
                for entry in values:
                    if isinstance(entry, str):
                        trimmed = entry.strip()
                        if trimmed:
                            items.append(trimmed)
            canonical_clean = canonical.strip()
            if canonical_clean and items:
                variants[canonical_clean] = tuple(dict.fromkeys(items))
    return variants


def _load_supplemental_phrases(path: Path) -> tuple[str, ...]:
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    except Exception:
        logger.exception("Failed to read supplemental transcript phrases from %s", path)
        return ()
    phrases = [line.strip() for line in contents.splitlines() if line.strip()]
    return tuple(dict.fromkeys(phrases))


def load_transcript_cleanup_agent(
    base_path: Optional[Path] = None,
    *,
    resources: Optional[BackgroundAgentResources] = None,
) -> Optional[TranscriptCleanupAgent]:
    resource_bundle = resources or load_background_agent_resources(base_path)
    if resource_bundle is None:
        return None
    return TranscriptCleanupAgent(resource_bundle)


_cached_agent_lock = threading.Lock()
_cached_agent: Optional[TranscriptCleanupAgent] = None


def _get_cached_agent() -> Optional[TranscriptCleanupAgent]:
    global _cached_agent
    if _cached_agent is not None:
        return _cached_agent
    with _cached_agent_lock:
        if _cached_agent is None:
            try:
                _cached_agent = load_transcript_cleanup_agent()
            except Exception:
                logger.exception("Failed to initialize transcript cleanup agent")
                _cached_agent = None
    return _cached_agent


def normalize_transcript(text: str) -> TranscriptCleanupResult:
    agent = _get_cached_agent()
    if agent is None:
        return TranscriptCleanupResult(text=text, corrections=())
    return agent.normalize(text)


__all__ = [
    "BackgroundAgentResources",
    "TranscriptCleanupAgent",
    "TranscriptCleanupResult",
    "TranscriptCorrection",
    "load_background_agent_resources",
    "load_transcript_cleanup_agent",
    "normalize_transcript",
]
