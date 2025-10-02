from __future__ import annotations

import hashlib
import json
import locale
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from utils.config_paths import get_data_dir, get_logger
from utils.io_atomic import atomic_write_text
from utils.memory_settings import load_identity_settings
from utils.vector_memory import VectorMemoryStore

_LOGGER = get_logger(__name__)

_CATEGORY = "temporal_context"
_TRACKER_DIR = "datetime_memory"
_COOLDOWN = timedelta(hours=24)


def _sanitize_identity(identity: str) -> str:
    normalized = (identity or "").strip()
    if not normalized:
        return "default"
    allowed = [ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in normalized]
    return ("".join(allowed) or "default")[:255]


def _tracker_path(identity: str) -> Path:
    root = get_data_dir() / _TRACKER_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{_sanitize_identity(identity)}.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _local_now() -> datetime:
    return datetime.now().astimezone()


def _to_iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        cleaned = value.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except Exception:
        return None


def _load_tracker(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _LOGGER.debug("Failed to read datetime tracker at %s", path, exc_info=True)
        return {}


def _write_tracker(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _format_offset(offset: timedelta | None) -> str:
    if offset is None:
        return "UTC+00:00"
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    minutes = abs(total_minutes)
    hours_component, minute_component = divmod(minutes, 60)
    return f"UTC{sign}{hours_component:02d}:{minute_component:02d}"


def _locale_candidates() -> Tuple[str | None, str | None]:
    codes: list[str] = []
    try:
        loc = locale.getlocale(locale.LC_TIME)
        if loc and loc[0]:
            codes.append(loc[0])
    except Exception:
        _LOGGER.debug("Failed to query LC_TIME locale", exc_info=True)
    try:
        loc = locale.getlocale()
        if loc and loc[0]:
            codes.append(loc[0])
    except Exception:
        _LOGGER.debug("Failed to query default locale", exc_info=True)
    try:
        loc = locale.getdefaultlocale()
        if loc and loc[0]:
            codes.append(loc[0])
    except Exception:
        _LOGGER.debug("Failed to query system default locale", exc_info=True)
    env_locale = os.environ.get("LC_ALL") or os.environ.get("LANG")
    if env_locale:
        codes.append(env_locale)

    normalized_code: str | None = None
    country: str | None = None
    for code in codes:
        cleaned = code.replace(".UTF-8", "").replace(".utf8", "")
        cleaned = cleaned.replace(".utf-8", "")
        normalized_code = cleaned
        if "_" in cleaned:
            parts = cleaned.split("_", 1)
            if len(parts) == 2 and parts[1]:
                country = parts[1].split(".")[0].upper()
                break
    return normalized_code, country


def _build_snapshot(now: datetime) -> Tuple[str, Dict[str, str]]:
    local_now = now
    if local_now.tzinfo is None:
        local_now = local_now.replace(tzinfo=timezone.utc)
    local_now = local_now.astimezone()
    date_iso = local_now.date().isoformat()
    weekday_name = local_now.strftime("%A")
    month_name = local_now.strftime("%B")
    tz_name = local_now.tzname() or "UTC"
    offset_label = _format_offset(local_now.utcoffset())

    locale_code, country = _locale_candidates()

    lines = [
        f"Today is {weekday_name}, {local_now.day} {month_name} {local_now.year}.",
        f"Local calendar date: {date_iso}.",
        f"Local timezone: {tz_name} ({offset_label}).",
    ]
    if locale_code:
        lines.append(f"Locale preference: {locale_code}.")
    if country:
        lines.append(f"Country/region code: {country}.")

    snapshot_text = "\n".join(lines)

    hash_builder = hashlib.sha256()
    hash_builder.update(snapshot_text.encode("utf-8"))
    snapshot_hash = hash_builder.hexdigest()

    metadata = {
        "category": _CATEGORY,
        "kind": "current_date",
        "date_iso": date_iso,
        "weekday": weekday_name,
        "month": month_name,
        "timezone": tz_name,
        "utc_offset": offset_label,
        "locale": locale_code or "",
        "country": country or "",
        "snapshot_hash": snapshot_hash,
    }
    return snapshot_text, metadata


def refresh_datetime_memory(
    identity: str,
    *,
    force: bool = False,
    reason: str | None = None,
) -> bool:
    """Refresh date and timezone context for ``identity`` if necessary."""

    tracker_path = _tracker_path(identity)
    tracker = _load_tracker(tracker_path)
    last_hash = tracker.get("snapshot_hash") if isinstance(tracker, dict) else None
    last_refresh = _from_iso(tracker.get("last_refreshed")) if isinstance(tracker, dict) else None

    now_utc = _utc_now()
    local_now = _local_now()
    snapshot_text, metadata = _build_snapshot(local_now)
    snapshot_hash = metadata.get("snapshot_hash", "")

    needs_refresh = False
    refresh_reason: str | None = None

    if force:
        needs_refresh = True
        refresh_reason = "Forced refresh requested"
    elif not last_refresh:
        needs_refresh = True
        refresh_reason = "No previous datetime refresh recorded"
    elif now_utc - last_refresh >= _COOLDOWN:
        needs_refresh = True
        elapsed_hours = (now_utc - last_refresh).total_seconds() / 3600.0
        refresh_reason = f"Date/time snapshot last refreshed {elapsed_hours:.1f}h ago"

    if not needs_refresh and last_hash != snapshot_hash:
        needs_refresh = True
        refresh_reason = "Date/time snapshot changed"

    if not needs_refresh:
        print("[DateTimeMemory] Date/time memory already current; skipping refresh.")
        return True

    if reason:
        refresh_reason = f"{refresh_reason or 'Refreshing date/time memory'} ({reason})"

    if refresh_reason:
        print(f"[DateTimeMemory] {refresh_reason}; injecting temporal context now...")
    else:
        print("[DateTimeMemory] Refreshing date/time memory...")

    settings = load_identity_settings(identity)
    if not settings.get("store_vector_memory", True):
        print(
            f"[DateTimeMemory] Vector memory disabled for identity '{identity}'; "
            "skipping date/time injection.",
        )
        _write_tracker(
            tracker_path,
            {"last_refreshed": _to_iso(now_utc), "snapshot_hash": snapshot_hash},
        )
        return True

    max_items = int(settings.get("max_vector_items", 5000) or 5000)
    ttl_days = settings.get("vector_ttl_days")
    pii_redaction = bool(settings.get("pii_redaction", False))

    try:
        store = VectorMemoryStore(identity)
        try:
            store.collection.delete(where={"category": _CATEGORY})  # type: ignore[arg-type]
        except TypeError:
            try:
                payload = store.collection.get(include=["ids", "metadatas"])
            except Exception:
                payload = {}
            ids = []
            for entry_id, meta in zip(payload.get("ids", []), payload.get("metadatas", [])):
                if isinstance(meta, dict) and meta.get("category") == _CATEGORY:
                    ids.append(entry_id)
            if ids:
                try:
                    store.collection.delete(ids=ids)
                except Exception:
                    _LOGGER.debug(
                        "Fallback deletion failed for datetime entries on %s", identity, exc_info=True
                    )
        except Exception:
            _LOGGER.debug("No existing date/time entries to delete for %s", identity, exc_info=True)

        result = store.add_memories(
            [snapshot_text],
            metadata=[metadata],
            max_items=max_items,
            ttl_days=ttl_days,
            pii_redaction=pii_redaction,
        )
        evicted = result.get("evicted", 0)
        print(
            "[DateTimeMemory] Date/time injection complete "
            f"(1 chunk, evicted={evicted}).",
        )
    except Exception:
        _LOGGER.exception("Failed to refresh date/time memory for %s", identity)
        print("[DateTimeMemory] Date/time refresh failed; check logs for details.")
        return False

    _write_tracker(
        tracker_path,
        {"last_refreshed": _to_iso(now_utc), "snapshot_hash": snapshot_hash},
    )
    return True


__all__ = ["refresh_datetime_memory"]
