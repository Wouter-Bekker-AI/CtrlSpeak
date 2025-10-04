"""Normalize legacy profile facts into slot-based documents."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

from utils.config_paths import get_data_dir, get_logger
from utils.vector_memory import VectorMemoryStore

logger = get_logger(__name__)

_PROFILE_PATTERNS = (
    ("name", re.compile(r"\bmy name is\s+(?P<value>[A-Za-z][A-Za-z\s'\-]{0,60})", re.IGNORECASE)),
    ("age", re.compile(r"\b(?:i am|i'm|my age is)\s+(?P<value>\d{1,3})", re.IGNORECASE)),
    (
        "favorite_sport",
        re.compile(
            r"\bmy\s+(?:favorite|favourite)\s+sport\s+is\s+(?P<value>[A-Za-z][A-Za-z\s'\-]{0,60})",
            re.IGNORECASE,
        ),
    ),
)


def _discover_identities(root: Path) -> Iterable[str]:
    for child in root.iterdir():
        if child.is_dir():
            yield child.name


def _parse_timestamp(value: Optional[str]) -> datetime:
    if not value:
        return datetime.utcnow()
    try:
        return datetime.fromisoformat(value.replace("Z", ""))
    except Exception:
        return datetime.utcnow()


def _extract_profile_fact(text: str) -> Optional[tuple[str, str]]:
    for attribute, pattern in _PROFILE_PATTERNS:
        match = pattern.search(text)
        if match:
            value = match.groupdict().get("value", "").strip()
            if value:
                return attribute, value
    return None


def migrate_identity(identity: str, *, dry_run: bool = False) -> dict:
    store = VectorMemoryStore(identity)
    summary: Dict[str, dict] = {}
    payload = store.collection.get(include=["documents", "metadatas"])
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    for doc, metadata in zip(documents, metadatas):
        if not isinstance(doc, str):
            continue
        fact = _extract_profile_fact(doc)
        if not fact:
            continue
        attribute, value = fact
        created = _parse_timestamp((metadata or {}).get("created_at"))
        record = summary.get(attribute)
        if record is None or created >= record["timestamp"]:
            summary[attribute] = {"value": value, "timestamp": created}
    written = 0
    for attribute, record in summary.items():
        existing = store.read_profile_slot("default_user", attribute)
        if existing and existing["metadata"].get("value") == record["value"]:
            continue
        if dry_run:
            logger.info("[DRY RUN] Would write %s=%s for %s", attribute, record["value"], identity)
            written += 1
            continue
        store.upsert_profile_slot(
            "default_user",
            attribute,
            record["value"],
            source="migration",
            valid_from=record["timestamp"],
        )
        written += 1
    return {"identity": identity, "written": written}


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy profile facts into slot documents.")
    parser.add_argument("--identity", help="Limit migration to a single identity")
    parser.add_argument("--dry-run", action="store_true", help="Inspect without writing any changes")
    args = parser.parse_args()

    root = get_data_dir() / "bot_memory"
    root.mkdir(parents=True, exist_ok=True)
    identities = [args.identity] if args.identity else list(_discover_identities(root))
    if not identities:
        logger.info("No identities found; nothing to migrate.")
        return

    for identity in identities:
        logger.info("Migrating profile facts for identity '%s'", identity)
        result = migrate_identity(identity, dry_run=args.dry_run)
        logger.info(
            "Profile migration for %s complete (facts written=%s)",
            result["identity"],
            result["written"],
        )


if __name__ == "__main__":
    main()
