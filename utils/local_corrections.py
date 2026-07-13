"""Persistent, exact-only correction learning for bundled transcriptions.

Whole-transcript edits are never decomposed into phrase substitutions. Each
approval is audited and may only override a future byte-for-byte raw transcript.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from utils import config_paths


LOGGER = config_paths.get_logger(__name__)
LOCAL_CORRECTIONS_FILENAME = "local-corrections.sqlite3"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class LocalTranscription:
    transcription_id: str
    raw_text: str
    corrected_text: str
    exact_override_id: str | None


@dataclass(frozen=True)
class LocalApproval:
    feedback_id: str
    override_id: str


class LocalCorrectionLibrary:
    """SQLite-backed audit index for user-approved local corrections."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        if not sys.platform.startswith("win"):
            self.database_path.touch(mode=0o600, exist_ok=True)
            self.database_path.chmod(0o600)
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS local_transcriptions (
                  id TEXT PRIMARY KEY,
                  raw_text TEXT NOT NULL,
                  returned_text TEXT NOT NULL,
                  metadata TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS local_edit_feedback (
                  id TEXT PRIMARY KEY,
                  transcription_id TEXT NOT NULL,
                  confirmed_text TEXT NOT NULL,
                  capture_method TEXT NOT NULL,
                  client_metadata TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(transcription_id) REFERENCES local_transcriptions(id) ON DELETE RESTRICT
                );
                CREATE TABLE IF NOT EXISTS local_exact_overrides (
                  id TEXT PRIMARY KEY,
                  raw_text TEXT NOT NULL UNIQUE,
                  corrected_text TEXT NOT NULL,
                  source_feedback_id TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  use_count INTEGER NOT NULL DEFAULT 0,
                  FOREIGN KEY(source_feedback_id) REFERENCES local_edit_feedback(id) ON DELETE RESTRICT
                );
                CREATE TABLE IF NOT EXISTS local_explicit_phrase_rules (
                  id TEXT PRIMARY KEY,
                  source_phrase TEXT NOT NULL,
                  replacement_phrase TEXT NOT NULL,
                  approval_note TEXT,
                  approved_at TEXT NOT NULL,
                  enabled INTEGER NOT NULL DEFAULT 1
                );
                """
            )

    def record_transcription(
        self,
        raw_text: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> LocalTranscription:
        """Apply only an exact approved override and retain an auditable result."""
        transcription_id = str(uuid.uuid4())
        now = _now()
        with self._connection() as connection:
            override = connection.execute(
                "SELECT id, corrected_text FROM local_exact_overrides WHERE raw_text = ?",
                (raw_text,),
            ).fetchone()
            if override:
                corrected_text = str(override["corrected_text"])
                override_id = str(override["id"])
                connection.execute(
                    """UPDATE local_exact_overrides
                       SET use_count = use_count + 1, updated_at = ? WHERE id = ?""",
                    (now, override_id),
                )
            else:
                corrected_text = raw_text
                override_id = None
            connection.execute(
                """INSERT INTO local_transcriptions
                   (id, raw_text, returned_text, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    transcription_id,
                    raw_text,
                    corrected_text,
                    json.dumps(dict(metadata or {})),
                    now,
                ),
            )
        return LocalTranscription(transcription_id, raw_text, corrected_text, override_id)

    def approve_exact_override(
        self,
        transcription_id: str,
        *,
        confirmed_text: str,
        capture_method: str,
        client_metadata: Mapping[str, Any] | None = None,
    ) -> LocalApproval | None:
        """Audit one approval and upsert only its complete exact transcript."""
        if not confirmed_text or not confirmed_text.strip():
            raise ValueError("confirmed_text must be non-empty")
        if not capture_method or not capture_method.strip():
            raise ValueError("capture_method must be non-empty")

        now = _now()
        feedback_id = str(uuid.uuid4())
        proposed_override_id = str(uuid.uuid4())
        with self._connection() as connection:
            transcription = connection.execute(
                "SELECT raw_text, returned_text FROM local_transcriptions WHERE id = ?",
                (transcription_id,),
            ).fetchone()
            if transcription is None:
                return None
            if confirmed_text == transcription["returned_text"]:
                raise ValueError("confirmed_text must differ from the returned transcript")
            connection.execute(
                """INSERT INTO local_edit_feedback
                   (id, transcription_id, confirmed_text, capture_method, client_metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    feedback_id,
                    transcription_id,
                    confirmed_text,
                    capture_method.strip(),
                    json.dumps(dict(client_metadata or {})),
                    now,
                ),
            )
            connection.execute(
                """INSERT INTO local_exact_overrides
                   (id, raw_text, corrected_text, source_feedback_id, created_at, updated_at, use_count)
                   VALUES (?, ?, ?, ?, ?, ?, 0)
                   ON CONFLICT(raw_text) DO UPDATE SET
                     corrected_text = excluded.corrected_text,
                     source_feedback_id = excluded.source_feedback_id,
                     updated_at = excluded.updated_at""",
                (
                    proposed_override_id,
                    str(transcription["raw_text"]),
                    confirmed_text,
                    feedback_id,
                    now,
                    now,
                ),
            )
            override = connection.execute(
                "SELECT id FROM local_exact_overrides WHERE raw_text = ?",
                (str(transcription["raw_text"]),),
            ).fetchone()
        return LocalApproval(feedback_id, str(override["id"]))

    def get_edit_feedback(self, feedback_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT feedback.*, transcription.raw_text,
                          transcription.returned_text,
                          transcription.metadata AS transcription_metadata
                   FROM local_edit_feedback AS feedback
                   JOIN local_transcriptions AS transcription
                     ON transcription.id = feedback.transcription_id
                   WHERE feedback.id = ?""",
                (feedback_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["client_metadata"] = json.loads(result.get("client_metadata") or "{}")
        result["transcription_metadata"] = json.loads(
            result.get("transcription_metadata") or "{}"
        )
        return result

    def list_explicit_phrase_rules(self) -> list[dict[str, Any]]:
        """Return only separately approved rules; exact-edit feedback never adds one."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM local_explicit_phrase_rules ORDER BY approved_at, id"
            ).fetchall()
        return [dict(row) for row in rows]


def get_local_correction_library() -> LocalCorrectionLibrary:
    """Open the per-user correction index outside the read-only application bundle."""
    return LocalCorrectionLibrary(
        config_paths.get_config_dir() / LOCAL_CORRECTIONS_FILENAME
    )
