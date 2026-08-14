from __future__ import annotations

import json
import re
import sqlite3
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_list(value: str | None) -> list[str]:
    return json.loads(value) if value else []


class CorrectionStore:
    """SQLite-backed explicit phrase correction rules and transcription audit data."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        if not sys.platform.startswith("win"):
            database_path.touch(mode=0o600, exist_ok=True)
            database_path.chmod(0o600)
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _initialize(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS correction_rules (
                  id TEXT PRIMARY KEY, source_phrase TEXT NOT NULL,
                  replacement_phrase TEXT NOT NULL, context_terms TEXT NOT NULL DEFAULT '[]',
                  tags TEXT NOT NULL DEFAULT '[]', enabled INTEGER NOT NULL DEFAULT 1,
                  priority INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL, use_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS transcriptions (
                  id TEXT PRIMARY KEY, raw_text TEXT NOT NULL, corrected_text TEXT NOT NULL,
                  language TEXT, segments TEXT NOT NULL, applied_rule_ids TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS transcription_feedback (
                  transcription_id TEXT NOT NULL, rule_id TEXT NOT NULL, note TEXT,
                  created_at TEXT NOT NULL, PRIMARY KEY (transcription_id, rule_id),
                  FOREIGN KEY(transcription_id) REFERENCES transcriptions(id) ON DELETE CASCADE,
                  FOREIGN KEY(rule_id) REFERENCES correction_rules(id) ON DELETE CASCADE
                );
                """
            )
            self._migrate_transcription_audit_columns(conn)
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS transcript_edit_feedback (
                  id TEXT PRIMARY KEY,
                  transcription_id TEXT NOT NULL,
                  confirmed_text TEXT NOT NULL,
                  capture_method TEXT NOT NULL,
                  client_metadata TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  FOREIGN KEY(transcription_id) REFERENCES transcriptions(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS exact_transcript_overrides (
                  id TEXT PRIMARY KEY,
                  raw_text TEXT NOT NULL UNIQUE,
                  corrected_text TEXT NOT NULL,
                  source_feedback_id TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  use_count INTEGER NOT NULL DEFAULT 0,
                  FOREIGN KEY(source_feedback_id) REFERENCES transcript_edit_feedback(id) ON DELETE RESTRICT
                );
                """
            )
            self._migrate_feedback_rule_cascade(conn)
            self._migrate_v06_scope_columns(conn)

    @staticmethod
    def _add_column_if_missing(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @classmethod
    def _migrate_v06_scope_columns(cls, conn: sqlite3.Connection) -> None:
        cls._add_column_if_missing(
            conn, "correction_rules", "scope", "TEXT NOT NULL DEFAULT 'global'"
        )
        cls._add_column_if_missing(conn, "correction_rules", "owner_id", "TEXT")
        cls._add_column_if_missing(
            conn, "correction_rules", "language_codes", "TEXT NOT NULL DEFAULT '[]'"
        )
        cls._add_column_if_missing(
            conn, "correction_rules", "send_as_keyword", "INTEGER NOT NULL DEFAULT 0"
        )
        cls._add_column_if_missing(
            conn, "transcriptions", "principal_id", "TEXT NOT NULL DEFAULT 'legacy'"
        )
        exact_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(exact_transcript_overrides)").fetchall()
        }
        if "principal_id" not in exact_columns:
            conn.executescript(
                """
                ALTER TABLE exact_transcript_overrides RENAME TO exact_transcript_overrides_v05;
                CREATE TABLE exact_transcript_overrides (
                  id TEXT PRIMARY KEY,
                  principal_id TEXT NOT NULL DEFAULT 'legacy',
                  raw_text TEXT NOT NULL,
                  corrected_text TEXT NOT NULL,
                  source_feedback_id TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  use_count INTEGER NOT NULL DEFAULT 0,
                  UNIQUE(principal_id, raw_text),
                  FOREIGN KEY(source_feedback_id) REFERENCES transcript_edit_feedback(id) ON DELETE RESTRICT
                );
                INSERT INTO exact_transcript_overrides
                  (id,principal_id,raw_text,corrected_text,source_feedback_id,created_at,updated_at,use_count)
                  SELECT id,'legacy',raw_text,corrected_text,source_feedback_id,created_at,updated_at,use_count
                  FROM exact_transcript_overrides_v05;
                DROP TABLE exact_transcript_overrides_v05;
                """
            )

    @staticmethod
    def _migrate_transcription_audit_columns(conn: sqlite3.Connection) -> None:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(transcriptions)").fetchall()}
        if "context" not in columns:
            conn.execute("ALTER TABLE transcriptions ADD COLUMN context TEXT")
        if "metadata" not in columns:
            conn.execute("ALTER TABLE transcriptions ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")

    @staticmethod
    def _migrate_feedback_rule_cascade(conn: sqlite3.Connection) -> None:
        foreign_keys = conn.execute("PRAGMA foreign_key_list(transcription_feedback)").fetchall()
        rule_fk = next((row for row in foreign_keys if row[3] == "rule_id"), None)
        if rule_fk is not None and str(rule_fk[6]).upper() == "CASCADE":
            return
        conn.executescript(
            """
            CREATE TABLE transcription_feedback_new (
              transcription_id TEXT NOT NULL, rule_id TEXT NOT NULL, note TEXT,
              created_at TEXT NOT NULL, PRIMARY KEY (transcription_id, rule_id),
              FOREIGN KEY(transcription_id) REFERENCES transcriptions(id) ON DELETE CASCADE,
              FOREIGN KEY(rule_id) REFERENCES correction_rules(id) ON DELETE CASCADE
            );
            INSERT INTO transcription_feedback_new (transcription_id, rule_id, note, created_at)
              SELECT transcription_id, rule_id, note, created_at FROM transcription_feedback;
            DROP TABLE transcription_feedback;
            ALTER TABLE transcription_feedback_new RENAME TO transcription_feedback;
            """
        )

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["enabled"] = bool(data["enabled"])
        data["send_as_keyword"] = bool(data.get("send_as_keyword", 0))
        data["context_terms"] = _json_list(data["context_terms"])
        data["tags"] = _json_list(data["tags"])
        data["language_codes"] = _json_list(data.get("language_codes"))
        return data

    def create_rule(
        self,
        *,
        source_phrase: str,
        replacement_phrase: str,
        context_terms: list[str] | None = None,
        tags: list[str] | None = None,
        enabled: bool = True,
        priority: int = 0,
        scope: str = "global",
        owner_id: str | None = None,
        language_codes: list[str] | None = None,
        send_as_keyword: bool = False,
    ) -> dict[str, Any]:
        if not source_phrase or not source_phrase.strip() or not replacement_phrase:
            raise ValueError("source_phrase and replacement_phrase must be non-empty")
        if scope not in {"global", "user"}:
            raise ValueError("scope must be global or user")
        if scope == "user" and not owner_id:
            raise ValueError("user-scoped correction rules require an owner_id")
        now, rule_id = _now(), str(uuid.uuid4())
        values = (
            rule_id,
            source_phrase.strip(),
            replacement_phrase,
            json.dumps(context_terms or []),
            json.dumps(tags or []),
            int(enabled),
            priority,
            now,
            now,
            scope,
            owner_id,
            json.dumps(language_codes or []),
            int(send_as_keyword),
        )
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO correction_rules
                (id,source_phrase,replacement_phrase,context_terms,tags,enabled,priority,created_at,
                 updated_at,scope,owner_id,language_codes,send_as_keyword)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                values,
            )
        return self.get_rule(rule_id)  # type: ignore[return-value]

    def list_rules(
        self,
        enabled: bool | None = None,
        *,
        principal_id: str | None = None,
        language: str | None = None,
        include_all: bool = False,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM correction_rules"
        clauses: list[str] = []
        params: list[Any] = []
        if enabled is not None:
            clauses.append("enabled = ?")
            params.append(int(enabled))
        if principal_id is not None and not include_all:
            clauses.append("(scope = 'global' OR (scope = 'user' AND owner_id = ?))")
            params.append(principal_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY priority DESC, length(source_phrase) DESC, created_at ASC"
        with self._connection() as conn:
            rules = [self._row(row) for row in conn.execute(query, tuple(params)).fetchall()]
        if language:
            language = language.casefold()
            rules = [
                rule
                for rule in rules
                if not rule["language_codes"] or language in rule["language_codes"]
            ]
        return rules

    def get_rule(
        self,
        rule_id: str,
        *,
        principal_id: str | None = None,
        include_all: bool = False,
    ) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM correction_rules WHERE id = ?", (rule_id,)).fetchone()
        rule = self._row(row) if row else None
        if (
            rule
            and principal_id is not None
            and not include_all
            and rule["scope"] != "global"
            and rule["owner_id"] != principal_id
        ):
            return None
        return rule

    def update_rule(
        self,
        rule_id: str,
        changes: dict[str, Any],
        *,
        principal_id: str | None = None,
        include_all: bool = False,
    ) -> dict[str, Any] | None:
        existing = self.get_rule(rule_id, principal_id=principal_id, include_all=include_all)
        if existing is None:
            return None
        allowed = {
            "source_phrase",
            "replacement_phrase",
            "context_terms",
            "tags",
            "enabled",
            "priority",
            "language_codes",
            "send_as_keyword",
        }
        fields, values = [], []
        for key, value in changes.items():
            if key not in allowed or value is None:
                continue
            if key in {"context_terms", "tags", "language_codes"}:
                value = json.dumps(value)
            if key in {"enabled", "send_as_keyword"}:
                value = int(value)
            fields.append(f"{key} = ?")
            values.append(value)
        if not fields:
            return existing
        if "source_phrase" in changes and not str(changes["source_phrase"]).strip():
            raise ValueError("source_phrase must be non-empty")
        if "replacement_phrase" in changes and not changes["replacement_phrase"]:
            raise ValueError("replacement_phrase must be non-empty")
        fields.append("updated_at = ?")
        values.extend([_now(), rule_id])
        with self._connection() as conn:
            cursor = conn.execute(f"UPDATE correction_rules SET {', '.join(fields)} WHERE id = ?", values)
        return self.get_rule(rule_id, principal_id=principal_id, include_all=include_all) if cursor.rowcount else None

    def delete_rule(
        self,
        rule_id: str,
        *,
        principal_id: str | None = None,
        include_all: bool = False,
    ) -> bool:
        if self.get_rule(rule_id, principal_id=principal_id, include_all=include_all) is None:
            return False
        with self._connection() as conn:
            return conn.execute("DELETE FROM correction_rules WHERE id = ?", (rule_id,)).rowcount > 0

    def apply(
        self,
        raw_text: str,
        context: str | None = None,
        *,
        principal_id: str = "legacy",
        language: str | None = None,
    ) -> tuple[str, list[str]]:
        corpus = f"{context or ''}\n{raw_text}".casefold()
        candidates = []
        for rule in self.list_rules(
            enabled=True,
            principal_id=principal_id,
            language=language,
        ):
            if all(term.casefold() in corpus for term in rule["context_terms"]):
                candidates.append(rule)
        candidates.sort(key=lambda rule: (-len(rule["source_phrase"]), -rule["priority"], rule["created_at"]))
        corrected, applied = raw_text, []
        if candidates:
            pattern = re.compile(
                "|".join(
                    f"(?P<R{index}>(?<!\\w){re.escape(rule['source_phrase'])}(?!\\w))"
                    for index, rule in enumerate(candidates)
                ),
                re.IGNORECASE,
            )

            def replace(match: re.Match[str]) -> str:
                index = int(str(match.lastgroup)[1:])
                rule = candidates[index]
                if rule["id"] not in applied:
                    applied.append(rule["id"])
                return str(rule["replacement_phrase"])

            corrected = pattern.sub(replace, raw_text)
        if applied:
            with self._connection() as conn:
                conn.executemany(
                    "UPDATE correction_rules SET use_count = use_count + 1, updated_at = ? WHERE id = ?",
                    [(_now(), rule_id) for rule_id in applied],
                )
        return corrected, applied

    def apply_with_metadata(
        self,
        raw_text: str,
        context: str | None = None,
        *,
        principal_id: str = "legacy",
        language: str | None = None,
    ) -> tuple[str, list[str], str | None]:
        with self._connection() as conn:
            override = conn.execute(
                """SELECT id, corrected_text FROM exact_transcript_overrides
                   WHERE principal_id = ? AND raw_text = ?""",
                (principal_id, raw_text),
            ).fetchone()
            if override:
                conn.execute(
                    "UPDATE exact_transcript_overrides SET use_count = use_count + 1, updated_at = ? WHERE id = ?",
                    (_now(), override["id"]),
                )
                return str(override["corrected_text"]), [], str(override["id"])
        corrected, applied = self.apply(
            raw_text,
            context,
            principal_id=principal_id,
            language=language,
        )
        return corrected, applied, None

    def keyword_hints(
        self,
        *,
        principal_id: str,
        language: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        hints: list[str] = []
        for rule in self.list_rules(
            enabled=True,
            principal_id=principal_id,
            language=language,
        ):
            if not rule["send_as_keyword"]:
                continue
            for phrase in (rule["source_phrase"], rule["replacement_phrase"]):
                if phrase not in hints:
                    hints.append(phrase)
                if len(hints) >= limit:
                    return hints
        return hints

    def create_transcription(
        self,
        *,
        raw_text: str,
        corrected_text: str,
        language: str | None,
        segments: list[dict[str, Any]],
        applied_rule_ids: list[str],
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
        principal_id: str = "legacy",
    ) -> str:
        transcription_id, now = str(uuid.uuid4()), _now()
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO transcriptions
                   (id,raw_text,corrected_text,language,segments,applied_rule_ids,created_at,context,
                    metadata,principal_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    transcription_id,
                    raw_text,
                    corrected_text,
                    language,
                    json.dumps(segments),
                    json.dumps(applied_rule_ids),
                    now,
                    context,
                    json.dumps(metadata or {}),
                    principal_id,
                ),
            )
        return transcription_id

    def get_transcription(
        self,
        transcription_id: str,
        *,
        principal_id: str | None = None,
        include_all: bool = False,
    ) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM transcriptions WHERE id = ?", (transcription_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        if principal_id is not None and not include_all and result["principal_id"] != principal_id:
            return None
        result["segments"] = _json_list(result.get("segments"))
        result["applied_rule_ids"] = _json_list(result.get("applied_rule_ids"))
        result["metadata"] = json.loads(result.get("metadata") or "{}")
        return result

    def approve_exact_override(
        self,
        transcription_id: str,
        *,
        confirmed_text: str,
        capture_method: str,
        client_metadata: dict[str, Any] | None = None,
        principal_id: str = "legacy",
        include_all: bool = False,
    ) -> dict[str, str] | None:
        if not confirmed_text or not confirmed_text.strip():
            raise ValueError("confirmed_text must be non-empty")
        if not capture_method or not capture_method.strip():
            raise ValueError("capture_method must be non-empty")
        transcription = self.get_transcription(
            transcription_id,
            principal_id=principal_id,
            include_all=include_all,
        )
        if transcription is None:
            return None
        if confirmed_text == transcription["corrected_text"]:
            raise ValueError("confirmed_text must differ from the returned transcript")

        now = _now()
        feedback_id = str(uuid.uuid4())
        proposed_override_id = str(uuid.uuid4())
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO transcript_edit_feedback
                   (id,transcription_id,confirmed_text,capture_method,client_metadata,created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    feedback_id,
                    transcription_id,
                    confirmed_text,
                    capture_method.strip(),
                    json.dumps(client_metadata or {}),
                    now,
                ),
            )
            conn.execute(
                """INSERT INTO exact_transcript_overrides
                   (id,principal_id,raw_text,corrected_text,source_feedback_id,created_at,updated_at,use_count)
                   VALUES (?,?,?,?,?,?,?,0)
                   ON CONFLICT(principal_id,raw_text) DO UPDATE SET
                     corrected_text=excluded.corrected_text,
                     source_feedback_id=excluded.source_feedback_id,
                     updated_at=excluded.updated_at""",
                (
                    proposed_override_id,
                    principal_id,
                    transcription["raw_text"],
                    confirmed_text,
                    feedback_id,
                    now,
                    now,
                ),
            )
            override = conn.execute(
                """SELECT id FROM exact_transcript_overrides
                   WHERE principal_id = ? AND raw_text = ?""",
                (principal_id, transcription["raw_text"]),
            ).fetchone()
        return {"feedback_id": feedback_id, "override_id": str(override["id"])}

    def get_edit_feedback(self, feedback_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            row = conn.execute(
                """SELECT f.*, t.raw_text, t.corrected_text AS returned_text, t.context,
                          t.metadata AS transcription_metadata
                   FROM transcript_edit_feedback f
                   JOIN transcriptions t ON t.id = f.transcription_id
                   WHERE f.id = ?""",
                (feedback_id,),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["client_metadata"] = json.loads(result.get("client_metadata") or "{}")
        result["transcription_metadata"] = json.loads(result.get("transcription_metadata") or "{}")
        return result

    def attach_feedback(
        self,
        transcription_id: str,
        rule_ids: list[str],
        note: str | None = None,
        *,
        principal_id: str = "legacy",
        include_all: bool = False,
    ) -> list[str] | None:
        with self._connection() as conn:
            exists = conn.execute(
                "SELECT principal_id FROM transcriptions WHERE id = ?",
                (transcription_id,),
            ).fetchone()
            if not exists or (
                not include_all and str(exists["principal_id"]) != principal_id
            ):
                return None
            found = (
                {
                    row[0]
                    for row in conn.execute(
                        "SELECT id FROM correction_rules WHERE id IN (%s)"
                        % ",".join("?" * len(rule_ids)),
                        rule_ids,
                    )
                }
                if rule_ids
                else set()
            )
            if found != set(rule_ids):
                raise KeyError("one or more correction rules do not exist")
            for rule_id in rule_ids:
                if self.get_rule(
                    rule_id,
                    principal_id=principal_id,
                    include_all=include_all,
                ) is None:
                    raise KeyError("one or more correction rules do not exist")
            conn.executemany(
                """INSERT INTO transcription_feedback (transcription_id,rule_id,note,created_at)
                   VALUES (?,?,?,?) ON CONFLICT(transcription_id,rule_id)
                   DO UPDATE SET note=excluded.note""",
                [(transcription_id, rule_id, note, _now()) for rule_id in rule_ids],
            )
        return rule_ids
