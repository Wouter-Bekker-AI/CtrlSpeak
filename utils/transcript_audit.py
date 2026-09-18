"""Bounded, local-only dictation audit; no credentials or request headers."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import threading

from utils.config_paths import get_logger, get_logs_dir
from utils.dictation_text import control_character_counts
from utils.version import APP_VERSION

_LOCK = threading.Lock()


def record_injection(result, insertion_text: str, *, preserve_formatting: bool,
                     attempt_id: str, outcome: str) -> None:
    """Record the exact input to the OS adapter, not a claim of target receipt.

    JSON escapes control characters so even malformed transcripts cannot forge
    log records. Three files of about 2 MiB bound plaintext transcript retention.
    Failure to log must not lose a successful transcription.
    """
    meta = result.metadata or {}
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "version": APP_VERSION,
        "attempt_id": attempt_id, "transcription_id": result.transcription_id,
        "outcome": outcome, "provider_used": meta.get("provider_used"),
        "raw_text": result.raw_text, "corrected_text": result.corrected_text,
        "s1_cleaned_text": meta.get("s1_cleaned_text"),
        "normalized_text": meta.get("normalized_text"),
        "selected_text": result.text, "insertion_text": insertion_text,
        "preserve_formatting": preserve_formatting,
        "selected_controls": control_character_counts(result.text),
        "insertion_controls": control_character_counts(insertion_text),
    }
    try:
        with _LOCK:
            path = get_logs_dir() / "dictation-audit.jsonl"
            if os.name != "nt":
                path.touch(mode=0o600, exist_ok=True)
                path.chmod(0o600)
            handler = RotatingFileHandler(path, maxBytes=2 * 1024 * 1024,
                                          backupCount=2, encoding="utf-8")
            try:
                message = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
                record = logging.LogRecord("dictation-audit", logging.INFO, "", 0, message, (), None)
                # Call the write path directly so I/O exceptions reach our safe
                # warning instead of logging.handleError dumping transcript data.
                if handler.shouldRollover(record):
                    handler.doRollover()
                handler.stream.write(message + "\n")
                handler.flush()
            finally:
                handler.close()
    except Exception:
        get_logger(__name__).warning("Dictation audit could not be written; transcript insertion remains available")
