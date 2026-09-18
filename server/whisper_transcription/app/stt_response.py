"""External Hermes command output contract; no imports from Hermes itself."""
import json
import os
from pathlib import Path

ATTRIBUTION = "S1-mini by Superwhisper"
STATUSES = {"disabled", "unavailable", "non_english", "empty_input", "input_too_long",
            "timeout", "runtime_error", "empty_output", "unsafe_output", "unchanged",
            "applied", "exact_override"}


def write_result(payload: dict, output_path: Path) -> None:
    corrected = payload.get("text")
    if not isinstance(corrected, str) or not corrected.strip():
        raise ValueError("CtrlSpeak gateway returned an empty transcript")
    raw = payload.get("raw_text")
    if not isinstance(raw, str) or not raw.strip():
        raw = corrected
    normalized = payload.get("normalized_text")
    metadata = payload.get("normalization")
    metadata = metadata if isinstance(metadata, dict) else {}
    confirmed = (
        payload.get("provider_used") == "ubuntu-gpu-large-v3-turbo"
        and payload.get("language") == "en"
        and payload.get("exact_override_id") is None
        and metadata.get("requested") is True
        and metadata.get("applied") is True
        and metadata.get("status") == "applied"
        and metadata.get("model") == ATTRIBUTION
        and metadata.get("location") == "ubuntu-worker"
        and metadata.get("device") == "cuda"
        and isinstance(normalized, str) and bool(normalized.strip())
        and len(normalized) <= 12000
    )
    selected = normalized.strip() if confirmed else corrected.strip()
    applied = confirmed and selected != corrected.strip()
    status = metadata.get("status")
    if not isinstance(status, str) or status not in STATUSES:
        status = "unavailable"
    if confirmed and not applied:
        status = "unchanged"
    elif not confirmed and status == "applied":
        status = "unavailable"
    sidecar = {
        "schema_version": 2, "raw_transcript": raw, "corrected_transcript": corrected,
        "cleaned_transcript": selected, "normalization_applied": applied,
        "normalization_status": status, "normalizer": ATTRIBUTION if applied else None,
        "ctrlspeak_feedback": {key: payload.get(key) for key in (
            "id", "raw_text", "text", "provider_used", "applied_correction_rule_ids",
            "normalized_applied_correction_rule_ids", "exact_override_id")},
    }
    for path, value in [(output_path, selected + "\n"),
                        (Path(str(output_path) + ".ctrlspeak.json"),
                         json.dumps(sidecar, ensure_ascii=False) + "\n")]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
