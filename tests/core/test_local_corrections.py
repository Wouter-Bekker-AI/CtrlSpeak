from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from utils.local_corrections import LocalCorrectionLibrary


pytestmark = pytest.mark.core_headless


def test_approved_edit_persists_as_an_exact_override_only(tmp_path: Path) -> None:
    database_path = tmp_path / "local-corrections.sqlite3"
    library = LocalCorrectionLibrary(database_path)
    if not sys.platform.startswith("win"):
        assert stat.S_IMODE(database_path.stat().st_mode) == 0o600
    first = library.record_transcription(
        "Meet Acme Corp tomorrow.",
        metadata={"model": "small"},
    )

    approval = library.approve_exact_override(
        first.transcription_id,
        confirmed_text="Meet ACME Corporation tomorrow.",
        capture_method="active_field_on_enter",
        client_metadata={"client": "CtrlSpeak", "version": "0.3.0"},
    )

    reopened = LocalCorrectionLibrary(database_path)
    repeated = reopened.record_transcription("Meet Acme Corp tomorrow.")
    similar = reopened.record_transcription("Meet Acme Corp today.")
    audit = reopened.get_edit_feedback(approval.feedback_id)

    assert repeated.corrected_text == "Meet ACME Corporation tomorrow."
    assert repeated.exact_override_id == approval.override_id
    assert similar.corrected_text == "Meet Acme Corp today."
    assert similar.exact_override_id is None
    assert audit is not None
    assert audit["raw_text"] == "Meet Acme Corp tomorrow."
    assert audit["returned_text"] == "Meet Acme Corp tomorrow."
    assert audit["confirmed_text"] == "Meet ACME Corporation tomorrow."
    assert audit["capture_method"] == "active_field_on_enter"
    assert audit["transcription_metadata"] == {"model": "small"}
    assert audit["client_metadata"] == {"client": "CtrlSpeak", "version": "0.3.0"}
    assert reopened.list_explicit_phrase_rules() == []


def test_local_approval_requires_a_changed_nonempty_transcript(tmp_path: Path) -> None:
    library = LocalCorrectionLibrary(tmp_path / "local-corrections.sqlite3")
    transcription = library.record_transcription("Already correct")

    with pytest.raises(ValueError, match="must differ"):
        library.approve_exact_override(
            transcription.transcription_id,
            confirmed_text="Already correct",
            capture_method="active_field_on_enter",
        )
    with pytest.raises(ValueError, match="non-empty"):
        library.approve_exact_override(
            transcription.transcription_id,
            confirmed_text="  ",
            capture_method="active_field_on_enter",
        )
    assert library.approve_exact_override(
        "missing-id",
        confirmed_text="Changed",
        capture_method="active_field_on_enter",
    ) is None
