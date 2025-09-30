from pathlib import Path

import pytest

from utils.io_atomic import atomic_append_lines, atomic_write_text

pytestmark = pytest.mark.core_headless


def test_atomic_write_text_creates_file(tmp_path):
    target = tmp_path / "config" / "settings.json"
    atomic_write_text(target, "{}")
    assert target.exists()
    assert target.read_text() == "{}"


def test_atomic_append_lines_rotates(tmp_path):
    log = tmp_path / "conversation.jsonl"

    atomic_append_lines(log, ["{\"role\": \"user\", \"content\": \"hi\"}"])
    first_snapshot = log.read_text()
    assert "hi" in first_snapshot

    atomic_append_lines(log, ["{\"role\": \"assistant\", \"content\": \"hello\"}"], max_bytes=64, keep=2)
    assert log.exists()
    assert "hello" in log.read_text()

    # Force rotation by appending a large entry
    large_entry = "{\"role\": \"assistant\", \"content\": \"" + ("x" * 80) + "\"}"
    atomic_append_lines(log, [large_entry], max_bytes=64, keep=2)

    rotated = Path(str(log) + ".1")
    assert rotated.exists()
    assert "hello" in rotated.read_text()
    assert "x" in log.read_text()
