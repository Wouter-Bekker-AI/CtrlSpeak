from pathlib import Path

import pytest

from utils import io_atomic
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


def test_atomic_write_text_retries_on_locked_target(tmp_path, monkeypatch):
    target = tmp_path / "data.json"
    call_count = {"value": 0}
    real_replace = io_atomic.os.replace

    def flaky_replace(src, dst):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise PermissionError("locked")
        return real_replace(src, dst)

    monkeypatch.setattr(io_atomic.os, "replace", flaky_replace)
    monkeypatch.setattr(io_atomic.time, "sleep", lambda *_args, **_kwargs: None)

    atomic_write_text(target, "{}")

    assert call_count["value"] == 2
    assert target.read_text() == "{}"


def test_atomic_append_lines_retries_on_locked_target(tmp_path, monkeypatch):
    log = tmp_path / "conversation.jsonl"
    call_count = {"value": 0}
    real_replace = io_atomic.os.replace

    def flaky_replace(src, dst):
        call_count["value"] += 1
        if call_count["value"] % 2 == 1:
            raise PermissionError("locked")
        return real_replace(src, dst)

    monkeypatch.setattr(io_atomic.os, "replace", flaky_replace)
    monkeypatch.setattr(io_atomic.time, "sleep", lambda *_args, **_kwargs: None)

    atomic_append_lines(log, ["{\"role\": \"user\", \"content\": \"hi\"}"])
    assert log.read_text().strip().endswith('"hi\"}')
    assert call_count["value"] == 2
