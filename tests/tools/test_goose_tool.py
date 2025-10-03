import io

import pytest

from tools import goose_tool


def test_goose_query_validates_prompt():
    with pytest.raises(ValueError):
        goose_tool.goose_query(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        goose_tool.goose_query("   ")


def test_goose_query_rejects_invalid_mode():
    with pytest.raises(ValueError):
        goose_tool.goose_query("do something", mode="invalid")


def test_goose_query_invokes_cli(monkeypatch):
    captured = {}

    def fake_run(command, *, stdout, stderr, text, encoding, errors, shell, env):
        captured["command"] = command
        captured["env"] = env

        class DummyCompletedProcess:
            def __init__(self):
                self.stdout = "line1\nline2\n"
                self.returncode = 0

        return DummyCompletedProcess()

    monkeypatch.setattr(goose_tool.subprocess, "run", fake_run)

    output = goose_tool.goose_query(
        "list files",
        model="qwen3:32b",
        mode="approve",
        provider="test-provider",
        goose_exe="/usr/bin/goose",
    )

    assert "line1\nline2" in output
    command = captured["command"]
    assert command[0] == "/usr/bin/goose"
    assert command[command.index("--model") + 1] == "qwen3:32b"
    assert command[command.index("--provider") + 1] == "test-provider"
    assert captured["env"]["GOOSE_MODE"] == "approve"


def test_goose_query_streams_when_requested(monkeypatch):
    captured = {}

    def fake_popen(command, *, stdout, stderr, text, encoding, errors, shell, env):
        captured["command"] = command
        captured["env"] = env

        class DummyProc:
            def __init__(self):
                self.stdout = io.StringIO("stream\nchunk\n")
                self.returncode = 0

            def wait(self):
                return 0

        return DummyProc()

    monkeypatch.setattr(goose_tool.subprocess, "Popen", fake_popen)

    output = goose_tool.goose_query("show status", stream=True)

    assert "stream\nchunk" in output
    command = captured["command"]
    assert command[0] == "goose"
    assert captured["env"]["GOOSE_MODE"] == "auto"


def test_goose_query_raises_on_failure(monkeypatch):
    def fake_run(command, *, stdout, stderr, text, encoding, errors, shell, env):
        class DummyCompletedProcess:
            def __init__(self):
                self.stdout = "error output\n"
                self.returncode = 1

        return DummyCompletedProcess()

    monkeypatch.setattr(goose_tool.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        goose_tool.goose_query("list files")

    assert "Goose failed" in str(exc_info.value)
    assert "error output" in str(exc_info.value)
