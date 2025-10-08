import importlib

import pytest

pytestmark = pytest.mark.core_headless


def test_refresh_document_memory_is_noop(capsys):
    module = importlib.reload(importlib.import_module("background_agents.document_memory_agent"))

    assert module.refresh_document_memory("vision") is True
    captured = capsys.readouterr()
    assert "Documentation ingestion disabled" in captured.out

    assert module.refresh_document_memory("reception", force=True, reason="hotkey") is True
    captured_second = capsys.readouterr()
    assert "Documentation ingestion disabled" in captured_second.out
