from __future__ import annotations

import json
import sys
import types

import pytest


def _raise_requests_error(*_args, **_kwargs):
    raise RuntimeError("requests is not installed in the test environment")


sys.modules.setdefault(
    "requests",
    types.SimpleNamespace(get=_raise_requests_error, post=_raise_requests_error),
)

from background_agents.transcript_cleanup_agent import (  # noqa: E402
    load_transcript_cleanup_agent,
    normalize_transcript,
)
from background_agents.transcript_cleanup_agent import background_agent as cleanup_module  # noqa: E402

pytestmark = pytest.mark.core_headless


def test_transcript_cleanup_agent_rewrites_common_variant():
    agent = load_transcript_cleanup_agent()
    assert agent is not None

    result = agent.normalize("Please look at my clubboard right now.")

    assert result.text == "Please look at my clipboard right now."
    assert result.corrections
    assert any(correction.reason.startswith("variant:") for correction in result.corrections)


def test_normalize_transcript_corrects_chat_with_defunct():
    result = normalize_transcript("could you chat with defunct this instant")

    assert result.text == "could you chat with reception this instant"
    assert any(
        correction.reason.startswith("variant:") or correction.reason.startswith("fuzzy:")
        for correction in result.corrections
    )


def _load_agent_with_custom_root(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup_module, "_cached_agent", None)
    monkeypatch.setattr(cleanup_module, "get_data_dir", lambda: tmp_path)
    return cleanup_module.load_transcript_cleanup_agent()


def test_custom_variant_map_extends_rewrites(tmp_path, monkeypatch):
    agent = _load_agent_with_custom_root(tmp_path, monkeypatch)

    custom_path = tmp_path / "langgraph_agents" / "transcript_cleanup_agent" / "custom_variant_map.json"
    payload = {"launch autopilot": ["launch auto pilot"]}
    custom_path.write_text(json.dumps(payload), encoding="utf-8")

    result = agent.normalize("please launch auto pilot now")

    assert result.text == "please launch autopilot now"
    assert any(correction.replacement.lower() == "launch autopilot" for correction in result.corrections)


def test_custom_variant_map_refreshes_on_change(tmp_path, monkeypatch):
    agent = _load_agent_with_custom_root(tmp_path, monkeypatch)

    custom_path = tmp_path / "langgraph_agents" / "transcript_cleanup_agent" / "custom_variant_map.json"
    custom_path.write_text(json.dumps({}, indent=2), encoding="utf-8")

    # First write introduces an entry.
    custom_path.write_text(
        json.dumps({"launch autopilot": ["launch auto pilot"]}),
        encoding="utf-8",
    )
    agent.normalize("please launch auto pilot")

    # Second write changes the payload to include a new mapping.
    custom_path.write_text(
        json.dumps(
            {
                "launch autopilot": ["launch auto pilot"],
                "open autopilot": ["open auto pilot"],
            }
        ),
        encoding="utf-8",
    )

    result = agent.normalize("could you open auto pilot")

    assert result.text == "could you open autopilot"
    assert any(
        correction.replacement.lower() == "open autopilot" for correction in result.corrections
    )
