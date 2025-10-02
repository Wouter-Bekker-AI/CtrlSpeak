import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "background_agents" / "manage_think.py"
_SPEC = importlib.util.spec_from_file_location("_manage_think_for_tests", _MODULE_PATH)
assert _SPEC and _SPEC.loader, "Failed to locate manage_think module"
_MANAGE_THINK = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MANAGE_THINK
_SPEC.loader.exec_module(_MANAGE_THINK)
ManageThinkAgent = _MANAGE_THINK.ManageThinkAgent

pytestmark = pytest.mark.core_headless


def test_manage_think_removes_plan_and_trims_response():
    agent = ManageThinkAgent()
    text = "<think>First plan</think>\nAnswer: Hello there."
    result = agent.filter_response(text)
    assert result.removed is True
    assert result.visible_text == "Hello there."
    assert result.hidden_think == "First plan"
    assert result.placeholder_text == "Thinking..."


def test_manage_think_handles_multiple_blocks():
    agent = ManageThinkAgent()
    text = "<think>Plan A</think>\n<THINK>Plan B</THINK>\nAnswer: Done."
    result = agent.filter_response(text)
    assert result.removed is True
    assert result.hidden_think == "Plan A\n\nPlan B"
    assert result.visible_text == "Done."


def test_manage_think_leaves_text_when_no_think_blocks():
    agent = ManageThinkAgent()
    text = "Answer: Nothing to hide."
    result = agent.filter_response(text)
    assert result.removed is False
    assert result.hidden_think is None
    assert result.visible_text == "Nothing to hide."


def test_manage_think_preserves_non_prefixed_text():
    agent = ManageThinkAgent()
    text = "Here is a direct reply."
    result = agent.filter_response(text)
    assert result.removed is False
    assert result.hidden_think is None
    assert result.visible_text == text
