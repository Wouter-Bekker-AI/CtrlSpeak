from __future__ import annotations

from pathlib import Path

import pytest

from utils import system


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.core_headless


def test_historical_v03_spec_is_preserved_but_standard_build_uses_v05() -> None:
    spec_path = ROOT / "packaging" / "CtrlSpeak_v0.3.spec"
    assert spec_path.is_file()
    spec = spec_path.read_text("utf-8")
    assert "name='CtrlSpeak_v0.3'" in spec
    assert "assets' / 'icon.ico'" in spec
    build_helper = (ROOT / "utils" / "build_exe.py").read_text("utf-8")
    assert '"CtrlSpeak_v0.5.spec"' in build_helper
    assert "standard_spec_path" in build_helper
    assert system.APP_VERSION == "0.5.1"


def test_build_documentation_names_stable_v05_artifact_and_command() -> None:
    documentation = (ROOT / "packaging" / "BUILDING.md").read_text("utf-8")
    assert "CtrlSpeak.exe" in documentation
    assert "python -m utils.build_exe" in documentation
    assert "console=False" in documentation
    assert "does not have a console window" in documentation
    assert "stable filename" in documentation.lower()
