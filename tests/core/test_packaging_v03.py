from __future__ import annotations

from pathlib import Path

import pytest

from utils import system


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.core_headless


def test_v03_packaging_uses_versioned_executable_name_and_existing_icon() -> None:
    spec_path = ROOT / "packaging" / "CtrlSpeak_v0.3.spec"
    assert spec_path.is_file()
    spec = spec_path.read_text("utf-8")
    assert "name='CtrlSpeak_v0.3'" in spec
    assert "assets' / 'icon.ico'" in spec
    build_helper = (ROOT / "utils" / "build_exe.py").read_text("utf-8")
    assert '"CtrlSpeak_v0.3.spec"' in build_helper
    assert system.APP_VERSION == "0.3.0"


def test_build_documentation_names_exact_v03_artifact_and_command() -> None:
    documentation = (ROOT / "packaging" / "BUILDING.md").read_text("utf-8")
    assert "CtrlSpeak_v0.3.exe" in documentation
    assert "python -m utils.build_exe" in documentation
    assert "console=False" in documentation
    assert "does not have a console window" in documentation
    assert ".\\dist\\CtrlSpeak_v0.3.exe --backend-status" not in documentation
