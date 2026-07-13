from __future__ import annotations

import importlib

import pytest


pytestmark = pytest.mark.core_headless


def test_local_corrections_imports_on_supported_python() -> None:
    importlib.import_module("utils.local_corrections")
