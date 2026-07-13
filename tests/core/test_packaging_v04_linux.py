from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest

from utils import system


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.core_headless


def test_linux_pyinstaller_spec_names_v04_artifact_and_bundles_linux_icon() -> None:
    spec_path = ROOT / "packaging" / "CtrlSpeak_v0.4.spec"

    assert spec_path.is_file()
    spec = spec_path.read_text("utf-8")
    assert "name='CtrlSpeak_v0.4'" in spec
    assert "assets' / 'icon.png'" in spec
    assert "console=False" in spec
    assert "utils.linux_input" in spec
    assert "pynput._util.xorg" in spec
    assert "pynput.keyboard._xorg" in spec
    assert "pynput.mouse._xorg" in spec
    assert system.APP_VERSION == "0.4.0"


def test_build_helper_routes_linux_to_v04_and_keeps_windows_spec() -> None:
    from utils.build_exe import _resolve_build_config

    linux = _resolve_build_config(False, platform_name="linux")
    windows = _resolve_build_config(False, platform_name="win32")

    assert linux.name == "CtrlSpeak v0.4 Linux"
    assert linux.spec_path.name == "CtrlSpeak_v0.4.spec"
    assert windows.spec_path.name == "CtrlSpeak_v0.3.spec"


def test_desktop_launcher_template_and_appstream_metadata_are_consistent() -> None:
    desktop_path = ROOT / "packaging" / "linux" / "ctrlspeak.desktop"
    metadata_path = ROOT / "packaging" / "linux" / "io.trueai.ctrlspeak.metainfo.xml"
    icon_path = ROOT / "assets" / "icon.png"

    assert desktop_path.is_file()
    desktop = desktop_path.read_text("utf-8")
    assert desktop.startswith("[Desktop Entry]\n")
    assert "Type=Application" in desktop
    assert "Name=CtrlSpeak" in desktop
    assert "Exec=@CTRLSPEAK_EXECUTABLE@" in desktop
    assert "Icon=@CTRLSPEAK_ICON@" in desktop
    assert "Terminal=false" in desktop
    assert "StartupWMClass=CtrlSpeak" in desktop

    assert metadata_path.is_file()
    component = ElementTree.parse(metadata_path).getroot()
    assert component.findtext("id") == "io.trueai.ctrlspeak"
    launchable = component.find("launchable")
    assert launchable is not None
    assert launchable.attrib["type"] == "desktop-id"
    assert launchable.text == "ctrlspeak.desktop"
    release = component.find("releases/release")
    assert release is not None
    assert release.attrib["version"] == "0.4.0"

    assert icon_path.is_file()
    assert icon_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_linux_build_documentation_names_prerequisites_and_uninstalled_template() -> None:
    documentation = (ROOT / "packaging" / "BUILDING.md").read_text("utf-8")

    assert "dist/CtrlSpeak_v0.4" in documentation
    assert "packaging/CtrlSpeak_v0.4.spec" in documentation
    assert "xclip" in documentation
    assert "Ubuntu on Xorg" in documentation
    assert "does not install" in documentation
    assert "ctrlspeak.desktop" in documentation
