from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest

from utils import system


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.core_headless


def test_v05_pyinstaller_spec_uses_stable_artifact_and_platform_adapters() -> None:
    spec_path = ROOT / "packaging" / "CtrlSpeak_v0.6.spec"

    assert spec_path.is_file()
    spec = spec_path.read_text("utf-8")
    assert "name='CtrlSpeak'" in spec
    assert "assets' / icon_name" in spec
    assert "icon_name = 'icon.ico' if IS_WINDOWS else 'icon.png'" in spec
    assert "console=False" in spec
    assert "utils.linux_input" in spec
    assert "utils.windows_input" in spec
    assert "utils.update_manager" in spec
    assert "cryptography" in spec
    assert "pynput._util.xorg" in spec
    assert "pynput.keyboard._xorg" in spec
    assert "pynput.mouse._xorg" in spec
    assert system.APP_VERSION == "0.6.1"


def test_build_helper_routes_windows_and_linux_to_v06_spec() -> None:
    from utils.build_exe import _resolve_build_config

    linux = _resolve_build_config(False, platform_name="linux")
    windows = _resolve_build_config(False, platform_name="win32")

    assert linux.name == "CtrlSpeak v0.6 Linux"
    assert linux.spec_path.name == "CtrlSpeak_v0.6.spec"
    assert windows.name == "CtrlSpeak v0.6 Windows"
    assert windows.spec_path.name == "CtrlSpeak_v0.6.spec"


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
    assert release.attrib["version"] == "0.6.1"

    assert icon_path.is_file()
    assert icon_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_linux_build_documentation_names_prerequisites_and_uninstalled_template() -> None:
    documentation = (ROOT / "packaging" / "BUILDING.md").read_text("utf-8")

    assert "dist/CtrlSpeak" in documentation
    assert "packaging/CtrlSpeak_v0.6.spec" in documentation
    assert "xclip" in documentation
    assert "Ubuntu on Xorg" in documentation
    assert "does not install" in documentation
    assert "ctrlspeak.desktop" in documentation
