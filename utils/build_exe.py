"""Select and run the native CtrlSpeak v0.7 PyInstaller specification.

The standard Windows and Linux artifact uses the stable name ``dist/CtrlSpeak``
(with ``.exe`` added by PyInstaller on Windows). From the project root run::

    python -m utils.build_exe
"""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
assets_dir = project_root / "assets"
standard_spec_path = project_root / "packaging" / "CtrlSpeak_v0.7.spec"
watcher_spec_path = project_root / "packaging" / "CtrlSpeak_Watcher.spec"


COMMON_REQUIRED_ASSETS = {
    "processing chime": assets_dir / "loading.wav",
    "automation clip": assets_dir / "test.wav",
    "fun facts list": assets_dir / "fun_facts.txt",
}


@dataclass(frozen=True)
class BuildConfig:
    name: str
    spec_path: Path
    intro_video_label: str
    intro_video_path: Path
    icon_path: Path


def _parse_args() -> bool:
    parser = ArgumentParser(description="Build the CtrlSpeak executable with PyInstaller")
    parser.add_argument(
        "--watcher",
        action="store_true",
        help="Build the white-label Watcher variant using the alternate intro video",
    )
    args = parser.parse_args()
    return bool(args.watcher)


def _resolve_build_config(
    watcher: bool,
    *,
    platform_name: str | None = None,
) -> BuildConfig:
    platform_value = (platform_name or sys.platform).lower()
    if platform_value.startswith("linux"):
        if watcher:
            raise SystemExit("The Watcher white-label build is Windows-only.")
        return BuildConfig(
            name="CtrlSpeak v0.7 Linux",
            spec_path=standard_spec_path,
            intro_video_label="welcome video",
            intro_video_path=assets_dir / "TrueAI_Intro_Video.mp4",
            icon_path=assets_dir / "icon.png",
        )
    if not platform_value.startswith("win"):
        raise SystemExit(
            f"CtrlSpeak packaging is supported only on Linux and Windows, not {platform_value!r}."
        )
    if watcher:
        return BuildConfig(
            name="CtrlSpeak Watcher",
            spec_path=watcher_spec_path,
            intro_video_label="Watcher intro video",
            intro_video_path=assets_dir / "Watcher_Intro_Video.mp4",
            icon_path=assets_dir / "icon.ico",
        )
    return BuildConfig(
        name="CtrlSpeak v0.7 Windows",
        spec_path=standard_spec_path,
        intro_video_label="welcome video",
        intro_video_path=assets_dir / "TrueAI_Intro_Video.mp4",
        icon_path=assets_dir / "icon.ico",
    )


def build(*, watcher: bool = False) -> None:
    config = _resolve_build_config(watcher)

    if not config.spec_path.exists():
        raise SystemExit(f"Missing PyInstaller spec at {config.spec_path}")

    required_assets = dict(COMMON_REQUIRED_ASSETS)
    required_assets["application icon"] = config.icon_path
    required_assets[config.intro_video_label] = config.intro_video_path

    missing_assets = [name for name, path in required_assets.items() if not path.exists()]
    if missing_assets:
        formatted = ", ".join(sorted(missing_assets))
        raise SystemExit(f"Missing required asset(s): {formatted}")

    print(f"Building {config.name} bundle...")
    args = ["--noconfirm", "--clean", str(config.spec_path)]
    print("Running PyInstaller with arguments:\n  " + "\n  ".join(args))
    try:
        from PyInstaller.__main__ import run as pyinstaller_run
    except ImportError as exc:  # pragma: no cover - developer convenience
        raise SystemExit(
            "PyInstaller is required to build the executable. "
            "Install it with 'pip install pyinstaller'."
        ) from exc
    pyinstaller_run(args)


if __name__ == '__main__':
    watcher_flag = _parse_args()
    build(watcher=watcher_flag)
