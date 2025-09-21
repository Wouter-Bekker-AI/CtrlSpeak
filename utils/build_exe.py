"""Helper script to build the CtrlSpeak executable with PyInstaller.

The resulting bundle lives in ``dist/CtrlSpeak/``. From the project root run::

    python -m utils.build_exe
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from PyInstaller.__main__ import run as pyinstaller_run
except ImportError as exc:  # pragma: no cover - developer convenience
    raise SystemExit(
        "PyInstaller is required to build the executable. "
        "Install it with 'pip install pyinstaller'."
    ) from exc


def build() -> None:
    root = Path(__file__).resolve().parent
    assets_dir = root / "assets"
    icon_path = assets_dir / "icon.ico"
    audio_path = assets_dir / "loading.wav"

    if not icon_path.exists():
        raise SystemExit(f"Missing icon at {icon_path}")
    if not audio_path.exists():
        raise SystemExit(f"Missing loading sound at {audio_path}")

    data_sep = ";" if os.name == "nt" else ":"
    datas = [
        f"{icon_path}{data_sep}assets",
        f"{audio_path}{data_sep}assets",
    ]

    args = [
        str(project_root / "main.py"),
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name=CtrlSpeak",
        f"--icon={icon_path}",
        "--collect-all=certifi",
        "--collect-submodules=huggingface_hub"
    ]
    args.extend(f"--add-data={entry}" for entry in datas)

    print("Running PyInstaller with arguments:\n  " + "\n  ".join(args))
    pyinstaller_run(args)


if __name__ == "__main__":
    build()
