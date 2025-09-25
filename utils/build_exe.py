"""Helper script to build the CtrlSpeak executable with PyInstaller.

The resulting bundle lives in ``dist/CtrlSpeak/``. From the project root run::

    python -m utils.build_exe
"""
from __future__ import annotations

from pathlib import Path

try:
    from PyInstaller.__main__ import run as pyinstaller_run
except ImportError as exc:  # pragma: no cover - developer convenience
    raise SystemExit(
        "PyInstaller is required to build the executable. "
        "Install it with 'pip install pyinstaller'."
    ) from exc

project_root = Path(__file__).resolve().parent.parent
assets_dir = project_root / 'assets'
spec_path = project_root / 'packaging' / 'CtrlSpeak.spec'

REQUIRED_ASSETS = {
    'application icon': assets_dir / 'icon.ico',
    'processing chime': assets_dir / 'loading.wav',
    'automation clip': assets_dir / 'test.wav',
    'fun facts list': assets_dir / 'fun_facts.txt',
    'welcome video': assets_dir / 'TrueAI_Intro_Video.mp4',
}


def build() -> None:
    if not spec_path.exists():
        raise SystemExit(f'Missing PyInstaller spec at {spec_path}')

    missing_assets = [name for name, path in REQUIRED_ASSETS.items() if not path.exists()]
    if missing_assets:
        formatted = ', '.join(sorted(missing_assets))
        raise SystemExit(f'Missing required asset(s): {formatted}')

    args = ['--noconfirm', '--clean', str(spec_path)]
    print('Running PyInstaller with arguments:\n  ' + '\n  '.join(args))
    pyinstaller_run(args)


if __name__ == '__main__':
    build()
