# -*- mode: python ; coding: utf-8 -*-
"""Maintained cross-platform PyInstaller specification for CtrlSpeak v0.5."""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


block_cipher = None

if sys.argv and sys.argv[-1].endswith('.spec'):
    SPEC_FILE = Path(sys.argv[-1]).resolve()
elif sys.argv and sys.argv[0].endswith('.spec'):
    SPEC_FILE = Path(sys.argv[0]).resolve()
elif '__file__' in globals():
    SPEC_FILE = Path(__file__).resolve()
elif '__spec__' in globals() and getattr(__spec__, 'origin', None):  # type: ignore[name-defined]
    SPEC_FILE = Path(__spec__.origin).resolve()  # type: ignore[name-defined]
else:
    SPEC_FILE = Path(os.path.abspath('.')).resolve()

BASE_DIR = SPEC_FILE.parent
PROJECT_ROOT = BASE_DIR.parent
IS_WINDOWS = sys.platform.startswith('win')


def _collect_or_empty(collector, package_name):
    try:
        return collector(package_name)
    except Exception:
        return []


def _dedupe(items):
    seen = set()
    unique = []
    for entry in items:
        key = tuple(entry) if isinstance(entry, (list, tuple)) else entry
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


third_party_datas = []
for pkg in (
    'faster_whisper',
    'ctranslate2',
    'huggingface_hub',
    'ffpyplayer',
    'pyautogui',
    'certifi',
    'requests',
    'cryptography',
):
    third_party_datas.extend(_collect_or_empty(collect_data_files, pkg))

icon_name = 'icon.ico' if IS_WINDOWS else 'icon.png'
asset_datas = [
    (str(PROJECT_ROOT / 'assets' / icon_name), 'assets'),
    (str(PROJECT_ROOT / 'assets' / 'loading.wav'), 'assets'),
    (str(PROJECT_ROOT / 'assets' / 'test.wav'), 'assets'),
    (str(PROJECT_ROOT / 'assets' / 'fun_facts.txt'), 'assets'),
    (str(PROJECT_ROOT / 'assets' / 'TrueAI_Intro_Video.mp4'), 'assets'),
]
datas = _dedupe(third_party_datas + asset_datas)

third_party_binaries = []
for pkg in ('ctranslate2', 'ffpyplayer', 'faster_whisper', 'cryptography'):
    third_party_binaries.extend(_collect_or_empty(collect_dynamic_libs, pkg))
binaries = _dedupe(third_party_binaries)

common_hiddenimports = [
    'cryptography',
    'cryptography.hazmat.primitives.asymmetric.ed25519',
    'ffpyplayer.player',
    'pyaudio',
    'pyautogui',
    'pystray',
    'pynput',
    'requests',
    'utils.update_manager',
    'utils.update_helper',
]
if IS_WINDOWS:
    platform_hiddenimports = [
        'pystray._win32',
        'pynput._util.win32',
        'pynput.keyboard._win32',
        'pynput.mouse._win32',
        'utils.windows_input',
    ]
    excludes = ['astroid', 'bs4', 'utils.linux_input']
else:
    platform_hiddenimports = [
        'pystray._appindicator',
        'pystray._gtk',
        'pystray._xorg',
        'pynput._util.xorg',
        'pynput.keyboard._xorg',
        'pynput.mouse._xorg',
        'utils.hotkeys',
        'utils.linux_input',
    ]
    excludes = ['astroid', 'bs4', 'utils.windows_input']

a = Analysis(
    [str(PROJECT_ROOT / 'main.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=common_hiddenimports + platform_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CtrlSpeak',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(PROJECT_ROOT / 'assets' / icon_name)],
    version=str(PROJECT_ROOT / 'packaging' / 'windows_version_info.txt') if IS_WINDOWS else None,
    onefile=True,
)
