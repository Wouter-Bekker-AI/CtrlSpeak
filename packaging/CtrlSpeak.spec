# -*- mode: python ; coding: utf-8 -*-
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
for pkg in ('faster_whisper', 'ctranslate2', 'huggingface_hub', 'ffpyplayer', 'pyautogui', 'certifi'):
    third_party_datas.extend(_collect_or_empty(collect_data_files, pkg))

asset_datas = [
    (str(PROJECT_ROOT / 'assets' / 'icon.ico'), 'assets'),
    (str(PROJECT_ROOT / 'assets' / 'loading.wav'), 'assets'),
    (str(PROJECT_ROOT / 'assets' / 'test.wav'), 'assets'),
    (str(PROJECT_ROOT / 'assets' / 'fun_facts.txt'), 'assets'),
    (str(PROJECT_ROOT / 'assets' / 'TrueAI_Intro_Video.mp4'), 'assets'),
]

datas = _dedupe(third_party_datas + asset_datas)

third_party_binaries = []
for pkg in ('ctranslate2', 'ffpyplayer', 'faster_whisper'):
    third_party_binaries.extend(_collect_or_empty(collect_dynamic_libs, pkg))

binaries = _dedupe(third_party_binaries)

a = Analysis(
    [str(PROJECT_ROOT / 'main.py')],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=['ffpyplayer.player', 'pyaudio', 'pyautogui', 'pynput'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    upx=True,
    console=False,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(PROJECT_ROOT / 'assets' / 'icon.ico')],
    onefile=True,
)
