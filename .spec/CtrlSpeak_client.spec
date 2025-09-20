# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CUDA_DIR = PROJECT_ROOT / "CUDA"

binaries = []
if CUDA_DIR.exists():
    for p in CUDA_DIR.glob("*.dll"):
        binaries.append((str(p), "."))

block_cipher = None

a = Analysis(
    ['CtrlSpeak.py'],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

exe = EXE(
    a.pure, a.scripts, a.binaries, a.zipfiles, a.datas,
    name='CtrlSpeakClient',
    debug=False,
    strip=False,
    upx=True,
    console=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='CtrlSpeakClient'
)
