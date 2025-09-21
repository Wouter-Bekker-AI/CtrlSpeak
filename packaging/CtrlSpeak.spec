# -*- mode: python ; coding: utf-8 -*-
# Run from packaging\ or project root; entry is one dir up.
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Include package data for faster_whisper (assets etc.)
fw_datas = collect_data_files('faster_whisper')

a = Analysis(
    ['..\\main.py'],        # entry script relative to this spec folder
    pathex=['..'],          # add project root to search path
    binaries=[],
    datas=fw_datas + [
        ('..\\icon.ico', '.'),
        ('..\\loading.wav', '.'),
    ],
    hiddenimports=[],
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
    console=False,                 # windowed; no terminal
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['..\\icon.ico'],         # app icon
    onefile=True,
)