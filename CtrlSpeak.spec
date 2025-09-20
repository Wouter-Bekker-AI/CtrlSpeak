# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['CtrlSpeak.py'],
    pathex=[],
    binaries=[],
    datas=[('icon.ico', '.'), ('loading.wav', '.'), ('C:\\Users\\woute\\AppData\\Local\\Programs\\Python\\Python310\\CtrSpeak\\Lib\\site-packages\\nvidia\\cudnn\\bin', 'nvidia/cudnn/bin'), ('C:\\Users\\woute\\AppData\\Local\\Programs\\Python\\Python310\\CtrSpeak\\Lib\\site-packages\\nvidia\\cublas\\bin', 'nvidia/cublas/bin'), ('C:\\Users\\woute\\AppData\\Local\\Programs\\Python\\Python310\\CtrSpeak\\Lib\\site-packages\\faster_whisper\\assets', 'faster_whisper/assets')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='CtrlSpeak-full',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
