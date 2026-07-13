# Building CtrlSpeak one-file executable

## Prereqs
- A 64-bit Windows 10/11 build host with Python 3.10+
- `pip install -r requirements.txt`
- `pip install pyinstaller` (already in requirements)
- Ensure all optional GPU or model components are **not** bundled; this app fetches them at first launch.

## Build
From the project root run:

```powershell
python -m utils.build_exe
```

Build on Windows from a clean virtual environment. The maintained spec is `packaging/CtrlSpeak_v0.3.spec`; the equivalent direct command is:

```powershell
pyinstaller --noconfirm --clean packaging/CtrlSpeak_v0.3.spec
```

The helper wraps PyInstaller and executes `packaging/CtrlSpeak_v0.3.spec`, keeping command-line and scripted builds aligned. You can still invoke `pyinstaller packaging/CtrlSpeak_v0.3.spec` directly if you need custom flags.
To produce a private white-label build that swaps in `assets/Watcher_Intro_Video.mp4` and emits `CtrlSpeak_Watcher.exe`, pass
`--watcher` to the helper. That path uses `packaging/CtrlSpeak_Watcher.spec` while leaving the standard build flow untouched.

The spec collects the native data required by `faster_whisper`, `ctranslate2`, and `ffpyplayer`, and embeds the existing `assets/icon.ico`, loading chime, onboarding video, fun-facts list, and regression test clip under the packaged `assets/` directory. The output is the single-file windowed executable `dist/CtrlSpeak_v0.3.exe`; CUDA runtimes and Whisper models remain external downloads performed by the app at runtime.

PyInstaller does not cross-compile this Windows executable from Linux. After the command finishes on Windows, verify the exact artifact and icon before distribution:

```powershell
Get-Item .\dist\CtrlSpeak_v0.3.exe
```

The spec uses `console=False`, so `CtrlSpeak_v0.3.exe` is a GUI executable and does not have a console window. Console-oriented flags such as status output and file transcription are useful when running `python main.py` from source, but their stdout/stderr is not visible when the packaged executable is launched normally. Startup configuration failures are therefore shown in a blocking GUI error dialog. Validate redacted CLI status from source with `python main.py --backend-status`; validate the packaged backend in the management window.

Code signing, malware scanning, and testing on a clean Windows 10/11 machine are release steps performed after this build; the helper does not sign, install, deploy, open firewall ports, or start services.

## First-run behavior
- Creates the per-user application data directory (for example `%APPDATA%\CtrlSpeak` on Windows) with subfolders `models/`, `cuda/`, `temp/`, and `logs/`.
- In the default embedded/local backend, automatically downloads the default `small` Whisper model so transcription works on CPU immediately.
- Defers CUDA preparation until the user runs `python main.py --download-cuda-only` (alias: `--setup-cuda`) or chooses **Install or repair CUDA** in the management window; when no CUDA-capable GPU is detected the command exits immediately and the UI leaves the GPU option hidden, so the packaged build never attempts a CUDA install on unsupported hardware. When invoked, the installer fetches the CUDA runtime, cuBLAS, and cuDNN wheels, caches the wheels under `%APPDATA%\CtrlSpeak\cuda\downloads` until validation succeeds, and stages the DLLs under `%APPDATA%\CtrlSpeak\cuda\12.3`.
- When API mode is selected before startup (for example with `--backend api` or `CTRLSPEAK_BACKEND=api`), bundled model download/warm-up is skipped. The default remains bundled mode.
- Creates settings and the exact-only local correction index under the per-user application-data directory, never beside the executable.
