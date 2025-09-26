# Building CtrlSpeak one-file executable

## Prereqs
- Python 3.10+ on Windows
- `pip install -r requirements.txt`
- `pip install pyinstaller` (already in requirements)
- Ensure all optional GPU or model components are **not** bundled; this app fetches them at first launch.

## Build
From the project root run:

```powershell
python -m utils.build_exe
```

The helper wraps PyInstaller and executes `packaging/CtrlSpeak.spec`, keeping command-line and scripted builds aligned. You can still invoke `pyinstaller packaging/CtrlSpeak.spec` directly if you need custom flags.
To produce a private white-label build that swaps in `assets/Watcher_Intro_Video.mp4` and emits `CtrlSpeak_Watcher.exe`, pass
`--watcher` to the helper. That path uses `packaging/CtrlSpeak_Watcher.spec` while leaving the standard build flow untouched.

The spec collects the native data required by `faster_whisper`, `ctranslate2`, and `ffpyplayer`, and embeds the tray icon, loading chime, onboarding video, fun-facts list, and regression test clip under the packaged `assets/` directory. The output is a single-file windowed executable at `dist/CtrlSpeak.exe`; CUDA runtimes and Whisper models remain external downloads performed by the app at runtime.

## First-run behavior
- Creates the per-user application data directory (for example `%APPDATA%\CtrlSpeak` on Windows) with subfolders `models/`, `cuda/`, `temp/`, and `logs/`.
- Automatically downloads the default `small` Whisper model so transcription works on CPU immediately.
- Defers CUDA preparation until the user runs `python main.py --download-cuda-only` (alias: `--setup-cuda`) or chooses **Install or repair CUDA** in the management window; when no CUDA-capable GPU is detected the command exits immediately and the UI leaves the GPU option hidden, so the packaged build never attempts a CUDA install on unsupported hardware. When invoked, the installer fetches the CUDA runtime, cuBLAS, and cuDNN wheels, caches the wheels under `%APPDATA%\CtrlSpeak\cuda\downloads` until validation succeeds, and stages the DLLs under `%APPDATA%\CtrlSpeak\cuda\12.3`.
