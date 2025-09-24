# Building CtrlSpeak one-file executable

## Prereqs
- Python 3.10+ on Windows
- `pip install -r requirements.txt`
- `pip install pyinstaller` (already in requirements)
- Ensure all optional GPU or model components are **not** bundled; this app fetches them at first launch.

## Build
From the project root (the folder containing `main.py`), run:

```powershell
pyinstaller packaging/CtrlSpeak.spec
```

Outputs are under `dist/CtrlSpeak/`.

## First-run behavior
- Creates the per-user application data directory (for example `%APPDATA%\CtrlSpeak` on Windows) with subfolders `models/`, `cuda/`, `temp/`, and `logs/`.
- Automatically downloads the default `small` Whisper model so transcription works on CPU immediately.
- Defers CUDA preparation until the user runs `python main.py --setup-cuda` or chooses **Install or repair CUDA** in the management window; no GPU installers are run during a normal startup.
