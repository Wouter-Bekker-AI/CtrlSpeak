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
- Creates `~/.ctrlspeak/` with subfolders:
  - `models/` for model weights
  - `runtimes/` for downloaded installers (CUDA/cuDNN on Windows)
- Prompts the user to download missing components with a GUI progress window showing **bytes**, **speed**, and **ETA**.
- Does **not** automatically run platform installers; the user must execute them if needed.