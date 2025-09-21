# CtrlSpeak

CtrlSpeak is a Windows speech-to-text assistant that records speech while the right `Ctrl` key is held down and injects the transcription into the active text control. It supports two deployment flavours:

- **Full (Client + Server)**: runs Whisper locally (GPU with CPU fallback) and serves other machines on the LAN.
- **Client Only**: lightweight build that discovers a LAN server and falls back to CPU transcription when none is available.

Both builds support Windows 10 and 11, single-instance enforcement, tray-based control, configurable sound cues, and AnyDesk-aware text injection.

## Repository Layout

- `main.py` – application entry point.
- `build_exe.py` – helper script that runs PyInstaller with the correct data files.
- `assets/` – static resources such as the tray icon (`icon.ico`) and processing chime (`loading.wav`).
- `utils/` – implementation modules (GUI, models, networking, configuration helpers, etc.).
- `serverportsetup.txt` – Windows PowerShell commands to open discovery/API firewall ports.

The `dist`, `build`, and other generated folders are ignored from version control.

## Python Requirements

Install the runtime dependencies inside a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

See `requirements.txt` for the package list (GPU acceleration requires NVIDIA CUDA-capable hardware and compatible drivers).

## Running from Source

```powershell
python main.py
```

On first launch you will be prompted to choose between **Client + Server** or **Client Only** modes. Settings, models, and logs live under `%APPDATA%\CtrlSpeak` (the folder is created automatically).

### Command-line Flags

- `--force-sendinput` - force the AnyDesk-compatible synthetic keystroke path.
- `--transcribe <wav>` - batch process an audio file without the hotkey workflow.
- `--uninstall` - remove the application data and executable (used by the packaged build).

Run `python main.py --help` for the full list.

## Building Executables

This project ships with a small helper that wraps PyInstaller with the correct options and assets. The generated build lives in `dist/CtrlSpeak/`.

```powershell
# From the project root
python build_exe.py
```

The script produces a windowed build that embeds `assets/icon.ico` as the executable icon and bundles both `assets/icon.ico` and `assets/loading.wav` as application data inside the package. CUDA runtime files or Whisper model weights are **not** bundled – they are downloaded by the running application if the user opts in.

## Manual Model Download

CtrlSpeak caches Whisper model weights under `%APPDATA%\CtrlSpeak\models`. The default configuration selects the lightweight `small` Whisper checkpoint and runs on the CPU. If you want to preload the model without launching the GUI, use the Hugging Face CLI:

```powershell
pip install huggingface_hub
$target = Join-Path $env:APPDATA 'CtrlSpeak\models\small'
huggingface-cli download Systran/faster-whisper-small --local-dir $target --local-dir-use-symlinks False
New-Item -ItemType File (Join-Path $target '.installed') -Force | Out-Null
```

- You can substitute a different `repo/model` name if you use another Whisper checkpoint.
- To point CtrlSpeak at a custom directory, set the `CTRLSPEAK_MODEL_DIR` environment variable to the parent folder that contains the models (defaults to `%APPDATA%\CtrlSpeak\models`).

## LAN Server Setup

After packaging the full build and copying it to a host machine, run the PowerShell commands in `serverportsetup.txt` (once, as Administrator) to open the discovery (UDP 54330) and API (TCP 65432) ports for the local network. Client machines can then find the server via the tray UI (**Refresh Servers**).

## Development Notes

- Temporary recordings, configuration, logs, and downloaded Whisper models live under `%APPDATA%\CtrlSpeak`.
- Test audio files such as `part1.wav` are intentionally excluded from Git to avoid large binaries.
- Use the tray menu to manage the client/server lifecycle or to uninstall (`Delete CtrlSpeak`).

## License

This project is released under the MIT License:

```
MIT License

Copyright (c) 2025 CtrlSpeak contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
