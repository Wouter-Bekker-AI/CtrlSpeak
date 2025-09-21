# CtrlSpeak

CtrlSpeak is a Windows speech-to-text assistant that records speech while the right `Ctrl` key is held down and injects the transcription into the active text control. It can run as a self-contained **Client + Server** bundle with Whisper hosted locally or as a lightweight **Client Only** build that discovers a LAN server.

Both flavours support Windows 10/11, enforce a single running instance, expose a tray UI for mode switching, and include AnyDesk-aware text injection with optional audio cues.

## Repository Layout

- `main.py` – application entry point.
- `build_exe.py` – helper script that runs PyInstaller with the correct data files.
- `assets/` – static resources such as the tray icon (`icon.ico`) and processing chime (`loading.wav`).
- `utils/` – implementation modules (GUI, models, networking, configuration helpers, etc.).
- `serverportsetup.txt` – Windows PowerShell commands to open discovery/API firewall ports.

Generated folders such as `dist/` and `build/` are ignored via `.gitignore`.

## Environment Setup

Create an isolated environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU acceleration requires an NVIDIA CUDA-capable GPU with compatible drivers. Whisper models are downloaded on demand when the application starts in server mode.

## Running from Source

```powershell
python main.py
```

On first launch you will be prompted to choose between **Client + Server** or **Client Only** modes. Settings, models, and logs live under `%APPDATA%\CtrlSpeak` (the folder is created automatically).

### Command-line Flags

- `--force-sendinput` – force the AnyDesk-compatible synthetic keystroke path.
- `--transcribe <wav>` – batch process an audio file without the hotkey workflow.
- `--uninstall` – remove the application data and executable (used by the packaged build).

Run `python main.py --help` for the full list.

## Packaging with PyInstaller

Use the helper module to build a distributable executable under `dist/CtrlSpeak/`:

```powershell
python -m utils.build_exe
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

- Substitute a different `repo/model` name if you prefer another Whisper checkpoint.
- To point CtrlSpeak at a custom directory, set the `CTRLSPEAK_MODEL_DIR` environment variable to the parent folder that contains the models (defaults to `%APPDATA%\CtrlSpeak\models`).

## Windows Server Provisioning

When deploying the combined Client + Server build to a dedicated host, run the following elevated PowerShell commands once per machine:

1. Copy the packaged executable onto the target PC (example assumes the Desktop):
   ```powershell
   Copy-Item "C:\Users\<user>\PycharmProjects\CtrlSpeak\dist\CtrlSpeak-full.exe" "$env:USERPROFILE\Desktop\CtrlSpeak-full.exe"
   ```
2. Prime the installation and download the Whisper model by running auto-setup mode:
   ```powershell
   Start-Process -FilePath "$env:USERPROFILE\Desktop\CtrlSpeak-full.exe" -ArgumentList '--auto-setup','client_server' -Wait
   ```
3. Allow the discovery and API ports through Windows Firewall (adjust the profile if you need different scopes):
   ```powershell
   netsh advfirewall firewall add rule name="CtrlSpeak API" dir=in action=allow protocol=TCP localport=65432 profile=private
   netsh advfirewall firewall add rule name="CtrlSpeak API (Public)" dir=in action=allow protocol=TCP localport=65432 profile=public
   netsh advfirewall firewall add rule name="CtrlSpeak Discovery In" dir=in action=allow protocol=UDP localport=54330 profile=private
   netsh advfirewall firewall add rule name="CtrlSpeak Discovery Out" dir=out action=allow protocol=UDP localport=54330 profile=private
   netsh advfirewall firewall add rule name="CtrlSpeak Discovery In (Public)" dir=in action=allow protocol=UDP localport=54330 profile=public
   netsh advfirewall firewall add rule name="CtrlSpeak Discovery Out (Public)" dir=out action=allow protocol=UDP localport=54330 profile=public
   ```
4. Launch CtrlSpeak normally (double-click the EXE) and confirm the **Manage CtrlSpeak** window reports:
   - Mode: `client_server`
   - Server thread: `Running`
   - Serving: `<server-IP>:65432`

After updates you can re-run `--auto-setup client_server` to refresh the installation silently.

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
