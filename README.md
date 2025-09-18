# CtrlSpeak

CtrlSpeak is a Windows speech-to-text assistant that records speech while the right `Ctrl` key is held down and injects the transcription into the active text control. It supports two deployment flavours:

- **Full (Client + Server)**: runs Whisper locally (GPU with CPU fallback) and serves other machines on the LAN.
- **Client Only**: lightweight build that discovers a LAN server and falls back to CPU transcription when none is available.

Both builds support Windows 10 and 11, single-instance enforcement, tray-based control, configurable sound cues, and AnyDesk-aware text injection.

## Repository Layout

- `CtrlSpeak.py` – main application code.
- `CtrlSpeak.spec` – PyInstaller spec used for packaging.
- `loading.wav` – chime played while transcription is running.
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
python CtrlSpeak.py
```

On first launch you will be prompted to choose between **Client + Server** or **Client Only** modes. Settings, models, and logs live under `%APPDATA%\CtrlSpeak`.

### Command-line Flags

- `--force-sendinput` – force the AnyDesk-compatible synthetic keystroke path.
- `--transcribe <wav>` – batch process an audio file without the hotkey workflow.
- `--delete` – remove the application data and executable (used by the packaged build).

Run `python CtrlSpeak.py --help` for the full list.

## Building Executables

This project uses PyInstaller. Builds store mode metadata by dropping a `client_only.flag` file next to the executable before packaging.

### Full (Client + Server) build

```powershell
# From the project root
Remove-Item client_only.flag -ErrorAction SilentlyContinue
pyinstaller CtrlSpeak.spec --clean --noconfirm --name CtrlSpeak-full
```

### Client Only build

```powershell
New-Item client_only.flag -ItemType File -Force | Out-Null
pyinstaller CtrlSpeak.spec --clean --noconfirm --name CtrlSpeak-client
Remove-Item client_only.flag
```

Generated binaries appear in `dist/`.

## LAN Server Setup

After packaging the full build and copying it to a host machine, run the PowerShell commands in `serverportsetup.txt` (once, as Administrator) to open the discovery (UDP 54330) and API (TCP 65432) ports for the local network. Client machines can then find the server via the tray UI (**Refresh Servers**).

## Development Notes

- Temporary recordings, configuration, logs, and downloaded Whisper models live under `%APPDATA%\CtrlSpeak`.
- Test audio files such as `part1.wav` are intentionally excluded from Git to avoid large binaries.
- Use the tray menu to manage the client/server lifecycle or to uninstall (`Delete CtrlSpeak`).

## License

TBD – add licensing information before public release.
