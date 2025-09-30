# CtrlSpeak

CtrlSpeak is a Windows speech-to-text assistant that records speech while the right `Ctrl` key is held down and injects the transcription into the active text control. It can run as a self-contained **Client + Server** bundle with Whisper hosted locally or as a lightweight **Client Only** build that discovers a LAN server.

Both flavours support Windows 10/11, enforce a single running instance, expose a tray UI for mode switching, and include AnyDesk-aware text injection with optional audio cues.

## Repository Layout

- `main.py` – application entry point.
- `background_agents/` – self-contained helpers that run alongside Chat with Bot. The bundled `tts_preprocessing_agent` rewrites LLM replies before Kokoro speaks them when an identity opts into text cleaning.
- `assets/` – static resources such as the tray icon (`icon.ico`), the welcome video (`TrueAI_Intro_Video.mp4`), the fun-fact rotation list (`fun_facts.txt`), and the processing chime (`loading.wav`).
- `utils/` – implementation modules (GUI, models, networking, configuration helpers, etc.).
- `utils/build_exe.py` – helper script that runs PyInstaller with the correct data files.
- `tools/` – reusable tooling (currently the shared `vision.py` capture helpers and keyword registry); see [`docs/tooling.md`](docs/tooling.md) for the authoritative API reference.
- `packaging/` – PyInstaller spec (`CtrlSpeak.spec`) and additional build documentation.

Generated folders such as `dist/` and `build/` are ignored via `.gitignore`.

## Environment Setup

Create an isolated environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Note: The optional "Chat with Bot" feature using the logo animation style requires PySide6, which is included in requirements.txt.
```

GPU acceleration requires an NVIDIA CUDA-capable GPU with compatible drivers, but CtrlSpeak always boots in CPU mode and skips CUDA validation unless you opt in. The Whisper `small` model is downloaded automatically on first launch so a fresh install is usable immediately. Selecting **GPU (CUDA)** in the management window now launches the same welcome-and-progress experience used for model downloads; the app installs the CUDA runtime, cuBLAS, and cuDNN automatically and only falls back to CPU if validation fails. The CUDA wheels are cached under `%APPDATA%\CtrlSpeak\cuda\downloads`, verified with the published SHA-256 digests, and reused on the next attempt so extraction failures no longer force a redownload; the cache is purged only after a validated install. You can also stage GPU support manually via `python main.py --download-cuda-only` (alias: `--setup-cuda`). When no CUDA-capable GPU is detected, the management UI hides the GPU option and the installer flag exits early with an explanatory message.
During the initial Whisper download, CtrlSpeak opens a centered welcome window sized to roughly 80% of a 1080p frame (about 1536×864) that plays the bundled intro clip (about five seconds for the default `TrueAI_Intro_Video.mp4`) with audio. Once the clip ends, the window transitions into a branded fun-facts card featuring the CtrlSpeak logo on a white tile and rotating onboarding tips sourced from `assets/fun_facts.txt`. A slim lockout window remains in the top-left corner with live status text and a red **Cancel download** button; cancelling stops the download subprocess immediately, exiting entirely if no model is available or otherwise returning you to the currently staged model.
The same welcome-and-progress experience now runs whenever Chat with Bot needs to download an Ollama model for a selected identity, keeping the assistant in the intro/fun-facts flow until the LLM weights finish caching.


## Running from Source

```powershell
python main.py
```

On first launch you will be prompted to choose between **Client + Server** or **Client Only** modes. The client-only card lets you refresh for LAN servers or manually enter a `host[:port]` so CtrlSpeak knows which remote host to target. Settings, models, and logs live under `%APPDATA%\CtrlSpeak` (the folder is created automatically).

### Command-line Flags

- `--auto-setup {client,client_server}` – pre-select the startup mode without showing the GUI prompts.
- `--force-sendinput` – force the AnyDesk-compatible synthetic keystroke path.
- `--download-cuda-only` (alias: `--setup-cuda`) – stage the CUDA runtime, cuBLAS, and cuDNN support packages (reusing any cached wheels before downloading fresh copies) and exit; the command aborts immediately when no CUDA-capable GPU is detected.
- `--transcribe <wav>` – batch process an audio file without the hotkey workflow.
- `--uninstall` – remove the application data and executable (used by the packaged build).

Run `python main.py --help` for the full list.

### Chat with Bot speech pipeline

When you launch **Chat with Bot**, identities that set `require_text_cleaning: true` in their `identity.json` route the LLM reply through `background_agents/tts_preprocessing_agent` before Kokoro generates speech. The helper reads its own configuration to decide whether to prepend `header_text.txt`, supply `system_prompt.txt`, or do both based on the `preamble` setting (`header`, `system`, or `both`, with the agent defaulting to `both`). That instruction set asks Gemma 3 1B to smooth the phrasing for text-to-speech and returns the rewritten script. Identities with `require_text_cleaning: false` hand their responses directly to Kokoro. If the agent fails or the directory is missing, CtrlSpeak falls back to speaking the original reply so conversations continue uninterrupted.

Voice keywords live in [`tools/keywords.py`](docs/tooling.md). Saying “look at my screen” or “look at my clipboard” captures an image via the shared vision tooling, “chat with <identity>” (for example, assistant or default) relaunches the bot with that persona unless it is already active, and “goodbye <identity>” stops the current conversation immediately.

Holding the right `Ctrl` hotkey also honours those keywords. When no bot is running, saying “chat with assistant” (or any configured identity) while using the push-to-talk workflow launches the requested bot instead of typing the phrase into the focused window. Saying “goodbye <identity>” through the same hotkey closes the active session using the same shutdown path as the tray menu, and switching identities issues a graceful stop before starting the new persona.

## Packaging with PyInstaller

Use the helper module to build a distributable executable under `dist/CtrlSpeak/`:

```powershell
python -m utils.build_exe
```

The helper executes the maintained `packaging/CtrlSpeak.spec` so manual `pyinstaller` runs stay aligned. The resulting one-file GUI build embeds the tray icon, loading chime, onboarding video, fun-facts rotation list, and regression test clip inside the internal `assets/` directory. Required runtime data for `faster_whisper`, `ctranslate2`, and `ffpyplayer` is collected automatically, while CUDA runtimes and Whisper model weights remain external downloads performed at runtime when the user opts in.

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
- Track future enhancements in [`docs/TODO.md`](docs/TODO.md); keep the list current as tasks are added or completed.

## Automation Flow

Run the regression harness to validate a workstation without touching the GUI:

```powershell
python main.py --automation-flow
```

The command performs a staged health-check entirely inside %APPDATA%\CtrlSpeak:

1. Ensure the default Whisper model is present under %APPDATA%\CtrlSpeak\models (downloading it when missing).
2. Reuse or install the NVIDIA CUDA runtime stack (nvidia-cuda-runtime-cu12, nvidia-cublas-cu12, nvidia-cudnn-cu12) so the DLLs live under %APPDATA%\CtrlSpeak\cuda\12.3 when GPU testing is required.
3. Transcribe assets/test.wav on the CPU.
4. Transcribe the same clip on the GPU using the DLLs staged in %APPDATA%\CtrlSpeak\cuda\12.3.
5. Simulate each text-injection strategy (direct insert, SendInput paste, clipboard paste, PyAutoGUI typing) and write a consolidated report to %APPDATA%\CtrlSpeak\automation\artifacts.

If any stage fails the workflow stops at that checkpoint and leaves detailed logs plus the partially populated automation_state.json in the same automation folder. Fix the underlying system issue (drivers, CUDA DLLs, networking, etc.) and re-run the flag - the script resumes where it left off.

### Handing the checklist to another operator or AI agent

Provide your helper with the single command above and the acceptance criteria:

- All stages complete without errors on a single pass.
- %APPDATA%\CtrlSpeak\automation\artifacts contains a report named automation_run_*.txt whose injection sections echo the canonical transcript.
- %APPDATA%\CtrlSpeak\cuda\12.3 holds the CUDA DLLs and `python main.py` can select both CPU and GPU devices without warnings.

An agent can loop on `python main.py --automation-flow`, examine automation_state.json, and only make host-level changes (install drivers, adjust PATH, etc.) until the run succeeds - no code edits are required.



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
