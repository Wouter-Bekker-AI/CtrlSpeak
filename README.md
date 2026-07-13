# CtrlSpeak

CtrlSpeak v0.3 is a Windows speech-to-text assistant that records speech while the right `Ctrl` key is held down and injects the transcription into the active text control. Its control center explicitly offers **Embedded / local** or **Remote API** transcription. The legacy **Client + Server** and **Client Only** choices remain available inside embedded/local mode.

Both flavours support Windows 10/11, enforce a single running instance, expose a tray UI for mode switching, and include AnyDesk-aware text injection with optional audio cues.

## Repository Layout

- `main.py` – application entry point.
- `assets/` – static resources such as the tray icon (`icon.ico`), the welcome video (`TrueAI_Intro_Video.mp4`), the fun-fact rotation list (`fun_facts.txt`), and the processing chime (`loading.wav`).
- `utils/` – implementation modules (GUI, models, networking, configuration helpers, etc.).
- `utils/build_exe.py` – helper script that runs PyInstaller with the correct data files.
- `packaging/` – PyInstaller spec (`CtrlSpeak.spec`) and additional build documentation.

Generated folders such as `dist/` and `build/` are ignored via `.gitignore`.

## Environment Setup

Create an isolated environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU acceleration requires an NVIDIA CUDA-capable GPU with compatible drivers, but CtrlSpeak always boots in CPU mode and skips CUDA validation unless you opt in. The Whisper `small` model is downloaded automatically on first launch so a fresh install is usable immediately. Selecting **GPU (CUDA)** in the management window now launches the same welcome-and-progress experience used for model downloads; the app installs the CUDA runtime, cuBLAS, and cuDNN automatically and only falls back to CPU if validation fails. The CUDA wheels are cached under `%APPDATA%\CtrlSpeak\cuda\downloads`, verified with the published SHA-256 digests, and reused on the next attempt so extraction failures no longer force a redownload; the cache is purged only after a validated install. You can also stage GPU support manually via `python main.py --download-cuda-only` (alias: `--setup-cuda`). When no CUDA-capable GPU is detected, the management UI hides the GPU option and the installer flag exits early with an explanatory message.
During the initial Whisper download, CtrlSpeak opens a centered welcome window sized to roughly 80% of a 1080p frame (about 1536×864) that plays the bundled intro clip (about five seconds for the default `TrueAI_Intro_Video.mp4`) with audio. Once the clip ends, the window transitions into a branded fun-facts card featuring the CtrlSpeak logo on a white tile and rotating onboarding tips sourced from `assets/fun_facts.txt`. A slim lockout window remains in the top-left corner with live status text and a red **Cancel download** button; cancelling stops the download subprocess immediately, exiting entirely if no model is available or otherwise returning you to the currently staged model.


## Running from Source

```powershell
python main.py
```

On first launch you will be prompted to choose between **Client + Server** or **Client Only** modes. The client-only card lets you refresh for LAN servers or manually enter a `host[:port]` so CtrlSpeak knows which remote host to target. Settings, models, and logs live under `%APPDATA%\CtrlSpeak` (the folder is created automatically).

## Transcription backends

The backend choice is independent of the legacy client/server operating mode:

- **Embedded / local** (`bundled`, the default) preserves existing offline functionality and does not use the configurable Whisper API. Client + Server runs the bundled model directly; the legacy Client Only role continues to use CtrlSpeak discovery/`/transcribe` and its existing local-server recovery flow.
- **Remote API** (`api`) uploads the WAV recording to `POST <api_url>/v1/transcribe`. `api_url` is a complete `http://` or `https://` base URL and may identify loopback, a LAN/VPN host, or a properly secured public/cloud service; no LAN-only assumption is made. CtrlSpeak retains the returned `id`, `raw_text`, corrected `text`, segments, language, correction IDs, and exact-override ID with the pending injection. An API error is actionable and never silently switches to the embedded backend.

Bundled results pass through `%APPDATA%\CtrlSpeak\local-corrections.sqlite3`. User-approved edits create only complete, exact raw-transcript → corrected-transcript overrides, so a different or merely similar transcript is not rewritten. This index is local and works without network access. Remote API corrections remain at the configured service.

The management window has backend, API URL, masked optional bearer-token, feedback-capture, and redacted status controls. Backend settings are pinned when CtrlSpeak starts; saving a changed backend, URL, token, or feedback method explicitly requires a CtrlSpeak restart, avoiding a partially switched runtime. Defaults and persisted settings are:

```json
{
  "transcription_backend": "bundled",
  "api_url": "http://127.0.0.1:8765",
  "api_token": null,
  "feedback_capture_method": "active_field_on_enter"
}
```

Environment variables override saved backend values at runtime:

- `CTRLSPEAK_BACKEND=bundled|api`
- `CTRLSPEAK_API_URL=http://127.0.0.1:8765`
- `CTRLSPEAK_API_TOKEN=...`

There is intentionally no bearer-token command-line flag, because command arguments can be exposed in process listings and shell history. Prefer `CTRLSPEAK_API_TOKEN`; a token entered in the UI is stored in the per-user `settings.json` as plain text, so protect that account and file. On POSIX systems CtrlSpeak creates the settings and correction-database files with user-only permissions; Windows protection relies on the user profile ACL. Status text and object representations report only whether a token exists and never print it.

### Automatic edit feedback and platform limitations

With the default `active_field_on_enter` method:

1. Dictate and let CtrlSpeak inject a transcript from either backend.
2. Edit the target field.
3. Press bare Enter to send/confirm it normally.

At the observed Enter press, CtrlSpeak best-effort snapshots the active field with Ctrl+A/C before returning from its non-suppressing keyboard callback. It restores the prior text clipboard value and submits the captured text only when it differs from the pending injected result. The user does not need to select or copy anything manually. The original Enter is never suppressed, replayed, duplicated, or synthesized. Pending state is single-use, replaced by the next injection, and expires after ten minutes. A pending remote result remains bound to the original API URL and in-memory token even if saved settings change before confirmation.

This is deliberately best-effort: some elevated, remote, custom-rendered, protected, password, terminal, or multiline controls may reject Ctrl+A/C, may expose only part of their content, or may interpret Enter as a newline. Capture is skipped when CtrlSpeak detects a non-text-only clipboard because that data cannot be restored safely by its lightweight Win32 helper. Ctrl+A leaves the field selected, and a field containing unrelated surrounding text would be treated as the complete final transcript. Remote feedback sends the changed final text and client audit metadata to `POST /v1/transcriptions/{id}/feedback`; embedded feedback stays in the local correction index. Both paths store an exact raw-transcript-to-approved-text override only. Neither path infers broad phrase substitutions from arbitrary whole-transcript edits; any future phrase rule must be separately and explicitly user-approved and auditable. Set the method to `disabled` in the management window if this behavior is unsuitable for a particular workflow.

### Command-line Flags

- `--auto-setup {client,client_server}` – pre-select the startup mode without showing the GUI prompts.
- `--force-sendinput` – force the AnyDesk-compatible synthetic keystroke path.
- `--backend {bundled,api}` – persist the selected transcription backend.
- `--api-url <http(s)-url>` – persist the API base URL; configure tokens through settings or `CTRLSPEAK_API_TOKEN`.
- `--backend-status` – print redacted backend/auth/feedback status and exit.
- `--download-cuda-only` (alias: `--setup-cuda`) – stage the CUDA runtime, cuBLAS, and cuDNN support packages (reusing any cached wheels before downloading fresh copies) and exit; the command aborts immediately when no CUDA-capable GPU is detected.
- `--transcribe <wav>` – batch process an audio file without the hotkey workflow.
- `--uninstall` – remove the application data and executable (used by the packaged build).

Run `python main.py --help` for the full list.

## Packaging with PyInstaller

Use the helper module to build the v0.3 executable:

```powershell
python -m utils.build_exe
```

The helper executes `packaging/CtrlSpeak_v0.3.spec` and produces `dist/CtrlSpeak_v0.3.exe`. The one-file GUI build uses `console=False`, so it opens no console window; startup configuration errors are shown in a GUI dialog, while stdout-oriented flags are intended for `python main.py` from source. The build reuses `assets/icon.ico` and embeds the loading chime, onboarding video, fun-facts rotation list, and regression clip. Required runtime data for `faster_whisper`, `ctranslate2`, `ffpyplayer`, and the API HTTP client is collected automatically, while CUDA runtimes and Whisper model weights remain external downloads used only by bundled mode.

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

## Controlled LAN and public API hosting

Configuring a remote URL in CtrlSpeak does not expose, rebind, deploy, restart, or open firewall access to any service. The companion Whisper service remains loopback-only by default. If an operator later enables remote access:

1. Prefer a specific private/VPN interface over all-interface binding, and restrict reachability to intended clients with separately managed network policy.
2. Configure a strong bearer token outside source control and use the same token in CtrlSpeak’s per-user settings or `CTRLSPEAK_API_TOKEN`. Never put it in a command line, repository file, or shared log.
3. Use HTTPS for public, cloud, or untrusted-network traffic. Put the loopback service behind a TLS gateway or VPN; validate certificates normally and do not disable TLS verification.
4. Remember that a reverse proxy connects to the service from loopback. The proxy is therefore inside the service’s trusted boundary and must authenticate remote clients itself before forwarding requests.
5. Protect and retain correction databases according to the sensitivity of dictated and edited text. Back them up only to approved encrypted storage.

The legacy bundled CtrlSpeak `/transcribe` and UDP-discovery protocol is for controlled trusted networks and does not implement the new bearer-auth contract. Do not publish it on the Internet. If it is required on a LAN, an operator must create narrowly scoped private-network firewall rules after reviewing the host/network design; this project does not make those changes automatically.

For a Windows embedded host, copy `dist\CtrlSpeak_v0.3.exe`, run it once with `--auto-setup client_server` to stage the model, and verify the control center reports the local server as running. Any installation, firewall, service, TLS, or restart action remains a deliberate post-build operator step.

## Development Notes

- Temporary recordings, configuration, logs, and downloaded Whisper models live under `%APPDATA%\CtrlSpeak`.
- Test audio files such as `part1.wav` are intentionally excluded from Git to avoid large binaries.
- Use the tray menu to manage the client/server lifecycle or to uninstall (`Delete CtrlSpeak`).

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
