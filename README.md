# CtrlSpeak

CtrlSpeak is a Windows speech-to-text assistant that records speech while the right `Ctrl` key is held down and injects the transcription into the active text control. It can run as a self-contained **Client + Server** bundle with Whisper hosted locally or as a lightweight **Client Only** build that discovers a LAN server.

Both flavours support Windows 10/11, enforce a single running instance, expose a tray UI for mode switching, and include AnyDesk-aware text injection with optional audio cues.

## Repository Layout

- `main.py` – application entry point.
- `background_agents/` – orchestrator-managed services that run alongside Chat with Bot. These agents refresh ingested documentation, stage current date/time snapshots, filter `<think>` plans, and perform other maintenance tasks automatically whenever the LangGraph workflow requires them.
- `assets/` – static resources such as the tray icon (`icon.ico`), the welcome video (`TrueAI_Intro_Video.mp4`), the fun-fact rotation list (`fun_facts.txt`), and the processing chime (`loading.wav`).
- `utils/` – implementation modules (GUI, models, networking, configuration helpers, etc.).
- `utils/build_exe.py` – helper script that runs PyInstaller with the correct data files.
- `tools/` – reusable tooling (currently the shared `vision.py` capture helpers, keyword registry, and `message_management.py` scrubbers that force assistant replies into safe plaintext); see [`docs/tooling.md`](docs/tooling.md) for the authoritative API reference.
- `packaging/` – PyInstaller spec (`CtrlSpeak.spec`) and additional build documentation.

Generated folders such as `dist/` and `build/` are ignored via `.gitignore`.

## Platform storage locations

CtrlSpeak never writes to the project directory at runtime. Instead it resolves platform-aware roots for persistent artifacts:

| Location    | Windows path                | Linux/macOS path                                   |
| ----------- | --------------------------- | -------------------------------------------------- |
| `data_root` | `%APPDATA%\\CtrlSpeak`      | `${XDG_DATA_HOME:-~/.local/share}/CtrlSpeak`       |
| `config_root` | `%APPDATA%\\CtrlSpeak`   | `${XDG_CONFIG_HOME:-~/.config}/CtrlSpeak`          |
| `logs_root` | `%APPDATA%\\CtrlSpeak\\logs` | `${XDG_DATA_HOME:-~/.local/share}/CtrlSpeak/logs` |

Models, CUDA runtimes, automation artifacts, screenshots, vector stores, and temporary recordings live under `data_root`, while user settings (`settings.json`) remain in `config_root`. Loggers write to `logs_root/ctrlspeak.log` via a rotating handler.

## Environment Setup

Create an isolated environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Note: The optional "Chat with Bot" feature using the logo animation style requires PySide6, which is included in requirements.txt.
```

GPU acceleration requires an NVIDIA CUDA-capable GPU with compatible drivers, but CtrlSpeak always boots in CPU mode and skips CUDA validation unless you opt in. The Whisper `small` model is downloaded automatically on first launch so a fresh install is usable immediately. Selecting **GPU (CUDA)** in the management window now launches the same welcome-and-progress experience used for model downloads; the app installs the CUDA runtime, cuBLAS, and cuDNN automatically and only falls back to CPU if validation fails. The CUDA wheels are cached under `${data_root}/cuda/downloads` (that is, `%APPDATA%\CtrlSpeak\cuda\downloads` on Windows or `${XDG_DATA_HOME:-~/.local/share}/CtrlSpeak/cuda/downloads` on Linux/macOS), verified with the published SHA-256 digests, and reused on the next attempt so extraction failures no longer force a redownload; the cache is purged only after a validated install. You can also stage GPU support manually via `python main.py --download-cuda-only` (alias: `--setup-cuda`). When no CUDA-capable GPU is detected, the management UI hides the GPU option and the installer flag exits early with an explanatory message.
During the initial Whisper download, CtrlSpeak opens a centered welcome window sized to roughly 80% of a 1080p frame (about 1536×864) that plays the bundled intro clip (about five seconds for the default `TrueAI_Intro_Video.mp4`) with audio. Once the clip ends, the window transitions into a branded fun-facts card featuring the CtrlSpeak logo on a white tile and rotating onboarding tips sourced from `assets/fun_facts.txt`. A slim lockout window remains in the top-left corner with live status text and a red **Cancel download** button; cancelling stops the download subprocess immediately, exiting entirely if no model is available or otherwise returning you to the currently staged model.
The same welcome-and-progress experience now runs whenever Chat with Bot needs to download an Ollama model for a selected identity, keeping the assistant, Einstein, and default personas in the intro/fun-facts flow until the LLM weights finish caching.


## Running from Source

```powershell
python main.py
```

On first launch you will be prompted to choose between **Client + Server** or **Client Only** modes. The client-only card lets you refresh for LAN servers or manually enter a `host[:port]` so CtrlSpeak knows which remote host to target. Settings live under `config_root`, while models, CUDA runtimes, logs, and other runtime files live under `data_root` (see [Platform storage locations](#platform-storage-locations) for the exact paths per operating system).

### Command-line Flags

- `--auto-setup {client,client_server}` – pre-select the startup mode without showing the GUI prompts.
- `--force-sendinput` – force the AnyDesk-compatible synthetic keystroke path.
- `--download-cuda-only` (alias: `--setup-cuda`) – stage the CUDA runtime, cuBLAS, and cuDNN support packages (reusing any cached wheels before downloading fresh copies) and exit; the command aborts immediately when no CUDA-capable GPU is detected.
- `--transcribe <wav>` – batch process an audio file without the hotkey workflow.
- `--uninstall` – remove the application data and executable (used by the packaged build).
- `--health` (with optional `--health-identity <name>`) – run the AppData health probe that verifies writability, lock acquisition, Chroma initialization, and embedder metadata for the selected identity; exits non-zero when any check fails.

Run `python main.py --help` for the full list.

### Chat with Bot speech pipeline

When you launch **Chat with Bot**, SocialRobot inspects each assistant reply and only invokes `tools.message_management.force_plaintext` when markdown bullets, control characters, or similar formatting artefacts are present. The deterministic scrub runs before persistence and Kokoro playback so the spoken reply, chat history, and vector store all receive the safe version; otherwise the untouched reply flows straight through.

Kokoro includes GPU acceleration because `requirements.txt` ships with `onnxruntime-gpu==1.23.0`, and the bundled identities now request the CUDA provider by default. CtrlSpeak automatically falls back to CPU when no GPU backend is available and prints a warning so operators know Kokoro could not stay on the GPU. You can override the provider or device by adding a `tts` block (or exporting `BOT_TTS_PROVIDER`/`BOT_TTS_DEVICE_ID`).

Chat with Bot now launches in a **text-first** workflow. SocialRobot opens a chat window that shows the live conversation log, provides a text input field, and exposes a microphone toggle. In this mode the floating logo stays hidden, CtrlSpeak does not start the VAD listener, and responses remain on screen without TTS playback. The window anchors itself to the bottom-right corner (matching the floating logo placement) and reuses `assets/icon.ico` for both the window chrome and microphone toggle so the entry point stays branded. Clicking the mic button switches to voice mode: the chat window shrinks to a read-only transcript, slides to the top-right corner, the logo reappears, the VAD listener starts, and replies are spoken aloud just like earlier releases. Clicking the button again returns to text chat, moves the window back to the bottom-right corner, and tears down the voice pipeline. Spoken “goodbye <identity>” requests (including ones routed through the stdin control channel) still stop the conversation by disabling voice mode and unwinding the audio stack, but the chat window now stays open so the user can close it manually. Typing the same phrase in text mode no longer triggers shutdown—it is delivered to the bot as a normal message.

Voice keywords live in [`tools/keywords.py`](docs/tooling.md). Saying “look at my screen” or “look at my clipboard” captures an image via the shared vision tooling, “chat with <identity>” (for example, assistant, Einstein, or default) relaunches the bot with that persona unless it is already active, and both “chat with …” and “goodbye …” now trigger the CtrlSpeak transcription server so the parent process owns the stop/start cycle before the utterance reaches SocialRobot. The detector treats these phrases fuzzily—light punctuation or common misrecognitions such as “chat was assistant” and “goodbye, assistant” still trip the handlers—unless a document states otherwise. See the **Keyword reference** table in [`docs/tooling.md`](docs/tooling.md#keyword-reference) for an explicit list of every supported phrase and the action each one performs (the clipboard command also accepts close variants such as “look at my slipboard”).

Holding the right `Ctrl` hotkey also honours those keywords. When no bot is running, saying “chat with assistant” (or any configured identity) while using the push-to-talk workflow launches the requested bot instead of typing the phrase into the focused window. Saying “goodbye <identity>” through the same hotkey closes the active session using the same shutdown path as the tray menu, and switching identities issues a graceful stop before starting the new persona.

### Bot memory storage

CtrlSpeak persists each identity’s runtime state under `${data_root}/bot_memory/<identity>/`, creating canonical `conversation/`, `screenshots/`, `chroma/`, and `traces/` subdirectories on demand. Conversation history is written as append-only JSONL at `conversation/conversation.jsonl`; entries are flushed via atomic temp-file swaps and the active file rotates at 10 MB (keeping five archives) to prevent unbounded growth. Vision captures now maintain a **single** PNG per identity (`screenshots/current.png`) plus a `metadata.json` descriptor in the same folder. When a new screenshot or clipboard image arrives, it atomically replaces the previous file so there is never more than one image stored per bot.

Vector memory lives in `chroma/`, which is backed by a single-writer Chroma collection tied to the per-identity lock. The collection stores embedder metadata (`ctrlspeak-minhash` by default), rejects concurrent writers, and automatically spawns a `_vN` suffix when the embedder name or version changes so upgrades never mix embeddings. Retention is enforced at 5 000 items per identity with LRU eviction, optional per-document TTL, and an opt-in PII redaction pass before embedding. All toggles are stored in per-identity configuration files under `${config_root}/identities/<identity>/memory.json`:

| Setting | Default | Description |
| --- | --- | --- |
| `store_vector_memory` | `true` | Enables Chroma persistence and retrieval. |
| `store_screenshots` | `true` | Controls whether captured images are saved to disk. |
| `retrieval_top_k` | `5` | Number of memories fetched per turn when similarity exceeds the threshold. |
| `retrieval_threshold` | `0.75` | Minimum cosine similarity required to inject retrieved context. |
| `max_vector_items` | `5000` | Upper bound for stored embeddings before LRU eviction. |
| `vector_ttl_days` | `null` | Optional time-to-live per embedding (in days). |
| `pii_redaction` | `false` | Redacts light PII (emails, phone numbers, IDs) before embedding. |

Each Chat with Bot turn flows through a LangGraph orchestrator (`use_langgraph_memory_orchestrator` setting) that sequences retrieval → planning → tool execution → LLM → persistence. Retrieval no-ops when the store is empty or below the similarity threshold, and asynchronous embedding/upsert keeps TTS playback responsive. Per-turn traces and a CSV metrics feed (`retrieval_hits`, `avg_similarity`, `persist_latency_ms`, `evictions`, `lock_wait_ms`) accumulate under `traces/` for observability.

To avoid corruption, CtrlSpeak acquires `${data_root}/.locks/<identity>.lock` before launching SocialRobot. If another process already owns the identity, the launcher prints “Identity in use. Close the running session before starting another.” and aborts. The management window’s **Clear Bot Memory** action targets the AppData-backed directories, deleting the JSONL log (and rotated archives) plus the `screenshots/`, `chroma/`, and `traces/` folders so packaged builds stay read-only.

## Packaging with PyInstaller

Use the helper module to build a distributable executable under `dist/CtrlSpeak/`:

```powershell
python -m utils.build_exe
```

The helper executes the maintained `packaging/CtrlSpeak.spec` so manual `pyinstaller` runs stay aligned. The resulting one-file GUI build embeds the tray icon, loading chime, onboarding video, fun-facts rotation list, and regression test clip inside the internal `assets/` directory. Required runtime data for `faster_whisper`, `ctranslate2`, and `ffpyplayer` is collected automatically, while CUDA runtimes and Whisper model weights remain external downloads performed at runtime when the user opts in.

## Manual Model Download

CtrlSpeak caches Whisper model weights under `${data_root}/models`. The default configuration selects the lightweight `small` Whisper checkpoint and runs on the CPU. If you want to preload the model without launching the GUI, use the Hugging Face CLI:

```powershell
pip install huggingface_hub
$target = Join-Path $env:APPDATA 'CtrlSpeak\models\small'
huggingface-cli download Systran/faster-whisper-small --local-dir $target --local-dir-use-symlinks False
New-Item -ItemType File (Join-Path $target '.installed') -Force | Out-Null
```

- Substitute a different `repo/model` name if you prefer another Whisper checkpoint.
- To point CtrlSpeak at a custom directory, set the `CTRLSPEAK_MODEL_DIR` environment variable to the parent folder that contains the models (defaults to `${data_root}/models`).

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
3. Allow the discovery and API ports through Windows Firewall (adjust the profile if you need different scopes). CtrlSpeak starts on TCP **65432** by default, but if Windows blocks that port the app automatically selects the next available port and updates the saved settings—mirror the new port in your firewall rules when that happens:
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
   - Serving: `<server-IP>:<port>` (65432 by default; the value reflects any automatic fallback)

After updates you can re-run `--auto-setup client_server` to refresh the installation silently.

## Development Notes

- Temporary recordings and other runtime artifacts live under `data_root`, while configuration files stay under `config_root` (see [Platform storage locations](#platform-storage-locations)).
- Test audio files such as `part1.wav` are intentionally excluded from Git to avoid large binaries.
- Use the tray menu to manage the client/server lifecycle or to uninstall (`Delete CtrlSpeak`).
- Track future enhancements in [`docs/TODO.md`](docs/TODO.md); keep the list current as tasks are added or completed.

## Automation Flow

Run the regression harness to validate a workstation without touching the GUI:

```powershell
python main.py --automation-flow
```

The command performs a staged health-check entirely inside `data_root`:

1. Ensure the default Whisper model is present under `${data_root}/models` (downloading it when missing).
2. Reuse or install the NVIDIA CUDA runtime stack (nvidia-cuda-runtime-cu12, nvidia-cublas-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12) so the DLLs live under `${data_root}/cuda/12.3` when GPU testing is required.
3. Transcribe `assets/test.wav` on the CPU.
4. Transcribe the same clip on the GPU using the DLLs staged in `${data_root}/cuda/12.3`.
5. Simulate each text-injection strategy (direct insert, SendInput paste, clipboard paste, PyAutoGUI typing) and write a consolidated report to `${data_root}/automation/artifacts`.

If any stage fails the workflow stops at that checkpoint and leaves detailed logs plus the partially populated automation_state.json in the same automation folder. Fix the underlying system issue (drivers, CUDA DLLs, networking, etc.) and re-run the flag - the script resumes where it left off.

### Handing the checklist to another operator or AI agent

Provide your helper with the single command above and the acceptance criteria:

- All stages complete without errors on a single pass.
- `${data_root}/automation/artifacts` contains a report named `automation_run_*.txt` whose injection sections echo the canonical transcript.
- `${data_root}/cuda/12.3` holds the CUDA DLLs and `python main.py` can select both CPU and GPU devices without warnings.

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
