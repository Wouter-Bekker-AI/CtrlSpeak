# CtrlSpeak Ground Rules (Authoritative Reference)

## Scope and immutability
- This document captures the non-negotiable architectural, operational, and process rules for CtrlSpeak. Treat it as binding policy—future changes to these requirements must be explicitly approved by maintainers, and implementations must continue to honor every rule recorded here.
- The ground rules themselves cannot be relaxed or ignored. When you extend the project, redesign within these boundaries rather than weakening them.

## Mandatory orientation for AI agents
- On your first pass through this repository, read **every** Markdown file so you understand the existing expectations before editing code. That includes `README.md`, everything under `docs/` (notably [`docs/tooling.md`](tooling.md) for shared tooling guidance), and the testing notes under `tests/` (especially `tests/TESTING.md`).
- When you make code changes that affect or invalidate existing documentation, update the impacted Markdown files in the same pull request so the written guidance never diverges from reality.

## Repository mutability lifecycle
- During development (source checkout inside Git), the repository can be freely read and written so contributors and AI agents can iterate. Use version control to manage those edits.
- Once CtrlSpeak is packaged into an executable, treat the installed directory as **read-only** at runtime. Assets resolve relative to the application base via `get_app_base_dir`, so the running program must never create or modify files beside the executable bundle.【F:utils/config_paths.py†L106-L120】

## Runtime storage boundaries
- All persistent runtime data lives under the CtrlSpeak data directory: `%APPDATA%\CtrlSpeak` on Windows or `${XDG_DATA_HOME:-~/.local/share}/CtrlSpeak` on Linux/macOS. `get_data_dir` eagerly creates the required subdirectories (`models/`, `cuda/`, `bot_memory/`, `logs/`, and `temp/`)—place every generated file under this tree.【F:utils/config_paths.py†L33-L87】
- User settings are stored exclusively in the CtrlSpeak config directory (`%APPDATA%\CtrlSpeak` on Windows or `${XDG_CONFIG_HOME:-~/.config}/CtrlSpeak` on Linux/macOS) as `settings.json`. Defaults guarantee CPU transcription with the `small` Whisper checkpoint until the user opts into different hardware or model sizes.【F:utils/config_paths.py†L19-L109】
- Every identity owns `${data_root}/bot_memory/<identity>/` with `conversation/`, `screenshots/`, `chroma/`, `traces/`, and `.locks/` enforced by `utils.memory_paths`. Conversation logs are append-only JSONL rotated at 10 MB (keep 5), screenshots respect the per-identity `store_screenshots` toggle, and the Chroma store caps out at 5 000 embeddings with optional TTL and PII redaction controlled via `${config_root}/identities/<identity>/memory.json`.【F:utils/memory_paths.py†L10-L56】【F:utils/memory_settings.py†L15-L62】【F:utils/vector_memory.py†L1-L232】
- Per-identity locks at `${data_root}/.locks/<identity>.lock` are mandatory before launching SocialRobot or touching vector memory. If the lock is held, CtrlSpeak must emit “Identity in use. Close the running session before starting another.” and abort to prevent corruption.【F:utils/memory_lock.py†L16-L73】【F:third_party/social_robot/main.py†L47-L88】【F:third_party/social_robot/main.py†L120-L162】

## Logging requirements
- Every module must obtain loggers through `utils.config_paths.get_logger`. This seeds a rotating file handler that writes to `${data_root}/logs/ctrlspeak.log` (for example, `%APPDATA%\CtrlSpeak\logs\ctrlspeak.log` on Windows) and wires global logging so warnings and exceptions are persisted. Never replace the logger wiring or redirect logs elsewhere.【F:utils/config_paths.py†L127-L168】
- When errors occur (I/O, GUI, CUDA, networking, etc.), catch the exception and log via the project logger so the failure is captured in AppData. Existing code uses `logger.exception(...)` as the pattern—follow it for new code paths.【F:utils/config_paths.py†L73-L81】【F:utils/system.py†L781-L788】【F:utils/models.py†L62-L83】
- Do not silence exceptions with `pass` inside `except` blocks. Always emit a log record (for example, `logger.exception(...)` or `logger.debug(..., exc_info=True)` for expected cases) so the failure is traceable in `${data_root}/logs/ctrlspeak.log` (for example, `%APPDATA%\CtrlSpeak\logs\ctrlspeak.log` on Windows).

## Conversation keyword ownership
- CtrlSpeak’s main process **must** remain the sole owner of conversation control keywords (for example, “chat with …” and “goodbye …”). Never relocate these detectors into SocialRobot or any worker thread. The parent process is the only component permitted to start or stop the bot so shutdown always follows the hardened tray/button workflow and avoids hangs.
- When you introduce new conversation triggers, extend `utils.system.handle_transcription_keyword` and `utils.system.handle_transcribed_text_from_hotkey` so the CtrlSpeak main loop processes them before any transcript is forwarded to SocialRobot. If the bot needs to react, have the parent send an explicit stdin control command (see `utils.bot_integration.request_goodbye`) rather than letting the child interpret raw speech.
- Document any new keywords alongside these helpers to keep future contributors from bypassing the main-thread enforcement. Regressions that reintroduce keyword handling inside the child process are blocked because they revive the historical bug where “goodbye” phrases left SocialRobot running.
- Keyword detectors must remain **fuzzy**. Match punctuation-insensitive and near-miss transcriptions (for example, “goodbye, assistant” or “chat was assistant”) unless a specification explicitly demands exact phrasing. Hard matching regressions risk stranding conversations when STT introduces minor substitutions.

## TTS preprocessing guardrails
- The background cleaning agent exists solely to delete `*` and `#` characters from LLM replies before TTS playback. Do not expand its responsibilities to rephrase, summarise, or otherwise alter the wording.
- Only invoke the agent when the reply actually contains those characters. Clean passages must go straight to Kokoro without touching the helper.
- Keep the prompt assets and Ollama options (Gemma 3 1B at temperature 0) locked down so the model echoes the input verbatim aside from the removed characters. Never let the helper answer questions, change pronouns, or inject extra narration.

## Tkinter and UI threading
- The hidden Tk root and all GUI windows are created on the **main thread** by `_initialize_management_ui_on_main_thread`. Do not create additional Tk roots or run `mainloop` outside the main thread.【F:utils/gui.py†L807-L870】
- Background threads may request UI work only through the management queue helpers (for example `_call_on_management_ui`). Direct Tk calls from worker threads are forbidden, and `pump_management_events_once` enforces that by raising if called off-thread.【F:utils/gui.py†L780-L899】
- Calling Tk APIs (even lightweight queries such as `winfo_exists`) from a worker thread will raise `RuntimeError: main thread is not in main loop` and can wedge shutdown paths. Always route those checks through the shared helpers in `utils.gui` so the main thread performs the Tk work.

## Mode selection and preference persistence
- `DEFAULT_SETTINGS` codifies first-run behavior: mode unset, device preference `cpu`, and Whisper model `small`. Keep these defaults intact to ensure safe startup on machines without GPU support.【F:utils/config_paths.py†L19-L28】
- The LangGraph orchestrator is guarded by the `use_langgraph_memory_orchestrator` flag, defaulting to `false` for staged rollout. When enabled, every turn must run through the `retrieve → plan → call_tools → llm → persist` graph, queue persistence asynchronously, and record traces/metrics under the identity’s `traces/` directory.【F:utils/config_paths.py†L19-L80】【F:utils/memory_orchestrator.py†L1-L285】
- `ensure_mode_selected` persists the user’s choice between **Client + Server** and **Client Only** modes, prompting only on first run. Client-only builds force the `client` mode once, while combined builds insist the local model download completes before proceeding. Do not bypass this workflow or write configuration elsewhere.【F:utils/gui.py†L712-L778】
- Device and model selections persist through `set_device_preference` and `set_current_model_name`, ensuring later launches remember CPU/GPU usage and the active Whisper checkpoint. Always update preferences through these setters so `settings.json` stays authoritative.【F:utils/models.py†L96-L146】
- GPU execution remains optional. `resolve_device` (via `cuda_runtime_ready`) automatically falls back to CPU when CUDA runtime files are missing or the user selected CPU. Never assume CUDA availability without consulting these helpers.【F:utils/models.py†L999-L1124】

## Model and CUDA assets
- Whisper weights are always staged under `${data_root}/models` (for example, `%APPDATA%\CtrlSpeak\models` on Windows). The helpers `model_store_path_for` and `model_files_present` encode the layout—do not introduce alternate caches or download directories.【F:utils/models.py†L148-L181】
- CUDA support searches `${data_root}/cuda/12.3` (plus bundled/wheel fallbacks) when staging DLLs, and `configure_cuda_paths` extends `PATH` accordingly. Keep GPU runtime files confined to this tree and never install them beside the executable.【F:utils/models.py†L340-L370】
- Model downloads and CUDA installers must emit human-readable traces under `${data_root}/logs` (e.g., `%APPDATA%\CtrlSpeak\logs` on Windows), preserving visibility into staged work.【F:utils/models.py†L62-L83】【F:utils/models.py†L190-L200】

## Client/server responsibilities
- In **Client Only** mode the app records audio locally but relies on a LAN server discovered by the listener threads. Keep the hotkey listener and discovery refresh logic running so remote inference stays responsive; do not disable them in client mode.【F:utils/system.py†L685-L753】
- The embedded HTTP server handles `/transcribe` uploads by writing them to a temp file under AppData, performing local transcription, and deleting the temporary audio before responding. Preserve this contract when making server-side changes.【F:utils/system.py†L754-L789】

## Text injection strategy
- Text insertion follows a strict priority: SendInput for consoles, AnyDesk, and other remote-hosted windows; direct message-based insertion for standard local controls; clipboard or simulated typing as fallbacks. Preserve this order so remote desktops and terminals continue working.【F:utils/winio.py†L385-L439】
- Remote-control contexts rely on the SendInput (“force auto input”) path. Do not revert to Win32 clipboard-only approaches for these windows because they break AnyDesk and console compatibility.【F:utils/winio.py†L413-L432】

## Temporary recordings and cleanup
- Microphone captures are staged in `${data_root}/temp` via `create_recording_file_path`, and `cleanup_recording_file` removes them when finished. Always use these helpers so no recordings are left beside the executable.【F:utils/config_paths.py†L65-L81】【F:utils/system.py†L650-L752】

## Testing obligations before submitting changes
- Run the compile smoke test to ensure every module still compiles to bytecode: `python -m compileall .`.
- Execute the core headless pytest suite on every change: `python -m pytest -m core_headless`. This fast suite guards configuration helpers, CLI parsing, and discovery utilities. The run must succeed; if it fails, stop, investigate, and explain the failure in your status update before proceeding.【F:tests/TESTING.md†L1-L34】
- Regularly run the full GUI/integration suite (`CTRLSPEAK_RUN_FULL_TESTS=1 python -m pytest -m full_gui`) and the combined run (`CTRLSPEAK_RUN_FULL_TESTS=1 python -m pytest`) to catch regressions that span the entire pipeline.【F:tests/TESTING.md†L18-L52】

## Documentation synchronization
- When code changes alter behavior, configuration, or user interaction, update the relevant Markdown documentation (`README.md`, files in `docs/`, `tests/TESTING.md`, etc.) in the same change set.
- Maintain the running backlog in [`docs/TODO.md`](TODO.md) so pending enhancements stay discoverable and current.
- This ground-rules document sets the guardrails those updates must respect; do not rewrite history to relax these requirements.

Adhering to this reference protects packaging constraints, logging visibility, client/server interoperability, UI stability, and operational readiness. Any change that conflicts with these principles must be redesigned until it complies.
