# CtrlSpeak End-to-End User Flow

This document walks through the complete runtime experience for CtrlSpeak, covering what the end user sees and the internal operations triggered by each decision path. It is organized chronologically from the first launch on a fresh machine through advanced management actions and follow-up sessions.

## 1. First Launch: Bootstrap Sequence
When `main.py` is executed for the first time on a system, CtrlSpeak performs a deterministic initialization pipeline before the user can interact with the UI:

1. **Argument handling and maintenance hooks** – Command-line switches such as `--backend`, `--api-url`, `--backend-status`, `--uninstall`, `--auto-setup`, `--download-cuda-only` (alias: `--setup-cuda`), and `--transcribe` are parsed first. These can persist a backend choice or short-circuit normal startup for maintenance, status, batch transcription, or automation flows.
2. **Single-instance enforcement** – The process creates `%APPDATA%/CtrlSpeak/CtrlSpeak.lock` (on Windows) or the equivalent in the platform config directory to ensure no other CtrlSpeak instance is already running. If the lock cannot be acquired, the user is notified and startup terminates.【F:main.py†L41-L47】【F:utils/system.py†L860-L912】
3. **Settings load** – Persistent configuration is read (or created with defaults) in `utils.config_paths.load_settings`. This populates the in-memory `settings` map with the legacy mode, device/model choices, transcription backend, full API base URL, optional token, and automatic active-field feedback method. API environment variables take precedence at runtime; token values are never printed in status. The effective backend configuration is pinned for the process lifetime, so management-window changes explicitly apply after restart.
4. **Optional automation profile** – If `--auto-setup` was supplied, the chosen profile is written to the settings before continuing, so subsequent logic knows which mode to assume without UI prompts.【F:main.py†L53-L55】【F:utils/system.py†L1115-L1121】
5. **Splash experience** – A splash window animates for `SPLASH_DURATION_MS` (default 1000 ms) to communicate application launch progress while background initialization continues.【F:main.py†L61-L63】【F:utils/gui.py†L492-L517】
6. **Management UI thread priming** – The hidden Tk root used by all later dialogs (mode picker, download overlays, management window) is created at this point so background threads can safely enqueue UI tasks once the app becomes interactive.【F:main.py†L65-L67】【F:utils/gui.py†L676-L764】

## 2. First-Run Mode Selection Experience
Immediately after the splash, CtrlSpeak determines whether the install already knows which legacy operating mode to use. This section applies only to **Embedded / local**; **Remote API** skips legacy mode selection and local-model setup.

1. **Mode lookup** – The cached `settings["mode"]` value is inspected. If it is already `"client"` or `"client_server"`, no prompt is shown. Otherwise, setup continues.【F:main.py†L69-L72】【F:utils/gui.py†L712-L738】
2. **Client-only builds** – If the binary was produced with the client-only flag, CtrlSpeak silently forces `mode="client"`, stores it, and proceeds without showing UI.【F:utils/gui.py†L740-L759】
3. **Mode selection dialog** – The user is shown a themed window with two cards: “Client + Server” (primary action) and “Client Only”. The client-only card exposes a manual `host[:port]` entry, discovery-backed server list, and **Refresh servers** action so the user can target a specific endpoint before committing to client mode.【F:utils/gui.py†L639-L859】【F:utils/gui.py†L912-L970】
4. **Server prerequisites check** – If the user chose “Client + Server”, `_ensure_server_mode_model_ready` verifies that the currently selected Whisper model exists on disk, prompting the download workflow if necessary. Cancelling the download causes CtrlSpeak to exit to avoid running in an incomplete state.【F:utils/gui.py†L1017-L1050】
5. **Settings persistence** – The selected mode is written to `settings.json` (`%APPDATA%/CtrlSpeak/settings.json` on Windows). Future boots will start directly in this mode unless changed manually later.【F:utils/gui.py†L1051-L1111】【F:utils/config_paths.py†L58-L72】

## 3. Automatic Model Preparation on First Launch
With a mode chosen, CtrlSpeak ensures the default speech model (small) is installed without further input:

1. **Eligibility check** – In the embedded/local backend, CtrlSpeak ensures the default Whisper model is present on disk. Remote API startup skips model download, CUDA/model warm-up, discovery, and the embedded server.
2. **Model acquisition** – `_ensure_model_files` auto-approves the download and drives `download_model_with_gui`, which now launches a centered welcome window rendered at roughly 80% of a 1080p frame (about 1536×864) that plays the bundled intro clip to completion (about five seconds for the default `assets/TrueAI_Intro_Video.mp4`) with audio before morphing into a fun-facts card. The refreshed card shows the CtrlSpeak logo on a white tile and cycles through onboarding tips from `assets/fun_facts.txt`. A slim lockout window docks in the top-left corner to stream status updates and expose the red **Cancel download** button. Cancelling stops the subprocess instantly, exiting if no model is staged or returning to the active model otherwise. Files are stored under `%APPDATA%/CtrlSpeak/models`.【F:utils/models.py†L720-L933】【F:utils/models.py†L968-L1145】
3. **Completion feedback** – After download succeeds, `model_auto_install_complete` is flagged in settings and a one-time `loading.wav` chime (or fallback tone) is scheduled via `play_model_ready_sound_once`, signalling readiness to the user.【F:utils/models.py†L940-L1008】【F:utils/system.py†L362-L441】
4. **Cancellation handling** – If the user closes the download dialog or the files fail verification, CtrlSpeak notifies the user and exits so they can retry later; startup does not continue in a partially configured state.【F:utils/gui.py†L758-L769】【F:utils/models.py†L952-L1008】

## 4. Transition into Active Mode
After prerequisites are satisfied, CtrlSpeak activates the runtime tailored to the chosen mode.

1. **Backend fetch** – The effective backend is resolved after settings and environment overrides. The choice is explicit and no failure path silently changes it.
2. **Embedded server-mode warmup** – For embedded/local plus `client_server`, `ensure_model_ready_for_local_server` double-checks the model and `initialize_transcriber(interactive=False)` preloads it.
3. **Embedded discovery** – Only the embedded/local legacy path starts UDP discovery. Client-mode instances can auto-detect legacy CtrlSpeak servers; Remote API uses only its configured HTTP/HTTPS URL.
4. **Embedded server activation** – Embedded/local plus `client_server` launches the legacy HTTP transcription endpoint and discovery threads. Remote API never starts this server.
5. **Remote API isolation** – API mode starts the hotkey/tray flow without downloading a bundled model, probing LAN discovery, or falling back locally.

## 5. System Tray Launch & Background Services
CtrlSpeak stays resident as a tray application once initialization is complete.

1. **Tray creation** – `run_tray` starts the global keyboard listener, constructs a pystray icon labeled with the active mode, and spawns the tray loop on a daemon thread. The main thread then enters the Tk mainloop to process all GUI windows (waveform overlay, notifications, management window).【F:main.py†L104-L107】【F:utils/system.py†L912-L968】
2. **Tray menu** – Right-clicking the tray icon surfaces two actions: “Manage CtrlSpeak” (opens the management dashboard) and “Quit” (invokes a clean shutdown after stopping client listeners, server threads, and Tk).【F:utils/system.py†L924-L968】
3. **Shutdown guarantees** – `atexit` handlers ensure the single-instance lock is released and all background threads (recording, discovery, server, management UI) are stopped even if the app exits unexpectedly.【F:main.py†L111-L116】【F:utils/system.py†L1105-L1121】

## 6. User Recording Flow (Ctrl+R)
Holding the right Control key drives the core speech-to-text workflow.

1. **Hotkey press** – `keyboard.Listener` invokes `on_press`. When `ctrl_r` is detected and no recording is in progress, CtrlSpeak creates a unique WAV file in `%APPDATA%/CtrlSpeak/temp`, starts a PyAudio capture thread, and displays an overlay with the live waveform sourced from the microphone.【F:utils/system.py†L600-L643】【F:utils/config_paths.py†L48-L64】
2. **Audio capture** – The recording thread reads 16-bit mono frames at 44.1 kHz until recording stops. Each chunk updates a waveform buffer so the overlay can animate in real time.【F:utils/system.py†L600-L620】【F:utils/system.py†L322-L364】
3. **Hotkey release** – When `ctrl_r` is released, CtrlSpeak transitions the overlay to a “Processing…” state, starts looping `loading.wav` via `start_processing_feedback`, and calls `transcribe_audio` with the recorded file.【F:utils/system.py†L644-L690】【F:utils/system.py†L362-L441】【F:utils/models.py†L1240-L1292】
4. **Inference path selection** – `transcribe_audio_result` calls only the selected backend. Remote API uploads multipart audio to `<base-url>/v1/transcribe` and treats connection/HTTP/schema errors as actionable failures. Embedded/local preserves its direct-model and legacy client/server behavior.
5. **Correction application and audit** – API results retain the service `id`, raw/corrected text, and response metadata. Embedded results are recorded in the per-user exact-only correction index and reuse a prior approval only for a byte-for-byte raw match.
6. **Result handling** – Successful text is injected using the established SendInput/direct-control/fallback priority. The injected result and immutable feedback destination become pending only when edit capture is explicitly enabled. Errors notify the user; processing feedback stops and the temporary WAV is deleted in all cases.
7. **Automatic confirmation** – When the user presses bare Enter, CtrlSpeak synchronously makes a best-effort Ctrl+A/C snapshot of the current active field, restores the prior text clipboard, then lets the original Enter continue exactly once. A changed, unexpired pending result is submitted in the background to its original API endpoint or local correction index. Custom/protected controls and non-text-only clipboards can prevent capture, and CtrlSpeak never suppresses or replays Enter.

## 7. Embedded/local Client-Only Runtime Specifics
When the embedded/local user initially selects “Client Only”, or later switches to it:

1. **Server discovery** – The background discovery listener keeps searching for LAN servers and updates the management UI badges via `schedule_management_refresh`.【F:utils/system.py†L1089-L1104】【F:utils/system.py†L300-L317】
2. **Preferred endpoint reuse** – Manual entries from the mode picker are validated, stored as the preferred host/port, and reused the next time discovery comes up empty, ensuring the client can target a known server even without broadcasts.【F:utils/gui.py†L679-L756】【F:utils/net_discovery.py†L85-L123】
3. **Recording behavior** – Recordings are still captured locally, but `transcribe_remote` uploads the audio over HTTP to the best discovered server. A guard (`_client_hotkey_available`) refuses to start recording when the mode is `client` and no server is connected, logging the block so operators understand why nothing happened. Processing audio feedback plays locally while waiting for the HTTP response.【F:utils/system.py†L606-L690】【F:utils/models.py†L1230-L1292】
4. **Missing server flow** – If no server answers, `handle_missing_server` prompts the user via a modal dialog to install the local server. Accepting switches the persisted mode to `client_server`, forces a local model load, starts the server, and then transcribes the pending recording locally.【F:utils/models.py†L1200-L1237】
5. **Re-launch experience** – Subsequent launches skip the mode picker and jump straight into client-only behavior because the mode is cached in settings. The user can revisit the picker later through the management window’s “Change mode” control.【F:utils/gui.py†L1051-L1111】【F:utils/gui.py†L1532-L1543】

## 8. Client + Server Mode Runtime Specifics
When operating as both client and server:

1. **Local inference** – All hotkey recordings are processed by the locally loaded Whisper model. Since `initialize_transcriber` was already called during startup, most requests avoid the cost of reloading weights. If the GPU preference was active and CUDA is available, inference uses GPU acceleration; otherwise it falls back to CPU and records a warning in the application log.【F:main.py†L92-L99】【F:utils/models.py†L1009-L1180】
2. **Network availability** – The HTTP server listens on the configured port (default 65432) for other clients. Discovery broadcasts advertise availability, and the management UI shows “Server · Online” badges when threads are healthy.【F:utils/system.py†L780-L838】【F:utils/gui.py†L1194-L1253】
3. **Remote clients** – External clients POST audio to `/transcribe`. The handler saves the payload to a temporary file, runs local transcription without playing audio feedback (because the originator already handles it), and returns JSON with the recognized text and elapsed processing time.【F:utils/system.py†L764-L838】

## 9. Management Window (Tray → “Manage CtrlSpeak”)
Right-clicking the tray icon and choosing “Manage CtrlSpeak” opens the full-featured dashboard where multiple subsystems can be controlled.

1. **Window layout** – The management UI presents explicit **Embedded / local** and **Remote API** choices, a full API base URL, masked optional token, automatic-on-Enter feedback control (which can be disabled), redacted backend status, legacy mode/network badges, device/model controls, input-device settings, and lifecycle actions.
2. **Live status refresh** – `refresh_status` populates the badges using the latest `settings`, CUDA availability, model inventory, and listener/server thread states. This method is called whenever the window opens and whenever background events schedule a refresh.【F:utils/gui.py†L1234-L1343】【F:utils/system.py†L300-L317】
3. **Start/Stop client** – Buttons invoke `start_client_listener` and `stop_client_listener`, enabling or disabling the Ctrl+R hotkey and cleaning up any in-progress recording safely (hiding overlays, deleting temp files).【F:utils/gui.py†L1562-L1589】【F:utils/system.py†L618-L752】
4. **Backend and mode changes** – Saving a backend validates and persists its complete HTTP/HTTPS URL, optional token, and feedback setting, then reports that CtrlSpeak must be restarted. The current process remains pinned to its startup backend so server/discovery/model state cannot become half-switched. The “Change mode” action remains available for the embedded legacy roles.

## 10. Device Preference Workflow (CPU ↔ GPU)
Within the management window, the “Device preference” panel allows selecting CPU or GPU execution. The internal flow is:

1. **Preference storage** - Clicking "Apply device" stores `device_preference` in settings (values: `cpu` or `cuda`).【F:utils/gui.py†L1494-L1529】【F:utils/models.py†L61-L86】
2. **CUDA availability check** – If `cuda` was selected, the workflow first attempts to reuse any staged runtime files. When staging or validation fails CtrlSpeak now launches the welcome/progress download window automatically, fetching fresh CUDA runtimes before it decides whether to stay on CPU. Cached wheel downloads under `%APPDATA%\CtrlSpeak\cuda\downloads` are validated against their published SHA-256 digest and reused on each retry so failed extraction attempts do not trigger another multi-gigabyte download.【F:utils/gui.py†L1498-L1533】【F:utils/models.py†L636-L924】
3. **Runtime detection** – `cuda_runtime_ready` ensures environment variables point at `%APPDATA%/CtrlSpeak/cuda/12.3`, verifies GPU devices via ctranslate2, and, on Windows, attempts to load the required DLLs (`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`, `cudnn_ops64_*.dll`). These checks are skipped until the user explicitly requests CUDA.【F:utils/models.py†L344-L392】
4. **Installation flow** - The Install/repair control (or `--download-cuda-only` alias) still runs `install_cuda_runtime_with_progress`, presenting the lockout window and re-validating availability. The downloader now stages the runtime into a temporary directory, verifies the DLLs can be loaded before swapping them into place, and only deletes the cached wheels after the validation step succeeds. When no CUDA-capable GPU is detected the button remains disabled and the CLI flag exits immediately, so CtrlSpeak stays on CPU with a clear warning.【F:utils/models.py†L825-L924】【F:main.py†L49-L86】
5. **Model reload** – Regardless of the branch, `_reload_transcriber_async` unloads the current Whisper model (if any), reloads it with the new device preference using `initialize_transcriber`, and updates UI status text. This ensures inference uses GPU or CPU accordingly on the next request.【F:utils/gui.py†L1345-L1447】【F:utils/gui.py†L1509-L1561】【F:utils/models.py†L1182-L1237】
6. **GPU → CPU switch** – Selecting CPU triggers the same reload path but forces `_force_cpu_env` during initialization to hide CUDA devices, guaranteeing inference runs on the CPU even if CUDA libraries remain installed.【F:utils/models.py†L1084-L1162】

## 11. Model Selection Workflow
Users can switch between the small and large Whisper models through the management window.

1. **Dropdown selection** – Choosing a different model updates `model_var` but does not immediately change the active model.【F:utils/gui.py†L1431-L1489】
2. **Apply model** – Clicking “Apply model” writes the selection to settings, ensures the files exist (prompting a GUI download if missing), and then reloads the transcriber asynchronously. Success produces a confirmation dialog.【F:utils/gui.py†L1523-L1561】【F:utils/models.py†L805-L838】
3. **Download only** – “Download model” lets users fetch a model without activating it. Completion messages update the badges to show availability. This is useful for preloading the large model before switching devices or modes.【F:utils/gui.py†L1563-L1589】
4. **Large model activation** – If the large model is chosen but not yet installed, the download dialog guides the user through the process, and a cancellation leaves the previous model active. Once installed, the reload flow handles the increased memory footprint and plays the ready sound as appropriate.【F:utils/models.py†L368-L540】【F:utils/gui.py†L1523-L1561】

## 12. Subsequent Launch Behavior
On every run after the first successful setup:

1. **Mode shortcut** – `ensure_mode_selected` reads the persisted mode and returns immediately without showing the selection dialog, shaving several seconds off startup. Only if the settings file is missing or corrupt will the prompt reappear.【F:utils/gui.py†L712-L738】
2. **Auto-install bypass** – Because `model_auto_install_complete` is set after the first download, `ensure_initial_model_installation` no longer launches the downloader; it simply confirms the flag and continues.【F:utils/models.py†L971-L1008】
3. **Warm start** – If the previous session already loaded the model (e.g., the process restarted quickly), `initialize_transcriber` reuses the in-memory model unless a device or model change forced an unload. The ready sound is played only once per session to avoid repetitive cues.【F:utils/models.py†L1009-L1180】【F:utils/system.py†L362-L441】
4. **Direct tray entry** – After splash, CtrlSpeak creates the tray/hotkey flow with prior preferences intact. Discovery/server/model work starts only for embedded/local; Remote API goes directly to its configured endpoint workflow.

## 13. Shutdown and Cleanup
When the user quits from the tray menu or the process exits:

1. **Listeners stopped** – Recording threads halt, overlays hide, PyAudio instances terminate, and temporary files are deleted.【F:utils/system.py†L708-L752】
2. **Server teardown** – HTTP and discovery threads are joined, sockets close, and the “last connected server” marker clears so stale state does not linger in the UI.【F:utils/system.py†L838-L860】
3. **Management UI exit** – Tk’s mainloop exits, any open management windows close, and background refresh timers are cancelled.【F:utils/system.py†L924-L968】【F:utils/gui.py†L1380-L1447】
4. **Lock release** – `release_single_instance_lock` removes the lock file so another CtrlSpeak process can launch later.【F:main.py†L111-L116】【F:utils/system.py†L860-L912】

This detailed map should serve as a reference for engineers and designers to understand both the visible user journey and the supporting subsystems that make CtrlSpeak operate smoothly across its various modes and advanced management paths.
