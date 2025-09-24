# CtrlSpeak End-to-End User Flow

This document walks through the complete runtime experience for CtrlSpeak, covering what the end user sees and the internal operations triggered by each decision path. It is organized chronologically from the first launch on a fresh machine through advanced management actions and follow-up sessions.

## 1. First Launch: Bootstrap Sequence
When `main.py` is executed for the first time on a system, CtrlSpeak performs a deterministic initialization pipeline before the user can interact with the UI:

1. **Argument handling and maintenance hooks** – Command-line switches such as `--uninstall`, `--auto-setup`, and `--transcribe` are parsed first. These can short-circuit normal startup for maintenance, batch transcription, or automation flows.【F:main.py†L29-L64】
2. **Single-instance enforcement** – The process creates `%APPDATA%/CtrlSpeak/CtrlSpeak.lock` (on Windows) or the equivalent in the platform config directory to ensure no other CtrlSpeak instance is already running. If the lock cannot be acquired, the user is notified and startup terminates.【F:main.py†L41-L47】【F:utils/system.py†L860-L912】
3. **Settings load** – Persistent configuration is read (or created with defaults) in `utils.config_paths.load_settings`. This populates the in-memory `settings` map with keys such as `mode`, `device_preference`, and stored server endpoints.【F:main.py†L49-L51】【F:utils/config_paths.py†L30-L72】
4. **Optional automation profile** – If `--auto-setup` was supplied, the chosen profile is written to the settings before continuing, so subsequent logic knows which mode to assume without UI prompts.【F:main.py†L53-L55】【F:utils/system.py†L1115-L1121】
5. **Splash experience** – A splash window animates for `SPLASH_DURATION_MS` (default 1000 ms) to communicate application launch progress while background initialization continues.【F:main.py†L61-L63】【F:utils/gui.py†L492-L517】
6. **Management UI thread priming** – The hidden Tk root used by all later dialogs (mode picker, download overlays, management window) is created at this point so background threads can safely enqueue UI tasks once the app becomes interactive.【F:main.py†L65-L67】【F:utils/gui.py†L676-L764】

## 2. First-Run Mode Selection Experience
Immediately after the splash, CtrlSpeak determines whether the install already knows which operating mode to use.

1. **Mode lookup** – The cached `settings["mode"]` value is inspected. If it is already `"client"` or `"client_server"`, no prompt is shown. Otherwise, setup continues.【F:main.py†L69-L72】【F:utils/gui.py†L712-L738】
2. **Client-only builds** – If the binary was produced with the client-only flag, CtrlSpeak silently forces `mode="client"`, stores it, and proceeds without showing UI.【F:utils/gui.py†L740-L759】
3. **Mode selection dialog** – The user is shown a themed window with two cards: “Client + Server” (primary action) and “Client Only”. Choosing either persists the selection. Closing the dialog aborts setup and exits CtrlSpeak until the user reruns it.【F:utils/gui.py†L533-L704】
4. **Server prerequisites check** – If the user chose “Client + Server”, `_ensure_server_mode_model_ready` verifies that the currently selected Whisper model exists on disk, prompting the download workflow if necessary. Cancelling the download causes CtrlSpeak to exit to avoid running in an incomplete state.【F:utils/gui.py†L705-L769】
5. **Settings persistence** – The selected mode is written to `settings.json` (`%APPDATA%/CtrlSpeak/settings.json` on Windows). Future boots will start directly in this mode unless changed manually later.【F:utils/gui.py†L770-L784】【F:utils/config_paths.py†L58-L72】

## 3. Automatic Model Preparation on First Launch
With a mode chosen, CtrlSpeak ensures the default speech model (small) is installed without further input:

1. **Eligibility check** – Auto-install is skipped for client-only mode because remote servers will handle transcription. It is also skipped if the install was already completed before (e.g., subsequent runs).【F:main.py†L74-L84】【F:utils/models.py†L971-L1000】
2. **Model acquisition** – `_ensure_model_files` orchestrates download using `download_model_with_gui`, which shows a full-screen progress dialog covering progress, speed, ETA, and allows cancellation. Files are stored under `%APPDATA%/CtrlSpeak/models`.【F:utils/models.py†L805-L838】【F:utils/models.py†L368-L540】
3. **Completion feedback** – After download succeeds, `model_auto_install_complete` is flagged in settings and a one-time `loading.wav` chime (or fallback tone) is scheduled via `play_model_ready_sound_once`, signalling readiness to the user.【F:utils/models.py†L940-L1008】【F:utils/system.py†L362-L441】
4. **Cancellation handling** – If the user closes the download dialog or the files fail verification, CtrlSpeak notifies the user and exits so they can retry later; startup does not continue in a partially configured state.【F:utils/gui.py†L758-L769】【F:utils/models.py†L952-L1008】

## 4. Transition into Active Mode
After prerequisites are satisfied, CtrlSpeak activates the runtime tailored to the chosen mode.

1. **Mode fetch** – The final `settings["mode"]` value is read under lock for consistency across threads.【F:main.py†L86-L89】
2. **Server-mode warmup** – For `client_server`, `ensure_model_ready_for_local_server` double-checks the model on disk (or triggers a guarded download if it somehow went missing) and `initialize_transcriber(interactive=False)` preloads the Whisper model into memory so the first transcription has minimal latency.【F:main.py†L90-L98】【F:utils/models.py†L805-L838】【F:utils/models.py†L1009-L1180】
3. **Discovery listener** – Regardless of mode, a UDP discovery listener starts so other clients can find this device and so client-mode instances can auto-detect servers on the LAN. It runs on its own thread and refreshes management UI status when new servers appear.【F:main.py†L100-L103】【F:utils/system.py†L1089-L1104】
4. **Server activation (client+server mode)** – `start_server` launches an HTTP transcription endpoint, discovery broadcaster, and discovery query listener. A background thread prints the listening port and updates the “last connected server” state used for UI badges.【F:main.py†L92-L99】【F:utils/system.py†L780-L838】
5. **Client discovery delay (client-only mode)** – In purely client mode, CtrlSpeak sleeps for 1 second after starting discovery to give server discovery packets time to populate before the tray icon appears.【F:main.py†L99-L102】

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
4. **Inference path selection** – `transcribe_audio` delegates to `transcribe_local` in client+server mode or `transcribe_remote` in client mode. If no remote server responds, a fallback prompt offers to enable the local server, switching the application to `client_server` after successful setup.【F:utils/models.py†L1265-L1292】【F:utils/models.py†L1084-L1237】
5. **Result handling** – Successful transcription text is inserted into the currently focused window using Win32 SendInput APIs. Errors trigger GUI or toast notifications. Regardless of success, the processing loop stops, the overlay hides, and the temporary WAV file is deleted.【F:utils/system.py†L660-L690】【F:utils/models.py†L1224-L1292】

## 7. Client-Only Mode Runtime Specifics
When the user initially selects “Client Only”, or later switches to it:

1. **Server discovery** – The background discovery listener keeps searching for LAN servers and updates the management UI badges via `schedule_management_refresh`.【F:utils/system.py†L1089-L1104】【F:utils/system.py†L300-L317】
2. **Recording behavior** – Recordings are still captured locally, but `transcribe_remote` uploads the audio over HTTP to the best discovered server. Processing audio feedback plays locally while waiting for the HTTP response.【F:utils/system.py†L644-L690】【F:utils/models.py†L1230-L1292】
3. **Missing server flow** – If no server answers, `handle_missing_server` prompts the user via a modal dialog to install the local server. Accepting switches the persisted mode to `client_server`, forces a local model load, starts the server, and then transcribes the pending recording locally.【F:utils/models.py†L1200-L1237】
4. **Re-launch experience** – Subsequent launches skip the mode picker and jump straight into client-only behavior because the mode is cached in settings. The user can revisit the picker later through the management window’s “Change mode” control.【F:utils/gui.py†L712-L784】【F:utils/gui.py†L1668-L1707】

## 8. Client + Server Mode Runtime Specifics
When operating as both client and server:

1. **Local inference** – All hotkey recordings are processed by the locally loaded Whisper model. Since `initialize_transcriber` was already called during startup, most requests avoid the cost of reloading weights. If the GPU preference was active and CUDA is available, inference uses GPU acceleration; otherwise it gracefully falls back to CPU with a notification.【F:main.py†L92-L99】【F:utils/models.py†L1009-L1180】
2. **Network availability** – The HTTP server listens on the configured port (default 65432) for other clients. Discovery broadcasts advertise availability, and the management UI shows “Server · Online” badges when threads are healthy.【F:utils/system.py†L780-L838】【F:utils/gui.py†L1194-L1253】
3. **Remote clients** – External clients POST audio to `/transcribe`. The handler saves the payload to a temporary file, runs local transcription without playing audio feedback (because the originator already handles it), and returns JSON with the recognized text and elapsed processing time.【F:utils/system.py†L764-L838】

## 9. Management Window (Tray → “Manage CtrlSpeak”)
Right-clicking the tray icon and choosing “Manage CtrlSpeak” opens the full-featured dashboard where multiple subsystems can be controlled.

1. **Window layout** – The management UI presents status badges (mode, network), device preference controls, model selectors, input device settings, and action buttons for starting/stopping client services or changing modes.【F:utils/gui.py†L976-L1447】
2. **Live status refresh** – `refresh_status` populates the badges using the latest `settings`, CUDA availability, model inventory, and listener/server thread states. This method is called whenever the window opens and whenever background events schedule a refresh.【F:utils/gui.py†L1234-L1343】【F:utils/system.py†L300-L317】
3. **Start/Stop client** – Buttons invoke `start_client_listener` and `stop_client_listener`, enabling or disabling the Ctrl+R hotkey and cleaning up any in-progress recording safely (hiding overlays, deleting temp files).【F:utils/gui.py†L1562-L1589】【F:utils/system.py†L618-L752】
4. **Mode changes** – The “Change mode” button re-opens the first-run dialog even on later sessions. Switching to `client_server` triggers `_ensure_server_mode_model_ready`, starts the server threads, and updates tray labels; switching to `client` stops the server and refreshes discovery, then persists the new choice.【F:utils/gui.py†L1620-L1707】

## 10. Device Preference Workflow (CPU ↔ GPU)
Within the management window, the “Device preference” panel allows selecting CPU or GPU execution. The internal flow is:

1. **Preference storage** – Clicking “Apply device” stores `device_preference` in settings (values: `cpu`, `cuda`, or `auto`).【F:utils/gui.py†L1499-L1521】【F:utils/models.py†L61-L86】
2. **CUDA availability check** – If `cuda` was selected but `cuda_runtime_ready` reports missing dependencies, a confirmation dialog asks whether to install CUDA support now.【F:utils/gui.py†L1509-L1521】【F:utils/models.py†L220-L252】
3. **Runtime detection** – `cuda_runtime_ready` ensures environment variables point at `%APPDATA%/CtrlSpeak/cuda`, verifies GPU devices via ctranslate2, and, on Windows, attempts to load required DLLs (`cudnn_ops64_9.dll`, `cublas64_12.dll`).【F:utils/models.py†L199-L252】
4. **Installation flow** – Accepting the prompt runs `install_cuda_runtime_with_progress`, which shows a lockout window, installs NVIDIA wheels with pip, and validates availability. Failures keep the system on CPU with warnings; success closes the lockout window and leaves CUDA ready for use.【F:utils/models.py†L253-L344】
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
4. **Direct tray entry** – After splash, CtrlSpeak creates the tray icon, starts discovery/client listeners, and becomes immediately available with prior preferences intact. The user can still access the mode picker from the management window if they need to switch roles.【F:main.py†L61-L107】【F:utils/gui.py†L1620-L1707】

## 13. Shutdown and Cleanup
When the user quits from the tray menu or the process exits:

1. **Listeners stopped** – Recording threads halt, overlays hide, PyAudio instances terminate, and temporary files are deleted.【F:utils/system.py†L708-L752】
2. **Server teardown** – HTTP and discovery threads are joined, sockets close, and the “last connected server” marker clears so stale state does not linger in the UI.【F:utils/system.py†L838-L860】
3. **Management UI exit** – Tk’s mainloop exits, any open management windows close, and background refresh timers are cancelled.【F:utils/system.py†L924-L968】【F:utils/gui.py†L1380-L1447】
4. **Lock release** – `release_single_instance_lock` removes the lock file so another CtrlSpeak process can launch later.【F:main.py†L111-L116】【F:utils/system.py†L860-L912】

This detailed map should serve as a reference for engineers and designers to understand both the visible user journey and the supporting subsystems that make CtrlSpeak operate smoothly across its various modes and advanced management paths.
