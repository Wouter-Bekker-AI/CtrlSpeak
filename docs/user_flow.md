# CtrlSpeak End-to-End User Flow

This document walks through the complete runtime experience for CtrlSpeak, covering what the end user sees and the internal operations triggered by each decision path. It is organized chronologically from the first launch on a fresh machine through advanced management actions and follow-up sessions.

## 1. First Launch: Bootstrap Sequence
When `main.py` is executed for the first time on a system, CtrlSpeak performs a deterministic initialization pipeline before the user can interact with the UI:

1. **Argument handling and maintenance hooks** â€“ Command-line switches such as `--uninstall`, `--auto-setup`, `--download-cuda-only` (alias: `--setup-cuda`), and `--transcribe` are parsed first. These can short-circuit normal startup for maintenance, batch transcription, or automation flows.ã€F:main.pyâ€ L29-L64ã€‘
2. **Single-instance enforcement** â€“ The process creates `CtrlSpeak.lock` inside the platform data directory (`%APPDATA%/CtrlSpeak` on Windows or `${XDG_DATA_HOME:-~/.local/share}/CtrlSpeak` on Linux/macOS) to ensure no other CtrlSpeak instance is already running. If the lock cannot be acquired, the user is notified and startup terminates.ã€F:main.pyâ€ L41-L47ã€‘ã€F:utils/system.pyâ€ L860-L912ã€‘
3. **Settings load** â€“ Persistent configuration is read (or created with defaults) in `utils.config_paths.load_settings`. This populates the in-memory `settings` map with keys such as `mode`, `device_preference`, and stored server endpoints.ã€F:main.pyâ€ L49-L51ã€‘ã€F:utils/config_paths.pyâ€ L30-L72ã€‘
4. **Optional automation profile** â€“ If `--auto-setup` was supplied, the chosen profile is written to the settings before continuing, so subsequent logic knows which mode to assume without UI prompts.ã€F:main.pyâ€ L53-L55ã€‘ã€F:utils/system.pyâ€ L1115-L1121ã€‘
5. **Splash experience** â€“ A splash window animates for `SPLASH_DURATION_MS` (default 1000â€¯ms) to communicate application launch progress while background initialization continues.ã€F:main.pyâ€ L61-L63ã€‘ã€F:utils/gui.pyâ€ L492-L517ã€‘
6. **Management UI thread priming** â€“ The hidden Tk root used by all later dialogs (mode picker, download overlays, management window) is created at this point so background threads can safely enqueue UI tasks once the app becomes interactive.ã€F:main.pyâ€ L65-L67ã€‘ã€F:utils/gui.pyâ€ L676-L764ã€‘

## 2. First-Run Mode Selection Experience
Immediately after the splash, CtrlSpeak determines whether the install already knows which operating mode to use.

1. **Mode lookup** â€“ The cached `settings["mode"]` value is inspected. If it is already `"client"` or `"client_server"`, no prompt is shown. Otherwise, setup continues.ã€F:main.pyâ€ L69-L72ã€‘ã€F:utils/gui.pyâ€ L712-L738ã€‘
2. **Client-only builds** â€“ If the binary was produced with the client-only flag, CtrlSpeak silently forces `mode="client"`, stores it, and proceeds without showing UI.ã€F:utils/gui.pyâ€ L740-L759ã€‘
3. **Mode selection dialog** â€“ The user is shown a themed window with two cards: â€œClient + Serverâ€ (primary action) and â€œClient Onlyâ€. The client-only card exposes a manual `host[:port]` entry, discovery-backed server list, and **Refresh servers** action so the user can target a specific endpoint before committing to client mode.ã€F:utils/gui.pyâ€ L639-L859ã€‘ã€F:utils/gui.pyâ€ L912-L970ã€‘
4. **Server prerequisites check** â€“ If the user chose â€œClient + Serverâ€, `_ensure_server_mode_model_ready` verifies that the currently selected Whisper model exists on disk, prompting the download workflow if necessary. Cancelling the download causes CtrlSpeak to exit to avoid running in an incomplete state.ã€F:utils/gui.pyâ€ L1017-L1050ã€‘
5. **Settings persistence** â€“ The selected mode is written to `settings.json` inside the platform config directory (`%APPDATA%/CtrlSpeak/settings.json` on Windows or `${XDG_CONFIG_HOME:-~/.config}/CtrlSpeak/settings.json` on Linux/macOS). Future boots will start directly in this mode unless changed manually later.ã€F:utils/gui.pyâ€ L1051-L1111ã€‘ã€F:utils/config_paths.pyâ€ L58-L72ã€‘

## 3. Automatic Model Preparation on First Launch
With a mode chosen, CtrlSpeak ensures the default speech model (small) is installed without further input:

1. **Eligibility check** â€“ CtrlSpeak always ensures the default Whisper model is present on disk, regardless of mode. The auto-install step is skipped only after the initial download has completed successfully.ã€F:main.pyâ€ L74-L84ã€‘ã€F:utils/models.pyâ€ L971-L1000ã€‘
2. **Model acquisition** â€“ `_ensure_model_files` auto-approves the download and drives `download_model_with_gui`, which now launches a centered welcome window rendered at roughly 80% of a 1080p frame (about 1536Ã—864) that plays the bundled intro clip to completion (about five seconds for the default `assets/TrueAI_Intro_Video.mp4`) with audio before morphing into a fun-facts card. The refreshed card shows the CtrlSpeak logo on a white tile and cycles through onboarding tips from `assets/fun_facts.txt`. A slim lockout window docks in the top-left corner to stream status updates and expose the red **Cancel download** button. Cancelling stops the subprocess instantly, exiting if no model is staged or returning to the active model otherwise. Files are stored under the platform data directory (for example, `%APPDATA%/CtrlSpeak/models` on Windows). The same welcome flow now runs before Chat with Bot downloads an Ollama model for a persona, so the assistant waits behind the video-and-fun-facts screen until the LLM is cached.ã€F:utils/models.pyâ€ L720-L933ã€‘ã€F:utils/models.pyâ€ L968-L1484ã€‘ã€F:utils/bot_integration.pyâ€ L61-L213ã€‘
3. **Completion feedback** â€“ After download succeeds, `model_auto_install_complete` is flagged in settings and a one-time `loading.wav` chime (or fallback tone) is scheduled via `play_model_ready_sound_once`, signalling readiness to the user.ã€F:utils/models.pyâ€ L940-L1008ã€‘ã€F:utils/system.pyâ€ L362-L441ã€‘
4. **Cancellation handling** â€“ If the user closes the download dialog or the files fail verification, CtrlSpeak notifies the user and exits so they can retry later; startup does not continue in a partially configured state.ã€F:utils/gui.pyâ€ L758-L769ã€‘ã€F:utils/models.pyâ€ L952-L1008ã€‘

## 4. Transition into Active Mode
After prerequisites are satisfied, CtrlSpeak activates the runtime tailored to the chosen mode.

1. **Mode fetch** â€“ The final `settings["mode"]` value is read under lock for consistency across threads.ã€F:main.pyâ€ L86-L89ã€‘
2. **Server-mode warmup** â€“ For `client_server`, `ensure_model_ready_for_local_server` double-checks the model on disk (or triggers a guarded download if it somehow went missing) and `initialize_transcriber(interactive=False)` preloads the Whisper model into memory so the first transcription has minimal latency.ã€F:main.pyâ€ L90-L98ã€‘ã€F:utils/models.pyâ€ L805-L838ã€‘ã€F:utils/models.pyâ€ L1009-L1180ã€‘
3. **Discovery listener** â€“ Regardless of mode, a UDP discovery listener starts so other clients can find this device and so client-mode instances can auto-detect servers on the LAN. It runs on its own thread and refreshes management UI status when new servers appear.ã€F:main.pyâ€ L100-L103ã€‘ã€F:utils/system.pyâ€ L1089-L1104ã€‘
4. **Server activation (client+server mode)** â€“ `start_server` launches an HTTP transcription endpoint, discovery broadcaster, and discovery query listener. A background thread prints the listening port and updates the â€œlast connected serverâ€ state used for UI badges.ã€F:main.pyâ€ L92-L99ã€‘ã€F:utils/system.pyâ€ L780-L838ã€‘
5. **Client discovery delay (client-only mode)** â€“ In purely client mode, CtrlSpeak sleeps for 1â€¯second after starting discovery to give server discovery packets time to populate before the tray icon appears.ã€F:main.pyâ€ L99-L102ã€‘

## 5. System Tray Launch & Background Services
CtrlSpeak stays resident as a tray application once initialization is complete.

1. **Tray creation** â€“ `run_tray` starts the global keyboard listener, constructs a pystray icon labeled with the active mode, and spawns the tray loop on a daemon thread. The main thread then enters the Tk mainloop to process all GUI windows (waveform overlay, notifications, management window).ã€F:main.pyâ€ L104-L107ã€‘ã€F:utils/system.pyâ€ L912-L968ã€‘
2. **Tray menu** â€“ Right-clicking the tray icon surfaces two actions: â€œManage CtrlSpeakâ€ (opens the management dashboard) and â€œQuitâ€ (invokes a clean shutdown after stopping client listeners, server threads, and Tk).ã€F:utils/system.pyâ€ L924-L968ã€‘
3. **Shutdown guarantees** â€“ `atexit` handlers ensure the single-instance lock is released and all background threads (recording, discovery, server, management UI) are stopped even if the app exits unexpectedly.ã€F:main.pyâ€ L111-L116ã€‘ã€F:utils/system.pyâ€ L1105-L1121ã€‘

## 6. User Recording Flow (Ctrl+R)
Holding the right Control key drives the core speech-to-text workflow.

1. **Hotkey press** â€“ `keyboard.Listener` invokes `on_press`. When `ctrl_r` is detected and no recording is in progress, CtrlSpeak creates a unique WAV file in `${data_root}/temp`, starts a PyAudio capture thread, and displays an overlay with the live waveform sourced from the microphone.ã€F:utils/system.pyâ€ L600-L643ã€‘ã€F:utils/config_paths.pyâ€ L48-L64ã€‘
2. **Audio capture** â€“ The recording thread reads 16-bit mono frames at 44.1â€¯kHz until recording stops. Each chunk updates a waveform buffer so the overlay can animate in real time.ã€F:utils/system.pyâ€ L600-L620ã€‘ã€F:utils/system.pyâ€ L322-L364ã€‘
3. **Hotkey release** â€“ When `ctrl_r` is released, CtrlSpeak transitions the overlay to a â€œProcessingâ€¦â€ state, starts looping `loading.wav` via `start_processing_feedback`, and calls `transcribe_audio` with the recorded file.ã€F:utils/system.pyâ€ L644-L690ã€‘ã€F:utils/system.pyâ€ L362-L441ã€‘ã€F:utils/models.pyâ€ L1240-L1292ã€‘
4. **Inference path selection** â€“ `transcribe_audio` delegates to `transcribe_local` in client+server mode or `transcribe_remote` in client mode. If no remote server responds, a fallback prompt offers to enable the local server, switching the application to `client_server` after successful setup.ã€F:utils/models.pyâ€ L1265-L1292ã€‘ã€F:utils/models.pyâ€ L1084-L1237ã€‘
5. **Result handling** â€“ Successful transcription text is inserted into the currently focused window using Win32 SendInput APIs. Errors trigger GUI or toast notifications. Regardless of success, the processing loop stops, the overlay hides, and the temporary WAV file is deleted.ã€F:utils/system.pyâ€ L660-L690ã€‘ã€F:utils/models.pyâ€ L1224-L1292ã€‘

## 7. Client-Only Mode Runtime Specifics
When the user initially selects â€œClient Onlyâ€, or later switches to it:

1. **Server discovery** â€“ The background discovery listener keeps searching for LAN servers and updates the management UI badges via `schedule_management_refresh`.ã€F:utils/system.pyâ€ L1089-L1104ã€‘ã€F:utils/system.pyâ€ L300-L317ã€‘
2. **Preferred endpoint reuse** â€“ Manual entries from the mode picker are validated, stored as the preferred host/port, and reused the next time discovery comes up empty, ensuring the client can target a known server even without broadcasts.ã€F:utils/gui.pyâ€ L679-L756ã€‘ã€F:utils/net_discovery.pyâ€ L85-L123ã€‘
3. **Recording behavior** â€“ Recordings are still captured locally, but `transcribe_remote` uploads the audio over HTTP to the best discovered server. A guard (`_client_hotkey_available`) refuses to start recording when the mode is `client` and no server is connected, logging the block so operators understand why nothing happened. Processing audio feedback plays locally while waiting for the HTTP response.ã€F:utils/system.pyâ€ L606-L690ã€‘ã€F:utils/models.pyâ€ L1230-L1292ã€‘
4. **Missing server flow** â€“ If no server answers, `handle_missing_server` prompts the user via a modal dialog to install the local server. Accepting switches the persisted mode to `client_server`, forces a local model load, starts the server, and then transcribes the pending recording locally.ã€F:utils/models.pyâ€ L1200-L1237ã€‘
5. **Re-launch experience** â€“ Subsequent launches skip the mode picker and jump straight into client-only behavior because the mode is cached in settings. The user can revisit the picker later through the management windowâ€™s â€œChange modeâ€ control.ã€F:utils/gui.pyâ€ L1051-L1111ã€‘ã€F:utils/gui.pyâ€ L1532-L1543ã€‘

## 8. Client + Server Mode Runtime Specifics
When operating as both client and server:

1. **Local inference** â€“ All hotkey recordings are processed by the locally loaded Whisper model. Since `initialize_transcriber` was already called during startup, most requests avoid the cost of reloading weights. If the GPU preference was active and CUDA is available, inference uses GPU acceleration; otherwise it falls back to CPU and records a warning in the application log.ã€F:main.pyâ€ L92-L99ã€‘ã€F:utils/models.pyâ€ L1009-L1180ã€‘
2. **Network availability** â€“ The HTTP server listens on the configured port (65432 by default). If Windows refuses that port, CtrlSpeak automatically walks up the high-port range until it finds an open socket, updates the saved settings, and surfaces a notification so operators know the new address. Discovery broadcasts advertise availability, and the management UI shows â€œServer Â· Onlineâ€ badges when threads are healthy.ã€F:utils/system.pyâ€ L780-L838ã€‘ã€F:utils/gui.pyâ€ L1194-L1253ã€‘
3. **Remote clients** â€“ External clients POST audio to `/transcribe`. The handler saves the payload to a temporary file, runs local transcription without playing audio feedback (because the originator already handles it), and returns JSON with the recognized text and elapsed processing time.ã€F:utils/system.pyâ€ L764-L838ã€‘

## 9. Management Window (Tray â†’ â€œManage CtrlSpeakâ€)
Right-clicking the tray icon and choosing â€œManage CtrlSpeakâ€ opens the full-featured dashboard where multiple subsystems can be controlled.

1. **Window layout** - The management UI presents status badges (mode, network), an Assistants card with Chat with Bot controls and identity availability badges, device preference controls, model selectors, input device settings, and action buttons for starting/stopping client services or changing modes.【F:utils/gui.py†L1404-L1575】
2. **Live status refresh** â€“ `refresh_status` populates the badges using the latest `settings`, CUDA availability, model inventory, and listener/server thread states. This method is called whenever the window opens and whenever background events schedule a refresh.ã€F:utils/gui.pyâ€ L1234-L1343ã€‘ã€F:utils/system.pyâ€ L300-L317ã€‘
3. **Start/Stop client** â€“ Buttons invoke `start_client_listener` and `stop_client_listener`, enabling or disabling the Ctrl+R hotkey and cleaning up any in-progress recording safely (hiding overlays, deleting temp files).ã€F:utils/gui.pyâ€ L1562-L1589ã€‘ã€F:utils/system.pyâ€ L618-L752ã€‘
4. **Mode changes** â€“ The â€œChange modeâ€ button re-opens the first-run dialog even on later sessions. Switching to `client_server` triggers `_ensure_server_mode_model_ready`, starts the server threads, and updates tray labels; switching to `client` stops the server and refreshes discovery, then persists the new choice.ã€F:utils/gui.pyâ€ L1620-L1707ã€‘

## 10. Device Preference Workflow (CPU â†” GPU)
Within the management window, the â€œDevice preferenceâ€ panel allows selecting CPU or GPU execution. The internal flow is:

1. **Preference storage** - Clicking "Apply device" stores `device_preference` in settings (values: `cpu` or `cuda`).ã€F:utils/gui.pyâ€ L1494-L1529ã€‘ã€F:utils/models.pyâ€ L61-L86ã€‘
2. **CUDA availability check** â€“ If `cuda` was selected, the workflow first attempts to reuse any staged runtime files. When staging or validation fails CtrlSpeak now launches the welcome/progress download window automatically, fetching fresh CUDA runtimes before it decides whether to stay on CPU. Cached wheel downloads under `${data_root}/cuda/downloads` are validated against their published SHA-256 digest and reused on each retry so failed extraction attempts do not trigger another multi-gigabyte download.ã€F:utils/gui.pyâ€ L1498-L1533ã€‘ã€F:utils/models.pyâ€ L636-L924ã€‘
3. **Runtime detection** â€“ `cuda_runtime_ready` ensures environment variables point at `${data_root}/cuda/12.3`, verifies GPU devices via ctranslate2, and, on Windows, attempts to load the required DLLs (`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`, `cudnn_ops64_*.dll`). These checks are skipped until the user explicitly requests CUDA.ã€F:utils/models.pyâ€ L344-L392ã€‘
4. **Installation flow** - The Install/repair control (or `--download-cuda-only` alias) still runs `install_cuda_runtime_with_progress`, presenting the lockout window and re-validating availability. The downloader now stages the runtime into a temporary directory, verifies the DLLs can be loaded before swapping them into place, and only deletes the cached wheels after the validation step succeeds. When no CUDA-capable GPU is detected the button remains disabled and the CLI flag exits immediately, so CtrlSpeak stays on CPU with a clear warning.ã€F:utils/models.pyâ€ L825-L924ã€‘ã€F:main.pyâ€ L49-L86ã€‘
5. **Model reload** â€“ Regardless of the branch, `_reload_transcriber_async` unloads the current Whisper model (if any), reloads it with the new device preference using `initialize_transcriber`, and updates UI status text. This ensures inference uses GPU or CPU accordingly on the next request.ã€F:utils/gui.pyâ€ L1345-L1447ã€‘ã€F:utils/gui.pyâ€ L1509-L1561ã€‘ã€F:utils/models.pyâ€ L1182-L1237ã€‘
6. **GPU â†’ CPU switch** â€“ Selecting CPU triggers the same reload path but forces `_force_cpu_env` during initialization to hide CUDA devices, guaranteeing inference runs on the CPU even if CUDA libraries remain installed.ã€F:utils/models.pyâ€ L1084-L1162ã€‘

## 11. Model Selection Workflow
Users can switch between the small and large Whisper models through the management window.

1. **Dropdown selection** â€“ Choosing a different model updates `model_var` but does not immediately change the active model.ã€F:utils/gui.pyâ€ L1431-L1489ã€‘
2. **Apply model** â€“ Clicking â€œApply modelâ€ writes the selection to settings, ensures the files exist (prompting a GUI download if missing), and then reloads the transcriber asynchronously. Success produces a confirmation dialog.ã€F:utils/gui.pyâ€ L1523-L1561ã€‘ã€F:utils/models.pyâ€ L805-L838ã€‘
3. **Download only** â€“ â€œDownload modelâ€ lets users fetch a model without activating it. Completion messages update the badges to show availability. This is useful for preloading the large model before switching devices or modes.ã€F:utils/gui.pyâ€ L1563-L1589ã€‘
4. **Large model activation** â€“ If the large model is chosen but not yet installed, the download dialog guides the user through the process, and a cancellation leaves the previous model active. Once installed, the reload flow handles the increased memory footprint and plays the ready sound as appropriate.ã€F:utils/models.pyâ€ L368-L540ã€‘ã€F:utils/gui.pyâ€ L1523-L1561ã€‘

## 12. Subsequent Launch Behavior
On every run after the first successful setup:

1. **Mode shortcut** â€“ `ensure_mode_selected` reads the persisted mode and returns immediately without showing the selection dialog, shaving several seconds off startup. Only if the settings file is missing or corrupt will the prompt reappear.ã€F:utils/gui.pyâ€ L712-L738ã€‘
2. **Auto-install bypass** â€“ Because `model_auto_install_complete` is set after the first download, `ensure_initial_model_installation` no longer launches the downloader; it simply confirms the flag and continues.ã€F:utils/models.pyâ€ L971-L1008ã€‘
3. **Warm start** â€“ If the previous session already loaded the model (e.g., the process restarted quickly), `initialize_transcriber` reuses the in-memory model unless a device or model change forced an unload. The ready sound is played only once per session to avoid repetitive cues.ã€F:utils/models.pyâ€ L1009-L1180ã€‘ã€F:utils/system.pyâ€ L362-L441ã€‘
4. **Direct tray entry** â€“ After splash, CtrlSpeak creates the tray icon, starts discovery/client listeners, and becomes immediately available with prior preferences intact. The user can still access the mode picker from the management window if they need to switch roles.ã€F:main.pyâ€ L61-L107ã€‘ã€F:utils/gui.pyâ€ L1620-L1707ã€‘

### Chat with Bot
After the tray icon appears you can launch SocialRobot from the management window or by saying “chat with <identity>” (for example, “chat with assistant” or “chat with Einstein”). The assistant, Einstein, and default personas now block on background memory ingestion before the window becomes interactive: `refresh_document_memory` hashes `README.md` plus every Markdown file under `docs/`, while `refresh_datetime_memory` captures the current local date, timezone, and locale hints. Each helper refreshes the identity’s vector store only when its content changed or the 24-hour cooldown elapsed, and logs terminal-only status so operators know when the preflight finished. Manual “update documentation” or “update datetime” requests from the hotkey flow or inside the chat window force the corresponding helper to run immediately for the active identity. Once a bot is active it now opens in text mode: the chat window anchors itself to the bottom-right corner (the traditional floating-logo spot), displays conversation history, accepts keyboard input, and keeps the floating logo hidden while VAD and TTS stay off. The window and microphone toggle reuse `assets/icon.ico` so the experience stays branded whether the session is text or voice driven. Click the microphone button to switch to voice mode; the window collapses to a read-only transcript, slides to the top-right corner, the logo reappears, VAD starts listening, and replies are played aloud. Click the button again to return to text-only interaction and move the window back to the bottom-right corner. Voice commands like “goodbye <identity>” still stop the current conversation instantly because the transcription server intercepts the utterance and tells CtrlSpeak to shut down the session before SocialRobot continues, but the chat window now stays open so the user can close it manually. Typing the same “goodbye <identity>” keyword in the text box leaves the session running—the phrase is logged as normal chat instead of triggering shutdown. Subsequent “chat with …” phrases relaunch the bot with the requested identity. The push-to-talk workflow (hold right Ctrl, speak, release) recognises the same keywords: if no bot is running the assistant launches immediately instead of pasting the phrase, switching identities shuts down the previous session before starting the new one, and saying “goodbye <identity>” exits using the same teardown path as the tray menu. Keyword detection is intentionally fuzzy, so light punctuation (“goodbye, assistant”) and typical STT slips (“chat was assistant”) still trigger the automation unless documentation calls for strict matching. Each identity (`third_party/social_robot/identities/<name>`) bundles its system prompt, default voice, LLM settings, and future memory storage. Refer to docs/bot_integration.md for prerequisites, environment overrides, and automation tips.


## 13. Shutdown and Cleanup
When the user quits from the tray menu or the process exits:

1. **Listeners stopped** â€“ Recording threads halt, overlays hide, PyAudio instances terminate, and temporary files are deleted.ã€F:utils/system.pyâ€ L708-L752ã€‘
2. **Server teardown** â€“ HTTP and discovery threads are joined, sockets close, and the â€œlast connected serverâ€ marker clears so stale state does not linger in the UI.ã€F:utils/system.pyâ€ L838-L860ã€‘
3. **Management UI exit** â€“ Tkâ€™s mainloop exits, any open management windows close, and background refresh timers are cancelled.ã€F:utils/system.pyâ€ L924-L968ã€‘ã€F:utils/gui.pyâ€ L1380-L1447ã€‘
4. **Lock release** â€“ `release_single_instance_lock` removes the lock file so another CtrlSpeak process can launch later.ã€F:main.pyâ€ L111-L116ã€‘ã€F:utils/system.pyâ€ L860-L912ã€‘

This detailed map should serve as a reference for engineers and designers to understand both the visible user journey and the supporting subsystems that make CtrlSpeak operate smoothly across its various modes and advanced management paths.






