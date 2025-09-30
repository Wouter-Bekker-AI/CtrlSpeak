# CtrlSpeak Thread Management Report

## Report Purpose and Maintenance Notes
This report inventories every long-lived and transient thread that CtrlSpeak relies on, explaining how they interact, what
synchronization primitives guard shared state, and how shutdown sequences avoid race conditions. It is intended to guide future
audits of concurrency decisions and ensure new workers integrate cleanly with the Tkinter UI loop, audio stack, and background
services. To regenerate or update this report, inspect the thread creation sites in `utils/system.py`, `utils/gui.py`,
`utils/net_discovery.py`, `utils/models.py`, and `utils/bot_integration.py`, along with any background helpers under
`background_agents/`. Confirm the lifecycle of each `Thread`, `Event`, and lock, note new daemon workers or listener loops, and
verify how teardown joins or signals them before revising the sections below.

## Threading Strategy Overview
CtrlSpeak uses a mix of long-lived daemon threads, short-lived workers, and the main Tk event loop to compartmentalize I/O, audio, and UI responsibilities. Shared state is guarded by locks, events, and queues defined in `utils.system`, ensuring background activity never touches Tkinter or GUI widgets directly.【F:utils/system.py†L120-L182】【F:utils/system.py†L301-L343】【F:utils.gui.py†L1180-L1309】

## Main/UI Thread
The Tk root is created and managed on the main thread by `utils.gui`, which records the thread identity, exposes a task queue, and enforces that `pump_management_events_once` only runs on the main thread. Worker code schedules UI work through `enqueue_management_task`, avoiding direct Tk calls from other threads.【F:utils.gui.py†L1180-L1309】【F:utils.system.py†L301-L343】 The pystray loop runs on a separate daemon thread so the main thread can continue servicing Tk events and queued UI callbacks.【F:utils/system.py†L1188-L1233】

## Audio Capture and Feedback Threads
- **Recording thread:** When the right Ctrl key is pressed, `start_client_listener` spawns a `keyboard.Listener` and, on activation, launches a daemon thread executing `record_audio` to pull microphone frames until the key is released. The main thread waits for the recorder to join before dispatching transcription and cleanup, ensuring orderly teardown.【F:utils/system.py†L846-L887】
- **Processing loop:** During transcription, `start_processing_feedback` spins up a daemon thread that loops the processing chime, computes RMS levels, and updates shared waveform buffers for the overlay until `processing_sound_stop_event` fires.【F:utils/system.py†L345-L498】
- **Ready sound worker:** `play_model_ready_sound_once` guards a single-shot worker thread that plays a notification clip without blocking other audio operations.【F:utils/system.py†L500-L534】

## Hotkey Listener and Discovery Refresh
`start_client_listener` protects listener lifecycle with `listener_lock`, starting the pynput listener and queueing a daemon `_refresh_best_server_async` thread to update discovery state in the background. `stop_client_listener` shuts down the listener, joins any active recording thread, hides GUI overlays, and removes residual files, preventing resource leaks.【F:utils/system.py†L846-L887】

## Networking and Server Threads
- **Discovery listener:** `utils.net_discovery.DiscoveryListener` subclasses `threading.Thread` to receive UDP broadcasts, prune stale entries, and expose the latest server info. It runs as a daemon and can be stopped via an event flag and socket close.【F:utils/net_discovery.py†L25-L83】 `utils.system.start_discovery_listener` ensures only one instance runs at a time and cleans up via `stop_discovery_listener`.【F:utils/system.py†L1351-L1376】
- **HTTP server:** `start_server` initializes a `ThreadingHTTPServer`, then launches a daemon thread to call `serve_forever`. Companion daemon threads broadcast presence and listen for discovery queries, all controlled by `threading.Event` objects for cooperative shutdown.【F:utils/system.py†L1094-L1166】
- **Tray orchestration:** `run_tray` creates the pystray icon and starts it in a daemon thread, while the main thread continues pumping Tk. Shutdown paths stop the listener, server, and UI cleanly.【F:utils/system.py†L1188-L1233】

## Bot and Model Management Threads
`utils.models.initialize_transcriber` can spawn background work during Whisper initialization (audio feedback, GPU fallbacks) but returns control once the model is ready; it relies on the processing feedback thread for status cues.【F:utils/models.py†L2140-L2289】 When Ollama assets are required, `utils.bot_integration` launches a worker process plus a daemon `OllamaModelMonitor` thread to watch availability and update GUI lockout dialogs.【F:utils/bot_integration.py†L414-L434】 Active SocialRobot sessions are monitored by a daemon `_monitor_bot_exit` thread so the active identity resets immediately when the subprocess terminates.【F:utils/bot_integration.py†L643-L833】

## Synchronization and Shutdown
Global locks (`listener_lock`, `_processing_level_lock`, `_proc_vis_lock`) and events (`processing_sound_stop_event`, discovery stop events) coordinate shared state across threads.【F:utils/system.py†L120-L182】【F:utils/system.py†L345-L498】【F:utils/system.py†L1094-L1166】 Shutdown helpers invoke these signals, join threads with timeouts, and refresh the management UI so users see accurate status as services stop.【F:utils/system.py†L1094-L1389】
