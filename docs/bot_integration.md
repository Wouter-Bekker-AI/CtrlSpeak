# Chat With Bot Integration

CtrlSpeak includes an optional "Chat with Bot" experience accessible from the management window. When enabled, it pairs the local transcription server with the vendored **SocialRobot** project to provide a voice conversation loop:

- **Speech to Text (STT)** - Uses CtrlSpeak's `/transcribe` endpoint. Audio captured by the VAD listener is sent to the running CtrlSpeak server (local or remote depending on mode). The server returns the recognized text.
- **Language Model (LLM)** - The recognized text is sent to the Ollama-compatible client inside SocialRobot. By default CtrlSpeak ships with a lightweight fallback response if no LLM endpoint is reachable, but you can supply your own by setting the `BOT_LLM_URL` and `BOT_LLM_MODEL` environment variables (or the matching CLI flags) before launching CtrlSpeak.
- **Text cleanup (background agent)** - Identities that opt into cleaning route the LLM reply through `background_agents/tts_preprocessing_agent`, which consults its `identity.json` to determine whether to prepend `header_text.txt`, load `system_prompt.txt`, or use both before sending the request to the Gemma 3 1B preprocessing helper. The agent only runs when the reply contains `*` or `#` characters and simply removes them—no paraphrasing or pronoun changes—before speech is generated.
- **Text to Speech (TTS)** - The LLM response is converted to audio via Kokoro-ONNX. CtrlSpeak defaults to the formal male `am_michael` voice; override it with `BOT_VOICE` or the `--voice` flag.
- **Animated Face / Logo** - SocialRobot renders the default TrueAI transparent logo with amplitude-based scaling for visual feedback. Identity folders can still supply alternate assets under `third_party/social_robot/identities/<name>` when a different look is desired.

## Identity Profiles

Personalities for the bot live under `third_party/social_robot/identities/<name>`. Each identity directory may contain:

- `identity.json` - configuration for the profile (voice, model, prompt settings, optional memory directory).
- `system_prompt.txt` (or another file referenced by the config) - the text injected as the LLM system prompt when the profile is loaded.
- Runtime memory lives under `${data_root}/bot_memory/<identity>/`, which CtrlSpeak creates automatically with `conversation/`, `screenshots/`, `chroma/`, and `traces/` subfolders. Packaged builds ignore any legacy `memory/` folders inside the repository tree so all writes land in AppData.
- Identity-specific defaults live under `${config_root}/identities/<identity>/memory.json`. CtrlSpeak seeds these files with `store_vector_memory: true`, `store_screenshots: true`, `retrieval_top_k: 5`, `retrieval_threshold: 0.75`, `max_vector_items: 5000`, `vector_ttl_days: null`, and `pii_redaction: false` so personas can independently tune retention and privacy.

The loader understands the following `identity.json` keys:

```json
{
  "name": "default",
  "description": "Friendly companion",
  "read_prompt_from_file": true,
  "prompt_file": "system_prompt.txt",
  "llm_model": "gemma3:1b",
  "llm_url": "http://localhost:11434/api/chat",
  "ollama_hardware": "cpu_and_gpu",
  "ollama_options": {
    "temperature": 0.5,
    "num_ctx": 8192
  },
  "voice": "am_michael",
  "require_text_cleaning": false,
  "vision": true,
  "tool": false,
  "memory_dir": "memory"
}
```

If `read_prompt_from_file` is `true`, the prompt file is read relative to the identity directory (unless an absolute path is provided). Omitting it falls back to the built-in default prompt.

When present, `ollama_options` is merged into the payload that SocialRobot sends to Ollama so you can tune generation parameters per identity. The optional `ollama_hardware` key controls how Ollama stages the weights:

- `cpu_only` – force the model to reside entirely in system RAM (`num_gpu` is set to zero).
- `cpu_and_gpu` – allow Ollama to combine VRAM and RAM (the default when unspecified).
- `gpu_only` – require the checkpoint to stay fully on the GPU (`gpu_only: true`).

CtrlSpeak applies the same options and hardware preference when it pre-warms the checkpoint via `/generate`, ensuring the residency chosen during warm-up matches the settings SocialRobot will use at runtime.

> **Note**
> The `memory_dir` field remains in legacy identity configs for compatibility, but CtrlSpeak always resolves runtime storage through the AppData helpers described above. Repository-relative memory paths are ignored so packaged builds stay read-only.

> **Tip**
> Update `${config_root}/identities/<identity>/memory.json` when you need to disable screenshot storage, change retrieval thresholds, adjust the vector-cap limit, apply a TTL, or enable the PII redactor for a specific persona.

The additional boolean keys control multimodal, cleaning, and future extensibility features:

- `require_text_cleaning` – When `true`, SocialRobot loads the TTS preprocessing agent and only sends replies that contain `*` or `#` characters through the cleanup pass before speech. When `false`, replies flow directly to Kokoro and CtrlSpeak skips staging the helper model.
- `vision` – Enables image capture tooling documented in [`docs/tooling.md`](tooling.md). When `true`, SocialRobot listens for the spoken “look at my screen” and “look at my clipboard” commands, exposes matching context-menu actions on the floating logo, and routes captured images to the LLM. When `false`, the commands are ignored, the context-menu items are hidden, and no images are taken.
- `tool` – Reserved flag for forthcoming external tool integrations. It defaults to `false` today but can be toggled once tool calling is implemented.

CtrlSpeak ships with two bundled identities: `assistant` (vision enabled, text cleaning enabled) and `default` (vision disabled, text cleaning disabled). Both currently set `tool` to `false` and can be expanded as the tool feature matures.

All face, mouth, and logo assets now live inside the identity directories; the legacy `third_party/social_robot/images/` placeholders have been removed so new personas should bundle their own art alongside `identity.json`. Likewise, shared prompt templates are deprecated—store any reusable system prompts with the identity that consumes them so packaging stays self-contained.

You can switch identities from the command line with:

```powershell
python third_party/social_robot/main.py --identity ross
```

or by setting `BOT_IDENTITY=ross` before launching CtrlSpeak so the management window uses that persona.

## Voice keywords

SocialRobot loads [`tools/keywords.py`](tooling.md) at startup and registers one keyword set per identity directory. The following phrases are recognized out of the box:

- **look at my screen** – Captures a screenshot via `tools/vision.py` when the active identity has `vision: true`.
- **look at my clipboard** – Pulls the most recent image from the system clipboard and forwards it like a screenshot.
- **chat with `<identity>`** – Immediately relaunches SocialRobot with the target persona through the transcription server so the parent process coordinates the shutdown and restart (requests that target the already-active identity are ignored).
- **goodbye `<identity>`** – Immediately ends the current conversation and shuts down the SocialRobot process from the CtrlSpeak transcription server so the parent process controls the teardown.

The push-to-talk workflow (hold the right Ctrl key while speaking) shares the same keyword registry. When VAD is idle because no bot session is active, saying “chat with assistant” through the hotkey launches that identity and skips text injection entirely. If another persona is already running, the helper first asks it to exit cooperatively via `utils.bot_integration.request_goodbye()` and falls back to `stop_bot()` only when the child process fails to exit within the timeout. Spoken “goodbye <identity>” commands are now intercepted in the CtrlSpeak main process (the transcription server), which issues the same graceful-then-hard stop sequence used by the tray menu so the parent always holds the shutdown controls. Each shutdown stage emits debug-level entries under `third_party.social_robot.main` in `${data_root}/logs/ctrlspeak.log` (for example, `%APPDATA%\CtrlSpeak\logs\ctrlspeak.log` on Windows), so you can see exactly which component executed when diagnosing a stalled goodbye. All of these phrases are matched fuzzily, so punctuation or light speech-to-text substitutions (for example, “chat was assistant”) still trigger the expected behaviour unless a document explicitly opts out.

> **Do not** move the goodbye/chat keyword detection back into SocialRobot or another worker thread. Keeping the logic in the CtrlSpeak main process is a hard requirement so the parent can enforce the proven shutdown path. Route any future conversation controls through the same helpers described above and send explicit stdin commands to the bot only after the parent has taken ownership of the request.

When you add new keywords, update `tools/keywords.py`, refresh [`docs/tooling.md`](tooling.md), and adjust the system prompts for any identities that should advertise the new commands. The assistant prompt bundled with CtrlSpeak now explicitly mentions the clipboard trigger and instructs the model to guide users toward the exact phrases when they hint at wanting a capture.

## LangGraph orchestration and persistence

Setting `use_langgraph_memory_orchestrator: true` in `settings.json` (or exporting `CTRLSPK_USE_LANGGRAPH_MEMORY_ORCHESTRATOR=1`) routes every conversation turn through `utils.memory_orchestrator`:

1. **retrieve** – query Chroma for up to `retrieval_top_k` memories above `retrieval_threshold`; empty stores or low scores short-circuit.
2. **plan_tools** – reserved for future branching/tooling (currently a pass-through node).
3. **call_tools** – executes planned tools (no-ops today).
4. **llm** – calls the Ollama client with the existing history plus any retrieved memory preamble.
5. **persist** – enqueues atomic JSONL appends and Chroma upserts on a background worker so Kokoro playback can start immediately.

Persistence enforces the 10 MB/5-file JSONL rotation, 5 000-vector cap with LRU eviction, optional TTL, and the `pii_redaction` toggle before embedding. Per-turn traces (`run_<timestamp>_<correlation>.json`) and CSV metrics (`retrieval_hits`, `avg_similarity`, `persist_latency_ms`, `evictions`, `lock_wait_ms`) accumulate under `${data_root}/bot_memory/<identity>/traces` for post-mortems. If initialization fails (missing AppData, Chroma errors, LangGraph import issues), SocialRobot prints “LangGraph orchestrator failure; falling back to legacy conversation.”, logs the exception, and resumes the synchronous history writer.

Run `python main.py --health [--health-identity <name>]` to probe the same pipeline. The command verifies data/log root writability, acquires/releases the identity lock, initializes the Chroma collection, and reports embedder metadata. Failures return a non-zero exit code and a JSON payload explaining the failing check.

## SocialRobot entrypoint responsibilities

`third_party/social_robot/main.py` remains the authoritative entrypoint for the Chat with Bot workflow. CtrlSpeak launches it as a subprocess from `utils.bot_integration.start_bot`, passes the resolved identity settings and CtrlSpeak STT URL, and relies on its stdin control channel for graceful shutdowns.【F:utils/bot_integration.py†L667-L760】【F:third_party/social_robot/main.py†L251-L399】 The module bootstraps speech recognition, Ollama chat streaming, Kokoro playback, optional preprocessing, and the animated face/logo UI before wiring the stdin listener that handles `goodbye` commands issued by the parent process.【F:third_party/social_robot/main.py†L146-L532】 Removing or partially deleting this file will break bot startup and teardown, so trim functionality only when you can update every caller and test path accordingly.

## Background agents

The speech rewrite pass lives under `background_agents/tts_preprocessing_agent`. The directory mirrors an identity folder:

- `identity.json` declares the Gemma 3 1B model, its Ollama URL, any `ollama_options` to send with each request, and the prompt files to load.
- `header_text.txt` is prepended to the raw reply when `preamble` is set to `header` or `both`.
- `system_prompt.txt` holds the system prompt used when `preamble` is set to `system` or `both`.

`background_agents/tts_preprocessing_agent/background_agent.py` loads these files, constructs a non-streaming `OllamaClient`, and exposes `load_tts_preprocessing_agent()` for the main loop. The `preamble` key in `identity.json` accepts `header`, `system`, or `both` to control which assets are required—missing files for the chosen mode disable the helper so playback still succeeds. Both prompt assets direct the model to return the reply verbatim except for deleting `*`/`#`, and the Ollama options pin `temperature` to `0.0` to prevent creative rewrites. If the resources are missing or the helper raises `OllamaUnavailableError`, SocialRobot logs the failure and falls back to the original reply so the session keeps flowing.【F:background_agents/tts_preprocessing_agent/background_agent.py†L1-L199】【F:third_party/social_robot/main.py†L180-L271】

CtrlSpeak treats the agent as a first-class asset for identities that request it: `utils.bot_integration.start_bot` only stages the Gemma weights and pre-warms the checkpoint when the selected identity advertises `require_text_cleaning: true`, skipping the additional download and warm-up otherwise.【F:utils/bot_integration.py†L116-L214】【F:utils/bot_integration.py†L635-L790】 Packaged builds bundle `background_agents/` so the helper is present in single-file executables.【F:packaging/CtrlSpeak.spec†L42-L63】【F:packaging/CtrlSpeak_Watcher.spec†L42-L63】【F:utils/build_exe.py†L22-L82】

## Runtime Requirements

Using Chat with Bot requires:

1. CtrlSpeak running in **Client + Server** mode with the transcription server active.
2. The dependencies listed in the repository-level `requirements.txt` (pygame, Kokoro-ONNX, faster-whisper, etc.). Installing the root requirements is sufficient; no separate `third_party/social_robot/requirements.txt` file exists anymore.
3. Optional: a reachable Ollama server if you want real LLM responses. Without one, the bot echoes a graceful fallback derived from the user's input.

## Launching the Bot Manually

The bot can be exercised from the command line via `utils.bot_integration.run_bot_test`:

```powershell
py -3 run_test_script.py
```

This script starts the CtrlSpeak server with `--start-server-only`, waits for `/ping` to report healthy, streams the sample `assets/test_16k_mono.wav` through SocialRobot, prints the transcription and LLM response, and shuts everything down cleanly via `/kill`.

To integrate with your own automation, call:

```python
from utils.bot_integration import run_bot_test
run_bot_test("assets/test_16k_mono.wav", identity="default")
```

The helper reuses an existing server when you pass `stt_url=...`, and you can forward identity-specific overrides such as `prompt_file`, `system_prompt`, or `voice`.

## Environment Overrides

| Variable | Purpose |
| --- | --- |
| `BOT_IDENTITY` | Name of the identity directory to load (default `default`). |
| `BOT_IDENTITIES_DIR` | Override the base directory that holds identity folders. |
| `BOT_PROMPT_FILE` | Explicit prompt file path to use when launching SocialRobot. |
| `BOT_SYSTEM_PROMPT` | Inline system prompt string (used when not reading from a file). |
| `BOT_MEMORY_DIR` | Override the memory directory for the active identity. |
| `BOT_LLM_URL` | Ollama endpoint (default `http://localhost:11434/api/chat`). |
| `BOT_LLM_MODEL` | Ollama model name (default `gemma3:1b`). |
| `BOT_VOICE` | Kokoro voice (default `hm_omega`). |
| `CTRLSPEAK_STT_URL` | Force SocialRobot to target a specific CtrlSpeak server. |

Each option also has a matching CLI flag in `third_party/social_robot/main.py` (for example `--identity`, `--prompt-file`, `--system-prompt`, `--memory-dir`).

## Assistant Model Downloads

Launching Chat with Bot from the management window now verifies that the Ollama model required by the selected identity is available before SocialRobot starts. CtrlSpeak queries the Ollama API to see whether the model is already cached and, when it is missing, reuses the same welcome workflow that Whisper and CUDA downloads use: the intro video plays, fun facts rotate, and the lockout window reports live status with a cancel button. Once the download succeeds the window closes automatically and the bot process launches; cancellation or errors stop the launch and surface the failure in the lockout message so the user can try again after resolving the issue.【F:utils/bot_integration.py†L61-L213】【F:utils/models.py†L1308-L1484】

## Screenshot workflow

When you launch an identity with `vision: true` (for example the bundled **Assistant** persona) and say “look at my screen,” SocialRobot captures the current desktop. The helper now writes the PNG to `${data_root}/bot_memory/<identity>/screenshots/current.png` (along with a `metadata.json` descriptor) so each bot keeps only its most recent image on disk; new captures atomically replace the previous file instead of accumulating a gallery. Saying “look at my clipboard” (or choosing **Look at my Clipboard** from the floating logo) pulls the most recent image from the system clipboard via `tools/vision.py` and flows through the same single-image store. Keyword detection for both phrases lives in `tools/keywords.py`, which keeps the trigger vocabulary centralized as new commands are added. LangGraph inspects the user’s wording every turn: if the request references the stored image or the capture workflow just ran, the orchestrator automatically loads `current.png`, attaches the PNG to the Ollama request, and records only a file reference in the JSONL history. If a capture fails—because PyAutoGUI cannot access the display, the clipboard has no image, or dependencies are missing—the bot logs the issue and continues as a text-only exchange. Identities with `vision: false` skip these hooks entirely; the floating logo omits both context-menu options and voice commands fall back to a standard text-only exchange.

As soon as you pick an identity, CtrlSpeak now pings the configured Ollama endpoint with that profile’s model so the checkpoint is fully loaded before you speak. This avoids the first-turn lag that previously occurred while Ollama initialized the weights after receiving the initial utterance.

## Clearing stored memory

Use the **Clear Bot Memory** button in the management window when you need to wipe a persona’s stored context. After you choose an identity and confirm the prompt, CtrlSpeak removes that profile’s `${data_root}/bot_memory/<identity>/conversation/conversation.jsonl` log (plus any rotated archives) and deletes the `screenshots/`, `chroma/`, and `traces/` subdirectories so cached captures, vector embeddings, traces, and metrics are cleared. If nothing exists under the identity’s AppData tree, the dialog reports that there is nothing to clear.

## GUI Workflow

1. Start CtrlSpeak in Client + Server mode.
2. Right-click the tray icon, choose **Manage CtrlSpeak**, then use the Assistants card to review identity availability badges and click **Chat with Bot**.【F:utils/gui.py†L1424-L1455】
3. The management UI toggles the bot: click once to launch, again to stop. The active persona's badge switches to **active** while other identities remain marked **available**.【F:utils/gui.py†L1730-L1812】
4. Speak once the "-> Starting the VAD listener..." message appears in the terminal; responses are spoken back and logged to the console as `-> Bot replied: ...`.
5. Right-click the transparent logo to open its context menu. Choose **Look at my Screen** to capture the desktop or **Look at my Clipboard** to forward the latest snip stored in the clipboard. Both actions mirror the spoken commands and share the tooling documented in [`docs/tooling.md`](tooling.md). The menu still includes **Quit** when you need to close the bot quickly.

The SocialRobot process keeps running if you close the management window—you can reopen it later without interrupting the conversation. To shut the bot down, either click **Stop Chat with Bot** in the management window or choose **Quit** from the floating logo's context menu.

