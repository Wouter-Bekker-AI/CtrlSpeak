# Chat With Bot Integration

CtrlSpeak includes an optional "Chat with Bot" experience accessible from the management window. When enabled, it pairs the local transcription server with the vendored **SocialRobot** project to provide a voice conversation loop:

- **Speech to Text (STT)** - Uses CtrlSpeak's `/transcribe` endpoint. Audio captured by the VAD listener is sent to the running CtrlSpeak server (local or remote depending on mode). The server returns the recognized text.
- **Language Model (LLM)** - The recognized text is sent to the Ollama-compatible client inside SocialRobot. By default CtrlSpeak ships with a lightweight fallback response if no LLM endpoint is reachable, but you can supply your own by setting the `BOT_LLM_URL` and `BOT_LLM_MODEL` environment variables (or the matching CLI flags) before launching CtrlSpeak.
- **Speech rewrite (background agent)** - The LLM reply is routed through `background_agents/tts_preprocessing_agent`, which consults its `identity.json` to determine whether to prepend `header_text.txt`, load `system_prompt.txt`, or use both before sending the request to the Gemma 3 1B preprocessing helper. The agent rewrites the reply for smoother narration before speech is generated.
- **Text to Speech (TTS)** - The LLM response is converted to audio via Kokoro-ONNX. CtrlSpeak defaults to the formal male `am_michael` voice; override it with `BOT_VOICE` or the `--voice` flag.
- **Animated Face / Logo** - SocialRobot renders the default TrueAI transparent logo with amplitude-based scaling for visual feedback. Identity folders can still supply alternate assets under `third_party/social_robot/identities/<name>` when a different look is desired.

## Identity Profiles

Personalities for the bot live under `third_party/social_robot/identities/<name>`. Each identity directory may contain:

- `identity.json` - configuration for the profile (voice, model, prompt settings, optional memory directory).
- `system_prompt.txt` (or another file referenced by the config) - the text injected as the LLM system prompt when the profile is loaded.
- `memory/` - reserved space for future long-term memory storage (per-identity state, LangChain vector stores, etc.).

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

The additional boolean keys control multimodal and future extensibility features:

- `vision` – Enables image capture tooling documented in [`docs/tooling.md`](tooling.md). When `true`, SocialRobot listens for the spoken “look at my screen” and “look at my clipboard” commands, exposes matching context-menu actions on the floating logo, and routes captured images to the LLM. When `false`, the commands are ignored, the context-menu items are hidden, and no images are taken.
- `tool` – Reserved flag for forthcoming external tool integrations. It defaults to `false` today but can be toggled once tool calling is implemented.

CtrlSpeak ships with two bundled identities: `assistant` (vision enabled) and `default` (vision disabled). Both currently set `tool` to `false` and can be expanded as the tool feature matures.

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

The push-to-talk workflow (hold the right Ctrl key while speaking) shares the same keyword registry. When VAD is idle because no bot session is active, saying “chat with assistant” through the hotkey launches that identity and skips text injection entirely. If another persona is already running, the helper first asks it to exit cooperatively via `utils.bot_integration.request_goodbye()` and falls back to `stop_bot()` only when the child process fails to exit within the timeout. Spoken “goodbye <identity>” commands are now intercepted in the CtrlSpeak main process (the transcription server), which issues the same graceful-then-hard stop sequence used by the tray menu so the parent always holds the shutdown controls. Each shutdown stage emits debug-level entries under `third_party.social_robot.main` in `%APPDATA%\CtrlSpeak\logs\ctrlspeak.log`, so you can see exactly which component executed when diagnosing a stalled goodbye.

> **Do not** move the goodbye/chat keyword detection back into SocialRobot or another worker thread. Keeping the logic in the CtrlSpeak main process is a hard requirement so the parent can enforce the proven shutdown path. Route any future conversation controls through the same helpers described above and send explicit stdin commands to the bot only after the parent has taken ownership of the request.

When you add new keywords, update `tools/keywords.py`, refresh [`docs/tooling.md`](tooling.md), and adjust the system prompts for any identities that should advertise the new commands. The assistant prompt bundled with CtrlSpeak now explicitly mentions the clipboard trigger and instructs the model to guide users toward the exact phrases when they hint at wanting a capture.

## Background agents

The speech rewrite pass lives under `background_agents/tts_preprocessing_agent`. The directory mirrors an identity folder:

- `identity.json` declares the Gemma 3 1B model, its Ollama URL, any `ollama_options` to send with each request, and the prompt files to load.
- `header_text.txt` is prepended to the raw reply when `preamble` is set to `header` or `both`.
- `system_prompt.txt` holds the system prompt used when `preamble` is set to `system` or `both`.

`background_agents/tts_preprocessing_agent/background_agent.py` loads these files, constructs a non-streaming `OllamaClient`, and exposes `load_tts_preprocessing_agent()` for the main loop. The `preamble` key in `identity.json` accepts `header`, `system`, or `both` to control which assets are required—missing files for the chosen mode disable the helper so playback still succeeds. If the resources are missing or the helper raises `OllamaUnavailableError`, SocialRobot logs the failure and falls back to the original reply so the session keeps flowing.【F:background_agents/tts_preprocessing_agent/background_agent.py†L1-L199】【F:third_party/social_robot/main.py†L180-L271】

CtrlSpeak treats the agent as a first-class asset: `utils.bot_integration.start_bot` stages the Gemma weights with the same welcome workflow used for identity models and pre-warms the checkpoint once it is available.【F:utils/bot_integration.py†L52-L226】【F:utils/bot_integration.py†L420-L637】 Packaged builds bundle `background_agents/` so the helper is present in single-file executables.【F:packaging/CtrlSpeak.spec†L42-L63】【F:packaging/CtrlSpeak_Watcher.spec†L42-L63】【F:utils/build_exe.py†L22-L82】

## Runtime Requirements

Using Chat with Bot requires:

1. CtrlSpeak running in **Client + Server** mode with the transcription server active.
2. The dependencies listed in `requirements.txt` (pygame, Kokoro-ONNX, faster-whisper, etc.). Installing CtrlSpeak's root requirements covers these.
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

When you launch an identity with `vision: true` (for example the bundled **Assistant** persona) and say “look at my screen,” SocialRobot captures the current desktop, stores a PNG copy under the identity’s `memory/screenshots` directory, and forwards the encoded image to the configured Ollama model. Saying “look at my clipboard” (or choosing **Look at my Clipboard** from the floating logo) pulls the most recent image from the system clipboard via `tools/vision.py` and follows the same storage and upload workflow. Keyword detection for both phrases lives in `tools/keywords.py`, which keeps the trigger vocabulary centralized as new commands are added. Both routes annotate history entries with the saved file path and the capture source so downstream agents can tell whether the data came from a screenshot or a clipboard snip. If a capture fails—because PyAutoGUI cannot access the display, the clipboard has no image, or dependencies are missing—the bot logs the issue and continues as a text-only exchange. Identities with `vision: false` skip these hooks entirely; the floating logo omits both context-menu options and voice commands fall back to a standard text-only exchange.

As soon as you pick an identity, CtrlSpeak now pings the configured Ollama endpoint with that profile’s model so the checkpoint is fully loaded before you speak. This avoids the first-turn lag that previously occurred while Ollama initialized the weights after receiving the initial utterance.

## Clearing stored memory

Use the **Clear Bot Memory** button in the management window when you need to wipe a persona’s stored context. After you choose an identity and confirm the prompt, CtrlSpeak removes that profile’s `memory/conversation.json` file and deletes the entire `memory/screenshots` directory so no cached captures remain. If neither artifact exists, the dialog reports that there is nothing to clear.

## GUI Workflow

1. Start CtrlSpeak in Client + Server mode.
2. Right-click the tray icon, choose **Manage CtrlSpeak**, then use the Assistants card to review identity availability badges and click **Chat with Bot**.【F:utils/gui.py†L1424-L1455】
3. The management UI toggles the bot: click once to launch, again to stop. The active persona's badge switches to **active** while other identities remain marked **available**.【F:utils/gui.py†L1730-L1812】
4. Speak once the "-> Starting the VAD listener..." message appears in the terminal; responses are spoken back and logged to the console as `-> Bot replied: ...`.
5. Right-click the transparent logo to open its context menu. Choose **Look at my Screen** to capture the desktop or **Look at my Clipboard** to forward the latest snip stored in the clipboard. Both actions mirror the spoken commands and share the tooling documented in [`docs/tooling.md`](tooling.md). The menu still includes **Quit** when you need to close the bot quickly.

The SocialRobot process keeps running if you close the management window—you can reopen it later without interrupting the conversation. To shut the bot down, either click **Stop Chat with Bot** in the management window or choose **Quit** from the floating logo's context menu.

