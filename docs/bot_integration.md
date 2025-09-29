# Chat With Bot Integration

CtrlSpeak includes an optional "Chat with Bot" experience accessible from the management window. When enabled, it pairs the local transcription server with the vendored **SocialRobot** project to provide a voice conversation loop:

- **Speech to Text (STT)** - Uses CtrlSpeak's `/transcribe` endpoint. Audio captured by the VAD listener is sent to the running CtrlSpeak server (local or remote depending on mode). The server returns the recognized text.
- **Language Model (LLM)** - The recognized text is sent to the Ollama-compatible client inside SocialRobot. By default CtrlSpeak ships with a lightweight fallback response if no LLM endpoint is reachable, but you can supply your own by setting the `BOT_LLM_URL` and `BOT_LLM_MODEL` environment variables (or the matching CLI flags) before launching CtrlSpeak.
- **Text to Speech (TTS)** - The LLM response is converted to audio via Kokoro-ONNX. CtrlSpeak defaults to the formal male `am_michael` voice; override it with `BOT_VOICE` or the `--voice` flag.
- **Animated Face** - SocialRobot renders the bundled desktop face for visual feedback. Assets live under `third_party/social_robot/images/Desktop`.

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
  "voice": "am_michael",
  "memory_dir": "memory"
}
```

If `read_prompt_from_file` is `true`, the prompt file is read relative to the identity directory (unless an absolute path is provided). Omitting it falls back to the built-in default prompt.

You can switch identities from the command line with:

```powershell
python third_party/social_robot/main.py --identity ross
```

or by setting `BOT_IDENTITY=ross` before launching CtrlSpeak so the management window uses that persona.

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

## Screenshot workflow

When you launch the **Assistant** identity from the management UI and say “look at my screen,” SocialRobot now captures the current desktop, stores a PNG copy under the identity’s `memory/screenshots` directory, and forwards the encoded image to the configured Ollama model. The spoken request is automatically augmented with a clarification asking the model to describe the screenshot, so multimodal checkpoints such as `gemma3:12b` can respond with contextual commentary. If the capture fails (for example, when `pyautogui` cannot access the display), the bot logs the issue and continues as a text-only exchange.

As soon as you pick an identity, CtrlSpeak now pings the configured Ollama endpoint with that profile’s model so the checkpoint is fully loaded before you speak. This avoids the first-turn lag that previously occurred while Ollama initialized the weights after receiving the initial utterance.

## GUI Workflow

1. Start CtrlSpeak in Client + Server mode.
2. Right-click the tray icon, choose **Manage CtrlSpeak**, then click **Chat with Bot**.
3. The management UI toggles the bot: click once to launch, again to stop. The button text reflects the current state.
4. Speak once the "-> Starting the VAD listener..." message appears in the terminal; responses are spoken back and logged to the console as `-> Bot replied: ...`.

Stopping the management window automatically terminates SocialRobot.

